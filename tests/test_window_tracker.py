"""Open-price anchoring.

Regression cover for the defect that invalidated the Sep 12-22 2026 dry
run: `open_price` was sampled at market-discovery time (~7.5s after the
window opened, given a 15s discovery poll) instead of at the window's own
open timestamp, giving a ~$10 mean absolute error against a ~$99 mean
absolute 15m move -- enough to mislabel the ~7% of windows that move less
than the error.
"""

from types import SimpleNamespace

from poly15m.config import Settings
from poly15m.data.market_finder import MarketInfo
from poly15m.data.window_tracker import OPEN_PRICE_SOURCE, WindowTracker
from poly15m.db import Database


def make_market(condition_id="cond1", open_ts=1000.0):
    return MarketInfo(
        condition_id=condition_id,
        slug=f"slug-{condition_id}",
        question_id=None,
        token_id_up="tok_up",
        token_id_down="tok_down",
        open_ts=open_ts,
        close_ts=open_ts + 900.0,
    )


def make_tracker(buffer, settings=None):
    db = Database(":memory:")
    binance = SimpleNamespace(
        buffer=list(buffer),
        price_at=lambda ts: next(((t, p) for t, p in reversed(list(buffer)) if t <= ts), None),
    )
    clob = SimpleNamespace(subscribe=lambda *a, **k: None)
    tracker = WindowTracker(db, binance, clob, cfg=settings or Settings())
    return tracker, db


def test_anchors_to_trade_at_window_open_not_at_discovery():
    # Price moved $40 between window open and discovery. The old behavior
    # recorded 100040.0; the anchor must be the 100000.0 trade at open.
    tracker, db = make_tracker([(995.0, 100000.0), (1007.5, 100040.0)])
    tracker.on_new_market(make_market())

    tracker.poll(now=1007.5)  # discovery-time poll, 7.5s after open

    assert tracker.open_price["cond1"] == 100000.0
    row = db._conn.execute(
        "SELECT open_price_source, open_price_ts FROM markets WHERE condition_id = 'cond1'"
    ).fetchone()
    # no market row was upserted in this unit test, so only the in-memory
    # value is asserted above; the DB write is covered below.
    assert row is None


def test_persists_anchor_provenance():
    tracker, db = make_tracker([(995.0, 100000.0)])
    market = make_market()
    db.upsert_market(
        {
            "condition_id": market.condition_id,
            "slug": market.slug,
            "question_id": None,
            "token_id_up": market.token_id_up,
            "token_id_down": market.token_id_down,
            "window_open_ts": market.open_ts,
            "window_close_ts": market.close_ts,
            "discovered_ts": 1007.5,
            "raw_json": "{}",
        }
    )
    tracker.on_new_market(market)

    tracker.poll(now=1007.5)

    price, source, anchor_ts = db._conn.execute(
        "SELECT open_price, open_price_source, open_price_ts FROM markets WHERE condition_id = 'cond1'"
    ).fetchone()
    assert price == 100000.0
    assert source == OPEN_PRICE_SOURCE
    assert anchor_ts == 995.0


def test_waits_until_window_opens_before_anchoring():
    """A market discovered while still upcoming must not be anchored to a
    pre-open trade."""
    tracker, _ = make_tracker([(900.0, 99000.0)])
    tracker.on_new_market(make_market(open_ts=1000.0))

    tracker.poll(now=950.0)  # still before open

    assert "cond1" not in tracker.open_price
    assert "cond1" in tracker._needs_open_price


def test_leaves_window_untraded_when_anchor_is_too_stale():
    """No trade near the open means no trustworthy reference. Better an
    untraded window than one traded against a guess."""
    settings = Settings(open_price_max_anchor_lag_seconds=5.0, open_price_anchor_deadline_seconds=60.0)
    tracker, _ = make_tracker([(900.0, 99000.0)], settings)  # 100s before open
    tracker.on_new_market(make_market(open_ts=1000.0))

    tracker.poll(now=1005.0)
    assert "cond1" not in tracker.open_price  # still retrying

    tracker.poll(now=1061.0)  # past the deadline
    assert "cond1" not in tracker.open_price
    assert "cond1" in tracker.unanchored
    assert "cond1" not in tracker._needs_open_price  # gave up, stopped retrying


def test_retries_until_a_usable_anchor_arrives():
    buffer = []
    tracker, _ = make_tracker(buffer)
    tracker.on_new_market(make_market(open_ts=1000.0))

    tracker.poll(now=1001.0)  # buffer empty -- nothing to anchor to yet
    assert "cond1" not in tracker.open_price

    buffer.append((999.0, 100000.0))  # feed catches up
    tracker.poll(now=1002.0)
    assert tracker.open_price["cond1"] == 100000.0


def test_falls_back_to_recorded_ticks_when_buffer_predates_the_window():
    """A process that starts mid-window has no buffered trade at the open,
    but a previous run's recorded ticks may still cover it."""
    db = Database(":memory:")
    db.insert_tick("binance", "BTCUSDT", 100000.0, 1.0, 998.0, False)
    binance = SimpleNamespace(buffer=[], price_at=lambda ts: None)
    clob = SimpleNamespace(subscribe=lambda *a, **k: None)
    tracker = WindowTracker(db, binance, clob, cfg=Settings())
    tracker.on_new_market(make_market(open_ts=1000.0))

    tracker.poll(now=1005.0)

    assert tracker.open_price["cond1"] == 100000.0


def test_close_price_ignores_ticks_after_window_close():
    # The 2026-09-25 07:00 window: +$4.44 at close, but a post-close tick
    # read ~1s later by the clock poll flipped the proxy to "down".
    tracker, _ = make_tracker([(1899.7, 100004.44), (1900.8, 99990.0)])
    tracker.on_new_market(make_market())  # close_ts = 1900.0

    assert tracker.close_price("cond1") == 100004.44


def test_close_price_is_none_when_last_tick_is_too_stale():
    tracker, _ = make_tracker([(1880.0, 100000.0)])
    tracker.on_new_market(make_market())

    assert tracker.close_price("cond1") is None
