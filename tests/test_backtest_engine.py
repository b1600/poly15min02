import math

from poly15m.backtest.engine import run_backtest
from poly15m.config import Settings
from poly15m.db import Database

WINDOW_SECONDS = 900.0
LOOKBACK = 90.0


def build_fixture_db(db_path, condition_id="cond1", open_ts=100000.0, trend_per_sec=0.05, up_ask=0.4, down_ask=0.6):
    """A market that trends steadily in one direction through the window,
    with a persistently cheap ask on the winning side -- engineered to
    produce an obvious, sustained divergence rather than to model realistic
    microstructure."""
    close_ts = open_ts + WINDOW_SECONDS
    db = Database(db_path)
    token_up, token_down = f"{condition_id}-up", f"{condition_id}-down"
    db.upsert_market(
        {
            "condition_id": condition_id,
            "slug": f"slug-{condition_id}",
            "question_id": None,
            "token_id_up": token_up,
            "token_id_down": token_down,
            "window_open_ts": open_ts,
            "window_close_ts": close_ts,
            "discovered_ts": open_ts,
            "raw_json": "{}",
        }
    )
    db.set_market_open_price(condition_id, 100.0, "test")

    price = 100.0
    t = open_ts - LOOKBACK
    while t < open_ts:  # flat lookback period so realized vol reflects the trend, not pre-window noise
        db.insert_tick("binance", "BTCUSDT", price, 1.0, t)
        t += 1.0

    final_price = price
    t = open_ts
    while t < close_ts:
        db.insert_tick("binance", "BTCUSDT", price, 1.0, t)
        final_price = price
        price += trend_per_sec
        t += 1.0

    t = open_ts
    while t < close_ts:
        db.insert_book_snapshot(condition_id, token_up, bids=[(up_ask - 0.02, 500.0)], asks=[(up_ask, 500.0)], event_ts=t)
        db.insert_book_snapshot(condition_id, token_down, bids=[(down_ask - 0.02, 500.0)], asks=[(down_ask, 500.0)], event_ts=t)
        t += 5.0

    db.close()
    return condition_id, open_ts, close_ts, final_price


def make_settings(**overrides):
    defaults = dict(min_edge_to_trade=0.01, paper_trade_size=10.0, paper_min_order_size=1.0)
    defaults.update(overrides)
    return Settings(**defaults)


def test_backtest_trending_up_market_is_profitable(tmp_path):
    db_path = tmp_path / "fixture.db"
    condition_id, open_ts, close_ts, final_price = build_fixture_db(db_path, trend_per_sec=0.05)
    assert final_price > 100.0

    settings = make_settings()
    engine, result = run_backtest(db_path, [condition_id], settings)

    assert result.num_windows_replayed == 1
    row = engine.db._conn.execute(
        "SELECT resolved_outcome FROM markets WHERE condition_id = ?", (condition_id,)
    ).fetchone()
    assert row[0] == "up"
    assert result.realized_pnl > 0  # bought a persistently cheap winning side


def test_backtest_trending_down_market_resolves_down(tmp_path):
    db_path = tmp_path / "fixture.db"
    condition_id, open_ts, close_ts, final_price = build_fixture_db(
        db_path, trend_per_sec=-0.05, up_ask=0.6, down_ask=0.4
    )
    assert final_price < 100.0

    settings = make_settings()
    engine, result = run_backtest(db_path, [condition_id], settings)

    row = engine.db._conn.execute(
        "SELECT resolved_outcome FROM markets WHERE condition_id = ?", (condition_id,)
    ).fetchone()
    assert row[0] == "down"
    assert result.realized_pnl > 0  # bought the (correctly cheap) down side


def test_backtest_records_fills_and_pnl_log(tmp_path):
    db_path = tmp_path / "fixture.db"
    condition_id, *_ = build_fixture_db(db_path)
    settings = make_settings()
    engine, result = run_backtest(db_path, [condition_id], settings)

    fills = engine.db._conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
    assert fills > 0
    pnl_rows = engine.db._conn.execute(
        "SELECT COUNT(*) FROM paper_pnl_log WHERE event = 'resolution'"
    ).fetchone()[0]
    assert pnl_rows == 1


def test_backtest_two_consecutive_windows_accumulate_state(tmp_path):
    db_path = tmp_path / "fixture.db"
    c1, open1, close1, _ = build_fixture_db(db_path, condition_id="condA", open_ts=100000.0, trend_per_sec=0.05)
    c2, open2, close2, _ = build_fixture_db(db_path, condition_id="condB", open_ts=close1, trend_per_sec=0.05)
    assert open2 == close1  # back-to-back windows, no gap

    settings = make_settings()
    engine, result = run_backtest(db_path, [c1, c2], settings)

    assert result.num_windows_replayed == 2
    assert set(result.condition_ids) == {c1, c2}
    resolved = dict(
        engine.db._conn.execute("SELECT condition_id, resolved_outcome FROM markets ORDER BY condition_id").fetchall()
    )
    assert resolved == {"condA": "up", "condB": "up"}
    # both windows profitable -> cumulative realized pnl reflects both
    assert result.realized_pnl > 0


def test_backtest_returns_empty_result_for_unknown_market(tmp_path):
    db_path = tmp_path / "fixture.db"
    build_fixture_db(db_path)  # unrelated market recorded, but we ask for a different id
    settings = make_settings()
    engine, result = run_backtest(db_path, ["nonexistent"], settings)
    assert result.num_windows_replayed == 0
    assert result.condition_ids == []
