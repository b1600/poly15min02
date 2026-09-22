from types import SimpleNamespace

from poly15m.config import Settings
from poly15m.data.clob_ws import OrderBook
from poly15m.db import Database
from poly15m.paper_trade import PaperTrader
from poly15m.positions.manager import PositionManager
from poly15m.risk.limits import RiskGate
from poly15m.sim.paper import PaperExecutor
from poly15m.signals.features import FeatureSnapshot


def make_trader(open_price: float, close_price: float | None):
    db = Database(":memory:")
    settings = Settings()
    executor = PaperExecutor(db, settings)
    position_manager = PositionManager(settings)
    risk_gate = RiskGate(settings, db)
    tracker = SimpleNamespace(open_price={"cond1": open_price})
    binance_feed = SimpleNamespace(last_price=close_price)
    trader = PaperTrader(db, binance_feed, None, tracker, None, executor, position_manager, risk_gate)
    return trader, db, executor, position_manager


def test_official_resolution_pays_the_winning_side():
    trader, db, executor, _ = make_trader(open_price=100.0, close_price=100.0)
    executor.submit_order("cond1", "tok_up", "up", [(0.5, 100.0)], target_size=10.0, limit_price=0.9, ts=1.0)

    trader.on_window_closed("cond1")
    trader.on_official_resolution("cond1", "up")

    assert executor.realized_pnl > 0  # bought the winning side cheap


def test_official_resolution_overrides_a_disagreeing_proxy():
    """The regression that inflated the Sep 12-22 2026 dry run.

    The Binance proxy says Up (close >= open) while Polymarket settled
    Down. Grading must follow settlement and book the loss, not the proxy
    -- proxy-graded windows were 9% of the sample and 72% of the
    "profit".
    """
    trader, db, executor, _ = make_trader(open_price=100.0, close_price=101.0)
    executor.submit_order("cond1", "tok_up", "up", [(0.5, 100.0)], target_size=10.0, limit_price=0.9, ts=1.0)

    trader.on_window_closed("cond1")
    assert trader._proxy_outcome["cond1"] == "up"

    trader.on_official_resolution("cond1", "down")

    assert executor.realized_pnl < 0
    assert "cond1" not in executor.positions


def test_window_close_does_not_realize_pnl():
    """Closing a window must not grade it -- settlement comes later."""
    trader, db, executor, _ = make_trader(open_price=100.0, close_price=101.0)
    executor.submit_order("cond1", "tok_up", "up", [(0.5, 100.0)], target_size=10.0, limit_price=0.9, ts=1.0)

    trader.on_window_closed("cond1")

    assert "cond1" in executor.positions
    assert executor.realized_pnl == 0.0
    assert db._conn.execute("SELECT COUNT(*) FROM paper_pnl_log").fetchone()[0] == 0


def test_abandoned_window_strands_cost_basis_instead_of_guessing():
    trader, db, executor, _ = make_trader(open_price=100.0, close_price=101.0)
    executor.submit_order("cond1", "tok_up", "up", [(0.5, 100.0)], target_size=10.0, limit_price=0.9, ts=1.0)

    trader.on_window_closed("cond1")
    trader.on_resolution_abandoned("cond1")

    assert "cond1" not in executor.positions
    assert executor.realized_pnl == 0.0  # never guessed
    assert executor.abandoned_cost_basis > 0
    assert executor.abandoned_windows == 1


def test_window_close_drops_position_manager_inventory():
    trader, db, executor, position_manager = make_trader(open_price=100.0, close_price=101.0)
    position_manager.record_fill("cond1", "up", 0.5, 10.0, 0.0)

    trader.on_window_closed("cond1")

    assert "cond1" not in position_manager.inventory


def _build_tick_test_trader():
    """A trader wired with real OrderBook/PaperExecutor/PositionManager but
    fake tracker/feature_engine, for testing on_binance_tick's runner-level
    logic (the book-staleness gate) in isolation from network/live feeds."""
    db = Database(":memory:")
    # generous position caps -- this test isolates the book-staleness gate,
    # not PositionManager's own position-cap behavior (covered separately
    # in test_position_manager.py). The notional cap has to be lifted too:
    # at the default $20/market a single Kelly-sized fill exhausts it, so
    # the second fill would be blocked by the cap rather than by the gate
    # under test.
    settings = Settings(max_inventory_imbalance=1000.0, max_notional_per_market=1000.0)
    executor = PaperExecutor(db, settings)
    position_manager = PositionManager(settings)
    risk_gate = RiskGate(settings, db)

    book_up = OrderBook(token_id="tok_up")
    book_up.apply_snapshot(bids=[], asks=[{"price": "0.5", "size": "1000"}], event_ts=100.0)
    book_down = OrderBook(token_id="tok_down")
    book_down.apply_snapshot(bids=[], asks=[], event_ts=100.0)
    clob_feed = SimpleNamespace(
        books={"tok_up": book_up, "tok_down": book_down}, last_msg_age=lambda now=None: 0.0
    )

    market = SimpleNamespace(condition_id="cond1", token_id_up="tok_up", token_id_down="tok_down", close_ts=10300.0)
    tracker = SimpleNamespace(latest_market=lambda: market, open_price={"cond1": 100.0})
    binance_feed = SimpleNamespace(last_price=100.0, last_trade_age=lambda now=None: 0.0)

    # deviation=1.0 -> P(Up)=Phi(1.0)~=0.84, comfortably clears min_edge_to_trade against a 0.5 ask
    snapshot = FeatureSnapshot(
        condition_id="cond1", ts=1.0, spot=100.0, open_price=100.0, t_remaining=300.0, sigma=1.0,
        deviation=1.0, momentum_1m=None, momentum_3m=None, momentum_5m=None,
        book_imbalance_up=None, aggressive_flow_up=None,
    )
    feature_engine = SimpleNamespace(compute=lambda *a, **k: snapshot)

    trader = PaperTrader(db, binance_feed, clob_feed, tracker, feature_engine, executor, position_manager, risk_gate)
    return trader, executor, book_up


def test_on_binance_tick_logs_fair_value_for_calibration():
    trader, executor, _ = _build_tick_test_trader()
    trader.on_binance_tick(0.0, 0.0)
    rows = trader.db._conn.execute("SELECT deviation, p_up_fair FROM fair_value_log").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 1.0  # matches the fixture's deviation
    assert rows[0][1] is not None


def test_on_binance_tick_does_not_refill_against_unchanged_books():
    trader, executor, book_up = _build_tick_test_trader()

    trader.on_binance_tick(0.0, 0.0)
    first_size = executor.position_size("cond1", "up")
    assert first_size > 0  # sanity: the constructed edge does clear the trade threshold

    trader._last_decision_ts["cond1"] = 0.0  # bypass the decision throttle to isolate the book-staleness gate
    trader.on_binance_tick(0.0, 0.0)
    assert executor.position_size("cond1", "up") == first_size  # no new fill -- books haven't changed

    book_up.apply_price_change([{"price": "0.5", "size": "500", "side": "SELL"}], event_ts=101.0)
    trader._last_decision_ts["cond1"] = 0.0
    trader.on_binance_tick(0.0, 0.0)
    assert executor.position_size("cond1", "up") > first_size


def test_on_binance_tick_blocked_by_kill_switch():
    trader, executor, _ = _build_tick_test_trader()
    trader.risk_gate.trigger_kill_switch("test")

    trader.on_binance_tick(0.0, 0.0)

    assert executor.position_size("cond1", "up") == 0.0


def test_on_binance_tick_blocked_by_stale_feeds():
    trader, executor, _ = _build_tick_test_trader()
    trader.binance_feed.last_trade_age = lambda now=None: 9999.0

    trader.on_binance_tick(0.0, 0.0)

    assert executor.position_size("cond1", "up") == 0.0


def test_pnl_log_written_on_resolution():
    trader, db, executor, _ = make_trader(open_price=100.0, close_price=101.0)
    executor.submit_order("cond1", "tok_up", "up", [(0.5, 100.0)], target_size=10.0, limit_price=0.9, ts=1.0)

    trader.on_window_closed("cond1")
    trader.on_official_resolution("cond1", "up")

    rows = db._conn.execute("SELECT event, realized_pnl FROM paper_pnl_log").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "resolution"
