import math

from poly15m.config import Settings
from poly15m.db import Database
from poly15m.sim.paper import PaperExecutor


def make_executor():
    db = Database(":memory:")
    settings = Settings()
    return PaperExecutor(db, settings), settings


def test_submit_order_applies_fill_ratio_haircut_and_fees():
    executor, settings = make_executor()
    ask_levels = [(0.5, 100.0)]
    fill = executor.submit_order("cond1", "tok_up", "up", ask_levels, target_size=20.0, limit_price=0.9, ts=1000.0)

    assert fill is not None
    expected_filled = 20.0 * settings.sim_fill_ratio
    assert math.isclose(fill.size, expected_filled)
    assert fill.price == 0.5
    expected_fee = (0.5 * expected_filled) * (settings.taker_fee_bps / 10000.0)
    assert math.isclose(fill.fee, expected_fee)
    assert executor.position_size("cond1", "up") == fill.size
    assert math.isclose(executor.fees_paid, expected_fee)


def test_submit_order_none_when_nothing_fillable():
    executor, _ = make_executor()
    assert executor.submit_order("cond1", "tok_up", "up", [], target_size=20.0, limit_price=0.9, ts=1000.0) is None


def test_resolve_market_winning_side_realizes_positive_pnl():
    executor, settings = make_executor()
    ask_levels = [(0.5, 100.0)]
    fill = executor.submit_order("cond1", "tok_up", "up", ask_levels, target_size=20.0, limit_price=0.9, ts=1000.0)

    resolution = executor.resolve_market("cond1", "up")

    assert resolution is not None
    assert resolution.payout == fill.size  # $1/share for the winning side
    assert math.isclose(resolution.cost_basis, fill.price * fill.size + fill.fee)
    assert math.isclose(resolution.pnl, resolution.payout - resolution.cost_basis)
    assert resolution.pnl > 0  # bought at 0.5, paid out at 1.0, minus a small fee
    assert math.isclose(executor.realized_pnl, resolution.pnl)


def test_resolve_market_losing_side_realizes_negative_pnl_equal_to_cost():
    executor, _ = make_executor()
    ask_levels = [(0.5, 100.0)]
    fill = executor.submit_order("cond1", "tok_down", "down", ask_levels, target_size=20.0, limit_price=0.9, ts=1000.0)

    resolution = executor.resolve_market("cond1", "up")  # down side loses

    assert resolution.payout == 0.0
    assert math.isclose(resolution.pnl, -(fill.price * fill.size + fill.fee))


def test_resolve_market_drops_position_afterward():
    executor, _ = make_executor()
    executor.submit_order("cond1", "tok_up", "up", [(0.5, 100.0)], target_size=20.0, limit_price=0.9, ts=1000.0)
    executor.resolve_market("cond1", "up")
    assert "cond1" not in executor.positions
    assert executor.position_size("cond1", "up") == 0.0


def test_resolve_market_none_for_unknown_condition():
    executor, _ = make_executor()
    assert executor.resolve_market("nonexistent", "up") is None


def test_both_sides_netted_at_resolution():
    executor, _ = make_executor()
    # buy both legs (hedged) -- winning leg pays out, losing leg doesn't
    up_fill = executor.submit_order("cond1", "tok_up", "up", [(0.4, 100.0)], target_size=10.0, limit_price=0.9, ts=1000.0)
    down_fill = executor.submit_order("cond1", "tok_down", "down", [(0.4, 100.0)], target_size=10.0, limit_price=0.9, ts=1000.0)

    resolution = executor.resolve_market("cond1", "up")

    total_cost = (up_fill.price * up_fill.size + up_fill.fee) + (down_fill.price * down_fill.size + down_fill.fee)
    assert math.isclose(resolution.cost_basis, total_cost)
    assert resolution.payout == up_fill.size
