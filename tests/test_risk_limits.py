import math
import time
from datetime import datetime, timedelta, timezone

from poly15m.config import Settings
from poly15m.db import Database
from poly15m.positions.manager import PositionManager, TradeIntent
from poly15m.risk.limits import RiskGate


def make_gate(**settings_kwargs):
    db = Database(":memory:")
    settings = Settings(**settings_kwargs)
    return RiskGate(settings, db), PositionManager(settings), db


def make_intent(reason="directional_kelly", outcome="up", size=10.0, price=0.5, condition_id="cond1"):
    return TradeIntent(condition_id, f"tok_{outcome}", outcome, size, price, reason, edge=0.1)


def test_normal_intent_passes_through_unchanged():
    gate, pm, _ = make_gate()
    intent = make_intent(size=10.0, price=0.5)
    result = gate.check_intent(intent, pm, t_remaining=500.0, deviation=1.0)
    assert result == intent


def test_kill_switch_triggers_on_daily_loss_limit_breach():
    gate, pm, _ = make_gate(daily_loss_limit=25.0)
    gate.record_realized_pnl(-30.0)
    assert gate.kill_switch_active is True

    intent = make_intent()
    assert gate.check_intent(intent, pm, t_remaining=500.0, deviation=1.0) is None


def test_kill_switch_does_not_clear_on_subsequent_gain():
    gate, pm, _ = make_gate(daily_loss_limit=25.0)
    gate.record_realized_pnl(-30.0)
    gate.record_realized_pnl(+100.0)
    assert gate.kill_switch_active is True


def test_kill_switch_persists_across_day_rollover():
    gate, pm, _ = make_gate(daily_loss_limit=25.0)
    day1 = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    day2 = datetime(2026, 1, 2, tzinfo=timezone.utc).timestamp()

    gate.record_realized_pnl(-30.0, ts=day1)
    assert gate.kill_switch_active is True

    gate.record_realized_pnl(1.0, ts=day2)  # a new day's trading -- kill switch must still hold
    assert gate.kill_switch_active is True


def test_daily_pnl_resets_on_day_rollover():
    gate, pm, _ = make_gate(daily_loss_limit=1000.0)  # high enough to not trigger
    day1 = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    day2 = datetime(2026, 1, 2, tzinfo=timezone.utc).timestamp()

    gate.record_realized_pnl(-10.0, ts=day1)
    assert gate.daily_pnl == -10.0

    gate.record_realized_pnl(-5.0, ts=day2)
    assert gate.daily_pnl == -5.0  # reset, not -15


def test_end_of_window_blocks_directional_with_low_conviction():
    gate, pm, _ = make_gate()
    intent = make_intent(reason="directional_kelly")
    result = gate.check_intent(intent, pm, t_remaining=60.0, deviation=1.0)  # inside final 2m, weak signal
    assert result is None


def test_end_of_window_allows_tail_capped_bet_with_many_sigma():
    gate, pm, _ = make_gate()
    intent = make_intent(reason="directional_kelly", size=50.0)
    result = gate.check_intent(intent, pm, t_remaining=60.0, deviation=4.0)
    assert result is not None
    assert result.size == gate.settings.near_resolution_max_size


def test_end_of_window_does_not_block_matched_arb_or_hedge():
    gate, pm, _ = make_gate()
    arb_intent = make_intent(reason="matched_arb")
    hedge_intent = make_intent(reason="temporal_hedge")
    assert gate.check_intent(arb_intent, pm, t_remaining=10.0, deviation=None) is not None
    assert gate.check_intent(hedge_intent, pm, t_remaining=10.0, deviation=None) is not None


def test_notional_cap_scales_down_oversized_intent():
    gate, pm, _ = make_gate(
        max_notional_per_market=20.0, max_inventory_imbalance=1000.0, max_net_directional_exposure=1000.0
    )
    intent = make_intent(size=100.0, price=0.5)  # notional 50, cap 20
    result = gate.check_intent(intent, pm, t_remaining=500.0, deviation=1.0)
    assert result is not None
    assert math.isclose(result.size, 40.0)  # 20 / 0.5
    assert math.isclose(result.size * result.limit_price, 20.0)


def test_notional_cap_rejects_once_exhausted():
    gate, pm, _ = make_gate(
        max_notional_per_market=10.0, max_inventory_imbalance=1000.0, max_net_directional_exposure=1000.0
    )
    first = make_intent(size=20.0, price=0.5)  # notional 10 -- exactly the cap
    assert gate.check_intent(first, pm, t_remaining=500.0, deviation=1.0) is not None

    second = make_intent(size=20.0, price=0.5)
    assert gate.check_intent(second, pm, t_remaining=500.0, deviation=1.0) is None


def test_imbalance_cap_rejects_when_projected_imbalance_too_large():
    gate, pm, _ = make_gate(max_inventory_imbalance=20.0)
    pm.record_fill("cond1", "up", 0.5, 15.0, 0.0)  # already 15 shares net long up
    intent = make_intent(outcome="up", size=10.0, price=0.5)  # would push to 25 > 20
    assert gate.check_intent(intent, pm, t_remaining=500.0, deviation=1.0) is None


def test_imbalance_cap_allows_when_within_bounds():
    gate, pm, _ = make_gate(max_inventory_imbalance=20.0)
    pm.record_fill("cond1", "up", 0.5, 5.0, 0.0)
    intent = make_intent(outcome="up", size=10.0, price=0.5)  # -> 15, within 20
    assert gate.check_intent(intent, pm, t_remaining=500.0, deviation=1.0) is not None


def test_portfolio_cap_rejects_second_market_when_aggregate_exceeds_limit():
    gate, pm, _ = make_gate(max_inventory_imbalance=20.0, max_net_directional_exposure=25.0)
    pm.record_fill("cond1", "up", 0.5, 18.0, 0.0)  # market 1: 18 shares net directional

    intent = make_intent(condition_id="cond2", outcome="up", size=10.0, price=0.5)  # would bring total to 28 > 25
    assert gate.check_intent(intent, pm, t_remaining=500.0, deviation=1.0) is None


def test_portfolio_cap_allows_when_aggregate_within_limit():
    gate, pm, _ = make_gate(max_inventory_imbalance=20.0, max_net_directional_exposure=100.0)
    pm.record_fill("cond1", "up", 0.5, 18.0, 0.0)

    intent = make_intent(condition_id="cond2", outcome="up", size=10.0, price=0.5)
    assert gate.check_intent(intent, pm, t_remaining=500.0, deviation=1.0) is not None


def test_feeds_stale_true_when_either_feed_too_old():
    gate, _, _ = make_gate(feed_staleness_seconds=5.0)
    assert gate.feeds_stale(binance_age=10.0, clob_age=1.0) is True
    assert gate.feeds_stale(binance_age=1.0, clob_age=10.0) is True


def test_feeds_stale_false_when_both_fresh_or_unknown():
    gate, _, _ = make_gate(feed_staleness_seconds=5.0)
    assert gate.feeds_stale(binance_age=1.0, clob_age=1.0) is False
    assert gate.feeds_stale(binance_age=None, clob_age=None) is False


def test_drop_market_resets_notional_tracking():
    gate, pm, _ = make_gate(
        max_notional_per_market=10.0, max_inventory_imbalance=1000.0, max_net_directional_exposure=1000.0
    )
    first = make_intent(size=20.0, price=0.5)  # exhausts the cap
    gate.check_intent(first, pm, t_remaining=500.0, deviation=1.0)

    gate.drop_market("cond1")

    second = make_intent(size=20.0, price=0.5)
    assert gate.check_intent(second, pm, t_remaining=500.0, deviation=1.0) is not None


def test_kill_switch_writes_lifecycle_event():
    gate, pm, db = make_gate(daily_loss_limit=25.0)
    gate.record_realized_pnl(-30.0)
    rows = db._conn.execute(
        "SELECT condition_id, event FROM lifecycle_events WHERE event = 'kill_switch_triggered'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "GLOBAL"
