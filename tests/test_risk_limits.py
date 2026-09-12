import math
import time

import pytest
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


DAY1 = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
DAY2 = datetime(2026, 1, 2, tzinfo=timezone.utc).timestamp()
DAY3 = datetime(2026, 1, 3, tzinfo=timezone.utc).timestamp()
DAY5 = datetime(2026, 1, 5, tzinfo=timezone.utc).timestamp()


def test_kill_switch_persists_across_day_rollover_without_auto_rearm():
    gate, pm, _ = make_gate(daily_loss_limit=25.0, kill_switch_auto_rearm=False)

    gate.record_realized_pnl(-30.0, ts=DAY1)
    assert gate.kill_switch_active is True

    gate.record_realized_pnl(1.0, ts=DAY2)  # a new day's trading -- must still hold
    assert gate.kill_switch_active is True


def test_kill_switch_rearms_on_new_utc_day():
    gate, pm, _ = make_gate(daily_loss_limit=25.0)
    gate.record_realized_pnl(-30.0, ts=DAY1)
    assert gate.kill_switch_active is True

    assert gate.check_intent(make_intent(), pm, 500.0, 1.0, ts=DAY1) is None
    assert gate.check_intent(make_intent(), pm, 500.0, 1.0, ts=DAY2) is not None
    assert gate.kill_switch_active is False


def test_rearm_clears_yesterdays_daily_pnl():
    """Otherwise the open-risk brake re-blocks everything on day two,
    using a budget that was already spent on day one."""
    gate, pm, _ = make_gate(daily_loss_limit=25.0)
    gate.record_realized_pnl(-30.0, ts=DAY1)
    gate.check_intent(make_intent(), pm, 500.0, 1.0, ts=DAY2)
    assert gate.daily_pnl == 0.0


def test_hard_halt_after_two_trips_in_three_days():
    gate, pm, _ = make_gate(daily_loss_limit=25.0)
    gate.record_realized_pnl(-30.0, ts=DAY1)
    assert gate.check_intent(make_intent(), pm, 500.0, 1.0, ts=DAY2) is not None  # re-armed

    gate.record_realized_pnl(-30.0, ts=DAY2)  # trips again, second day running
    assert gate.check_intent(make_intent(), pm, 500.0, 1.0, ts=DAY3) is None
    assert gate.hard_halted is True
    assert gate.kill_switch_active is True

    # a hard halt does not time out -- only a human clears it
    assert gate.check_intent(make_intent(), pm, 500.0, 1.0, ts=DAY5) is None
    assert gate.hard_halted is True


def test_trips_outside_the_window_do_not_hard_halt():
    # cap pinned high: this test spends into the same market twice, and
    # _notional_spent is per-market for the life of the window, not per-day
    gate, pm, _ = make_gate(daily_loss_limit=25.0, max_notional_per_market=100.0)
    gate.record_realized_pnl(-30.0, ts=DAY1)
    assert gate.check_intent(make_intent(), pm, 500.0, 1.0, ts=DAY2) is not None

    gate.record_realized_pnl(-30.0, ts=DAY5)  # four days later, not a pattern
    assert gate.check_intent(make_intent(), pm, 500.0, 1.0, ts=DAY5 + 86400) is not None
    assert gate.hard_halted is False


def test_hard_halt_survives_a_process_restart():
    """The operator's actual habit on 2026-09-03..11 was to restart after
    every kill. An escalation that a restart clears is not an escalation."""
    db = Database(":memory:")
    settings = Settings(daily_loss_limit=25.0, max_notional_per_market=100.0)
    now = time.time()

    first = RiskGate(settings, db)
    first.record_realized_pnl(-30.0, ts=now - 86400)  # yesterday
    first.record_realized_pnl(-30.0, ts=now)  # ...and again today
    assert first.kill_switch_active is True

    restarted = RiskGate(settings, db)  # same db == same event log
    pm = PositionManager(settings)
    assert restarted.check_intent(make_intent(), pm, 500.0, 1.0) is None
    assert restarted.hard_halted is True


def test_restart_after_a_single_trip_comes_up_armed():
    db = Database(":memory:")
    settings = Settings(daily_loss_limit=25.0, max_notional_per_market=100.0)

    first = RiskGate(settings, db)
    first.record_realized_pnl(-30.0, ts=time.time() - 86400)
    assert first.kill_switch_active is True

    restarted = RiskGate(settings, db)
    pm = PositionManager(settings)
    assert restarted.check_intent(make_intent(), pm, 500.0, 1.0) is not None
    assert restarted.hard_halted is False


def test_operator_clear_resets_hard_halt_immediately():
    db = Database(":memory:")
    settings = Settings(daily_loss_limit=25.0, max_notional_per_market=100.0)
    gate = RiskGate(settings, db)
    gate.record_realized_pnl(-30.0, ts=time.time() - 86400)
    gate.record_realized_pnl(-30.0, ts=time.time())
    assert gate.hard_halted is True

    gate.clear_hard_halt("investigated -- stale losses from before a config fix")
    pm = PositionManager(settings)
    assert gate.hard_halted is False
    assert gate.check_intent(make_intent(), pm, 500.0, 1.0) is not None


def test_operator_clear_survives_restart_and_ignores_stale_trips():
    db = Database(":memory:")
    settings = Settings(daily_loss_limit=25.0, max_notional_per_market=100.0)
    first = RiskGate(settings, db)
    t0 = time.time() - 86400  # yesterday
    t1 = time.time()  # today -- two distinct trip *days*, as the escalation requires
    first.record_realized_pnl(-30.0, ts=t0)
    first.record_realized_pnl(-30.0, ts=t1)
    assert first.hard_halted is True

    first.clear_hard_halt("acknowledged", ts=t1)

    restarted = RiskGate(settings, db)
    pm = PositionManager(settings)
    assert restarted.hard_halted is False
    assert restarted.check_intent(make_intent(), pm, 500.0, 1.0) is not None

    # a trip *after* the clear still counts fresh
    restarted.record_realized_pnl(-30.0, ts=time.time())
    reloaded = RiskGate(settings, db)
    assert reloaded.hard_halted is False  # only one trip since the clear


def test_trip_history_is_ignored_when_auto_rearm_is_off():
    db = Database(":memory:")
    settings = Settings(daily_loss_limit=25.0, kill_switch_auto_rearm=False)
    first = RiskGate(settings, db)
    first.record_realized_pnl(-30.0, ts=time.time() - 86400)
    first.record_realized_pnl(-30.0, ts=time.time())

    restarted = RiskGate(settings, db)  # backtests must not inherit live history
    assert restarted.hard_halted is False
    assert restarted.kill_switch_active is False


def test_open_risk_counts_against_the_daily_budget():
    """The 2026-09-11 overshoot: eleven breaches, every one past the limit,
    because only *resolved* PnL was counted while positions were in flight."""
    gate, pm, _ = make_gate(daily_loss_limit=25.0, max_notional_per_market=100.0)
    gate.record_realized_pnl(-18.0)

    # $6 of open, unhedged risk -- realized -18 plus worst-case -6 is -24,
    # still inside the budget, but another $5 bet would project to -29
    pm.record_fill("cond1", "up", price=0.6, size=10.0, fee=0.0)
    assert gate.worst_case_open_loss(pm) == pytest.approx(-6.0)
    assert gate.projected_daily_pnl(pm) == pytest.approx(-24.0)
    assert gate.check_intent(make_intent(size=10.0, price=0.5), pm, 500.0, 1.0) is None


def test_matched_inventory_is_not_counted_as_open_risk():
    """A completed pair pays $1/share whichever way it resolves."""
    gate, pm, _ = make_gate(daily_loss_limit=25.0, max_notional_per_market=100.0)
    pm.record_fill("cond1", "up", price=0.5, size=10.0, fee=0.0)
    pm.record_fill("cond1", "down", price=0.45, size=10.0, fee=0.0)
    assert gate.worst_case_open_loss(pm) == pytest.approx(0.0)


def test_unrealized_gains_cannot_fund_new_risk():
    """A locked-in profit contributes 0, not a credit against the budget."""
    gate, pm, _ = make_gate(daily_loss_limit=25.0, max_notional_per_market=100.0)
    gate.record_realized_pnl(-24.0)
    pm.record_fill("cond1", "up", price=0.4, size=10.0, fee=0.0)
    pm.record_fill("cond1", "down", price=0.4, size=10.0, fee=0.0)  # $8 cost, $10 guaranteed
    assert gate.worst_case_open_loss(pm) == pytest.approx(0.0)
    assert gate.check_intent(make_intent(size=10.0, price=0.5), pm, 500.0, 1.0) is None


def test_risk_reducing_intents_are_exempt_from_the_budget():
    gate, pm, _ = make_gate(daily_loss_limit=25.0, max_notional_per_market=100.0)
    gate.record_realized_pnl(-30.0)
    gate.reset_kill_switch_for_new_day()
    gate.record_realized_pnl(-24.0)
    for reason in ("matched_arb", "temporal_hedge"):
        assert gate.check_intent(make_intent(reason=reason), pm, 500.0, 1.0) is not None


def test_reset_kill_switch_for_new_day_clears_active_switch():
    gate, pm, _ = make_gate(daily_loss_limit=25.0)
    gate.record_realized_pnl(-30.0)
    assert gate.kill_switch_active is True

    gate.reset_kill_switch_for_new_day()
    assert gate.kill_switch_active is False

    intent = make_intent()
    assert gate.check_intent(intent, pm, t_remaining=500.0, deviation=1.0) is not None


def test_reset_kill_switch_for_new_day_is_noop_when_inactive():
    gate, _, _ = make_gate(daily_loss_limit=25.0)
    assert gate.kill_switch_active is False
    gate.reset_kill_switch_for_new_day()  # should not raise or otherwise misbehave
    assert gate.kill_switch_active is False


def test_kill_switch_trigger_count_survives_reset():
    gate, _, _ = make_gate(daily_loss_limit=25.0)
    assert gate.kill_switch_trigger_count == 0

    gate.record_realized_pnl(-30.0)
    assert gate.kill_switch_trigger_count == 1

    gate.reset_kill_switch_for_new_day()
    gate.record_realized_pnl(-30.0)  # trips again on the new day
    assert gate.kill_switch_trigger_count == 2


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
    # cap pinned high: this test is about the end-of-window exemption, and
    # two $5 intents in one market would otherwise be rejected by whatever
    # max_notional_per_market happens to be today
    gate, pm, _ = make_gate(max_notional_per_market=100.0)
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
