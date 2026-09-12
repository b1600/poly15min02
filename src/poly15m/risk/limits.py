"""Risk module (Implementation_Plan.md Phase 5, item 20-21): a hard gate in
front of the executor. Every `TradeIntent` from `PositionManager` passes
through `RiskGate.check_intent` before either executor ever sees it --
this is deliberately a second, independent layer on top of
`PositionManager`'s own caps (defense in depth: a bug in one layer
shouldn't be the only thing standing between the bot and an oversized
position).

Enforces, in order:
  1. Kill switch -- if active, reject everything. Set by a breached daily
     loss limit. It used to latch forever, on the argument that a loss
     limit which silently un-halts itself at midnight is a foot-gun. In
     practice (2026-09-03..11) it tripped eleven times in nine days and
     the operator simply restarted the process each time, so the limit
     protected nothing while truncating every trading day. It now
     re-arms at the UTC day boundary when `kill_switch_auto_rearm` is
     set -- the honest version of what was already happening by hand --
     with a *hard halt* that a restart cannot clear once it trips on
     `kill_switch_hard_halt_trips` distinct days inside
     `kill_switch_hard_halt_window_days`. That escalation is the part
     that actually detects a broken model: at a ~5%/day false-alarm
     rate, two trips in three days is p ~ 0.007.
     `reset_kill_switch_for_new_day` remains for `backtest/engine.py`,
     which constructs its gate with auto re-arm disabled and drives the
     day rollover itself over simulated time.
  2. Daily loss *budget*, counting open risk (not just realized PnL).
     `record_realized_pnl` only sees a window at resolution, so the old
     gate was blind to positions already in flight: every breach
     overshot the limit, by an average of $7 on a $25 limit (-31.14,
     -27.49, -31.82, -25.14, -37.76, -38.12, -25.78, -41.33, -36.72,
     -37.43, -30.58). A single resolution took the day from inside the
     budget to well past it. So new *directional* risk is refused once
     realized PnL plus the worst case on everything currently open plus
     this intent's own cost would breach the limit. This is a brake, not
     a latch -- it blocks a trade without halting the bot, since the
     projected loss is hypothetical until it resolves. Matched-arb and
     hedge-fulfillment intents are exempt for the same reason they are
     exempt from the end-of-window gate: they reduce risk.
  3. End-of-window handling (item 21): no *new* directional risk in the
     final `end_of_window_seconds`, unless deviation is many sigma (a
     near-certain outcome), in which case a small tail-risk-capped bet is
     still allowed. Matched-arb and hedge-fulfillment trades are exempt --
     both reduce risk (lock in a riskless pair, or complete an existing
     hedge) rather than add it, which is fine right up to resolution.
  4. Per-market cumulative notional cap.
  5. Per-market inventory imbalance cap.
  6. Portfolio-wide net directional exposure cap (sum of |imbalance|
     across every currently-open market -- uncorrelated directional bets
     in different windows don't net against each other, so this sums
     absolute values, not signed ones).

`feeds_stale` is the other watchdog item (item 22): callers should check
it before even computing a decision, not just before submitting one --
reconnect logic already lives in `data/binance_ws.py` and
`data/clob_ws.py` (Phase 1).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone

from ..config import Settings
from ..db import Database
from ..positions.manager import MarketInventory, PositionManager, TradeIntent

logger = logging.getLogger(__name__)

GLOBAL_SENTINEL = "GLOBAL"


@dataclass
class RiskLimits:
    max_notional_per_market: float
    max_net_directional_exposure: float
    max_inventory_imbalance: float
    daily_loss_limit: float


class RiskGate:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self._notional_spent: dict[str, float] = {}
        self._daily_pnl: float = 0.0
        self._daily_reset_date: date = datetime.now(tz=timezone.utc).date()
        self._kill_switch_active: bool = False
        self._kill_switch_reason: str | None = None
        self._kill_switch_trigger_count: int = 0
        self._hard_halted: bool = False
        self._trip_dates: list[date] = self._load_trip_dates()
        # Evaluate immediately: reloaded history that already qualifies
        # must show up in `hard_halted` (and get logged) from the moment
        # of construction, not only once the first `check_intent` call
        # happens to run -- an operator (or a tool like
        # `poly15m-clear-halt`) reading state right after startup must see
        # the same answer the trader is about to act on.
        if self.settings.kill_switch_auto_rearm and self._too_many_recent_trips(
            datetime.fromtimestamp(time.time(), tz=timezone.utc).date()
        ):
            self._enter_hard_halt(time.time(), datetime.fromtimestamp(time.time(), tz=timezone.utc).date())

    def _load_trip_dates(self) -> list[date]:
        """Rebuild recent trip history from the event log. Without this the
        hard-halt escalation lives only in memory, and `pkill && restart`
        -- the exact reflex it exists to catch -- wipes it clean.

        An operator clear (see `clear_hard_halt`) moves the effective
        start of the window forward: trips at or before it are what got
        looked at and dealt with, not a pattern still in progress."""
        if not self.settings.kill_switch_auto_rearm:
            return []
        window = self.settings.kill_switch_hard_halt_window_days
        since = time.time() - window * 86400
        try:
            clear_ts = self.db.latest_operator_clear_ts()
            if clear_ts is not None:
                since = max(since, clear_ts)
            tss = self.db.recent_kill_switch_trip_ts(since)
        except Exception:  # a DB without the table (or a stub in tests) must not block startup
            logger.warning("kill_switch_trip_history_unavailable", exc_info=True)
            return []
        seen: list[date] = []
        for ts in tss:
            d = datetime.fromtimestamp(ts, tz=timezone.utc).date()
            if d not in seen:
                seen.append(d)
        if seen:
            logger.warning("kill_switch_trip_history_loaded", extra={"trips": [d.isoformat() for d in seen]})
        return seen

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_active

    @property
    def kill_switch_trigger_count(self) -> int:
        return self._kill_switch_trigger_count

    @property
    def hard_halted(self) -> bool:
        """Tripped too often in too few days. Never auto-re-arms; a human
        has to look at why the model is losing and restart deliberately."""
        return self._hard_halted

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    def trigger_kill_switch(self, reason: str, ts: float | None = None) -> None:
        if self._kill_switch_active:
            return
        ts = ts if ts is not None else time.time()
        self._kill_switch_active = True
        self._kill_switch_reason = reason
        self._kill_switch_trigger_count += 1
        trip_date = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        if trip_date not in self._trip_dates:
            self._trip_dates.append(trip_date)
        logger.critical("kill_switch_triggered", extra={"reason": reason, "daily_pnl": round(self._daily_pnl, 2)})
        self.db.insert_lifecycle_event(
            GLOBAL_SENTINEL, "kill_switch_triggered", ts, {"reason": reason}
        )
        if self.settings.kill_switch_auto_rearm and self._too_many_recent_trips(trip_date):
            self._enter_hard_halt(ts, trip_date)

    def _maybe_rearm(self, ts: float) -> None:
        """Re-arm at the UTC day boundary, or escalate to a hard halt if
        the switch has tripped on too many days recently. No-op unless
        `kill_switch_auto_rearm` is set -- `backtest/engine.py` disables
        it and drives the rollover itself over simulated time."""
        if self._hard_halted:
            return
        if not self.settings.kill_switch_auto_rearm:
            return
        today = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        # Checked even when the switch is not currently active: a restart
        # comes up armed with an empty daily PnL but a reloaded trip
        # history, and must re-halt itself rather than trade on.
        if self._too_many_recent_trips(today):
            self._enter_hard_halt(ts, today)
            return
        if not self._kill_switch_active:
            return
        if not self._trip_dates or today <= self._trip_dates[-1]:
            return  # still the day it tripped on

        logger.warning("kill_switch_rearmed", extra={"prior_reason": self._kill_switch_reason})
        self.db.insert_lifecycle_event(
            GLOBAL_SENTINEL, "kill_switch_rearmed", ts, {"prior_reason": self._kill_switch_reason}
        )
        self._kill_switch_active = False
        self._kill_switch_reason = None
        self._daily_pnl = 0.0  # new day, new budget

    def _recent_trips(self, today: date) -> list[date]:
        window = self.settings.kill_switch_hard_halt_window_days
        return [d for d in self._trip_dates if 0 <= (today - d).days < window]

    def _too_many_recent_trips(self, today: date) -> bool:
        return len(self._recent_trips(today)) >= self.settings.kill_switch_hard_halt_trips

    def _enter_hard_halt(self, ts: float, today: date) -> None:
        recent = self._recent_trips(today)
        self._hard_halted = True
        self._kill_switch_active = True
        self._kill_switch_reason = f"hard_halt: {len(recent)} trips in {self.settings.kill_switch_hard_halt_window_days}d"
        logger.critical(
            "kill_switch_hard_halt",
            extra={
                "trips": [d.isoformat() for d in recent],
                "window_days": self.settings.kill_switch_hard_halt_window_days,
            },
        )
        self.db.insert_lifecycle_event(
            GLOBAL_SENTINEL,
            "kill_switch_hard_halt",
            ts,
            {
                "trips": [d.isoformat() for d in recent],
                "window_days": self.settings.kill_switch_hard_halt_window_days,
            },
        )

    def clear_hard_halt(self, reason: str, ts: float | None = None) -> None:
        """The only sanctioned way off a hard halt: a human looked at why
        it tripped repeatedly, decided it's safe to resume, and says why.
        Writes a marker to the event log so a fresh restart also comes up
        clear -- clearing in memory alone would just be re-hidden by
        `_load_trip_dates` on the next process start.

        Not exposed as a bare "un-halt" -- callers (the `poly15m-clear-halt`
        CLI) must supply `reason` and it is logged and persisted, so the
        override is auditable rather than silent."""
        ts = ts if ts is not None else time.time()
        logger.warning("kill_switch_operator_cleared", extra={"reason": reason})
        self.db.insert_lifecycle_event(GLOBAL_SENTINEL, "kill_switch_operator_cleared", ts, {"reason": reason})
        self._trip_dates = []
        self._hard_halted = False
        self._kill_switch_active = False
        self._kill_switch_reason = None
        self._daily_pnl = 0.0

    def reset_kill_switch_for_new_day(self) -> None:
        """Backtest-only day-rollover reset -- see module docstring. A no-op
        if the kill switch isn't currently active."""
        if not self._kill_switch_active:
            return
        logger.warning(
            "kill_switch_reset_new_backtest_day", extra={"prior_reason": self._kill_switch_reason}
        )
        self._kill_switch_active = False
        self._kill_switch_reason = None
        self._daily_pnl = 0.0  # new day, new budget -- see _maybe_rearm

    def record_realized_pnl(self, pnl: float, ts: float | None = None) -> None:
        ts = ts if ts is not None else time.time()
        self._maybe_rearm(ts)
        self._maybe_reset_daily(ts)
        self._daily_pnl += pnl
        if self._daily_pnl <= -abs(self.settings.daily_loss_limit):
            self.trigger_kill_switch(f"daily_loss_limit_breached: {self._daily_pnl:.2f}", ts=ts)

    def _maybe_reset_daily(self, ts: float) -> None:
        today = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        if today != self._daily_reset_date:
            self._daily_reset_date = today
            self._daily_pnl = 0.0

    def feeds_stale(self, binance_age: float | None, clob_age: float | None) -> bool:
        threshold = self.settings.feed_staleness_seconds
        if binance_age is not None and binance_age > threshold:
            return True
        if clob_age is not None and clob_age > threshold:
            return True
        return False

    def drop_market(self, condition_id: str) -> None:
        self._notional_spent.pop(condition_id, None)

    def worst_case_open_loss(self, position_manager: PositionManager) -> float:
        """Largest loss still possible from inventory already held, summed
        over open markets. Never positive: a market that is locked in
        profitable (a completed matched pair) contributes 0 rather than a
        credit, so unrealized gains can't be spent as risk budget.

        Per market the two outcomes pay `up.size` or `down.size`, so the
        worst case pays `min(up.size, down.size)` -- the matched portion,
        which settles at $1/share either way."""
        total = 0.0
        for inv in position_manager.inventory.values():
            guaranteed_payout = min(inv.up.size, inv.down.size)
            cost_basis = inv.up.cost_basis + inv.down.cost_basis
            total += min(0.0, guaranteed_payout - cost_basis)
        return total

    def projected_daily_pnl(self, position_manager: PositionManager) -> float:
        """Where today ends up if every open position resolves against us.
        This is what the daily loss limit is actually about -- `daily_pnl`
        alone lags by up to a full window."""
        return self._daily_pnl + self.worst_case_open_loss(position_manager)

    def _within_daily_loss_budget(
        self, position_manager: PositionManager, intent: TradeIntent
    ) -> bool:
        # Risk-reducing intents are exempt: a matched pair or a hedge
        # completion shrinks the worst case rather than growing it, and
        # refusing them near the limit would strand exactly the positions
        # most in need of closing.
        if intent.reason != "directional_kelly":
            return True
        projected = self.projected_daily_pnl(position_manager) - intent.size * intent.limit_price
        return projected > -abs(self.settings.daily_loss_limit)

    def check_intent(
        self,
        intent: TradeIntent,
        position_manager: PositionManager,
        t_remaining: float,
        deviation: float | None,
        ts: float | None = None,
    ) -> TradeIntent | None:
        ts = ts if ts is not None else time.time()
        self._maybe_rearm(ts)
        self._maybe_reset_daily(ts)
        if self._kill_switch_active:
            return None

        if intent.reason == "directional_kelly" and t_remaining < self.settings.end_of_window_seconds:
            if deviation is None or abs(deviation) < self.settings.near_resolution_deviation_threshold:
                return None
            intent = replace(intent, size=min(intent.size, self.settings.near_resolution_max_size))

        intent = self._apply_notional_cap(intent)
        if intent is None:
            return None

        inv = position_manager.get_inventory(intent.condition_id)
        if not self._within_imbalance_cap(inv, intent):
            return None
        if not self._within_portfolio_cap(position_manager, inv, intent):
            return None
        # Last, so it sees the final size after every cap has scaled the
        # intent down -- a trade that doesn't fit at full size may fit
        # after clipping.
        if not self._within_daily_loss_budget(position_manager, intent):
            return None

        spent = self._notional_spent.get(intent.condition_id, 0.0)
        self._notional_spent[intent.condition_id] = spent + intent.size * intent.limit_price
        return intent

    def _apply_notional_cap(self, intent: TradeIntent) -> TradeIntent | None:
        spent = self._notional_spent.get(intent.condition_id, 0.0)
        room = self.settings.max_notional_per_market - spent
        if room <= 0:
            return None
        notional = intent.size * intent.limit_price
        if notional <= room:
            return intent
        if intent.limit_price <= 0:
            return None
        scaled_size = room / intent.limit_price
        if scaled_size < self.settings.paper_min_order_size:
            return None
        return replace(intent, size=scaled_size)

    def _projected_imbalance(self, inv: MarketInventory, intent: TradeIntent) -> float:
        up = inv.up.size + (intent.size if intent.outcome == "up" else 0.0)
        down = inv.down.size + (intent.size if intent.outcome == "down" else 0.0)
        return up - down

    def _within_imbalance_cap(self, inv: MarketInventory, intent: TradeIntent) -> bool:
        return abs(self._projected_imbalance(inv, intent)) <= self.settings.max_inventory_imbalance

    def _within_portfolio_cap(
        self, position_manager: PositionManager, inv: MarketInventory, intent: TradeIntent
    ) -> bool:
        current_total = sum(abs(m.net_directional) for m in position_manager.inventory.values())
        projected_this_market = abs(self._projected_imbalance(inv, intent))
        projected_total = current_total - abs(inv.net_directional) + projected_this_market
        return projected_total <= self.settings.max_net_directional_exposure
