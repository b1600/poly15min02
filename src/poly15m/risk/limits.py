"""Risk module (Implementation_Plan.md Phase 5, item 20-21): a hard gate in
front of the executor. Every `TradeIntent` from `PositionManager` passes
through `RiskGate.check_intent` before either executor ever sees it --
this is deliberately a second, independent layer on top of
`PositionManager`'s own caps (defense in depth: a bug in one layer
shouldn't be the only thing standing between the bot and an oversized
position).

Enforces, in order:
  1. Kill switch -- if active, reject everything. Set by a breached daily
     loss limit; does not clear itself, including across a day rollover,
     by design (an automatic loss limit that silently un-halts itself at
     midnight is a foot-gun, not a safety feature).
  2. End-of-window handling (item 21): no *new* directional risk in the
     final `end_of_window_seconds`, unless deviation is many sigma (a
     near-certain outcome), in which case a small tail-risk-capped bet is
     still allowed. Matched-arb and hedge-fulfillment trades are exempt --
     both reduce risk (lock in a riskless pair, or complete an existing
     hedge) rather than add it, which is fine right up to resolution.
  3. Per-market cumulative notional cap.
  4. Per-market inventory imbalance cap.
  5. Portfolio-wide net directional exposure cap (sum of |imbalance|
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

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_active

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    def trigger_kill_switch(self, reason: str) -> None:
        if self._kill_switch_active:
            return
        self._kill_switch_active = True
        self._kill_switch_reason = reason
        logger.critical("kill_switch_triggered", extra={"reason": reason, "daily_pnl": round(self._daily_pnl, 2)})
        self.db.insert_lifecycle_event(
            GLOBAL_SENTINEL, "kill_switch_triggered", time.time(), {"reason": reason}
        )

    def record_realized_pnl(self, pnl: float, ts: float | None = None) -> None:
        ts = ts if ts is not None else time.time()
        self._maybe_reset_daily(ts)
        self._daily_pnl += pnl
        if self._daily_pnl <= -abs(self.settings.daily_loss_limit):
            self.trigger_kill_switch(f"daily_loss_limit_breached: {self._daily_pnl:.2f}")

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

    def check_intent(
        self,
        intent: TradeIntent,
        position_manager: PositionManager,
        t_remaining: float,
        deviation: float | None,
    ) -> TradeIntent | None:
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
