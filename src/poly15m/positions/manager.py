"""Position manager (Implementation_Plan.md Phase 4, item 16): the hybrid
structure recommended in Strategy_v1.md --

  - Matched pairs: buying Up+Down together for < $1 combined locks in a
    riskless profit regardless of outcome. Checked first, every decision,
    since it's the safest trade available and doesn't depend on the
    fair-value model being right.
  - Directional imbalance: sized by fractional Kelly on net_edge (Phase 3),
    capped at `max_inventory_imbalance` shares of net exposure.
  - Temporal arb: once a directional position exists, a standing hedge
    target is implicit in the held leg's average cost -- any price for the
    other leg that keeps the combined cost under $1 (minus a margin)
    completes the hedge. Checked before opening any *new* directional
    exposure, since reducing existing risk takes priority over adding more.

This module only decides what to trade -- it returns `TradeIntent`s and
never touches a book or an executor. `record_fill` is how callers (paper or
live) feed fills back in to keep inventory accurate. That split is what
lets Phase 3's `PaperExecutor` and Phase 4's live executor share this same
decision logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..config import Settings
from ..pricing.edge import Level, compute_edge, walk_book_vwap
from ..pricing.fair_value import FairValue


@dataclass
class TokenInventory:
    size: float = 0.0
    cost_basis: float = 0.0

    @property
    def avg_cost(self) -> float | None:
        return self.cost_basis / self.size if self.size > 0 else None


@dataclass
class MarketInventory:
    condition_id: str
    up: TokenInventory = field(default_factory=TokenInventory)
    down: TokenInventory = field(default_factory=TokenInventory)

    @property
    def matched_size(self) -> float:
        return min(self.up.size, self.down.size)

    @property
    def net_directional(self) -> float:
        """Positive = net long Up, negative = net long Down."""
        return self.up.size - self.down.size


@dataclass
class TradeIntent:
    condition_id: str
    token_id: str
    outcome: str  # "up" | "down"
    size: float
    limit_price: float
    reason: str  # "matched_arb" | "temporal_hedge" | "directional_kelly"
    edge: float | None = None  # net_edge (directional) or locked-in margin (arb/hedge), for logging


def kelly_fraction(fair_p: float, price: float) -> float:
    """f* = (p - price) / (1 - price): standard Kelly fraction for a binary
    bet costing `price` per $1 payout with true win probability `fair_p`.
    Clamped to [0, 1]."""
    if price <= 0.0 or price >= 1.0:
        return 0.0
    f = (fair_p - price) / (1.0 - price)
    return max(0.0, min(1.0, f))


class PositionManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.inventory: dict[str, MarketInventory] = {}

    def get_inventory(self, condition_id: str) -> MarketInventory:
        return self.inventory.setdefault(condition_id, MarketInventory(condition_id))

    def record_fill(self, condition_id: str, outcome: str, price: float, size: float, fee: float) -> None:
        inv = self.get_inventory(condition_id)
        token = inv.up if outcome == "up" else inv.down
        token.size += size
        token.cost_basis += price * size + fee

    def drop_market(self, condition_id: str) -> MarketInventory | None:
        return self.inventory.pop(condition_id, None)

    def compute_resolution_pnl(self, condition_id: str, outcome: str) -> float | None:
        """Estimated PnL if this market resolves to `outcome`: the winning
        side pays $1/share (minus our cost basis), the losing side pays $0.
        This is what PnL *should* be given our recorded fills -- it does
        not depend on on-chain redemption having actually happened, which
        makes it suitable for risk-tracking (the daily loss limit) even
        where automated redemption isn't wired up (see live_trade.py)."""
        inv = self.inventory.get(condition_id)
        if inv is None:
            return None
        payout = inv.up.size if outcome == "up" else inv.down.size if outcome == "down" else 0.0
        cost_basis = inv.up.cost_basis + inv.down.cost_basis
        return payout - cost_basis

    def decide_trades(
        self,
        condition_id: str,
        token_id_up: str,
        token_id_down: str,
        fair_value: FairValue,
        asks_up: Sequence[Level],
        asks_down: Sequence[Level],
        t_remaining: float,
        sigma: float | None,
    ) -> list[TradeIntent]:
        inv = self.get_inventory(condition_id)

        matched = self._check_matched_arb(condition_id, token_id_up, token_id_down, asks_up, asks_down)
        if matched:
            return matched

        hedge = self._check_hedge_fulfillment(condition_id, token_id_up, token_id_down, inv, asks_up, asks_down)
        if hedge is not None:
            return [hedge]

        if abs(inv.net_directional) >= self.settings.max_inventory_imbalance:
            return []  # at the directional cap -- only matched/hedge trades until it shrinks

        candidates = []
        for outcome, token_id, fair_p, asks in (
            ("up", token_id_up, fair_value.p_up, asks_up),
            ("down", token_id_down, fair_value.p_down, asks_down),
        ):
            intent = self._check_directional_kelly(
                condition_id, token_id, outcome, fair_p, asks, t_remaining, sigma, inv
            )
            if intent is not None:
                candidates.append(intent)
        if not candidates:
            return []
        candidates.sort(key=lambda i: i.edge or 0.0, reverse=True)
        return [candidates[0]]

    def _check_matched_arb(
        self, condition_id: str, token_id_up: str, token_id_down: str, asks_up: Sequence[Level], asks_down: Sequence[Level]
    ) -> list[TradeIntent]:
        s = self.settings
        walk_up = walk_book_vwap(asks_up, s.paper_trade_size)
        walk_down = walk_book_vwap(asks_down, s.paper_trade_size)
        if walk_up is None or walk_down is None:
            return []
        _, filled_up = walk_up
        _, filled_down = walk_down
        matched_size = min(filled_up, filled_down)
        if matched_size < s.paper_min_order_size:
            return []

        vwap_up, _ = walk_book_vwap(asks_up, matched_size)
        vwap_down, _ = walk_book_vwap(asks_down, matched_size)
        combined = vwap_up + vwap_down
        fee_frac = s.taker_fee_bps / 10000.0
        net_lock_in = 1.0 - combined - combined * fee_frac
        if net_lock_in < s.matched_arb_min_margin:
            return []

        notional = matched_size * combined
        if notional > s.max_notional_per_market:
            matched_size *= s.max_notional_per_market / notional
            if matched_size < s.paper_min_order_size:
                return []

        return [
            TradeIntent(condition_id, token_id_up, "up", matched_size, vwap_up, "matched_arb", net_lock_in),
            TradeIntent(condition_id, token_id_down, "down", matched_size, vwap_down, "matched_arb", net_lock_in),
        ]

    def _check_hedge_fulfillment(
        self,
        condition_id: str,
        token_id_up: str,
        token_id_down: str,
        inv: MarketInventory,
        asks_up: Sequence[Level],
        asks_down: Sequence[Level],
    ) -> TradeIntent | None:
        s = self.settings
        imbalance = inv.net_directional
        if abs(imbalance) < s.paper_min_order_size:
            return None

        if imbalance > 0:
            held, other_outcome, other_token_id, other_asks = inv.up, "down", token_id_down, asks_down
        else:
            held, other_outcome, other_token_id, other_asks = inv.down, "up", token_id_up, asks_up

        avg_cost = held.avg_cost
        if avg_cost is None:
            return None
        max_hedge_price = 1.0 - avg_cost - s.matched_arb_min_margin
        if max_hedge_price <= 0:
            return None

        target_size = min(abs(imbalance), s.paper_trade_size)
        walk = walk_book_vwap(other_asks, target_size, limit_price=max_hedge_price)
        if walk is None:
            return None
        vwap, filled = walk
        size = min(filled, abs(imbalance))
        if size < s.paper_min_order_size:
            return None

        locked_in = 1.0 - (avg_cost + vwap)
        return TradeIntent(condition_id, other_token_id, other_outcome, size, vwap, "temporal_hedge", locked_in)

    def _check_directional_kelly(
        self,
        condition_id: str,
        token_id: str,
        outcome: str,
        fair_p: float,
        asks: Sequence[Level],
        t_remaining: float,
        sigma: float | None,
        inv: MarketInventory,
    ) -> TradeIntent | None:
        s = self.settings
        edge = compute_edge(
            outcome, fair_p, asks, s.paper_trade_size, t_remaining, s.market_window_seconds, sigma, s
        )
        if edge is None or edge.net_edge < s.min_edge_to_trade:
            return None

        price = edge.vwap_fill
        f_star = kelly_fraction(fair_p, price)
        if f_star <= 0:
            return None

        notional = s.bankroll * s.kelly_fraction * f_star
        size = notional / price if price > 0 else 0.0
        size = min(size, edge.filled_size, s.paper_trade_size)

        room = s.max_inventory_imbalance - abs(inv.net_directional)
        size = min(size, max(0.0, room))
        if size < s.paper_min_order_size:
            return None

        if size * price > s.max_notional_per_market:
            size = s.max_notional_per_market / price
            if size < s.paper_min_order_size:
                return None

        return TradeIntent(condition_id, token_id, outcome, size, price, "directional_kelly", edge.net_edge)
