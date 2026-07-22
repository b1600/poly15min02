"""Paper trading simulator (Implementation_Plan.md Phase 3, item 14).

`PaperExecutor.submit_order` is deliberately shaped the way a live executor
(Phase 4) will be: give it a condition/token id, a side, a target size, and
the current book, and it hands back a fill (or None). Everything about
*how* the fill is produced is simulator-only -- walking the given book with
a `sim_fill_ratio` haircut to model competing order flow eating some of the
visible depth before our order arrives, plus a limit price so it never
fills worse than intended. The book itself is just whatever the caller
passes in: the live in-memory book now, or a replayed historical snapshot
once Phase 6 reuses this same simulator for backtesting.

Only BUY orders are supported. Exiting a position early (selling before
resolution) and the more sophisticated hedged-directional / temporal-arb
position structure are Phase 4's position manager (item 16) -- this phase
only needs to answer "is net_edge actually realizable as PnL after costs",
which requires nothing fancier than: buy the underpriced side, hold to
resolution, collect $1/$0 per share.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from ..config import Settings
from ..db import Database
from ..pricing.edge import Level, walk_book_vwap

logger = logging.getLogger(__name__)


@dataclass
class TokenPosition:
    size: float = 0.0
    cost_basis: float = 0.0  # total $ spent (incl. fees) acquiring `size`


@dataclass
class MarketPosition:
    condition_id: str
    up: TokenPosition = field(default_factory=TokenPosition)
    down: TokenPosition = field(default_factory=TokenPosition)


@dataclass
class SimFill:
    condition_id: str
    token_id: str
    outcome: str
    price: float
    size: float
    fee: float
    ts: float


@dataclass
class Resolution:
    condition_id: str
    outcome: str
    payout: float
    cost_basis: float
    pnl: float
    up_size: float
    down_size: float


class PaperExecutor:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings
        self.positions: dict[str, MarketPosition] = {}
        self.realized_pnl: float = 0.0
        self.fees_paid: float = 0.0

    def position_size(self, condition_id: str, outcome: str) -> float:
        pos = self.positions.get(condition_id)
        if pos is None:
            return 0.0
        return pos.up.size if outcome == "up" else pos.down.size

    def submit_order(
        self,
        condition_id: str,
        token_id: str,
        outcome: str,
        ask_levels: Sequence[Level],
        target_size: float,
        limit_price: float,
        ts: float,
    ) -> SimFill | None:
        walk = walk_book_vwap(ask_levels, target_size, limit_price=limit_price)
        if walk is None:
            return None
        vwap, filled = walk
        filled *= self.settings.sim_fill_ratio  # competing order flow haircut
        if filled <= 0:
            return None

        cost = vwap * filled
        fee = cost * (self.settings.taker_fee_bps / 10000.0)

        pos = self.positions.setdefault(condition_id, MarketPosition(condition_id))
        token_pos = pos.up if outcome == "up" else pos.down
        token_pos.size += filled
        token_pos.cost_basis += cost + fee
        self.fees_paid += fee

        self.db.insert_order(condition_id, token_id, "BUY", vwap, filled, "filled", ts, mode="paper")
        self.db.insert_fill(condition_id, token_id, outcome, vwap, filled, fee, ts, mode="paper")
        logger.info(
            "paper_fill",
            extra={
                "condition_id": condition_id,
                "outcome": outcome,
                "price": round(vwap, 4),
                "size": round(filled, 2),
                "fee": round(fee, 4),
            },
        )
        return SimFill(condition_id, token_id, outcome, vwap, filled, fee, ts)

    def resolve_market(self, condition_id: str, outcome: str) -> Resolution | None:
        """Realize PnL for a resolved window: the winning side pays $1/share,
        the losing side pays $0. Drops the position afterward -- the
        window's tokens cease to exist once resolved."""
        pos = self.positions.pop(condition_id, None)
        if pos is None:
            return None

        payout = pos.up.size if outcome == "up" else pos.down.size if outcome == "down" else 0.0
        cost_basis = pos.up.cost_basis + pos.down.cost_basis
        pnl = payout - cost_basis
        self.realized_pnl += pnl

        resolution = Resolution(
            condition_id=condition_id,
            outcome=outcome,
            payout=payout,
            cost_basis=cost_basis,
            pnl=pnl,
            up_size=pos.up.size,
            down_size=pos.down.size,
        )
        logger.info(
            "paper_resolution",
            extra={
                "condition_id": condition_id,
                "outcome": outcome,
                "payout": round(payout, 2),
                "cost_basis": round(cost_basis, 2),
                "pnl": round(pnl, 2),
                "realized_pnl_total": round(self.realized_pnl, 2),
            },
        )
        return resolution
