"""Window-lifecycle tracking shared by the Phase 1 recorder and the Phase 2
paper-signals runner.

Registers newly discovered markets (subscribing the CLOB feed, starting a
clock), retries capturing a reference open price once Binance data is
available -- discovery can complete before Binance has connected, so a
single attempt at discovery time is not enough -- and emits lifecycle
events as each clock crosses a milestone.
"""

from __future__ import annotations

import logging
import time

from .binance_ws import BinanceFeed
from .clob_ws import ClobFeed
from .clock import WindowClock
from .market_finder import MarketInfo
from ..db import Database

logger = logging.getLogger(__name__)


class WindowTracker:
    def __init__(self, db: Database, binance_feed: BinanceFeed, clob_feed: ClobFeed):
        self.db = db
        self.binance_feed = binance_feed
        self.clob_feed = clob_feed
        self.clocks: dict[str, WindowClock] = {}
        self.markets: dict[str, MarketInfo] = {}
        self.open_price: dict[str, float] = {}
        self._needs_open_price: set[str] = set()

    def on_new_market(self, market: MarketInfo) -> None:
        self.clob_feed.subscribe(market.condition_id, [market.token_id_up, market.token_id_down])
        self.clocks[market.condition_id] = WindowClock(market.condition_id, market.open_ts, market.close_ts)
        self.markets[market.condition_id] = market
        self._needs_open_price.add(market.condition_id)

    def latest_market(self) -> MarketInfo | None:
        if not self.markets:
            return None
        return max(self.markets.values(), key=lambda m: m.open_ts)

    def poll(self, now: float | None = None) -> list[tuple[str, str]]:
        """Advance clocks and retry open-price capture. Returns newly-crossed
        (condition_id, event) pairs since the last call."""
        now = time.time() if now is None else now

        if self.binance_feed.last_price is not None and self._needs_open_price:
            price = self.binance_feed.last_price
            for condition_id in self._needs_open_price:
                self.open_price[condition_id] = price
                self.db.set_market_open_price(condition_id, price, "binance_at_discovery")
            self._needs_open_price.clear()

        crossed: list[tuple[str, str]] = []
        expired = []
        for condition_id, clock in list(self.clocks.items()):
            for event in clock.poll(now):
                logger.info("lifecycle_event", extra={"condition_id": condition_id, "event": event})
                self.db.insert_lifecycle_event(condition_id, event, now)
                crossed.append((condition_id, event))
            if now > clock.close_ts + 30:
                expired.append(condition_id)

        for condition_id in expired:
            self.clocks.pop(condition_id, None)
            self.markets.pop(condition_id, None)
            self.open_price.pop(condition_id, None)
            self._needs_open_price.discard(condition_id)

        return crossed
