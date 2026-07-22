"""Binance trade/bookTicker WebSocket feed.

Public data, no auth. Maintains a rolling (ts, price) buffer of trades
covering the last `binance_buffer_seconds` for downstream momentum /
volatility features, and persists every trade tick to SQLite.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Callable

import websockets
from websockets.exceptions import ConnectionClosed

from ..config import Settings
from ..db import Database

logger = logging.getLogger(__name__)

OnTick = Callable[[float, float], None]


class BinanceFeed:
    def __init__(self, settings: Settings, db: Database, on_tick: OnTick | None = None):
        self.settings = settings
        self.db = db
        self.symbol = settings.binance_symbol.lower()
        self._on_tick = on_tick

        self.buffer: deque[tuple[float, float]] = deque()
        self.last_price: float | None = None
        self.best_bid: float | None = None
        self.best_ask: float | None = None
        self.last_msg_ts: float | None = None  # local receive time, for staleness checks

    def set_on_tick(self, callback: OnTick | None) -> None:
        self._on_tick = callback

    @property
    def url(self) -> str:
        streams = f"{self.symbol}@trade/{self.symbol}@bookTicker"
        return f"{self.settings.binance_ws_base}/stream?streams={streams}"

    def last_trade_age(self, now: float | None = None) -> float | None:
        if self.last_msg_ts is None:
            return None
        now = time.time() if now is None else now
        return now - self.last_msg_ts

    def price_since(self, seconds_ago: float, now: float | None = None) -> float | None:
        """Nearest buffered price at or before `now - seconds_ago` (for momentum features)."""
        now = time.time() if now is None else now
        target = now - seconds_ago
        candidate = None
        for ts, price in self.buffer:
            if ts <= target:
                candidate = price
            else:
                break
        return candidate

    async def run(self) -> None:
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(self.url, ping_interval=15, ping_timeout=10) as ws:
                    logger.info("binance_connected", extra={"url": self.url})
                    backoff = 1.0
                    async for raw in ws:
                        self._handle_message(raw)
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "binance_disconnected", extra={"error": str(exc), "retry_in": backoff}
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _handle_message(self, raw: str | bytes) -> None:
        self.last_msg_ts = time.time()
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("binance_bad_json", extra={"raw": str(raw)[:200]})
            return

        data = msg.get("data", msg)
        event = data.get("e")

        if event == "trade":
            ts = float(data["T"]) / 1000.0
            price = float(data["p"])
            qty = float(data["q"])
            self.last_price = price
            self._append_buffer(ts, price)
            self.db.insert_tick(
                "binance", self.symbol.upper(), price, qty, ts, bool(data.get("m"))
            )
            if self._on_tick:
                self._on_tick(ts, price)
        elif "b" in data and "a" in data and "B" in data and "A" in data:
            # bookTicker (no explicit "e" field in the combined-stream payload)
            self.best_bid = float(data["b"])
            self.best_ask = float(data["a"])
        else:
            logger.debug("binance_unhandled_event", extra={"keys": list(data.keys())})

    def _append_buffer(self, ts: float, price: float) -> None:
        self.buffer.append((ts, price))
        cutoff = ts - self.settings.binance_buffer_seconds
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()
