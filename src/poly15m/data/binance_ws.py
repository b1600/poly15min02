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
    def __init__(
        self,
        settings: Settings,
        db: Database,
        on_tick: OnTick | None = None,
        persist_ticks: bool = True,
    ):
        self.settings = settings
        self.db = db
        self.symbol = settings.binance_symbol.lower()
        self._on_tick = on_tick
        # Backtest replay feeds tens of millions of historical ticks through
        # this same handler; persisting each one to `db` (there, a throwaway
        # in-memory replay DB nothing ever reads back) grew unbounded across
        # the full run and was the actual OOM-killer trigger -- upstream
        # per-day event batching only bounded the read side, not this.
        self._persist_ticks = persist_ticks

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
        """Nearest buffered price at or before `now - seconds_ago` (for momentum features).

        Momentum lookbacks (60-300s) are typically much shorter than
        `binance_buffer_seconds` (the buffer's full retention window), so
        walking backward from the newest tick and stopping at the first
        one at/before `target` is far cheaper than scanning the whole
        buffer from the front on every call -- the same reasoning as
        `resample_to_bars` in signals/features.py.
        """
        now = time.time() if now is None else now
        target = now - seconds_ago
        for ts, price in reversed(self.buffer):
            if ts <= target:
                return price
        return None

    def price_at(self, target_ts: float) -> tuple[float, float] | None:
        """Newest buffered trade at or before `target_ts`, as (ts, price).

        Used to anchor a window's reference open price to the window's own
        open timestamp instead of to whenever discovery happened. Walks
        backward from the newest tick for the same reason `price_since`
        does -- the anchor is normally only seconds old, so scanning from
        the front of a 30-minute buffer would be wasteful.

        Returns None when the buffer holds nothing at or before the
        target, which is the honest answer: the caller must not invent an
        anchor from a later tick.
        """
        for ts, price in reversed(self.buffer):
            if ts <= target_ts:
                return ts, price
        return None

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
        self._handle_payload(msg)

    def _handle_payload(self, msg: dict) -> None:
        """The parsed body of `_handle_message`, split out so a caller that
        already has a dict -- the backtest replay loop, which builds these
        messages itself rather than receiving JSON bytes off a socket --
        can skip the encode/decode round trip. Same logic either way; only
        the transport layer (bytes-in) differs from this entry point
        (dict-in). Field values below use `float()`/comparisons that work
        identically whether they arrive as native numbers or as strings,
        so this makes no behavioral distinction between the two."""
        data = msg.get("data", msg)
        event = data.get("e")

        if event == "trade":
            ts = float(data["T"]) / 1000.0
            price = float(data["p"])
            qty = float(data["q"])
            self.last_price = price
            self._append_buffer(ts, price)
            if self._persist_ticks:
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
