"""Polymarket CLOB market-data WebSocket feed.

Subscribes to the public "market" channel (no auth) for a set of token
(asset) ids and maintains an in-memory order book per token, persisting
every snapshot/delta and trade print to SQLite.

Message shapes (confirmed live against wss://ws-subscriptions-clob.polymarket.com/ws/market):
  - On subscribe: a JSON array of `event_type: "book"` full snapshots, one
    per asset_id -- {market, asset_id, timestamp, hash, bids, asks,
    tick_size, event_type, last_trade_price}, bids/asks as [{price, size}].
  - Incremental updates arrive individually (dict, not array) per
    Polymarket's documented channel: `price_change` (delta -- size "0"
    means remove that price level) and `last_trade_price` (trade print).
    Handling for both is defensive: unrecognized shapes are logged and
    skipped rather than raising.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field

import websockets
from websockets.exceptions import ConnectionClosed

from ..config import Settings
from ..db import Database

logger = logging.getLogger(__name__)


@dataclass
class OrderBook:
    token_id: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    tick_size: float | None = None
    last_trade_price: float | None = None
    last_update_ts: float | None = None
    # (ts, price, size, side) trade prints, for the Phase 2 aggressive-flow
    # feature -- kept in memory rather than round-tripped through SQLite
    # since it's read on every feature computation.
    recent_trades: deque[tuple[float, float, float, str | None]] = field(
        default_factory=lambda: deque(maxlen=5000)
    )

    def record_trade(self, ts: float, price: float, size: float, side: str | None, buffer_seconds: float) -> None:
        self.recent_trades.append((ts, price, size, side))
        cutoff = ts - buffer_seconds
        while self.recent_trades and self.recent_trades[0][0] < cutoff:
            self.recent_trades.popleft()

    def apply_snapshot(self, bids: list[dict], asks: list[dict], event_ts: float | None, tick_size=None) -> None:
        self.bids = {float(b["price"]): float(b["size"]) for b in bids}
        self.asks = {float(a["price"]): float(a["size"]) for a in asks}
        self.last_update_ts = event_ts
        if tick_size:
            self.tick_size = float(tick_size)

    def apply_price_change(self, changes: list[dict], event_ts: float | None) -> None:
        for change in changes:
            try:
                price = float(change["price"])
                size = float(change["size"])
            except (KeyError, ValueError, TypeError):
                continue
            side = str(change.get("side", "")).upper()
            book_side = self.bids if side == "BUY" else self.asks
            if size == 0:
                book_side.pop(price, None)
            else:
                book_side[price] = size
        self.last_update_ts = event_ts

    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    def as_sorted(self) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        bids = sorted(self.bids.items(), key=lambda kv: -kv[0])
        asks = sorted(self.asks.items(), key=lambda kv: kv[0])
        return bids, asks


class ClobFeed:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.books: dict[str, OrderBook] = {}
        self.condition_id_by_token: dict[str, str] = {}
        self.last_msg_ts: float | None = None  # local receive time, for staleness checks

        self._desired_assets: set[str] = set()
        self._reconnect_needed = asyncio.Event()

    def subscribe(self, condition_id: str, token_ids: list[str]) -> None:
        """Point the feed at a (new) market's tokens; triggers a resubscribe."""
        for token_id in token_ids:
            self.condition_id_by_token[token_id] = condition_id
            self.books.setdefault(token_id, OrderBook(token_id))
        new_assets = set(token_ids)
        if new_assets != self._desired_assets:
            self._desired_assets = new_assets
            self._reconnect_needed.set()

    def last_msg_age(self, now: float | None = None) -> float | None:
        if self.last_msg_ts is None:
            return None
        now = time.time() if now is None else now
        return now - self.last_msg_ts

    async def run(self) -> None:
        backoff = 1.0
        while True:
            if not self._desired_assets:
                await asyncio.sleep(0.5)
                continue
            try:
                await self._connect_and_listen()
                backoff = 1.0
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as exc:
                logger.warning("clob_ws_disconnected", extra={"error": str(exc), "retry_in": backoff})
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _connect_and_listen(self) -> None:
        url = f"{self.settings.clob_ws_base}/market"
        async with websockets.connect(url, ping_interval=10, ping_timeout=10) as ws:
            assets = sorted(self._desired_assets)
            await ws.send(json.dumps({"assets_ids": assets, "type": "market"}))
            logger.info("clob_ws_connected", extra={"assets": assets})
            self._reconnect_needed.clear()

            recv_task = asyncio.ensure_future(ws.recv())
            reconnect_task = asyncio.ensure_future(self._reconnect_needed.wait())
            try:
                while True:
                    done, _ = await asyncio.wait(
                        {recv_task, reconnect_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if reconnect_task in done:
                        return  # desired asset set changed -> reconnect with fresh subscribe
                    raw = recv_task.result()
                    self._handle_message(raw)
                    recv_task = asyncio.ensure_future(ws.recv())
            finally:
                for task in (recv_task, reconnect_task):
                    if not task.done():
                        task.cancel()

    def _handle_message(self, raw: str | bytes) -> None:
        self.last_msg_ts = time.time()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("clob_ws_bad_json", extra={"raw": str(raw)[:200]})
            return
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if isinstance(item, dict):
                self._handle_event(item)

    def _handle_event(self, item: dict) -> None:
        event_type = item.get("event_type")
        asset_id = item.get("asset_id")
        if not asset_id:
            return
        condition_id = self.condition_id_by_token.get(asset_id) or item.get("market")
        book = self.books.setdefault(asset_id, OrderBook(asset_id))
        ts_raw = item.get("timestamp")
        event_ts = float(ts_raw) / 1000.0 if ts_raw else None

        if event_type == "book":
            book.apply_snapshot(item.get("bids", []), item.get("asks", []), event_ts, item.get("tick_size"))
            bids, asks = book.as_sorted()
            self.db.insert_book_snapshot(condition_id, asset_id, bids, asks, event_ts)
        elif event_type == "price_change":
            changes = item.get("changes") or item.get("price_changes") or []
            book.apply_price_change(changes, event_ts)
            bids, asks = book.as_sorted()
            self.db.insert_book_snapshot(condition_id, asset_id, bids, asks, event_ts)
        elif event_type == "last_trade_price":
            price = item.get("price")
            if price is not None:
                size = float(item.get("size") or 0)
                side = item.get("side")
                book.last_trade_price = float(price)
                book.record_trade(
                    event_ts if event_ts is not None else time.time(),
                    float(price),
                    size,
                    side,
                    self.settings.clob_trade_buffer_seconds,
                )
                self.db.insert_clob_trade(
                    condition_id,
                    asset_id,
                    float(price),
                    size,
                    side,
                    item.get("trade_id"),
                    event_ts,
                )
        elif event_type == "tick_size_change":
            new_tick = item.get("new_tick_size") or item.get("tick_size")
            if new_tick:
                book.tick_size = float(new_tick)
        else:
            logger.debug("clob_ws_unhandled_event", extra={"event_type": event_type, "keys": list(item.keys())})
