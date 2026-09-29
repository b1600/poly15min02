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

# The market channel pushes ~600 frames/s for a live BTC 15m market. With
# websockets' default max_queue (16 frames, ~25 ms of traffic) any event-loop
# stall pauses socket reads, so keepalive pongs go unread ("keepalive ping
# timeout") and Polymarket's send buffer fills ("1013 slow consumer"). 20k
# frames (~30 s, ~12 MB) rides out stalls instead of dropping the connection.
_MAX_QUEUE = 20_000
# A connection that stayed up this long resets the reconnect backoff, so a
# drop after hours of uptime retries in 1 s rather than the last delay.
_BACKOFF_RESET_UPTIME_S = 60.0


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

    def prune_to_best(self, best_bid: float | None, best_ask: float | None) -> None:
        """Drop levels better than the exchange's reported best prices. A
        trade can consume a level without a price_change removing it, which
        leaves a stale level (seen live: a 0.86 ask left under a 0.87 best
        ask, crossing the 0.86 bid) until the next full book snapshot."""
        if best_bid is not None:
            for price in [p for p in self.bids if p > best_bid]:
                del self.bids[price]
        if best_ask is not None:
            for price in [p for p in self.asks if p < best_ask]:
                del self.asks[price]

    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    def as_sorted(self) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        bids = sorted(self.bids.items(), key=lambda kv: -kv[0])
        asks = sorted(self.asks.items(), key=lambda kv: kv[0])
        return bids, asks


def _opt_float(value) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


class ClobFeed:
    def __init__(self, settings: Settings, db: Database, persist_events: bool = True):
        self.settings = settings
        self.db = db
        self.books: dict[str, OrderBook] = {}
        self.condition_id_by_token: dict[str, str] = {}
        self.last_msg_ts: float | None = None  # local receive time, for staleness checks

        self._desired_assets: set[str] = set()
        self._reconnect_needed = asyncio.Event()
        self._connected_at: float | None = None
        # See BinanceFeed's identical flag: backtest replay pushes millions
        # of historical book/trade events through this handler, and
        # persisting each into `db` (a throwaway in-memory replay DB
        # nothing reads back) is the main driver of the OOM this avoids.
        self._persist_events = persist_events
        # price_change arrives hundreds of times a second, so its book
        # snapshots are throttled per token: token -> local time of the last
        # write, and tokens changed since then whose latest state is unwritten.
        self._last_persist_recv: dict[str, float] = {}
        self._unpersisted: dict[str, float | None] = {}  # token -> event_ts of its latest change

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
            self._connected_at = None
            try:
                await self._connect_and_listen()
                backoff = 1.0
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as exc:
                uptime = time.time() - self._connected_at if self._connected_at is not None else None
                if uptime is not None and uptime >= _BACKOFF_RESET_UPTIME_S:
                    backoff = 1.0
                logger.warning(
                    "clob_ws_disconnected",
                    extra={
                        "error": str(exc),
                        "uptime_s": round(uptime, 1) if uptime is not None else None,
                        "retry_in": backoff,
                    },
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _connect_and_listen(self) -> None:
        url = f"{self.settings.clob_ws_base}/market"
        async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_queue=_MAX_QUEUE) as ws:
            # Clear before reading the asset set so a subscribe() landing
            # during the send below still triggers a resubscribe.
            self._reconnect_needed.clear()
            assets = sorted(self._desired_assets)
            await ws.send(json.dumps({"assets_ids": assets, "type": "market"}))
            self._connected_at = time.time()
            logger.info("clob_ws_connected", extra={"assets": assets})

            # Closing the socket on a resubscribe request ends the `async for`
            # cleanly. A plain read loop costs half the CPU per frame of racing
            # a fresh recv() task against the event with asyncio.wait.
            watcher = asyncio.ensure_future(self._close_on_resubscribe(ws))
            try:
                async for raw in ws:
                    self._handle_message(raw)
            finally:
                watcher.cancel()
            if not self._reconnect_needed.is_set():
                logger.warning("clob_ws_closed_by_server", extra={"close_code": ws.close_code})

    async def _close_on_resubscribe(self, ws) -> None:
        await self._reconnect_needed.wait()
        await ws.close()

    def _handle_message(self, raw: str | bytes) -> None:
        self.last_msg_ts = time.time()
        if self._unpersisted:
            self._flush_due_snapshots(self.last_msg_ts)
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
        if event_type == "price_change" and "price_changes" in item:
            self._handle_price_changes(item)
            return
        asset_id = item.get("asset_id")
        if not asset_id:
            return
        condition_id = self.condition_id_by_token.get(asset_id) or item.get("market")
        if asset_id not in self.condition_id_by_token and condition_id:
            self.condition_id_by_token[asset_id] = condition_id
        book = self.books.setdefault(asset_id, OrderBook(asset_id))
        ts_raw = item.get("timestamp")
        event_ts = float(ts_raw) / 1000.0 if ts_raw else None

        if event_type == "book":
            book.apply_snapshot(item.get("bids", []), item.get("asks", []), event_ts, item.get("tick_size"))
            if self._persist_events:
                self._persist_book(asset_id, event_ts, time.time())
        elif event_type == "price_change":
            # older shape: one asset per message, deltas under "changes"
            book.apply_price_change(item.get("changes") or [], event_ts)
            if self._persist_events:
                self._persist_book_throttled(asset_id, event_ts)
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
                if self._persist_events:
                    self.db.insert_clob_trade(
                        condition_id,
                        asset_id,
                        float(price),
                        size,
                        side,
                        item.get("trade_id") or item.get("transaction_hash"),
                        event_ts,
                    )
        elif event_type == "tick_size_change":
            new_tick = item.get("new_tick_size") or item.get("tick_size")
            if new_tick:
                book.tick_size = float(new_tick)
        else:
            logger.debug("clob_ws_unhandled_event", extra={"event_type": event_type, "keys": list(item.keys())})

    def _handle_price_changes(self, item: dict) -> None:
        """Current market-channel shape: one message per market, with
        `price_changes` entries each naming their own asset_id -- there is
        no top-level asset_id. Entries are applied per asset in order."""
        ts_raw = item.get("timestamp")
        event_ts = float(ts_raw) / 1000.0 if ts_raw else None
        by_asset: dict[str, list[dict]] = {}
        for change in item.get("price_changes") or []:
            asset_id = change.get("asset_id")
            if asset_id:
                by_asset.setdefault(asset_id, []).append(change)
        for asset_id, changes in by_asset.items():
            book = self.books.setdefault(asset_id, OrderBook(asset_id))
            book.apply_price_change(changes, event_ts)
            book.prune_to_best(_opt_float(changes[-1].get("best_bid")), _opt_float(changes[-1].get("best_ask")))
            if self._persist_events:
                if asset_id not in self.condition_id_by_token and item.get("market"):
                    self.condition_id_by_token[asset_id] = item["market"]
                self._persist_book_throttled(asset_id, event_ts)

    def _persist_book_throttled(self, asset_id: str, event_ts: float | None) -> None:
        now = time.time()
        last = self._last_persist_recv.get(asset_id)
        if last is None or now - last >= self.settings.clob_book_persist_interval_seconds:
            self._persist_book(asset_id, event_ts, now)
        else:
            self._unpersisted[asset_id] = event_ts

    def _flush_due_snapshots(self, now: float) -> None:
        """Writes the latest state of any token whose last change was held
        back by the throttle, once its interval has passed, so a burst's final
        book always reaches the DB (stamped with that change's own time)."""
        interval = self.settings.clob_book_persist_interval_seconds
        for asset_id, event_ts in list(self._unpersisted.items()):
            if now - self._last_persist_recv.get(asset_id, 0.0) >= interval:
                self._persist_book(asset_id, event_ts, now)

    def _persist_book(self, asset_id: str, event_ts: float | None, now: float) -> None:
        condition_id = self.condition_id_by_token.get(asset_id)
        if condition_id is None:
            return
        bids, asks = self.books[asset_id].as_sorted()
        self.db.insert_book_snapshot(condition_id, asset_id, bids, asks, event_ts)
        self._last_persist_recv[asset_id] = now
        self._unpersisted.pop(asset_id, None)
