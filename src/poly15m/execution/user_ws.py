"""Authenticated CLOB user-channel WebSocket (Implementation_Plan.md Phase 4,
item 18): our own order status updates and fills, which drive
`OrderStateMachine` and `PositionManager` reconciliation in real time.

`py-clob-client` has no WebSocket support at all (REST only), so this
mirrors `data/clob_ws.py`'s structure but for the authenticated "user"
channel instead of the public "market" channel.

Unverified: this is built against Polymarket's publicly documented user
channel shape (subscribe with `{"auth": {apiKey, secret, passphrase},
"markets": [...], "type": "user"}`; `order` events for status changes,
`trade` events for fills) but -- like `execution/executor.py` -- has never
been exercised against a live account. `_handle_event` is deliberately
defensive (unrecognized shapes are logged and skipped, never raised) so a
field-name mismatch degrades to "fills stop updating local state and get
caught by the next `reconcile()`" rather than crashing the process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import websockets
from websockets.exceptions import ConnectionClosed

from ..config import Settings
from ..positions.manager import PositionManager
from .order_state import OrderStateMachine

logger = logging.getLogger(__name__)


class UserFeed:
    def __init__(self, settings: Settings, order_state: OrderStateMachine, position_manager: PositionManager):
        self.settings = settings
        self.order_state = order_state
        self.position_manager = position_manager
        self.last_msg_ts: float | None = None

        self._condition_ids: set[str] = set()
        self._reconnect_needed = asyncio.Event()

    def subscribe(self, condition_id: str) -> None:
        if condition_id not in self._condition_ids:
            self._condition_ids.add(condition_id)
            self._reconnect_needed.set()

    def last_msg_age(self, now: float | None = None) -> float | None:
        if self.last_msg_ts is None:
            return None
        now = time.time() if now is None else now
        return now - self.last_msg_ts

    async def run(self) -> None:
        backoff = 1.0
        while True:
            if not self._condition_ids:
                await asyncio.sleep(0.5)
                continue
            try:
                await self._connect_and_listen()
                backoff = 1.0
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as exc:
                logger.warning("user_ws_disconnected", extra={"error": str(exc), "retry_in": backoff})
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _connect_and_listen(self) -> None:
        url = f"{self.settings.clob_ws_base}/user"
        async with websockets.connect(url, ping_interval=10, ping_timeout=10) as ws:
            auth = {
                "apiKey": self.settings.polymarket_api_key,
                "secret": self.settings.polymarket_api_secret,
                "passphrase": self.settings.polymarket_api_passphrase,
            }
            markets = sorted(self._condition_ids)
            await ws.send(json.dumps({"auth": auth, "markets": markets, "type": "user"}))
            logger.info("user_ws_connected", extra={"markets": markets})
            self._reconnect_needed.clear()

            recv_task = asyncio.ensure_future(ws.recv())
            reconnect_task = asyncio.ensure_future(self._reconnect_needed.wait())
            try:
                while True:
                    done, _ = await asyncio.wait(
                        {recv_task, reconnect_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if reconnect_task in done:
                        return
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
            logger.warning("user_ws_bad_json", extra={"raw": str(raw)[:200]})
            return
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if isinstance(item, dict):
                self._handle_event(item)

    def _handle_event(self, item: dict) -> None:
        event_type = item.get("event_type") or item.get("type")
        if event_type == "order":
            self._handle_order_event(item)
        elif event_type == "trade":
            self._handle_trade_event(item)
        else:
            logger.debug("user_ws_unhandled_event", extra={"event_type": event_type, "keys": list(item.keys())})

    def _handle_order_event(self, item: dict) -> None:
        order_id = item.get("id") or item.get("order_id")
        status = str(item.get("status") or "").upper()
        if not order_id:
            return
        if status in ("CANCELED", "CANCELLED"):
            self.order_state.mark_cancelled(order_id)
        elif status == "REJECTED":
            self.order_state.mark_rejected(order_id)

    def _handle_trade_event(self, item: dict) -> None:
        order_id = item.get("taker_order_id") or item.get("order_id") or item.get("id")
        size = item.get("size") or item.get("matched_size")
        price = item.get("price")
        if order_id is None or size is None:
            logger.debug("user_ws_trade_missing_fields", extra={"keys": list(item.keys())})
            return

        fill_size = float(size)
        record = self.order_state.apply_fill(order_id, fill_size)
        if record is None:
            return

        fee = float(item.get("fee", 0.0) or 0.0)
        fill_price = float(price) if price is not None else record.price
        self.position_manager.record_fill(record.condition_id, record.outcome, fill_price, fill_size, fee)
        logger.info(
            "live_fill",
            extra={
                "condition_id": record.condition_id,
                "outcome": record.outcome,
                "price": fill_price,
                "size": fill_size,
            },
        )
