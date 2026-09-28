"""Chainlink 60s BTC/USD TWAP recorder -- the price 15m markets settle on.

Each market's rules name Chainlink's `btc-usd-twap-60s-streams` as the
resolution source, while the strategy prices off Binance spot. Recording
the settlement feed next to Binance makes that basis measurable (proxy vs
official divergence ran ~6% of windows in the Sep 23-26 shakeout).

Record-only: ticks go to `price_ticks` under `source = SOURCE` and nothing
in the trading path reads them.

Polymarket relays the TWAP on PolyBolt's `price.crypto.twap` channel
(docs.polymarket.com/api-reference/live-data/overview). It is gated on CLOB
API credentials; without them `run()` logs once and returns, so the rest of
the bot is unaffected. (The plain `price.crypto` channel is Pyth, not
Chainlink, and the old RTDS `crypto_prices_chainlink` topics are deprecated.)

Every (re)subscribe replays the preceding two minutes as a snapshot. Points
at or before the newest stored `event_ts` are skipped, so reconnects fill
short gaps without writing duplicates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time

import websockets
from websockets.exceptions import ConnectionClosed

from ..config import Settings
from ..db import Database

logger = logging.getLogger(__name__)

SOURCE = "chainlink_twap60"
CHANNEL = "price.crypto.twap"

# Close codes that retrying cannot fix without operator action (bad
# credentials, a malformed request) -- back off hard instead of hammering.
_FATAL_CLOSE_CODES = {4001, 4008}
_FATAL_RETRY_SECONDS = 300.0


class _AuthRejected(Exception):
    pass


class SettlementTwapFeed:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.symbol = settings.settlement_twap_symbol.lower()
        self.last_value: float | None = None
        self.last_event_ts: float | None = db.latest_tick_ts(SOURCE)
        self.last_msg_ts: float | None = None  # local receive time of the last data frame

    @property
    def has_credentials(self) -> bool:
        s = self.settings
        return bool(s.polymarket_api_key and s.polymarket_api_secret and s.polymarket_api_passphrase)

    def last_msg_age(self, now: float | None = None) -> float | None:
        if self.last_msg_ts is None:
            return None
        now = time.time() if now is None else now
        return now - self.last_msg_ts

    async def run(self) -> None:
        if not self.has_credentials:
            logger.warning(
                "settlement_twap_disabled",
                extra={"reason": "POLY15M_POLYMARKET_API_KEY/SECRET/PASSPHRASE not set"},
            )
            return

        backoff = 1.0
        while True:
            try:
                await self._connect_and_listen()
                backoff = 1.0
                continue  # stale-feed reconnect: go straight back in
            except _AuthRejected as exc:
                delay = _FATAL_RETRY_SECONDS
                logger.error("settlement_twap_auth_rejected", extra={"code": str(exc), "retry_in": delay})
            except ConnectionClosed as exc:
                code = exc.rcvd.code if exc.rcvd is not None else None
                if code in _FATAL_CLOSE_CODES:
                    delay = _FATAL_RETRY_SECONDS
                    logger.error("settlement_twap_closed", extra={"close_code": code, "retry_in": delay})
                else:
                    # full jitter, per PolyBolt's reconnect guidance
                    delay = random.uniform(0, backoff)
                    backoff = min(backoff * 2, 30.0)
                    logger.warning(
                        "settlement_twap_disconnected",
                        extra={"close_code": code, "error": str(exc), "retry_in": round(delay, 1)},
                    )
            except (OSError, asyncio.TimeoutError) as exc:
                delay = random.uniform(0, backoff)
                backoff = min(backoff * 2, 30.0)
                logger.warning(
                    "settlement_twap_disconnected", extra={"error": str(exc), "retry_in": round(delay, 1)}
                )
            await asyncio.sleep(delay)

    async def _connect_and_listen(self) -> None:
        s = self.settings
        async with websockets.connect(s.polybolt_ws_base, ping_interval=20, ping_timeout=20) as ws:
            await ws.send(
                json.dumps(
                    {
                        "op": "auth",
                        "rid": "a1",
                        "auth": {
                            "apiKey": s.polymarket_api_key,
                            "secret": s.polymarket_api_secret,
                            "passphrase": s.polymarket_api_passphrase,
                        },
                    }
                )
            )
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=s.settlement_twap_stale_seconds)
                except asyncio.TimeoutError:
                    logger.warning(
                        "settlement_twap_stale",
                        extra={"silent_for_s": s.settlement_twap_stale_seconds},
                    )
                    return
                for item in _parse(raw):
                    if item.get("op") == "authed":
                        await ws.send(json.dumps(self._subscribe_frame()))
                    else:
                        self._handle_item(item)

    def _subscribe_frame(self) -> dict:
        return {
            "op": "subscribe",
            "rid": "s1",
            "subscriptions": [
                {"channel": CHANNEL, "filter": {"symbol": self.symbol, "window_seconds": 60}}
            ],
        }

    def _handle_item(self, item: dict) -> None:
        op = item.get("op")
        if op == "subscribed":
            logger.info("settlement_twap_subscribed", extra={"symbol": self.symbol})
            return
        if op == "error":
            code = item.get("code")
            if code == "auth_invalid":
                raise _AuthRejected(code)
            logger.warning("settlement_twap_error", extra={"code": code, "channel": item.get("channel")})
            return
        if op is not None:
            return  # pong / unsubscribed

        # Data envelope. The raw API uses `channel` + `snapshot: true`; the
        # SDK docs show `topic` + `type: "subscribe"`. Accept either.
        channel = item.get("channel") or item.get("topic") or ""
        if "twap" not in channel:
            return
        payload = item.get("payload") or {}
        if item.get("dropped"):
            logger.warning("settlement_twap_dropped", extra={"dropped": item.get("dropped")})
        self.last_msg_ts = time.time()
        points = payload.get("data") if "data" in payload else [payload]
        for point in points or []:
            self._record(point)

    def _record(self, point: dict) -> None:
        raw_ts = point.get("timestamp")
        raw_value = point.get("full_accuracy_value", point.get("value"))
        if raw_ts is None or raw_value is None:
            return
        ts = float(raw_ts) / 1000.0
        if self.last_event_ts is not None and ts <= self.last_event_ts:
            return
        value = float(raw_value)
        self.db.insert_tick(SOURCE, self.symbol.upper(), value, None, ts)
        self.last_event_ts = ts
        self.last_value = value


def _parse(raw: str | bytes) -> list[dict]:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("settlement_twap_bad_json", extra={"raw": str(raw)[:200]})
        return []
    items = msg if isinstance(msg, list) else [msg]
    return [i for i in items if isinstance(i, dict)]
