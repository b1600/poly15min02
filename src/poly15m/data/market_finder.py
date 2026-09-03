"""Polymarket market discovery.

Polls the Gamma API for the "BTC Up or Down 15m" recurring series and
figures out which event is the one currently *inside* its trading window
(open_ts <= now < close_ts), handling rollover to the next window
automatically. No auth required -- this is public read-only data.

Discovery is scoped to the series via its numeric series id (resolved once
via GET /series?slug=<market_series_slug> and cached) rather than paging
through the generic GET /events firehose. That firehose is shared by every
market on the platform ordered by creation time, and this series alone
creates a new 15-minute window every 15 minutes across seven-plus coins at
both 5m and 15m granularity -- often enough that the currently in-progress
BTC-15m window ages out of any reasonably-sized page before the very old
listing that hasn't been cleaned up yet does. Scoping to series_id narrows
the candidate set to roughly the ~24h of windows this series keeps
pre-listed, which reliably contains the active one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import aiohttp

from ..config import Settings
from ..db import Database

logger = logging.getLogger(__name__)

OnNewMarket = Callable[["MarketInfo"], None]


@dataclass
class MarketInfo:
    condition_id: str
    slug: str
    question_id: str | None
    token_id_up: str
    token_id_down: str
    open_ts: float
    close_ts: float
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def to_db_row(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "slug": self.slug,
            "question_id": self.question_id,
            "token_id_up": self.token_id_up,
            "token_id_down": self.token_id_down,
            "window_open_ts": self.open_ts,
            "window_close_ts": self.close_ts,
            "discovered_ts": time.time(),
            "raw_json": json.dumps(self.raw)[:20000],
        }


def _parse_iso(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    except ValueError:
        return None


def _select_market(
    events: list[dict[str, Any]], slug_prefix: str, window_seconds: int, now: float
) -> dict[str, Any] | None:
    candidates = [e for e in events if e.get("slug", "").startswith(slug_prefix)]

    in_window: dict[str, Any] | None = None
    upcoming: dict[str, Any] | None = None
    upcoming_open_ts = float("inf")

    for event in candidates:
        close_ts = _parse_iso(event.get("endDate"))
        if close_ts is None:
            continue
        # `eventStartTime` (the true window-open / price-anchor timestamp) is
        # present on the single-event endpoint but comes back null on the
        # list endpoint we poll here; `startDate` is just when the market
        # record was created for trading, which can be ~24h before the
        # window and is NOT the window open. Every window in this series is
        # exactly `window_seconds` long and `endDate` is always populated,
        # so derive open_ts from it rather than trusting either field above.
        open_ts = _parse_iso(event.get("eventStartTime")) or (close_ts - window_seconds)
        event["_open_ts"] = open_ts
        event["_close_ts"] = close_ts
        if open_ts <= now < close_ts:
            if in_window is None or open_ts > in_window["_open_ts"]:
                in_window = event
        elif open_ts > now and open_ts < upcoming_open_ts:
            upcoming = event
            upcoming_open_ts = open_ts

    return in_window or upcoming


class MarketFinder:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.current: MarketInfo | None = None
        self._series_id: str | None = None

    async def run(self, on_new_market: OnNewMarket) -> None:
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    if self._series_id is None:
                        self._series_id = await self._resolve_series_id(session)
                    market = await self._fetch(session)
                except Exception:
                    logger.exception("market_finder_poll_failed")
                    market = None

                if market and (self.current is None or market.condition_id != self.current.condition_id):
                    logger.info(
                        "market_rollover",
                        extra={
                            "slug": market.slug,
                            "condition_id": market.condition_id,
                            "open_ts": market.open_ts,
                            "close_ts": market.close_ts,
                        },
                    )
                    self.current = market
                    self.db.upsert_market(market.to_db_row())
                    on_new_market(market)

                await asyncio.sleep(self.settings.market_poll_interval_seconds)

    async def _resolve_series_id(self, session: aiohttp.ClientSession) -> str:
        url = f"{self.settings.gamma_api_base}/series"
        params = {"slug": self.settings.market_series_slug}
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            series_list = await resp.json()
        if not series_list:
            raise RuntimeError(f"no series found for slug={self.settings.market_series_slug!r}")
        series_id = series_list[0]["id"]
        logger.info(
            "market_finder_series_resolved",
            extra={"series_slug": self.settings.market_series_slug, "series_id": series_id},
        )
        return series_id

    async def _fetch(self, session: aiohttp.ClientSession) -> MarketInfo | None:
        url = f"{self.settings.gamma_api_base}/events"
        params = {
            "series_id": self._series_id,
            "closed": "false",
            "limit": 100,
            "order": "startDate",
            "ascending": "true",
        }
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            events = await resp.json()

        event = _select_market(
            events, self.settings.market_slug_prefix, self.settings.market_window_seconds, time.time()
        )
        if event is None:
            return None

        markets = event.get("markets") or []
        if not markets:
            return None
        m = markets[0]

        try:
            token_ids = json.loads(m.get("clobTokenIds", "[]"))
            outcomes = json.loads(m.get("outcomes", "[]"))
        except json.JSONDecodeError:
            logger.warning("market_finder_bad_payload", extra={"slug": event.get("slug")})
            return None

        token_by_outcome = dict(zip(outcomes, token_ids))
        token_up = token_by_outcome.get("Up") or (token_ids[0] if token_ids else None)
        token_down = token_by_outcome.get("Down") or (token_ids[1] if len(token_ids) > 1 else None)
        if not token_up or not token_down or not m.get("conditionId"):
            logger.warning("market_finder_missing_fields", extra={"slug": event.get("slug")})
            return None

        return MarketInfo(
            condition_id=m["conditionId"],
            slug=event["slug"],
            question_id=m.get("questionID"),
            token_id_up=token_up,
            token_id_down=token_down,
            open_ts=event["_open_ts"],
            close_ts=event["_close_ts"],
            raw=event,
        )
