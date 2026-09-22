"""Official settlement grading.

Paper PnL is graded against Polymarket's own settlement, never against our
Binance proxy.

The proxy (`close_price >= open_price`, both from our Binance feed) looks
like a reasonable stand-in, but it shares `open_price` with the
fair-value model that decides the trades. A scorer that shares a
corrupted input with the strategy does not produce noisy PnL -- it
produces *biased* PnL, because the grading error is correlated with the
position the error caused. Measured over Sep 12-22 2026 that turned a
true +0.55% ROI into a reported +23.6%: 72% of "profit" came from the 9%
of windows where the proxy contradicted the market's own verdict, at
+181% ROI, because a mislabel converts a ~$0.05 losing token into a $1.00
winner.

So settlement comes from Gamma, and a window that never settles is
abandoned rather than guessed at -- see `PaperExecutor.abandon_market`.

Gamma shape (verified against live settled markets 2026-09-22): each
market carries `closed`, `outcomes` ('["Up", "Down"]') and
`outcomePrices` ('["1", "0"]' once settled, live mid prices like
'["0.505", "0.495"]' before then). Both list fields come back as JSON
*strings*, not arrays.

Lookup is by event slug, and that choice matters -- the obvious
condition-id routes on /markets are both traps:

  /markets?condition_ids=<id>  -> HTTP 200 with an empty list
  /markets?condition_id=<id>   -> HTTP 200 with a default page of 20
                                  unrelated markets, filter silently ignored

Neither errors, so both would have looked like "not settled yet" forever
and abandoned every window. /events?slug=<slug> filters correctly and
accepts the param repeated, so a batch of pending windows costs one
request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable

import aiohttp

from ..config import Settings

if TYPE_CHECKING:  # pragma: no cover
    from ..db import Database

logger = logging.getLogger(__name__)

OnResolved = Callable[[str, str, float], None]
OnAbandoned = Callable[[str], None]

RESOLUTION_SOURCE = "polymarket_official"

# A settled binary market pays exactly 1/0. Anything less decisive than
# this is a market still trading (or a half-written record), not a
# settlement -- grading on it would reintroduce exactly the guessing this
# module exists to remove.
_SETTLED_WIN = 0.99
_SETTLED_LOSS = 0.01

# Gamma accepts `slug` repeated; keep the query string bounded anyway.
_SLUG_BATCH = 20

_OUTCOME_ALIASES = {"up": "up", "yes": "up", "down": "down", "no": "down"}


def _as_list(value: Any) -> list[Any]:
    """Gamma returns `outcomes`/`outcomePrices` as JSON strings; tolerate
    real lists too in case that ever changes."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def parse_official_outcome(market: dict[str, Any]) -> str | None:
    """Return "up"/"down" if this payload is decisively settled, else None.

    None means "not settled yet, ask again" -- never "assume a side".
    """
    if not market.get("closed"):
        return None

    outcomes = [str(o).strip().lower() for o in _as_list(market.get("outcomes"))]
    try:
        prices = [float(p) for p in _as_list(market.get("outcomePrices"))]
    except (TypeError, ValueError):
        logger.warning(
            "resolution_unparseable_prices",
            extra={"condition_id": market.get("conditionId"), "raw": str(market.get("outcomePrices"))[:120]},
        )
        return None

    if len(outcomes) != len(prices) or not outcomes:
        return None

    winners = [o for o, p in zip(outcomes, prices) if p >= _SETTLED_WIN]
    losers = [o for o, p in zip(outcomes, prices) if p <= _SETTLED_LOSS]
    if len(winners) != 1 or len(losers) != len(outcomes) - 1:
        return None  # still trading, or an indecisive/void settlement

    return _OUTCOME_ALIASES.get(winners[0])


class ResolutionPoller:
    """Polls Gamma for settlement of windows whose trading has closed.

    `fetch` is injectable so the whole retry/timeout/callback state machine
    is testable without network access.
    """

    def __init__(
        self,
        settings: Settings,
        on_resolved: OnResolved,
        on_abandoned: OnAbandoned,
        db: "Database | None" = None,
        fetch: Callable[[Iterable[str]], Awaitable[dict[str, dict[str, Any]]]] | None = None,
    ):
        self.settings = settings
        self.on_resolved = on_resolved
        self.on_abandoned = on_abandoned
        self.db = db
        self._fetch = fetch
        # condition_id -> deadline after which we stop waiting
        self.pending: dict[str, float] = {}
        # condition_id -> event slug, the key Gamma actually filters on
        self._slugs: dict[str, str] = {}

    def enqueue(self, condition_id: str, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if condition_id in self.pending:
            return
        self.pending[condition_id] = now + self.settings.resolution_timeout_seconds
        logger.info("resolution_pending", extra={"condition_id": condition_id})

    async def poll_once(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if not self.pending:
            return

        try:
            found = await self._fetch(list(self.pending))
        except Exception:
            logger.exception("resolution_fetch_failed", extra={"pending": len(self.pending)})
            found = {}

        for condition_id, payload in found.items():
            if condition_id not in self.pending:
                continue
            outcome = parse_official_outcome(payload)
            if outcome is None:
                continue
            del self.pending[condition_id]
            self.on_resolved(condition_id, outcome, now)

        for condition_id, deadline in list(self.pending.items()):
            if now >= deadline:
                del self.pending[condition_id]
                logger.error(
                    "resolution_timeout",
                    extra={
                        "condition_id": condition_id,
                        "waited_s": round(self.settings.resolution_timeout_seconds, 1),
                        "consequence": "cost_basis_stranded_not_graded",
                    },
                )
                self.on_abandoned(condition_id)

    async def run(self) -> None:
        async with aiohttp.ClientSession() as session:
            if self._fetch is None:
                self._fetch = lambda ids: self._fetch_from_gamma(session, ids)
            while True:
                await self.poll_once()
                await asyncio.sleep(self.settings.resolution_poll_interval_seconds)

    def _slug_for(self, condition_id: str) -> str | None:
        """Event slug for a condition id, cached. Read from the markets
        table, which the finder populated at discovery."""
        if condition_id in self._slugs:
            return self._slugs[condition_id]
        if self.db is None:
            return None
        row = self.db._conn.execute(
            "SELECT slug FROM markets WHERE condition_id = ?", (condition_id,)
        ).fetchone()
        if row and row[0]:
            self._slugs[condition_id] = row[0]
            return row[0]
        return None

    async def _fetch_from_gamma(
        self, session: aiohttp.ClientSession, condition_ids: Iterable[str]
    ) -> dict[str, dict[str, Any]]:
        """Look up pending windows by event slug, keyed by conditionId.

        The `closed` filter is deliberately left off -- we want these
        events precisely once they close.
        """
        wanted = {}
        for cid in condition_ids:
            slug = self._slug_for(cid)
            if slug is None:
                logger.warning("resolution_no_slug", extra={"condition_id": cid})
                continue
            wanted[slug] = cid
        if not wanted:
            return {}

        url = f"{self.settings.gamma_api_base}/events"
        by_id: dict[str, dict[str, Any]] = {}
        slugs = list(wanted)
        # Chunked to keep the query string bounded; pending is normally 1-2.
        for start in range(0, len(slugs), _SLUG_BATCH):
            batch = slugs[start : start + _SLUG_BATCH]
            params = [("slug", slug) for slug in batch]
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                resp.raise_for_status()
                payload = await resp.json()

            events = payload if isinstance(payload, list) else payload.get("data", [])
            for event in events:
                if not isinstance(event, dict):
                    continue
                markets = event.get("markets") or []
                if not markets:
                    continue
                m = markets[0]
                cid = m.get("conditionId")
                # Trust the payload's own id, but only for something we asked
                # about -- never grade a window off an unrelated event.
                if cid and cid in wanted.values():
                    by_id[cid] = m

        if not by_id:
            logger.warning("resolution_lookup_empty", extra={"requested": len(wanted)})
        return by_id
