"""One-shot backfill of official settlement outcomes from Gamma, for
windows resolved before `ResolutionPoller` existed.

Every window resolved by the pre-fix code has `resolved_outcome` set from
the Binance close-vs-open proxy and `resolved_source` NULL (the migration
added that column after these rows were written). The proxy shares
`open_price` with the fair-value model that chose the trades, so its
errors correlate with the positions it graded rather than being
independent noise -- see `data/resolution.py` and `sim/paper.py` for the
full story and the measured impact.

This walks every window whose trading has closed and whose
`resolved_source` is not `polymarket_official`, looks it up on Gamma by
event slug (the only route that actually filters -- the condition-id
routes on /markets return HTTP 200 while silently failing to filter, see
`data/resolution.py`'s docstring), and overwrites `resolved_outcome` /
`resolved_source` with the real settlement. The old proxy value is
preserved in `proxy_outcome` before being overwritten, so the divergence
stays auditable rather than disappearing.

`open_price` is a separate, independent problem -- see `backfill.py` for
re-anchoring that.

Dry-run by default; pass --apply to write. Safe to re-run: rows already
marked `polymarket_official` are skipped, so a partial run (network
failure partway through) can simply be repeated.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp

from .config import settings
from .data.resolution import RESOLUTION_SOURCE, parse_official_outcome

logger = logging.getLogger(__name__)

# Batched the same way ResolutionPoller batches live lookups.
_SLUG_BATCH = 20
_BATCH_DELAY_SECONDS = 0.15


async def _fetch_events_from_gamma(
    session: aiohttp.ClientSession, slugs: list[str], batch_size: int = _SLUG_BATCH
) -> dict[str, dict[str, Any]]:
    """slug -> market payload, for every slug Gamma actually has an event for."""
    url = f"{settings.gamma_api_base}/events"
    by_slug: dict[str, dict[str, Any]] = {}
    for start in range(0, len(slugs), batch_size):
        batch = slugs[start : start + batch_size]
        params = [("slug", s) for s in batch]
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            resp.raise_for_status()
            payload = await resp.json()

        events = payload if isinstance(payload, list) else payload.get("data", [])
        for event in events:
            if not isinstance(event, dict):
                continue
            markets = event.get("markets") or []
            if not markets:
                continue
            slug = event.get("slug")
            if slug:
                by_slug[slug] = markets[0]

        if start + batch_size < len(slugs):
            await asyncio.sleep(_BATCH_DELAY_SECONDS)
    return by_slug


async def backfill_outcomes(
    db_path: Path | str,
    apply: bool = False,
    limit: int | None = None,
    batch_size: int = _SLUG_BATCH,
    fetch_events: Callable[[list[str]], Awaitable[dict[str, dict[str, Any]]]] | None = None,
) -> dict[str, int]:
    """`fetch_events` is injectable so this is testable without network
    access; the CLI wires up the real Gamma call."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    now = time.time()

    query = """
        SELECT condition_id, slug, resolved_outcome, window_close_ts
        FROM markets
        WHERE slug IS NOT NULL AND slug != '' AND window_close_ts <= ?
          AND (resolved_source IS NULL OR resolved_source != ?)
        ORDER BY window_close_ts
    """
    params: list = [now, RESOLUTION_SOURCE]
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(query, params).fetchall()
    stats = {
        "examined": len(rows),
        "settled": 0,
        "diverged_from_proxy": 0,
        "still_open_on_gamma": 0,
        "not_found_on_gamma": 0,
    }

    if not rows:
        conn.close()
        return stats

    slug_to_row = {r["slug"]: r for r in rows}

    if fetch_events is None:
        async def fetch_events(slugs: list[str]) -> dict[str, dict[str, Any]]:
            async with aiohttp.ClientSession() as session:
                return await _fetch_events_from_gamma(session, slugs, batch_size)

    by_slug = await fetch_events(list(slug_to_row.keys()))

    for slug, row in slug_to_row.items():
        market = by_slug.get(slug)
        if market is None:
            stats["not_found_on_gamma"] += 1
            continue

        outcome = parse_official_outcome(market)
        if outcome is None:
            stats["still_open_on_gamma"] += 1
            continue

        old_proxy = row["resolved_outcome"]
        if old_proxy is not None and old_proxy != outcome:
            stats["diverged_from_proxy"] += 1

        if apply:
            conn.execute(
                """UPDATE markets
                   SET resolved_outcome = ?, resolved_source = ?,
                       proxy_outcome = COALESCE(proxy_outcome, ?),
                       resolved_ts = COALESCE(resolved_ts, ?)
                   WHERE condition_id = ?""",
                (outcome, RESOLUTION_SOURCE, old_proxy, row["window_close_ts"], row["condition_id"]),
            )
        stats["settled"] += 1

    if apply:
        conn.commit()
    conn.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(settings.db_path), help="path to poly15m.db")
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    parser.add_argument("--limit", type=int, default=None, help="only examine the first N windows")
    parser.add_argument("--batch-size", type=int, default=_SLUG_BATCH, help="slugs per Gamma request")
    args = parser.parse_args()

    stats = asyncio.run(
        backfill_outcomes(args.db, apply=args.apply, limit=args.limit, batch_size=args.batch_size)
    )

    mode = "APPLIED" if args.apply else "DRY RUN (nothing written; pass --apply to write)"
    print(f"official-outcome backfill -- {mode}")
    print(f"  windows examined         : {stats['examined']}")
    print(f"  settled on Gamma         : {stats['settled']}")
    print(f"    diverged from old proxy: {stats['diverged_from_proxy']}")
    print(f"  still open on Gamma      : {stats['still_open_on_gamma']}")
    print(f"  not found on Gamma       : {stats['not_found_on_gamma']}")
    if stats["settled"]:
        rate = 100.0 * stats["diverged_from_proxy"] / stats["settled"]
        print(f"\n  proxy disagreement rate: {rate:.1f}% (historical baseline: ~6.8%)")


if __name__ == "__main__":
    main()
