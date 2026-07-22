"""Crash-safe restart, position half (Implementation_Plan.md Phase 5, item 22).

`OrderStateMachine.reconcile()` (Phase 4) already handles rebuilding *order*
state from the exchange. Rebuilding actual token *inventory* is a harder
problem: py-clob-client's REST API is about orders and the book, not
on-chain balances. Polymarket's public Data API
(https://data-api.polymarket.com/positions?user=<address>) does expose
current holdings -- confirmed live against a real address, response shape
below is not guessed.

Deliberately narrow scope: this fetches and summarizes the remote snapshot
so an operator restarting the bot can SEE what they actually hold before
trusting anything else. It does not attempt to auto-populate
`PositionManager`'s Kelly-sizing inventory from it, because correctly
attributing a raw `asset` (token id) back to "Up" or "Down" for a market
that may have already closed requires looking up that specific market's
`clobTokenIds`/`outcomes` -- solvable, but adds real surface for a
15-minute-bounded edge case (any pre-crash position resolves on its own
within at most one window regardless of whether this reconstructs it).
Treat a restart with open positions as "start flat, but go verify the
logged snapshot," not "seamlessly resumed."

Example response shape (fields used here; live-verified):
    {"proxyWallet": "0x...", "asset": "<token_id>", "conditionId": "0x...",
     "size": 5000, "avgPrice": 0, "outcome": "No", "redeemable": true,
     "title": "...", ...}
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from ..config import Settings

logger = logging.getLogger(__name__)


async def fetch_remote_positions(settings: Settings, address: str) -> list[dict[str, Any]]:
    url = f"{settings.polymarket_data_api_base}/positions"
    async with aiohttp.ClientSession() as session:
        async with session.get(
            url, params={"user": address}, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            resp.raise_for_status()
            return await resp.json()


def summarize_positions(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "condition_id": p.get("conditionId"),
            "token_id": p.get("asset"),
            "outcome": p.get("outcome"),
            "size": p.get("size"),
            "avg_price": p.get("avgPrice"),
            "redeemable": p.get("redeemable"),
            "title": p.get("title"),
        }
        for p in positions
    ]


async def log_remote_positions(settings: Settings, address: str) -> list[dict[str, Any]]:
    """Fetch and log the current on-chain position snapshot for `address`.
    Best-effort: a failure here should never block startup, just warn."""
    try:
        raw = await fetch_remote_positions(settings, address)
    except Exception:
        logger.exception("remote_positions_fetch_failed")
        return []

    summary = summarize_positions(raw)
    if not summary:
        logger.info("remote_positions_empty")
    for p in summary:
        logger.warning("remote_position_on_restart", extra=p)
    return summary
