"""Process-level stale-feed watchdog.

Each feed reconnects on its own when its socket errors, but on Oct 1-3 2026
the CLOB socket hung for 3.5 days without erroring, and on Oct 15 2026
Binance, CLOB and Chainlink all went silent at once while the process kept
running (and logging pnl_status) for 5+ days. Nothing was recorded or
traded, and nothing noticed.

This loop doesn't try to repair a feed in-process. If any expected feed has
been silent longer than `feed_watchdog_stale_seconds`, it logs an error
(mirrored to Telegram when configured) and asks the runner to shut down
with a non-zero exit code, so an external supervisor
(scripts/run_paper_forever.sh or systemd) starts a fresh process.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Protocol

logger = logging.getLogger(__name__)

EXIT_CODE_STALE_FEED = 3


class _Feed(Protocol):
    last_msg_ts: float | None


def stale_feeds(
    feeds: dict[str, _Feed], started_at: float, threshold: float, now: float
) -> dict[str, float]:
    """Feeds silent for longer than `threshold`, as name -> seconds silent.
    A feed that hasn't sent anything yet counts from `started_at`, so one
    that never connects is caught too."""
    stale = {}
    for name, feed in feeds.items():
        last = feed.last_msg_ts if feed.last_msg_ts is not None else started_at
        silent = now - last
        if silent > threshold:
            stale[name] = round(silent, 1)
    return stale


async def feed_watchdog_loop(
    feeds: dict[str, _Feed],
    threshold: float,
    on_stale: Callable[[dict[str, float]], None],
    check_interval: float = 10.0,
    clock: Callable[[], float] = time.time,
) -> None:
    started_at = clock()
    while True:
        await asyncio.sleep(check_interval)
        stale = stale_feeds(feeds, started_at, threshold, clock())
        if stale:
            logger.error(
                "feed_watchdog_stale",
                extra={"stale_feeds": stale, "threshold_s": threshold, "action": "exiting_for_restart"},
            )
            on_stale(stale)
            return
