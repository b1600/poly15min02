"""Phase 1 milestone runner.

Wires the read-only data layer together: Binance feed, Polymarket market
discovery/rollover, Polymarket CLOB feed, and the window clock, all
recording to SQLite. Run this for a few hours and then check:
  - opening prices got recorded for each window (see `markets.open_price`)
  - book snapshots are arriving steadily for both tokens of the active
    market (see `book_snapshots`, grouped by token_id)
  - lifecycle events fire in order with no gaps (see `lifecycle_events`)
  - neither feed goes stale for long stretches (watch the WARNING logs)

Known gap (left for a later phase, not required for Phase 1): actual
Up/Down resolution outcomes aren't fetched yet, so `markets.resolved_outcome`
stays NULL. The Gamma API doesn't expose the window's Chainlink-anchored
opening price directly either -- `open_price` here is our own
Binance-sourced snapshot taken at market-discovery time, used later as the
fair-value reference; treat it as an approximation, not the authoritative
resolution price.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time

from .config import settings
from .data.binance_ws import BinanceFeed
from .data.clob_ws import ClobFeed
from .data.market_finder import MarketFinder
from .data.window_tracker import WindowTracker
from .db import Database
from .logging_setup import setup_logging

logger = logging.getLogger(__name__)


async def clock_loop(tracker: WindowTracker) -> None:
    while True:
        tracker.poll()
        await asyncio.sleep(1.0)


async def status_loop(tracker: WindowTracker, binance_feed: BinanceFeed, clob_feed: ClobFeed, interval: float = 60.0) -> None:
    while True:
        await asyncio.sleep(interval)
        active = tracker.latest_market()
        clock = tracker.clocks.get(active.condition_id) if active else None
        binance_age = binance_feed.last_trade_age()
        clob_age = clob_feed.last_msg_age()
        logger.info(
            "status",
            extra={
                "active_condition_id": active.condition_id if active else None,
                "time_remaining_s": round(clock.time_remaining(), 1) if clock else None,
                "binance_last_price": binance_feed.last_price,
                "binance_feed_age_s": round(binance_age, 1) if binance_age is not None else None,
                "clob_feed_age_s": round(clob_age, 1) if clob_age is not None else None,
                "tracked_windows": len(tracker.clocks),
            },
        )
        if binance_age is not None and binance_age > settings.feed_staleness_seconds * 5:
            logger.warning("binance_feed_stale", extra={"age_s": round(binance_age, 1)})
        if clob_age is not None and clob_age > settings.feed_staleness_seconds * 5:
            logger.warning("clob_feed_stale", extra={"age_s": round(clob_age, 1)})


async def run() -> None:
    settings.ensure_dirs()
    setup_logging(
        settings.log_level,
        settings.log_json,
        telegram_bot_token=settings.telegram_bot_token,
        telegram_chat_id=settings.telegram_chat_id,
    )
    logger.info("starting", extra={"environment": settings.environment, "db_path": str(settings.db_path)})

    db = Database(settings.db_path)
    binance_feed = BinanceFeed(settings, db)
    clob_feed = ClobFeed(settings, db)
    market_finder = MarketFinder(settings, db)
    tracker = WindowTracker(db, binance_feed, clob_feed)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass  # e.g. Windows

    tasks = [
        asyncio.create_task(binance_feed.run(), name="binance_feed"),
        asyncio.create_task(clob_feed.run(), name="clob_feed"),
        asyncio.create_task(market_finder.run(tracker.on_new_market), name="market_finder"),
        asyncio.create_task(clock_loop(tracker), name="clock_loop"),
        asyncio.create_task(status_loop(tracker, binance_feed, clob_feed), name="status_loop"),
        asyncio.create_task(db.run_flush_loop(), name="db_flush"),
    ]

    await stop.wait()
    logger.info("shutting_down")
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    db.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
