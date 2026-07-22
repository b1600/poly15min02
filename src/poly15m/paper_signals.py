"""Phase 2 milestone runner: paper-mode signal logging.

On every Binance tick (throttled to `fair_value_log_interval_seconds`),
computes the feature snapshot and analytic fair value P(Up) = Phi(deviation)
for the currently active market, compares it against the live Polymarket
mid price for the Up token, and logs both to SQLite (`fair_value_log`).

This is read-only / paper-mode: no orders are placed. The Phase 2
milestone (Implementation_Plan.md item 12) isn't a pass/fail check --
it's confirming, from the recorded data, that fair-value/market
dislocations actually appear and gauging how long they persist, before
any execution code gets written. Run this for a while, then inspect
`fair_value_log` (grouped by condition_id, ordered by ts): a `divergence`
column that's frequently non-trivial and stays non-trivial for several
ticks in a row is the signal that funds Phase 3.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from collections import deque

from .config import settings
from .data.binance_ws import BinanceFeed
from .data.clob_ws import ClobFeed
from .data.market_finder import MarketFinder
from .data.window_tracker import WindowTracker
from .db import Database
from .logging_setup import setup_logging
from .pricing.fair_value import compute_fair_value
from .signals.features import FeatureEngine, FeatureSnapshot

logger = logging.getLogger(__name__)


class SignalRunner:
    def __init__(
        self,
        db: Database,
        binance_feed: BinanceFeed,
        clob_feed: ClobFeed,
        tracker: WindowTracker,
        feature_engine: FeatureEngine,
    ):
        self.db = db
        self.binance_feed = binance_feed
        self.clob_feed = clob_feed
        self.tracker = tracker
        self.feature_engine = feature_engine
        self._last_log_ts: dict[str, float] = {}
        self._recent: deque[tuple[float, float]] = deque()  # (ts, divergence), for the status summary

    def on_binance_tick(self, _ts: float, _price: float) -> None:
        market = self.tracker.latest_market()
        if market is None:
            return

        now = time.time()
        last = self._last_log_ts.get(market.condition_id, 0.0)
        if now - last < settings.fair_value_log_interval_seconds:
            return

        open_price = self.tracker.open_price.get(market.condition_id)
        if open_price is None:
            return

        snapshot = self.feature_engine.compute(
            market.condition_id, market.token_id_up, open_price, market.close_ts, now
        )
        if snapshot is None:
            return
        self._last_log_ts[market.condition_id] = now

        fair_value = compute_fair_value(snapshot.deviation)
        mid = self._market_mid_up(market.token_id_up)
        divergence = fair_value.p_up - mid if (fair_value is not None and mid is not None) else None

        self._log(snapshot, fair_value.p_up if fair_value else None, mid, divergence)

        if divergence is not None:
            self._recent.append((now, divergence))
            cutoff = now - 300.0
            while self._recent and self._recent[0][0] < cutoff:
                self._recent.popleft()
            if abs(divergence) >= settings.divergence_alert_threshold:
                logger.info(
                    "large_divergence",
                    extra={
                        "condition_id": market.condition_id,
                        "t_remaining_s": round(snapshot.t_remaining, 1),
                        "deviation": round(snapshot.deviation, 3) if snapshot.deviation is not None else None,
                        "p_up_fair": round(fair_value.p_up, 4),
                        "market_mid_up": round(mid, 4),
                        "divergence": round(divergence, 4),
                    },
                )

    def _market_mid_up(self, token_id_up: str) -> float | None:
        book = self.clob_feed.books.get(token_id_up)
        if book is None:
            return None
        bb, ba = book.best_bid(), book.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def _log(
        self,
        snapshot: FeatureSnapshot,
        p_up_fair: float | None,
        mid: float | None,
        divergence: float | None,
    ) -> None:
        self.db.insert_fair_value_log(
            snapshot.condition_id,
            snapshot.ts,
            snapshot.spot,
            snapshot.open_price,
            snapshot.t_remaining,
            snapshot.sigma,
            snapshot.deviation,
            snapshot.momentum_1m,
            snapshot.momentum_3m,
            snapshot.momentum_5m,
            snapshot.book_imbalance_up,
            snapshot.aggressive_flow_up,
            p_up_fair,
            mid,
            divergence,
        )
        logger.debug(
            "fair_value",
            extra={
                "condition_id": snapshot.condition_id,
                "t_remaining_s": round(snapshot.t_remaining, 1),
                "spot": snapshot.spot,
                "deviation": round(snapshot.deviation, 3) if snapshot.deviation is not None else None,
                "p_up_fair": round(p_up_fair, 4) if p_up_fair is not None else None,
                "market_mid_up": round(mid, 4) if mid is not None else None,
                "divergence": round(divergence, 4) if divergence is not None else None,
            },
        )

    async def status_loop(self, interval: float = 60.0) -> None:
        while True:
            await asyncio.sleep(interval)
            if not self._recent:
                logger.info("signal_status", extra={"divergence_samples_5m": 0})
                continue
            values = [abs(d) for _, d in self._recent]
            logger.info(
                "signal_status",
                extra={
                    "divergence_samples_5m": len(values),
                    "divergence_mean_abs_5m": round(sum(values) / len(values), 4),
                    "divergence_max_abs_5m": round(max(values), 4),
                },
            )


async def clock_loop(tracker: WindowTracker) -> None:
    while True:
        tracker.poll()
        await asyncio.sleep(1.0)


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
    feature_engine = FeatureEngine(settings, binance_feed, clob_feed)
    runner = SignalRunner(db, binance_feed, clob_feed, tracker, feature_engine)
    binance_feed.set_on_tick(runner.on_binance_tick)

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
        asyncio.create_task(runner.status_loop(), name="status_loop"),
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
