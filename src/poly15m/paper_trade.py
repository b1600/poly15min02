"""Phase 3/4 milestone runner: paper trading.

On every throttled Binance tick, computes fair value and asks
`PositionManager` (Phase 4 item 16) what to do about it -- matched-pair
arbitrage, temporal-hedge fulfillment, or a Kelly-sized directional bet --
and simulates each resulting `TradeIntent` via `PaperExecutor` (Phase 3
item 14). At window resolution, PnL is realized: winning side pays
$1/share, losing side $0.

This is the safe way to validate Phase 4's decision logic: real live data,
real book depth, zero financial risk. The live counterpart (`live_trade.py`)
reuses the exact same `PositionManager` -- only the execution layer
(simulated fill vs. a real `py-clob-client` order) differs.

Known approximation: Polymarket resolves against Chainlink's BTC/USD
stream, which we don't have a feed for. Resolution outcome here uses our
own recorded Binance open price vs. Binance price at window close instead
-- the same proxy already used for `open_price` (see record.py) -- so paper
PnL is only as accurate as that proxy.

Milestone (item 15): let this run for >=1-2 days of live data, then check
`paper_pnl_log` -- only move to live trading (Phase 4) if realized PnL is
positive after fees.
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
from .pricing.fair_value import compute_fair_value
from .positions.manager import PositionManager
from .risk.limits import RiskGate
from .signals.features import FeatureEngine
from .sim.paper import PaperExecutor

logger = logging.getLogger(__name__)


class PaperTrader:
    def __init__(
        self,
        db: Database,
        binance_feed: BinanceFeed,
        clob_feed: ClobFeed,
        tracker: WindowTracker,
        feature_engine: FeatureEngine,
        executor: PaperExecutor,
        position_manager: PositionManager,
        risk_gate: RiskGate,
    ):
        self.db = db
        self.binance_feed = binance_feed
        self.clob_feed = clob_feed
        self.tracker = tracker
        self.feature_engine = feature_engine
        self.executor = executor
        self.position_manager = position_manager
        self.risk_gate = risk_gate
        self._last_decision_ts: dict[str, float] = {}
        # condition_id -> latest of both books' last_update_ts as of our
        # last trade attempt. The live order book is shared market state,
        # not something our own simulated fills deplete -- without this,
        # re-walking the same unchanged books every throttled tick would
        # let us "fill" against the same resting liquidity over and over.
        # Only attempt another decision once a book has genuinely moved.
        self._last_traded_book_ts: dict[str, float] = {}

    def on_binance_tick(self, _ts: float, _price: float, now: float | None = None) -> None:
        """`now` is overridable so the backtest engine (Phase 6) can drive
        this exact method with a replayed timestamp instead of the wall
        clock -- live callers never pass it, so behavior there is
        unchanged."""
        market = self.tracker.latest_market()
        if market is None:
            return

        now = now if now is not None else time.time()
        if self.risk_gate.feeds_stale(self.binance_feed.last_trade_age(now), self.clob_feed.last_msg_age(now)):
            return

        last = self._last_decision_ts.get(market.condition_id, 0.0)
        if now - last < settings.fair_value_log_interval_seconds:
            return
        self._last_decision_ts[market.condition_id] = now

        open_price = self.tracker.open_price.get(market.condition_id)
        if open_price is None:
            return
        snapshot = self.feature_engine.compute(
            market.condition_id, market.token_id_up, open_price, market.close_ts, now
        )
        if snapshot is None:
            return
        fair_value = compute_fair_value(snapshot.deviation)
        if fair_value is None:
            return

        book_up = self.clob_feed.books.get(market.token_id_up)
        book_down = self.clob_feed.books.get(market.token_id_down)
        if book_up is None or book_down is None:
            return

        # calibration (Phase 6 item 24) needs (deviation, actual outcome)
        # pairs -- log every decision-eligible tick regardless of whether a
        # trade happens, so a single paper_trade.py run builds that dataset
        # instead of requiring a separate paper_signals.py run alongside it.
        mid_up = self._market_mid(book_up)
        divergence = fair_value.p_up - mid_up if mid_up is not None else None
        self.db.insert_fair_value_log(
            market.condition_id,
            now,
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
            fair_value.p_up,
            mid_up,
            divergence,
        )

        latest_book_ts = max(
            (t for t in (book_up.last_update_ts, book_down.last_update_ts) if t is not None), default=None
        )
        last_traded = self._last_traded_book_ts.get(market.condition_id)
        if latest_book_ts is not None and last_traded is not None and latest_book_ts <= last_traded:
            return

        _, asks_up = book_up.as_sorted()
        _, asks_down = book_down.as_sorted()
        intents = self.position_manager.decide_trades(
            market.condition_id,
            market.token_id_up,
            market.token_id_down,
            fair_value,
            asks_up,
            asks_down,
            snapshot.t_remaining,
            snapshot.sigma,
        )
        if not intents:
            return
        if latest_book_ts is not None:
            self._last_traded_book_ts[market.condition_id] = latest_book_ts

        for intent in intents:
            intent = self.risk_gate.check_intent(intent, self.position_manager, snapshot.t_remaining, snapshot.deviation)
            if intent is None:
                continue
            asks = asks_up if intent.outcome == "up" else asks_down
            fill = self.executor.submit_order(
                intent.condition_id, intent.token_id, intent.outcome, asks, intent.size, intent.limit_price, now
            )
            if fill is None:
                continue
            self.position_manager.record_fill(fill.condition_id, fill.outcome, fill.price, fill.size, fill.fee)
            logger.info(
                "trade_decision",
                extra={
                    "condition_id": intent.condition_id,
                    "outcome": intent.outcome,
                    "reason": intent.reason,
                    "edge": round(intent.edge, 4) if intent.edge is not None else None,
                    "limit_price": intent.limit_price,
                    "filled_price": round(fill.price, 4),
                    "filled_size": round(fill.size, 2),
                    "t_remaining_s": round(snapshot.t_remaining, 1),
                },
            )

    def _market_mid(self, book) -> float | None:
        bb, ba = book.best_bid(), book.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def on_window_resolved(self, condition_id: str, now: float | None = None) -> None:
        open_price = self.tracker.open_price.get(condition_id)
        close_price = self.binance_feed.last_price
        now = now if now is not None else time.time()
        if open_price is None or close_price is None:
            logger.warning("resolution_skipped_missing_price", extra={"condition_id": condition_id})
            return

        outcome = "up" if close_price >= open_price else "down"
        self.db.set_market_resolution(condition_id, outcome, now)
        resolution = self.executor.resolve_market(condition_id, outcome)

        self.db.insert_paper_pnl_log(
            now,
            condition_id,
            "resolution",
            self.executor.realized_pnl,
            self.executor.fees_paid,
            None,
            None,
            None,
            None,
        )
        if resolution is not None:
            logger.info(
                "window_resolved",
                extra={
                    "condition_id": condition_id,
                    "outcome": outcome,
                    "open_price": open_price,
                    "close_price": close_price,
                    "pnl": round(resolution.pnl, 2),
                    "realized_pnl_total": round(self.executor.realized_pnl, 2),
                },
            )
            self.risk_gate.record_realized_pnl(resolution.pnl, ts=now)
        self.position_manager.drop_market(condition_id)
        self.risk_gate.drop_market(condition_id)

    async def status_loop(self, interval: float = 60.0) -> None:
        while True:
            await asyncio.sleep(interval)
            logger.info(
                "pnl_status",
                extra={
                    "realized_pnl": round(self.executor.realized_pnl, 2),
                    "fees_paid": round(self.executor.fees_paid, 2),
                    "open_positions": len(self.executor.positions),
                    "daily_pnl": round(self.risk_gate.daily_pnl, 2),
                    "kill_switch_active": self.risk_gate.kill_switch_active,
                },
            )
            self.db.insert_paper_pnl_log(
                time.time(),
                None,
                "status",
                self.executor.realized_pnl,
                self.executor.fees_paid,
                None,
                None,
                None,
                None,
            )


async def clock_loop(tracker: WindowTracker, trader: PaperTrader) -> None:
    while True:
        for condition_id, event in tracker.poll():
            if event == "resolved":
                trader.on_window_resolved(condition_id)
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
    executor = PaperExecutor(db, settings)
    position_manager = PositionManager(settings)
    risk_gate = RiskGate(settings, db)
    trader = PaperTrader(db, binance_feed, clob_feed, tracker, feature_engine, executor, position_manager, risk_gate)
    binance_feed.set_on_tick(trader.on_binance_tick)

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
        asyncio.create_task(clock_loop(tracker, trader), name="clock_loop"),
        asyncio.create_task(trader.status_loop(), name="status_loop"),
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
