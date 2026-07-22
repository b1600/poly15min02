"""Phase 4 milestone runner: LIVE trading with real money (item 19 --
"live with minimum size ($5-20 per market), one market at a time").

This is the only runner in this project capable of spending real funds.
It refuses to start unless `settings.live_trading_enabled` is explicitly
True *and* every Polymarket credential is configured -- see
`execution/executor.py` for why. Nothing in this codebase sets that flag
or supplies credentials; both must come from the operator's own `.env`.
Read `execution/executor.py` and `execution/user_ws.py`'s module
docstrings before ever enabling this -- both are reviewed against
`py-clob-client`'s real API and Polymarket's documented WS shape, but
neither has been exercised against a live account.

Known open question, unverified: whether resolved conditional-token
positions get redeemed to USDC automatically or require an explicit
on-chain redemption call. This runner drops the position from
`PositionManager`'s inventory on resolution (mirroring the paper
simulator) but does not attempt redemption -- confirm how your account
handles this before relying on it.

Reuses the same `PositionManager` decision logic validated in paper mode
(paper_trade.py) -- only the execution layer differs.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time

from .config import settings
from .data.binance_ws import BinanceFeed
from .data.clob_ws import ClobFeed
from .data.market_finder import MarketFinder
from .data.window_tracker import WindowTracker
from .db import Database
from .execution.executor import LiveExecutor, _missing_credentials
from .execution.order_state import OrderStateMachine
from .execution.position_sync import log_remote_positions
from .execution.user_ws import UserFeed
from .logging_setup import setup_logging
from .pricing.fair_value import compute_fair_value
from .positions.manager import PositionManager
from .risk.limits import RiskGate
from .signals.features import FeatureEngine

logger = logging.getLogger(__name__)


class LiveTrader:
    def __init__(
        self,
        db: Database,
        binance_feed: BinanceFeed,
        clob_feed: ClobFeed,
        tracker: WindowTracker,
        feature_engine: FeatureEngine,
        position_manager: PositionManager,
        executor: LiveExecutor,
        user_feed: UserFeed,
        risk_gate: RiskGate,
    ):
        self.db = db
        self.binance_feed = binance_feed
        self.clob_feed = clob_feed
        self.tracker = tracker
        self.feature_engine = feature_engine
        self.position_manager = position_manager
        self.executor = executor
        self.user_feed = user_feed
        self.risk_gate = risk_gate
        self._last_decision_ts: dict[str, float] = {}
        self._last_traded_book_ts: dict[str, float] = {}
        self._subscribed_user_feed: set[str] = set()
        self._kill_switch_handled = False

    def on_binance_tick(self, _ts: float, _price: float) -> None:
        market = self.tracker.latest_market()
        if market is None:
            return

        now = time.time()
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

        # only re-decide once at least one book has moved since our last
        # attempt -- see paper_trade.py for why (avoids re-acting on a
        # static snapshot).
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
            book = book_up if intent.outcome == "up" else book_down
            tick_size = book.tick_size or 0.01
            records = self.executor.submit_split_order(intent, tick_size)
            logger.info(
                "live_order_submitted",
                extra={
                    "condition_id": intent.condition_id,
                    "outcome": intent.outcome,
                    "reason": intent.reason,
                    "size": round(intent.size, 2),
                    "limit_price": intent.limit_price,
                    "levels": len(records),
                },
            )

    def on_new_market(self, market) -> None:
        self.tracker.on_new_market(market)
        if market.condition_id not in self._subscribed_user_feed:
            self._subscribed_user_feed.add(market.condition_id)
            self.user_feed.subscribe(market.condition_id)

    def on_window_resolved(self, condition_id: str) -> None:
        self.executor.cancel_all_for_condition(condition_id)

        now = time.time()
        open_price = self.tracker.open_price.get(condition_id)
        close_price = self.binance_feed.last_price
        if open_price is not None and close_price is not None:
            outcome = "up" if close_price >= open_price else "down"
            self.db.set_market_resolution(condition_id, outcome, now)
            pnl_estimate = self.position_manager.compute_resolution_pnl(condition_id, outcome)
            if pnl_estimate is not None:
                self.risk_gate.record_realized_pnl(pnl_estimate, ts=now)
                logger.warning(
                    "live_resolution_pnl_estimate",
                    extra={
                        "condition_id": condition_id,
                        "outcome": outcome,
                        "pnl_estimate": round(pnl_estimate, 2),
                        "note": "assumes $1/$0 per share at redemption -- verify against your account",
                    },
                )
        else:
            logger.warning("live_resolution_outcome_unknown", extra={"condition_id": condition_id})

        self.position_manager.drop_market(condition_id)
        self.risk_gate.drop_market(condition_id)
        self._handle_kill_switch_if_needed()
        logger.warning(
            "window_resolved_live",
            extra={"condition_id": condition_id, "note": "verify redemption is handled by your account"},
        )

    def _handle_kill_switch_if_needed(self) -> None:
        if self.risk_gate.kill_switch_active and not self._kill_switch_handled:
            self._kill_switch_handled = True
            logger.critical("kill_switch_active_cancelling_all_orders")
            self.executor.cancel_everything()

    async def kill_switch_watchdog_loop(self, interval: float = 5.0) -> None:
        while True:
            await asyncio.sleep(interval)
            self._handle_kill_switch_if_needed()

    async def reprice_loop(self, interval: float = 5.0) -> None:
        while True:
            await asyncio.sleep(interval)
            market = self.tracker.latest_market()
            if market is None:
                continue
            open_price = self.tracker.open_price.get(market.condition_id)
            if open_price is None:
                continue
            now = time.time()
            snapshot = self.feature_engine.compute(
                market.condition_id, market.token_id_up, open_price, market.close_ts, now
            )
            if snapshot is None:
                continue
            fair_value = compute_fair_value(snapshot.deviation)
            if fair_value is None:
                continue
            self.executor.reprice_if_needed(
                market.condition_id, {"up": fair_value.p_up, "down": fair_value.p_down}
            )

    async def reconcile_loop(self, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            market = self.tracker.latest_market()
            try:
                self.executor.reconcile(market.condition_id if market else None)
            except Exception:
                logger.exception("reconcile_failed")


async def clock_loop(tracker: WindowTracker, trader: LiveTrader) -> None:
    while True:
        for condition_id, event in tracker.poll():
            if event == "resolved":
                trader.on_window_resolved(condition_id)
        await asyncio.sleep(1.0)


def _refuse_to_start() -> bool:
    """Returns True if it's safe to refuse (i.e., safety gates aren't met)."""
    if not settings.live_trading_enabled:
        logger.error("refusing_to_start", extra={"reason": "live_trading_enabled is False"})
        return True
    missing = _missing_credentials(settings)
    if missing:
        logger.error("refusing_to_start", extra={"reason": "missing_credentials", "missing": missing})
        return True
    return False


async def startup_reconciliation(executor: LiveExecutor) -> None:
    """Crash-safe restart (Implementation_Plan.md Phase 5, item 22), order
    half: adopt-or-cancel any order the exchange thinks is open that we
    have no local record of -- the safe default for something we restarted
    with no decision-state confidence about is to cancel it, not guess.
    Position half: log the real on-chain snapshot so the operator can
    verify what's actually held (see execution/position_sync.py for why
    that's surfaced rather than auto-merged into PositionManager)."""
    try:
        result = executor.reconcile()
    except Exception:
        logger.exception("startup_reconcile_failed")
    else:
        if result.unknown_remote_orders:
            logger.warning(
                "startup_cancelling_unrecognized_orders", extra={"count": len(result.unknown_remote_orders)}
            )
            for remote in result.unknown_remote_orders:
                order_id = remote.get("id")
                if order_id:
                    executor.cancel_order(order_id)

    address = settings.polymarket_funder_address or executor.client.get_address()
    await log_remote_positions(settings, address)


async def run() -> None:
    settings.ensure_dirs()
    setup_logging(
        settings.log_level,
        settings.log_json,
        telegram_bot_token=settings.telegram_bot_token,
        telegram_chat_id=settings.telegram_chat_id,
    )

    if _refuse_to_start():
        sys.exit(1)

    logger.warning(
        "starting_live_trading",
        extra={"environment": settings.environment, "bankroll": settings.bankroll, "db_path": str(settings.db_path)},
    )

    db = Database(settings.db_path)
    binance_feed = BinanceFeed(settings, db)
    clob_feed = ClobFeed(settings, db)
    market_finder = MarketFinder(settings, db)
    tracker = WindowTracker(db, binance_feed, clob_feed)
    feature_engine = FeatureEngine(settings, binance_feed, clob_feed)
    position_manager = PositionManager(settings)
    risk_gate = RiskGate(settings, db)
    order_state = OrderStateMachine()
    executor = LiveExecutor(settings, order_state)
    user_feed = UserFeed(settings, order_state, position_manager)

    await startup_reconciliation(executor)

    trader = LiveTrader(
        db, binance_feed, clob_feed, tracker, feature_engine, position_manager, executor, user_feed, risk_gate
    )
    binance_feed.set_on_tick(trader.on_binance_tick)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    tasks = [
        asyncio.create_task(binance_feed.run(), name="binance_feed"),
        asyncio.create_task(clob_feed.run(), name="clob_feed"),
        asyncio.create_task(user_feed.run(), name="user_feed"),
        asyncio.create_task(market_finder.run(trader.on_new_market), name="market_finder"),
        asyncio.create_task(clock_loop(tracker, trader), name="clock_loop"),
        asyncio.create_task(trader.reprice_loop(), name="reprice_loop"),
        asyncio.create_task(trader.reconcile_loop(settings.order_poll_interval_seconds), name="reconcile_loop"),
        asyncio.create_task(trader.kill_switch_watchdog_loop(), name="kill_switch_watchdog"),
        asyncio.create_task(db.run_flush_loop(), name="db_flush"),
    ]

    await stop.wait()
    logger.warning("shutting_down_live_trading")
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    db.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
