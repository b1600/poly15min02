"""Backtester (Implementation_Plan.md Phase 6, item 23): replays recorded
book data through the identical strategy code path, paper simulator
reused, per the plan's own wording.

The design choice that makes "identical" literally true rather than
approximately true: this doesn't reimplement any strategy logic. It
reconstructs synthetic WebSocket-shaped messages from the recorded rows
and feeds them through `BinanceFeed._handle_message` /
`ClobFeed._handle_message` -- the exact same parsers live trading uses --
and drives `PaperTrader.on_binance_tick` / `on_window_resolved` with an
explicit replayed `now` (both accept one; see paper_trade.py). Everything
downstream (`FeatureEngine`, `compute_fair_value`, `PositionManager`,
`RiskGate`, `PaperExecutor`) is untouched, unmodified, the same objects
live/paper trading uses.

Multiple markets replay through *one* engine instance sharing one
`BinanceFeed` (so its rolling buffer accumulates continuously across
windows, like a real always-on bot's would -- not reset per market) and one
`PositionManager`/`RiskGate`/`PaperExecutor` (so daily-loss tracking and
cross-window state behave the way they would live). Each market is only
registered into `WindowTracker` at the simulated moment replay reaches its
`open_ts` -- registering everything up front would make `latest_market()`
jump straight to the last (future) window instead of tracking whichever
one is actually "current" at that point in the replay.

One deliberate divergence from "identical to live": `RiskGate`'s kill
switch is sticky forever in live trading (see risk/limits.py) because a
human has to investigate before the bot resumes. A historical replay has
no human in the loop and can span weeks -- without intervention, one bad
early day would permanently zero out every subsequent day's evaluation
for the rest of the dataset, which defeats the point of a multi-day
backtest/sweep. So `run()` calls `RiskGate.reset_kill_switch_for_new_day()`
once per simulated UTC day, approximating an operator re-arming the bot
each trading day.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from ..config import Settings
from ..data.binance_ws import BinanceFeed
from ..data.clob_ws import ClobFeed
from ..data.clock import WindowClock
from ..data.market_finder import MarketInfo
from ..data.window_tracker import WindowTracker
from ..db import Database
from ..paper_trade import PaperTrader
from ..positions.manager import PositionManager
from ..risk.limits import RiskGate
from ..signals.features import FeatureEngine
from ..sim.paper import PaperExecutor
from . import data_loader
from .data_loader import BinanceTick, BookEvent, MarketRow, TradeEvent

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    condition_ids: list[str]
    num_windows_replayed: int
    realized_pnl: float
    fees_paid: float
    daily_pnl: float
    kill_switch_active: bool
    kill_switch_trigger_count: int = 0


def _next_utc_midnight(ts: float) -> float:
    day = datetime.fromtimestamp(ts, tz=timezone.utc).date()
    next_day = day + timedelta(days=1)
    return datetime(next_day.year, next_day.month, next_day.day, tzinfo=timezone.utc).timestamp()


class BacktestEngine:
    def __init__(self, settings: Settings, output_db: Database | None = None):
        self.settings = settings
        self.db = output_db if output_db is not None else Database(":memory:")
        self.binance_feed = BinanceFeed(settings, self.db)
        self.clob_feed = ClobFeed(settings, self.db)
        self.tracker = WindowTracker(self.db, self.binance_feed, self.clob_feed)
        self.feature_engine = FeatureEngine(settings, self.binance_feed, self.clob_feed)
        self.position_manager = PositionManager(settings)
        # A replay drives its own day rollover via
        # reset_kill_switch_for_new_day() below. Auto re-arm would work
        # here too (decision timestamps are replayed, not wall-clock), but
        # it also brings the hard-halt escalation, which would silently
        # change what every existing sweep result means. Replays keep the
        # old semantics; live/paper get the new ones.
        self.risk_gate = RiskGate(settings.model_copy(update={"kill_switch_auto_rearm": False}), self.db)
        self.executor = PaperExecutor(self.db, settings)
        self.trader = PaperTrader(
            self.db,
            self.binance_feed,
            self.clob_feed,
            self.tracker,
            self.feature_engine,
            self.executor,
            self.position_manager,
            self.risk_gate,
            cfg=settings,
        )
        self._windows_replayed = 0
        self._next_day_boundary_ts: float | None = None

    def _register_market(self, market_row: MarketRow) -> None:
        """Equivalent to WindowTracker.on_new_market, but sources open_price
        directly from the historical record rather than the live
        retry-until-Binance-connects mechanism, which has nothing to key
        off of during replay."""
        info = MarketInfo(
            condition_id=market_row.condition_id,
            slug=market_row.slug,
            question_id=None,
            token_id_up=market_row.token_id_up,
            token_id_down=market_row.token_id_down,
            open_ts=market_row.open_ts,
            close_ts=market_row.close_ts,
            raw={},
        )
        self.db.upsert_market(
            {
                "condition_id": info.condition_id,
                "slug": info.slug,
                "question_id": info.question_id,
                "token_id_up": info.token_id_up,
                "token_id_down": info.token_id_down,
                "window_open_ts": info.open_ts,
                "window_close_ts": info.close_ts,
                "discovered_ts": info.open_ts,
                "raw_json": "{}",
            }
        )
        self.clob_feed.subscribe(info.condition_id, [info.token_id_up, info.token_id_down])
        self.tracker.clocks[info.condition_id] = WindowClock(info.condition_id, info.open_ts, info.close_ts)
        self.tracker.markets[info.condition_id] = info
        if market_row.open_price is not None:
            self.tracker.open_price[info.condition_id] = market_row.open_price
            self.db.set_market_open_price(info.condition_id, market_row.open_price, "backtest_input")
        self._windows_replayed += 1

    def _feed_binance_tick(self, tick: BinanceTick) -> None:
        msg = {"e": "trade", "T": int(tick.ts * 1000), "p": str(tick.price), "q": str(tick.qty or 0), "m": tick.is_buyer_maker}
        self.binance_feed._handle_message(json.dumps(msg))
        self.trader.on_binance_tick(tick.ts, tick.price, now=tick.ts)

    def _feed_book_event(self, condition_id: str, event: BookEvent) -> None:
        msg = {
            "event_type": "book",
            "asset_id": event.token_id,
            # A live message always carries its own "market" field, so
            # ClobFeed can resolve condition_id even before subscribe() has
            # populated condition_id_by_token for this token -- e.g. a
            # market whose recorded book activity started fractionally
            # before its own window_open_ts landed in this replay. Passing
            # the known condition_id here restores that fallback instead of
            # dropping it (which surfaces as a NOT NULL crash on insert).
            "market": condition_id,
            "timestamp": str(int(event.ts * 1000)),
            "bids": [{"price": str(p), "size": str(s)} for p, s in event.bids],
            "asks": [{"price": str(p), "size": str(s)} for p, s in event.asks],
        }
        self.clob_feed._handle_message(json.dumps(msg))

    def _feed_trade_event(self, condition_id: str, event: TradeEvent) -> None:
        msg = {
            "event_type": "last_trade_price",
            "asset_id": event.token_id,
            "market": condition_id,
            "timestamp": str(int(event.ts * 1000)),
            "price": str(event.price),
            "size": str(event.size),
            "side": event.side,
        }
        self.clob_feed._handle_message(json.dumps(msg))

    def run(
        self,
        db_path: str | Path,
        condition_ids: list[str],
        binance_lookback_seconds: float = 1800.0,
        on_window_resolved: Callable[[int, int], None] | None = None,
    ) -> BacktestResult:
        market_rows = [r for r in (data_loader.load_market(db_path, cid) for cid in condition_ids) if r is not None]
        market_rows = [r for r in market_rows if r.open_price is not None]
        if not market_rows:
            return BacktestResult([], 0, 0.0, 0.0, 0.0, False)

        start_ts = min(r.open_ts for r in market_rows) - binance_lookback_seconds
        end_ts = max(r.close_ts for r in market_rows)
        binance_ticks = data_loader.load_binance_ticks(db_path, start_ts, end_ts)

        events: list[tuple[float, int, str, object]] = []
        # priority: market registration and book state should land before a
        # binance tick at the exact same timestamp, so a decision made on
        # that tick sees fresh state
        for r in market_rows:
            events.append((r.open_ts, 0, "market_open", r))
        for t in binance_ticks:
            events.append((t.ts, 2, "binance", t))
        for r in market_rows:
            for e in data_loader.load_book_events(db_path, r.condition_id):
                events.append((e.ts, 1, "book", (r.condition_id, e)))
            for e in data_loader.load_trade_events(db_path, r.condition_id):
                events.append((e.ts, 1, "trade", (r.condition_id, e)))
        events.sort(key=lambda x: (x[0], x[1]))

        total_windows = len(market_rows)
        windows_resolved = 0

        if events:
            self._next_day_boundary_ts = _next_utc_midnight(events[0][0])

        for ts, _priority, kind, payload in events:
            # cheap float comparison per event; the (rare) actual reset +
            # midnight recompute only runs when a day boundary is crossed
            while self._next_day_boundary_ts is not None and ts >= self._next_day_boundary_ts:
                self.risk_gate.reset_kill_switch_for_new_day()
                self._next_day_boundary_ts = _next_utc_midnight(self._next_day_boundary_ts)

            if kind == "market_open":
                self._register_market(payload)
            elif kind == "binance":
                self._feed_binance_tick(payload)
            elif kind == "book":
                self._feed_book_event(*payload)
            elif kind == "trade":
                self._feed_trade_event(*payload)

            for condition_id, event_name in self.tracker.poll(now=ts):
                if event_name == "resolved":
                    self.trader.on_window_resolved(condition_id, now=ts)
                    windows_resolved += 1
                    if on_window_resolved is not None:
                        on_window_resolved(windows_resolved, total_windows)

        # recorded data can realistically end slightly before the exact
        # close_ts (the last live book/tick update just happened to land a
        # moment early) -- one final poll at the latest close_ts makes sure
        # "resolved" still fires for every window rather than silently
        # never triggering.
        final_ts = max(r.close_ts for r in market_rows)
        for condition_id, event_name in self.tracker.poll(now=final_ts):
            if event_name == "resolved":
                self.trader.on_window_resolved(condition_id, now=final_ts)
                windows_resolved += 1
                if on_window_resolved is not None:
                    on_window_resolved(windows_resolved, total_windows)

        self.db.flush()
        return BacktestResult(
            condition_ids=[r.condition_id for r in market_rows],
            num_windows_replayed=self._windows_replayed,
            realized_pnl=self.executor.realized_pnl,
            fees_paid=self.executor.fees_paid,
            daily_pnl=self.risk_gate.daily_pnl,
            kill_switch_active=self.risk_gate.kill_switch_active,
            kill_switch_trigger_count=self.risk_gate.kill_switch_trigger_count,
        )


def run_backtest(
    db_path: str | Path,
    condition_ids: list[str] | None,
    settings: Settings,
    output_db: Database | None = None,
    on_window_resolved: Callable[[int, int], None] | None = None,
) -> tuple[BacktestEngine, BacktestResult]:
    ids = condition_ids if condition_ids is not None else data_loader.list_backtestable_markets(db_path)
    engine = BacktestEngine(settings, output_db)
    result = engine.run(db_path, ids, on_window_resolved=on_window_resolved)
    return engine, result
