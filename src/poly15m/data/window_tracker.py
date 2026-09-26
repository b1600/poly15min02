"""Window-lifecycle tracking shared by the Phase 1 recorder and the Phase 2
paper-signals runner.

Registers newly discovered markets (subscribing the CLOB feed, starting a
clock), anchors each window's reference open price to the window's own
open timestamp, and emits lifecycle events as each clock crosses a
milestone.

Open-price anchoring is deliberately strict. An earlier version stamped
`open_price` with `binance_feed.last_price` at the first poll after
discovery (source tag `binance_at_discovery`). Discovery runs on
`market_poll_interval_seconds`, so that sample landed ~7.5s late on
average and up to a full poll interval late -- a mean absolute error of
about $10 against BTC. That is small next to the ~$99 mean absolute 15m
move, but it is *not* small next to the move in the ~7% of windows that
travel less than $10, and those are exactly the windows it mislabels.

The damage was worse than mislabeling alone, because `open_price` feeds
both the fair-value model and (via the Binance proxy) outcome grading:
the scorer shared a corrupted input with the strategy, so the errors
correlated in the strategy's favour and manufactured profit that was not
there.

So: anchor to the last trade at or before `window_open_ts`, require it to
be recent enough to trust, and if no trustworthy anchor can be found
before the deadline, leave `open_price` unset -- which leaves the window
untraded. An untraded window costs nothing; a window traded against a
guessed reference corrupts the track record.
"""

from __future__ import annotations

import logging
import time

from .binance_ws import BinanceFeed
from .clob_ws import ClobFeed
from .clock import WindowClock
from .market_finder import MarketInfo
from ..config import Settings, settings as default_settings
from ..db import Database

logger = logging.getLogger(__name__)

OPEN_PRICE_SOURCE = "binance_at_open"


class WindowTracker:
    def __init__(
        self,
        db: Database,
        binance_feed: BinanceFeed,
        clob_feed: ClobFeed,
        cfg: Settings | None = None,
    ):
        self.db = db
        self.binance_feed = binance_feed
        self.clob_feed = clob_feed
        self.settings = cfg if cfg is not None else default_settings
        self.clocks: dict[str, WindowClock] = {}
        self.markets: dict[str, MarketInfo] = {}
        self.open_price: dict[str, float] = {}
        self._needs_open_price: set[str] = set()
        # condition_ids that hit the anchor deadline without a trustworthy
        # anchor -- kept so we log the give-up exactly once per window.
        self.unanchored: set[str] = set()

    def on_new_market(self, market: MarketInfo) -> None:
        self.clob_feed.subscribe(market.condition_id, [market.token_id_up, market.token_id_down])
        self.clocks[market.condition_id] = WindowClock(market.condition_id, market.open_ts, market.close_ts)
        self.markets[market.condition_id] = market
        self._needs_open_price.add(market.condition_id)

    def close_price(self, condition_id: str) -> float | None:
        """Last trade at or before this window's `close_ts`, for the Binance
        proxy outcome -- the close-side twin of open-price anchoring.

        The window-closed event fires on a 1s clock poll, so reading
        `binance_feed.last_price` there sampled up to ~1s *after* close and
        could flip small-move windows on post-close ticks. Returns None
        when no trustworthy tick exists (same lag tolerance as the open
        anchor), leaving the proxy unset rather than guessed.
        """
        market = self.markets.get(condition_id)
        if market is None:
            return None
        tick = self.binance_feed.price_at(market.close_ts)
        if tick is None:
            tick = self.db.price_at("binance", market.close_ts)
        if tick is None or market.close_ts - tick[0] > self.settings.open_price_max_anchor_lag_seconds:
            return None
        return tick[1]

    def latest_market(self) -> MarketInfo | None:
        if not self.markets:
            return None
        return max(self.markets.values(), key=lambda m: m.open_ts)

    def _anchor_open_price(self, condition_id: str, now: float) -> bool:
        """Try to pin this window's open price to a trade at or before its
        open timestamp. Returns True once the window is settled one way or
        the other (anchored, or given up on) and should stop being retried.
        """
        market = self.markets.get(condition_id)
        if market is None:
            return True  # market gone; nothing left to anchor

        # The anchoring trade cannot exist yet if the window hasn't opened.
        if now < market.open_ts:
            return False

        anchor = self.binance_feed.price_at(market.open_ts)
        if anchor is None:
            # Window opened before this process's buffer starts -- fall back
            # to the recorded tick history, which a previous run may have.
            anchor = self.db.price_at("binance", market.open_ts)

        if anchor is not None:
            anchor_ts, price = anchor
            lag = market.open_ts - anchor_ts
            if lag <= self.settings.open_price_max_anchor_lag_seconds:
                self.open_price[condition_id] = price
                self.db.set_market_open_price(condition_id, price, OPEN_PRICE_SOURCE, anchor_ts)
                logger.info(
                    "open_price_anchored",
                    extra={
                        "condition_id": condition_id,
                        "open_price": price,
                        "window_open_ts": market.open_ts,
                        "anchor_lag_s": round(lag, 3),
                    },
                )
                return True

        if now - market.open_ts >= self.settings.open_price_anchor_deadline_seconds:
            self.unanchored.add(condition_id)
            logger.warning(
                "open_price_anchor_failed",
                extra={
                    "condition_id": condition_id,
                    "window_open_ts": market.open_ts,
                    "anchor_ts": anchor[0] if anchor else None,
                    "reason": "no_trade_at_or_before_open" if anchor is None else "anchor_too_stale",
                    "consequence": "window_left_untraded",
                },
            )
            return True

        return False  # keep retrying until the deadline

    def poll(self, now: float | None = None) -> list[tuple[str, str]]:
        """Advance clocks and retry open-price anchoring. Returns
        newly-crossed (condition_id, event) pairs since the last call."""
        now = time.time() if now is None else now

        for condition_id in list(self._needs_open_price):
            if self._anchor_open_price(condition_id, now):
                self._needs_open_price.discard(condition_id)

        crossed: list[tuple[str, str]] = []
        expired = []
        for condition_id, clock in list(self.clocks.items()):
            for event in clock.poll(now):
                logger.info("lifecycle_event", extra={"condition_id": condition_id, "event": event})
                self.db.insert_lifecycle_event(condition_id, event, now)
                crossed.append((condition_id, event))
            if now > clock.close_ts + 30:
                expired.append(condition_id)

        for condition_id in expired:
            self.clocks.pop(condition_id, None)
            self.markets.pop(condition_id, None)
            self.open_price.pop(condition_id, None)
            self._needs_open_price.discard(condition_id)
            self.unanchored.discard(condition_id)

        return crossed
