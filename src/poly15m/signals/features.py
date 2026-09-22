"""Signal feature computation, recomputed on each Binance tick.

Implements the four feature groups from Strategy_v1.md / Implementation_Plan.md:
  - Deviation: (spot - open_price) / (sigma * sqrt(t_remaining))
  - Momentum: 1m / 3m / 5m simple returns
  - Realized volatility: EWMA of 1s price-difference variance
  - Book imbalance and aggressive trade flow on the CLOB

`resample_to_bars` is called once per throttled decision (by default
every ~1s per FeatureEngine, unthrottled in backtest replay) and, taken
literally, rescans the full `vol_lookback_seconds` window of raw ticks
from scratch every time -- roughly 12,000 ticks at Binance's live trade
rate. That's invisible in live trading (there's slack between real
ticks), but a backtest replays ticks back-to-back with no idle time, so
this dominates replay runtime (profiled at >50% of total). `_RollingBars`
below reproduces the exact same output incrementally -- O(new ticks since
the last call) instead of O(the whole window) -- and is the only thing
`FeatureEngine` actually calls; `resample_to_bars` stays untouched as the
reference implementation, exercised directly by its own tests and by the
differential tests in test_features.py that check `_RollingBars` against
it on real recorded tick sequences.

The vol/momentum helpers are pure functions over a plain (ts, price) buffer
so they're testable without a live feed; `FeatureEngine` just wires them up
against the live `BinanceFeed` / `ClobFeed` instances.

Note the deviation formula models the *price level* (not log-price) as a
driftless Brownian motion -- sigma is therefore in dollars per sqrt(second),
not a log-return volatility. That's the same simplification
Implementation_Plan.md specifies for the analytic fair-value model: it's a
reasonable approximation over a 15-minute horizon where moves are small
relative to price, not a claim that BTC actually follows arithmetic
Brownian motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..config import Settings
from ..data.binance_ws import BinanceFeed
from ..data.clob_ws import ClobFeed

Trade = tuple[float, float, float, "str | None"]  # (ts, price, size, side)


def resample_to_bars(
    buffer: Sequence[tuple[float, float]], bar_seconds: float, now: float, lookback_seconds: float
) -> list[float]:
    """Forward-filled price at each bar boundary within [now - lookback, now].

    `buffer` (BinanceFeed's rolling tick buffer) is retained for
    `binance_buffer_seconds`, which is deliberately much longer than
    `lookback_seconds` here -- so walking it front-to-back on every call
    would rescan far more history than this call needs. Instead walk
    backward from the newest tick (cheap on a deque) and stop as soon as
    we've passed `start`, keeping only the relevant tail plus one seed
    price (the latest tick at/before `start`) to forward-fill the first bar.
    """
    if not buffer:
        return []
    start = now - lookback_seconds
    relevant: list[tuple[float, float]] = []
    seed_price: float | None = None
    for ts, price in reversed(buffer):
        if ts <= start:
            seed_price = price
            break
        relevant.append((ts, price))
    relevant.reverse()

    bars: list[float] = []
    last_price = seed_price
    i = 0
    n = len(relevant)
    t = start
    while t <= now:
        while i < n and relevant[i][0] <= t:
            last_price = relevant[i][1]
            i += 1
        if last_price is not None:
            bars.append(last_price)
        t += bar_seconds
    return bars


class _RollingBars:
    """Incremental equivalent of `resample_to_bars(buffer, bar_seconds,
    now, lookback_seconds)`, for a single (bar_seconds, lookback_seconds)
    pair called repeatedly with non-decreasing `now`.

    Why this is exact, not approximate: `resample_to_bars`'s bar
    boundaries are anchored to `now - lookback_seconds`, recomputed fresh
    every call -- they are NOT a fixed clock grid, because `now` is a real
    tick timestamp and consecutive calls are not exactly `bar_seconds`
    apart. So the *bars themselves* can't be reused across calls. What
    doesn't change is the underlying raw-tick window `resample_to_bars`
    scans to build them (its `relevant` list plus one seed tick) -- that
    window only grows at the newest end (new ticks) and shrinks at the
    oldest end (ticks aging out of the lookback) between calls. Maintaining
    that window incrementally and re-running the same bar-fill loop over
    it every call produces identical output to calling `resample_to_bars`
    fresh -- verified by the differential tests in test_features.py.

    Storage is a plain `list` with a lazy `_head` index for eviction, not a
    `deque`: the bar-fill loop below needs O(1) random access
    (`self._window[i]`) for every one of the ~lookback/bar_seconds*rate
    entries it visits, and `deque` indexing is O(n) -- using one turns that
    loop from O(window) into O(window^2) per call, which is *slower* than
    the original function it's replacing. Evicted entries are dropped by
    advancing `_head` (O(1) per eviction) rather than removed immediately;
    `_compact` reclaims the dead prefix only every `_COMPACT_EVERY` calls,
    amortizing that O(window) cost across many calls instead of paying it
    every time.
    """

    _COMPACT_EVERY = 256

    def __init__(self) -> None:
        self._window: list[tuple[float, float]] = []
        self._head = 0  # first live (non-evicted) index into _window
        self._seed_ts: float | None = None
        self._seed_price: float | None = None
        # The exact tuple object (identity, not value) most recently folded
        # in -- see the "new ticks" comment below for why this can't be a
        # timestamp comparison.
        self._newest_seen: tuple[float, float] | None = None
        self._last_now: float | None = None
        self._calls_since_compact = 0

    def bars(
        self, buffer: Sequence[tuple[float, float]], bar_seconds: float, now: float, lookback_seconds: float
    ) -> list[float]:
        start = now - lookback_seconds

        if self._last_now is None or now < self._last_now:
            self._rebuild(buffer, start)
        else:
            # "New ticks" must be found by walking backward until we reach
            # the exact tuple object already folded in -- NOT until we
            # reach a timestamp <= the last-known one. Binance batches
            # simultaneous trades at identical millisecond timestamps
            # (observed: single timestamps shared by 100+ trades in the
            # recorded data), so a tie can straddle a call boundary: some
            # ticks at a given ts processed on one call, more ticks at that
            # SAME ts arriving before the next call. A value comparison
            # (`ts <= newest_known`) would silently drop those as
            # "already seen"; identity comparison can't, since each tick is
            # a freshly-created tuple even when its value duplicates one
            # already in the window.
            new_ticks: list[tuple[float, float]] = []
            for item in reversed(buffer):
                if item is self._newest_seen:
                    break
                new_ticks.append(item)
            new_ticks.reverse()
            if new_ticks:
                self._window.extend(new_ticks)
                self._newest_seen = new_ticks[-1]

            n = len(self._window)
            while self._head < n and self._window[self._head][0] <= start:
                self._seed_ts, self._seed_price = self._window[self._head]
                self._head += 1

            self._calls_since_compact += 1
            if self._calls_since_compact >= self._COMPACT_EVERY:
                self._compact()

        self._last_now = now

        bars: list[float] = []
        last_price = self._seed_price
        i = self._head
        n = len(self._window)
        t = start
        while t <= now:
            while i < n and self._window[i][0] <= t:
                last_price = self._window[i][1]
                i += 1
            if last_price is not None:
                bars.append(last_price)
            t += bar_seconds
        return bars

    def _compact(self) -> None:
        if self._head > 0:
            self._window = self._window[self._head :]
            self._head = 0
        self._calls_since_compact = 0

    def _rebuild(self, buffer: Sequence[tuple[float, float]], start: float) -> None:
        self._seed_ts = None
        self._seed_price = None
        self._newest_seen = None
        relevant: list[tuple[float, float]] = []
        for item in reversed(buffer):
            if self._newest_seen is None:
                self._newest_seen = item  # buffer's actual newest tick, seen regardless of relevance
            ts, price = item
            if ts <= start:
                self._seed_ts, self._seed_price = ts, price
                break
            relevant.append(item)
        relevant.reverse()
        self._window = relevant
        self._head = 0
        self._calls_since_compact = 0


def _ewma_variance(diffs: Sequence[float], bar_seconds: float, halflife_seconds: float) -> float | None:
    if not diffs:
        return None
    decay = 0.5 ** (bar_seconds / halflife_seconds)
    var = diffs[0] ** 2
    for d in diffs[1:]:
        var = decay * var + (1 - decay) * d * d
    return var


def _vol_from_bars(
    bars: list[float], bar_seconds: float, halflife_seconds: float, min_bars: int
) -> float | None:
    if len(bars) < min_bars:
        return None
    diffs = [bars[i] - bars[i - 1] for i in range(1, len(bars))]
    variance = _ewma_variance(diffs, bar_seconds, halflife_seconds)
    if variance is None or variance <= 0:
        return None
    return math.sqrt(variance)


def realized_vol_dollar_per_sqrt_s(
    buffer: Sequence[tuple[float, float]],
    now: float,
    lookback_seconds: float,
    bar_seconds: float,
    halflife_seconds: float,
    min_bars: int = 10,
) -> float | None:
    """EWMA realized volatility, in dollars per sqrt(second)."""
    bars = resample_to_bars(buffer, bar_seconds, now, lookback_seconds)
    return _vol_from_bars(bars, bar_seconds, halflife_seconds, min_bars)


def book_imbalance(
    bids: Sequence[tuple[float, float]], asks: Sequence[tuple[float, float]], depth: int
) -> float | None:
    """(bid_vol - ask_vol) / (bid_vol + ask_vol) over the top `depth` levels each side."""
    bid_vol = sum(size for _, size in bids[:depth])
    ask_vol = sum(size for _, size in asks[:depth])
    total = bid_vol + ask_vol
    if total <= 0:
        return None
    return (bid_vol - ask_vol) / total


def aggressive_flow(trades: Sequence[Trade], now: float, lookback_seconds: float) -> float | None:
    """Net signed trade volume (BUY - SELL) / total volume within the lookback window."""
    cutoff = now - lookback_seconds
    net = 0.0
    total = 0.0
    for ts, _price, size, side in trades:
        if ts < cutoff:
            continue
        sign = 1.0 if str(side).upper() == "BUY" else -1.0
        net += sign * size
        total += size
    if total <= 0:
        return None
    return net / total


@dataclass
class FeatureSnapshot:
    condition_id: str
    ts: float
    spot: float
    open_price: float
    t_remaining: float
    sigma: float | None
    deviation: float | None
    momentum_1m: float | None
    momentum_3m: float | None
    momentum_5m: float | None
    book_imbalance_up: float | None
    aggressive_flow_up: float | None


class FeatureEngine:
    def __init__(self, settings: Settings, binance_feed: BinanceFeed, clob_feed: ClobFeed):
        self.settings = settings
        self.binance_feed = binance_feed
        self.clob_feed = clob_feed
        # One instance for this engine's lifetime -- `now` is monotonically
        # non-decreasing across every call `compute()` ever makes (ticks
        # are processed in time order, live or replayed), which is exactly
        # what `_RollingBars` needs to stay incremental instead of falling
        # back to a full rebuild.
        self._vol_bars = _RollingBars()

    def compute(
        self, condition_id: str, token_id_up: str, open_price: float | None, close_ts: float, now: float
    ) -> FeatureSnapshot | None:
        spot = self.binance_feed.last_price
        if spot is None or open_price is None:
            return None

        t_remaining = max(0.0, close_ts - now)
        s = self.settings
        bars = self._vol_bars.bars(self.binance_feed.buffer, s.vol_bar_seconds, now, s.vol_lookback_seconds)
        sigma = _vol_from_bars(bars, s.vol_bar_seconds, s.vol_halflife_seconds, min_bars=10)
        deviation = None
        if sigma is not None and sigma > 0 and t_remaining > 0:
            deviation = (spot - open_price) / (sigma * math.sqrt(t_remaining))

        momentum_1m = self._momentum(now, 60)
        momentum_3m = self._momentum(now, 180)
        momentum_5m = self._momentum(now, 300)

        book = self.clob_feed.books.get(token_id_up)
        imbalance = None
        flow = None
        if book is not None:
            bids, asks = book.as_sorted()
            imbalance = book_imbalance(bids, asks, s.book_imbalance_depth)
            flow = aggressive_flow(book.recent_trades, now, s.flow_lookback_seconds)

        return FeatureSnapshot(
            condition_id=condition_id,
            ts=now,
            spot=spot,
            open_price=open_price,
            t_remaining=t_remaining,
            sigma=sigma,
            deviation=deviation,
            momentum_1m=momentum_1m,
            momentum_3m=momentum_3m,
            momentum_5m=momentum_5m,
            book_imbalance_up=imbalance,
            aggressive_flow_up=flow,
        )

    def _momentum(self, now: float, lookback_seconds: float) -> float | None:
        last = self.binance_feed.last_price
        past = self.binance_feed.price_since(lookback_seconds, now)
        if last is None or past is None or past == 0:
            return None
        return last / past - 1.0
