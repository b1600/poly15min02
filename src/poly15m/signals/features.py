"""Signal feature computation, recomputed on each Binance tick.

Implements the four feature groups from Strategy_v1.md / Implementation_Plan.md:
  - Deviation: (spot - open_price) / (sigma * sqrt(t_remaining))
  - Momentum: 1m / 3m / 5m simple returns
  - Realized volatility: EWMA of 1s price-difference variance
  - Book imbalance and aggressive trade flow on the CLOB

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
    """Forward-filled price at each bar boundary within [now - lookback, now]."""
    if not buffer:
        return []
    start = now - lookback_seconds
    bars: list[float] = []
    last_price: float | None = None
    i = 0
    n = len(buffer)
    t = start
    while t <= now:
        while i < n and buffer[i][0] <= t:
            last_price = buffer[i][1]
            i += 1
        if last_price is not None:
            bars.append(last_price)
        t += bar_seconds
    return bars


def _ewma_variance(diffs: Sequence[float], bar_seconds: float, halflife_seconds: float) -> float | None:
    if not diffs:
        return None
    decay = 0.5 ** (bar_seconds / halflife_seconds)
    var = diffs[0] ** 2
    for d in diffs[1:]:
        var = decay * var + (1 - decay) * d * d
    return var


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
    if len(bars) < min_bars:
        return None
    diffs = [bars[i] - bars[i - 1] for i in range(1, len(bars))]
    variance = _ewma_variance(diffs, bar_seconds, halflife_seconds)
    if variance is None or variance <= 0:
        return None
    return math.sqrt(variance)


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

    def compute(
        self, condition_id: str, token_id_up: str, open_price: float | None, close_ts: float, now: float
    ) -> FeatureSnapshot | None:
        spot = self.binance_feed.last_price
        if spot is None or open_price is None:
            return None

        t_remaining = max(0.0, close_ts - now)
        s = self.settings
        sigma = realized_vol_dollar_per_sqrt_s(
            self.binance_feed.buffer, now, s.vol_lookback_seconds, s.vol_bar_seconds, s.vol_halflife_seconds
        )
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
