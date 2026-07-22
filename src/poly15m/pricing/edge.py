"""Executable edge calculation (Implementation_Plan.md Phase 3, item 13).

net_edge = fair_value - vwap_fill - fees - slippage_buffer - uncertainty_buffer

`vwap_fill` comes from walking the real order book for a target size (so it
already reflects book-implied slippage/depth); `slippage_buffer` is an
*additional* haircut for latency between our decision and the order
reaching the exchange, which a static book snapshot can't capture.
`uncertainty_buffer` is model risk, not execution risk: it shrinks as
t_remaining -> 0 (the outcome becomes more determined), scales with current
realized vol (a shakier vol estimate means a shakier fair-value estimate),
but widens again in the final minute because that's when a small
Binance-vs-Chainlink resolution-feed divergence matters most relative to
the (by-then tiny) remaining uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..config import Settings

Level = tuple[float, float]  # (price, size)


def walk_book_vwap(
    levels: Sequence[Level], target_size: float, limit_price: float | None = None
) -> tuple[float, float] | None:
    """Volume-weighted average fill price for `target_size`, walking `levels`
    (best-first, ascending price -- i.e. an ask side for a BUY). Stops at
    `limit_price` if given, so this never returns a fill worse than a limit
    order's price. Returns (vwap, filled_size); filled_size < target_size
    when the book doesn't have enough depth (down to and including 0, via
    None, when nothing fillable at all)."""
    if target_size <= 0:
        return None
    remaining = target_size
    cost = 0.0
    filled = 0.0
    for price, size in levels:
        if limit_price is not None and price > limit_price:
            break
        take = min(remaining, size)
        cost += take * price
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    if filled <= 0:
        return None
    return cost / filled, filled


def uncertainty_buffer(
    t_remaining: float, window_seconds: float, sigma: float | None, settings: Settings
) -> float:
    time_frac = max(0.0, min(1.0, t_remaining / window_seconds)) if window_seconds > 0 else 0.0
    vol_factor = 1.0
    if sigma is not None and settings.reference_sigma > 0:
        vol_factor = sigma / settings.reference_sigma
    base = settings.uncertainty_buffer_base * vol_factor * math.sqrt(time_frac)

    widen = 0.0
    if t_remaining < settings.uncertainty_final_minute_seconds:
        widen_frac = 1.0 - max(0.0, t_remaining) / settings.uncertainty_final_minute_seconds
        widen = settings.uncertainty_final_minute_extra * widen_frac

    return base + widen


@dataclass
class EdgeResult:
    side: str  # "up" | "down"
    fair_value: float
    vwap_fill: float
    filled_size: float
    fees: float
    slippage_buffer: float
    uncertainty_buffer: float
    net_edge: float


def compute_edge(
    side: str,
    fair_value_p: float,
    ask_levels: Sequence[Level],
    target_size: float,
    t_remaining: float,
    window_seconds: float,
    sigma: float | None,
    settings: Settings,
) -> EdgeResult | None:
    """Edge for BUYing `side` up to `target_size` shares. The walk is capped
    at `fair_value_p`: paying more than our own fair-value estimate is a
    guaranteed-negative-EV fill before fees/buffers even apply, so it's
    excluded from the VWAP rather than counted against the edge."""
    walk = walk_book_vwap(ask_levels, target_size, limit_price=fair_value_p)
    if walk is None:
        return None
    vwap_fill, filled_size = walk

    fees = vwap_fill * (settings.taker_fee_bps / 10000.0)
    slippage = vwap_fill * (settings.slippage_buffer_bps / 10000.0)
    unc = uncertainty_buffer(t_remaining, window_seconds, sigma, settings)

    net_edge = fair_value_p - vwap_fill - fees - slippage - unc
    return EdgeResult(
        side=side,
        fair_value=fair_value_p,
        vwap_fill=vwap_fill,
        filled_size=filled_size,
        fees=fees,
        slippage_buffer=slippage,
        uncertainty_buffer=unc,
        net_edge=net_edge,
    )
