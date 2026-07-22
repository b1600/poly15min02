"""Analytic fair-value model (Implementation_Plan.md Phase 2, item 10).

P(Up) = Phi(deviation), i.e. the probability BTC ends above the window's
opening price given current distance, recent volatility, and time left,
under a driftless-Brownian-motion assumption on the price level. This is
deliberately not ML: no training dependency, and it's fast enough to
reprice on every tick, which is the actual edge this strategy is chasing
(repricing faster than stale resting orders).

`compute_fair_value` returns None when `deviation` is None (e.g. not
enough realized-vol history yet, or the window has already closed) --
callers should treat "no fair value yet" as "don't trade this tick", not
default to a specific probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class FairValue:
    p_up: float
    p_down: float
    deviation: float


def compute_fair_value(deviation: float | None) -> FairValue | None:
    if deviation is None:
        return None
    p_up = normal_cdf(deviation)
    return FairValue(p_up=p_up, p_down=1.0 - p_up, deviation=deviation)
