"""Polymarket taker fees (Implementation_Plan.md Phase 3, item 13).

Polymarket charges takers a *price-dependent* fee, not a flat rate:

    fee = shares * fee_rate * p * (1 - p)

so the cost peaks at p=0.50 and tapers toward zero at both extremes.
Crypto is the most expensive category (fee_rate 0.07 -- $1.75 per 100
shares at 50c). Makers pay nothing; taker fees are rebated to makers.

See https://help.polymarket.com/en/articles/13364478-trading-fees

Modelling this as a flat percentage of notional -- which `config.py`
previously did, with a `taker_fee_bps` placeholder its own comment flagged
as needing verification -- understates the cost of every fill below
p ~= 0.71 and overstates it above. On this project's own recorded paper
fills (467 fills, average price 0.45) the flat 200bps model came in 1.56x
too cheap, which is the difference between a strategy that clears its
costs and one that doesn't.
"""

from __future__ import annotations

from ..config import Settings


def taker_fee_per_share(price: float, settings: Settings) -> float:
    """Fee in dollars per share transacted at `price`."""
    p = min(max(price, 0.0), 1.0)
    return settings.taker_fee_rate * p * (1.0 - p)


def taker_fee(price: float, size: float, settings: Settings) -> float:
    """Total fee in dollars for `size` shares transacted at `price`."""
    return size * taker_fee_per_share(price, settings)
