"""Regression guard for the inert-Kelly bug found in the 2026-09-03 sweep.

`bankroll` and `kelly_fraction` only mean anything if the Kelly stake can
land *below* the position caps. When it can't, every order is clipped to a
flat size, the strategy is silently fixed-stake, and sweeping
`kelly_fraction` produces byte-identical results across the whole axis --
which is exactly what happened: 36 combinations, 9 distinct outcomes.
"""

import pytest

from poly15m.config import Settings
from poly15m.positions.manager import kelly_headroom

PRICES = (0.2, 0.35, 0.5, 0.65, 0.8)


def test_default_settings_leave_kelly_live():
    settings = Settings()
    for price in PRICES:
        assert kelly_headroom(settings, price) < 1.0, (
            f"Kelly is inert at price {price}: every order is clipped by a cap, "
            "so kelly_fraction has no effect. Raise the caps or lower kelly_fraction."
        )


def test_detects_the_original_broken_configuration():
    """The pre-fix values, which produced a no-op kelly_fraction axis."""
    broken = Settings(
        kelly_fraction=0.15,
        paper_trade_size=20.0,
        max_inventory_imbalance=20.0,
        max_notional_per_market=20.0,
    )
    for price in PRICES:
        assert kelly_headroom(broken, price) > 1.0


def test_raising_kelly_fraction_without_caps_reintroduces_the_bug():
    # read the default rather than hardcoding it: kelly_fraction and
    # max_notional_per_market have to move together, and pinning one of
    # them here made this test fail the next time they did
    default = Settings().kelly_fraction
    assert kelly_headroom(Settings(kelly_fraction=default), 0.5) < 1.0
    assert kelly_headroom(Settings(kelly_fraction=default * 10), 0.5) > 1.0


def test_headroom_scales_linearly_with_kelly_fraction():
    a = kelly_headroom(Settings(kelly_fraction=0.05), 0.5)
    b = kelly_headroom(Settings(kelly_fraction=0.10), 0.5)
    assert b == pytest.approx(2.0 * a)


def test_rejects_degenerate_prices():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            kelly_headroom(Settings(), bad)
