import math

from poly15m.config import Settings
from poly15m.pricing.fees import taker_fee, taker_fee_per_share

SETTINGS = Settings()


def test_matches_polymarket_published_example():
    """Polymarket documents $1.75 per 100 crypto shares at 50c."""
    assert math.isclose(taker_fee(0.50, 100.0, SETTINGS), 1.75)


def test_fee_peaks_at_fifty_cents():
    peak = taker_fee_per_share(0.50, SETTINGS)
    assert peak > taker_fee_per_share(0.30, SETTINGS)
    assert peak > taker_fee_per_share(0.70, SETTINGS)


def test_fee_symmetric_about_fifty_cents():
    assert math.isclose(taker_fee_per_share(0.20, SETTINGS), taker_fee_per_share(0.80, SETTINGS))


def test_fee_tapers_to_zero_at_extremes():
    assert taker_fee_per_share(0.0, SETTINGS) == 0.0
    assert taker_fee_per_share(1.0, SETTINGS) == 0.0


def test_flat_bps_model_understates_below_71_cents():
    """The old flat-200bps model charged price*0.02 per share. The real
    schedule is more expensive below ~0.71 and cheaper above -- the reason
    the pre-fix sweep overstated PnL."""
    for p in (0.2, 0.4, 0.5, 0.6):
        assert taker_fee_per_share(p, SETTINGS) > p * 0.02
    for p in (0.8, 0.9):
        assert taker_fee_per_share(p, SETTINGS) < p * 0.02
