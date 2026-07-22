import math

from poly15m.config import Settings
from poly15m.pricing.edge import compute_edge, uncertainty_buffer, walk_book_vwap

SETTINGS = Settings()


def test_walk_book_vwap_single_level():
    levels = [(0.5, 100.0)]
    vwap, filled = walk_book_vwap(levels, target_size=20.0)
    assert vwap == 0.5
    assert filled == 20.0


def test_walk_book_vwap_across_levels():
    levels = [(0.5, 10.0), (0.6, 10.0)]
    vwap, filled = walk_book_vwap(levels, target_size=15.0)
    assert filled == 15.0
    assert math.isclose(vwap, (10 * 0.5 + 5 * 0.6) / 15.0)


def test_walk_book_vwap_partial_fill_when_book_thin():
    levels = [(0.5, 5.0)]
    vwap, filled = walk_book_vwap(levels, target_size=20.0)
    assert filled == 5.0
    assert vwap == 0.5


def test_walk_book_vwap_none_when_book_empty():
    assert walk_book_vwap([], target_size=10.0) is None


def test_walk_book_vwap_none_when_limit_price_below_best():
    levels = [(0.5, 100.0)]
    assert walk_book_vwap(levels, target_size=10.0, limit_price=0.4) is None


def test_walk_book_vwap_stops_at_limit_price():
    levels = [(0.4, 10.0), (0.6, 10.0)]
    vwap, filled = walk_book_vwap(levels, target_size=20.0, limit_price=0.5)
    assert filled == 10.0  # only the 0.4 level is within the limit
    assert vwap == 0.4


def test_uncertainty_buffer_shrinks_with_time_outside_final_minute():
    far = uncertainty_buffer(t_remaining=900.0, window_seconds=900.0, sigma=None, settings=SETTINGS)
    mid = uncertainty_buffer(t_remaining=300.0, window_seconds=900.0, sigma=None, settings=SETTINGS)
    assert far > mid > 0


def test_uncertainty_buffer_scales_with_vol():
    low_vol = uncertainty_buffer(t_remaining=300.0, window_seconds=900.0, sigma=SETTINGS.reference_sigma, settings=SETTINGS)
    high_vol = uncertainty_buffer(t_remaining=300.0, window_seconds=900.0, sigma=SETTINGS.reference_sigma * 2, settings=SETTINGS)
    assert math.isclose(high_vol, low_vol * 2)


def test_uncertainty_buffer_widens_in_final_minute_despite_less_time_remaining():
    just_outside = uncertainty_buffer(t_remaining=61.0, window_seconds=900.0, sigma=None, settings=SETTINGS)
    well_inside = uncertainty_buffer(t_remaining=30.0, window_seconds=900.0, sigma=None, settings=SETTINGS)
    assert well_inside > just_outside


def test_compute_edge_matches_manual_formula():
    ask_levels = [(0.5, 100.0)]
    result = compute_edge(
        "up", fair_value_p=0.7, ask_levels=ask_levels, target_size=20.0,
        t_remaining=300.0, window_seconds=900.0, sigma=None, settings=SETTINGS,
    )
    assert result is not None
    expected_fees = 0.5 * (SETTINGS.taker_fee_bps / 10000.0)
    expected_slippage = 0.5 * (SETTINGS.slippage_buffer_bps / 10000.0)
    expected_unc = uncertainty_buffer(300.0, 900.0, None, SETTINGS)
    expected_net_edge = 0.7 - 0.5 - expected_fees - expected_slippage - expected_unc
    assert math.isclose(result.net_edge, expected_net_edge)
    assert result.vwap_fill == 0.5
    assert result.filled_size == 20.0


def test_compute_edge_none_when_book_priced_above_fair_value():
    ask_levels = [(0.8, 100.0)]
    result = compute_edge(
        "up", fair_value_p=0.7, ask_levels=ask_levels, target_size=20.0,
        t_remaining=300.0, window_seconds=900.0, sigma=None, settings=SETTINGS,
    )
    assert result is None
