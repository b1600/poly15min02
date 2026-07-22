import math

from poly15m.signals.features import (
    aggressive_flow,
    book_imbalance,
    realized_vol_dollar_per_sqrt_s,
    resample_to_bars,
)


def test_resample_to_bars_forward_fills():
    buffer = [(0.0, 100.0), (2.5, 101.0), (5.5, 99.0)]
    bars = resample_to_bars(buffer, bar_seconds=1.0, now=5.0, lookback_seconds=5.0)
    # bar boundaries at t=0,1,2,3,4,5 -> price is last observed at/before each boundary
    assert bars == [100.0, 100.0, 100.0, 101.0, 101.0, 101.0]


def test_resample_to_bars_empty_when_no_data():
    assert resample_to_bars([], bar_seconds=1.0, now=10.0, lookback_seconds=5.0) == []


def test_realized_vol_is_none_with_too_few_bars():
    buffer = [(0.0, 100.0), (1.0, 100.5)]
    assert realized_vol_dollar_per_sqrt_s(buffer, now=1.0, lookback_seconds=5.0, bar_seconds=1.0, halflife_seconds=60.0) is None


def test_realized_vol_is_none_for_constant_price():
    buffer = [(float(i), 100.0) for i in range(30)]
    vol = realized_vol_dollar_per_sqrt_s(
        buffer, now=29.0, lookback_seconds=30.0, bar_seconds=1.0, halflife_seconds=60.0
    )
    assert vol is None  # zero variance -> treated as "unavailable", not 0.0


def test_realized_vol_scales_with_step_size():
    small_steps = [(float(i), 100.0 + (1 if i % 2 else -1)) for i in range(60)]
    big_steps = [(float(i), 100.0 + (10 if i % 2 else -10)) for i in range(60)]
    kwargs = dict(now=59.0, lookback_seconds=60.0, bar_seconds=1.0, halflife_seconds=60.0)
    vol_small = realized_vol_dollar_per_sqrt_s(small_steps, **kwargs)
    vol_big = realized_vol_dollar_per_sqrt_s(big_steps, **kwargs)
    assert vol_small is not None and vol_big is not None
    assert vol_big > vol_small
    assert math.isclose(vol_big / vol_small, 10.0, rel_tol=0.05)


def test_book_imbalance_symmetric_book_is_zero():
    bids = [(0.49, 100.0), (0.48, 50.0)]
    asks = [(0.51, 100.0), (0.52, 50.0)]
    assert book_imbalance(bids, asks, depth=10) == 0.0


def test_book_imbalance_favors_larger_side():
    bids = [(0.49, 300.0)]
    asks = [(0.51, 100.0)]
    imb = book_imbalance(bids, asks, depth=10)
    assert imb == (300.0 - 100.0) / (300.0 + 100.0)


def test_book_imbalance_none_when_empty():
    assert book_imbalance([], [], depth=10) is None


def test_book_imbalance_respects_depth_cutoff():
    bids = [(0.49, 100.0), (0.48, 100.0), (0.47, 900.0)]  # 3rd level excluded at depth=2
    asks = [(0.51, 100.0)]
    imb = book_imbalance(bids, asks, depth=2)
    assert imb == (200.0 - 100.0) / (200.0 + 100.0)


def test_aggressive_flow_all_buys_is_one():
    trades = [(10.0, 0.5, 5.0, "BUY"), (11.0, 0.51, 3.0, "buy")]
    assert aggressive_flow(trades, now=12.0, lookback_seconds=60.0) == 1.0


def test_aggressive_flow_mixed_sides():
    trades = [(10.0, 0.5, 6.0, "BUY"), (11.0, 0.5, 4.0, "SELL")]
    flow = aggressive_flow(trades, now=12.0, lookback_seconds=60.0)
    assert flow == (6.0 - 4.0) / (6.0 + 4.0)


def test_aggressive_flow_ignores_trades_outside_lookback():
    trades = [(0.0, 0.5, 100.0, "SELL"), (59.0, 0.5, 1.0, "BUY")]
    flow = aggressive_flow(trades, now=60.0, lookback_seconds=10.0)
    assert flow == 1.0  # only the recent BUY counts


def test_aggressive_flow_none_when_no_trades_in_window():
    assert aggressive_flow([], now=10.0, lookback_seconds=60.0) is None
