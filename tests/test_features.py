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


# -- _RollingBars: incremental equivalence -----------------------------
#
# FeatureEngine calls _RollingBars instead of resample_to_bars for
# performance (profiled at >50% of backtest replay runtime -- see
# features.py's module docstring). These tests prove it produces
# bit-identical output to the reference implementation across the actual
# calling pattern (throttled, non-bar-aligned `now`, irregular real tick
# spacing) rather than just on the toy cases above.

import random

from poly15m.signals.features import _RollingBars


def _check_matches_reference(ticks, call_times, bar_seconds, lookback_seconds):
    """Feed `ticks` into a growing buffer, calling both implementations at
    each of `call_times` (mimicking on_binance_tick's throttle), and assert
    identical bars every time."""
    buffer: list[tuple[float, float]] = []
    tick_iter = iter(ticks)
    next_tick = next(tick_iter, None)
    roller = _RollingBars()

    for now in call_times:
        while next_tick is not None and next_tick[0] <= now:
            buffer.append(next_tick)
            next_tick = next(tick_iter, None)

        expected = resample_to_bars(buffer, bar_seconds, now, lookback_seconds)
        actual = roller.bars(buffer, bar_seconds, now, lookback_seconds)
        assert actual == expected, f"mismatch at now={now}: expected {expected}, got {actual}"


def test_rolling_bars_matches_reference_on_regular_ticks():
    ticks = [(float(i) * 0.1, 100.0 + math.sin(i)) for i in range(3000)]
    call_times = [t for t in [i * 1.0 for i in range(1, 250)]]
    _check_matches_reference(ticks, call_times, bar_seconds=1.0, lookback_seconds=60.0)


def test_rolling_bars_matches_reference_on_irregular_real_world_timing():
    """The actual failure mode this needs to survive: `now` is a real tick
    timestamp, so consecutive calls are >= the throttle interval apart but
    essentially never exactly that far apart -- bar boundaries do not
    align between calls."""
    rng = random.Random(42)
    ts = 0.0
    ticks = []
    while ts < 400.0:
        ts += rng.uniform(0.01, 0.08)  # ~15-100 ticks/sec, like live Binance
        ticks.append((ts, 100.0 + rng.uniform(-5, 5)))

    call_times = []
    last = 0.0
    for t, _ in ticks:
        if t - last >= 1.0:  # mimics fair_value_log_interval_seconds throttle
            call_times.append(t)
            last = t

    _check_matches_reference(ticks, call_times, bar_seconds=1.0, lookback_seconds=60.0)


def test_rolling_bars_matches_reference_with_gaps_in_ticks():
    """A stretch with no ticks at all (feed stall) must forward-fill
    identically to the reference once ticks resume."""
    ticks = [(float(i), 100.0 + i * 0.01) for i in range(30)]
    ticks += [(float(i), 100.3 + (i - 80) * 0.02) for i in range(80, 140)]  # 50s gap
    call_times = [float(i) for i in range(10, 139, 3)]
    _check_matches_reference(ticks, call_times, bar_seconds=1.0, lookback_seconds=60.0)


def test_rolling_bars_matches_reference_across_multiple_bar_seconds():
    rng = random.Random(7)
    ts = 0.0
    ticks = []
    while ts < 500.0:
        ts += rng.uniform(0.02, 0.2)
        ticks.append((ts, 100.0 + rng.uniform(-3, 3)))
    call_times = [t for i, (t, _) in enumerate(ticks) if i % 5 == 0]

    for bar_seconds in (0.5, 1.0, 2.5):
        _check_matches_reference(ticks, call_times, bar_seconds=bar_seconds, lookback_seconds=120.0)


def test_rolling_bars_rebuilds_correctly_after_now_goes_backward():
    """A fresh FeatureEngine (or a replay discontinuity) must not carry
    stale state forward -- `now` regressing must trigger a clean rebuild
    that still matches the reference."""
    buffer = [(float(i), 100.0 + i * 0.1) for i in range(50)]
    roller = _RollingBars()

    first = roller.bars(buffer, bar_seconds=1.0, now=40.0, lookback_seconds=20.0)
    assert first == resample_to_bars(buffer, bar_seconds=1.0, now=40.0, lookback_seconds=20.0)

    # now goes backward relative to the previous call
    second = roller.bars(buffer, bar_seconds=1.0, now=25.0, lookback_seconds=20.0)
    assert second == resample_to_bars(buffer, bar_seconds=1.0, now=25.0, lookback_seconds=20.0)


def test_rolling_bars_matches_reference_on_real_recorded_ticks():
    """The differential test that actually matters: real Binance ticks
    from the recorded database, at the real throttle cadence, across a
    span long enough to exercise window eviction many times over."""
    import sqlite3

    conn = sqlite3.connect("var/poly15m.db")
    rows = conn.execute(
        """SELECT event_ts, price FROM price_ticks WHERE source='binance'
           AND event_ts >= strftime('%s','2026-09-21') ORDER BY event_ts LIMIT 200000"""
    ).fetchall()
    conn.close()
    if len(rows) < 10000:
        return  # no recorded data available in this environment -- skip rather than fail

    ticks = [(ts, price) for ts, price in rows]
    call_times = []
    last = None
    for t, _ in ticks:
        if last is None or t - last >= 1.0:
            call_times.append(t)
            last = t
        if len(call_times) >= 600:
            break

    _check_matches_reference(ticks, call_times, bar_seconds=1.0, lookback_seconds=300.0)


def test_rolling_bars_handles_duplicate_timestamps_straddling_a_call():
    """Binance batches simultaneous trades at identical millisecond
    timestamps (observed: single timestamps shared by 100+ trades in the
    recorded data). A tick arriving with the same ts as the last tick
    already folded in must still be counted -- this is the exact bug found
    when validating against real data: a value comparison (`ts <=
    newest_known`) silently drops such ties if they straddle a call
    boundary, since they look identical to "already seen"."""
    roller = _RollingBars()
    buffer = [(0.0, 100.0), (1.0, 101.0), (2.0, 102.0), (2.0, 103.0)]

    # first call sees only the first tie member
    first = roller.bars(buffer[:3], bar_seconds=1.0, now=2.0, lookback_seconds=5.0)
    assert first == resample_to_bars(buffer[:3], bar_seconds=1.0, now=2.0, lookback_seconds=5.0)

    # second tie member (same ts=2.0, different tuple object) arrives before the next call
    second = roller.bars(buffer, bar_seconds=1.0, now=2.5, lookback_seconds=5.0)
    expected = resample_to_bars(buffer, bar_seconds=1.0, now=2.5, lookback_seconds=5.0)
    assert second == expected
    assert second[-1] == 103.0  # the second same-ts tick must win the forward-fill, not vanish


def test_rolling_bars_matches_reference_with_many_ties_at_one_timestamp():
    """A burst of same-millisecond trades (common in real data) landing
    exactly at a call boundary."""
    buffer = [(float(i), 100.0 + i * 0.01) for i in range(20)]
    buffer += [(20.0, 100.2 + j * 0.001) for j in range(50)]  # 50-way tie at ts=20.0
    buffer += [(float(i), 100.2 + i * 0.01) for i in range(21, 40)]

    call_times = [10.0, 19.5, 20.0, 20.0, 25.0, 39.0]  # straddles the tie from both sides
    roller = _RollingBars()
    for now in call_times:
        visible = [t for t in buffer if t[0] <= now]
        expected = resample_to_bars(visible, bar_seconds=1.0, now=now, lookback_seconds=15.0)
        actual = roller.bars(visible, bar_seconds=1.0, now=now, lookback_seconds=15.0)
        assert actual == expected, f"mismatch at now={now}"
