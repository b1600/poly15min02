"""A fit can improve average log-loss while being unsafe to trade.

Log-loss is an average over ticks; trading happens in specific price
bands. A model can win on the average and still overstate P(up) badly in
the band it actually trades -- and an overstated probability is an
overpaid trade. `reliability` reports predicted vs realised per band so
that failure is visible.

The significance half matters just as much. Ticks are not independent:
every tick inside a 15-minute window shares that window's single outcome.
Scoring per-tick would shrink the error bars by an order of magnitude and
make ordinary base-rate drift look like proof of miscalibration -- on this
project's own data the held-out base rate sits 3.4pp below the training
one by chance alone, which shows up as a uniform positive overstatement in
every single band.
"""

import numpy as np
import pytest

from poly15m.calibration.fit import ReliabilityBucket, reliability


def _cid(n, n_windows):
    """Window ids spread across n rows, so rows cluster into windows."""
    return np.array([f"cond{i % n_windows}" for i in range(n)], dtype=object)


def test_flags_a_model_that_overstates_a_band():
    rng = np.random.default_rng(0)
    p = np.full(5000, 0.20)
    y = (rng.random(5000) < 0.02).astype(float)  # model says 0.20, reality 0.02
    buckets = reliability(y, p, _cid(5000, 200))
    assert len(buckets) == 1
    assert buckets[0].overstatement > 0.15
    assert buckets[0].z > 2.0


def test_well_calibrated_model_has_near_zero_overstatement():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 40_000)
    y = (rng.random(40_000) < p).astype(float)
    for b in reliability(y, p, _cid(40_000, 2000)):
        assert abs(b.overstatement) < 0.05, f"band [{b.lo},{b.hi}) off by {b.overstatement}"


def test_std_error_is_clustered_on_windows_not_ticks():
    """Same rows, same outcome, fewer independent windows -> wider error."""
    p = np.full(4000, 0.5)
    y = np.concatenate([np.ones(2000), np.zeros(2000)])
    many = reliability(y, p, _cid(4000, 400))[0]
    few = reliability(y, p, _cid(4000, 10))[0]
    assert many.n == few.n == 4000
    assert few.std_error > many.std_error
    # 40x fewer windows -> sqrt(40) ~= 6.3x wider
    assert few.std_error / many.std_error == pytest.approx(np.sqrt(40.0))


def test_uniform_base_rate_drift_is_not_significant_on_few_windows():
    """A 6pp shift over 30 windows is noise, not evidence of a bad model."""
    p = np.full(3000, 0.50)
    y = (np.arange(3000) % 100 < 44).astype(float)  # realised 0.44
    b = reliability(y, p, _cid(3000, 30))[0]
    assert b.overstatement > 0.05  # materially overstated...
    assert b.z < 2.0  # ...but not distinguishable from noise


def test_sparse_bands_are_skipped_not_reported_as_perfect():
    assert reliability(np.zeros(10), np.full(10, 0.5), _cid(10, 5)) == []
    # enough ticks but nearly all from one window -> no real information
    assert reliability(np.zeros(400), np.full(400, 0.5), _cid(400, 2)) == []


def test_overstatement_sign_convention():
    over = ReliabilityBucket(lo=0.1, hi=0.2, n=100, n_windows=50, predicted=0.15, realised=0.02)
    assert over.overstatement > 0  # claimed more than reality -> would overpay
    under = ReliabilityBucket(lo=0.8, hi=0.9, n=100, n_windows=50, predicted=0.85, realised=0.95)
    assert under.overstatement < 0
    assert under.z < 0
