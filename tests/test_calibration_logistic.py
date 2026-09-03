import math

import numpy as np

from poly15m.calibration.logistic import LogisticRegression, log_loss


def test_fit_recovers_a_clear_positive_relationship():
    rng = np.random.default_rng(42)
    x = rng.uniform(-3, 3, size=2000)
    true_p = 1.0 / (1.0 + np.exp(-(1.5 * x + 0.3)))
    y = (rng.uniform(size=x.size) < true_p).astype(float)

    model = LogisticRegression().fit(x, y)

    intercept, slope = model.coef_
    assert slope > 1.0  # recovers the strong positive relationship, roughly near 1.5
    assert abs(intercept - 0.3) < 0.3


def test_fit_recovers_a_clear_negative_relationship():
    rng = np.random.default_rng(7)
    x = rng.uniform(-3, 3, size=2000)
    true_p = 1.0 / (1.0 + np.exp(-(-2.0 * x)))
    y = (rng.uniform(size=x.size) < true_p).astype(float)

    model = LogisticRegression().fit(x, y)
    assert model.coef_[1] < -1.0


def test_predict_proba_in_unit_interval():
    rng = np.random.default_rng(1)
    x = rng.uniform(-5, 5, size=200)
    y = (x > 0).astype(float)
    model = LogisticRegression().fit(x, y)

    p = model.predict_proba(np.array([-10.0, 0.0, 10.0]))
    assert np.all((p > 0.0) & (p < 1.0))
    assert p[0] < p[1] < p[2]  # monotonic in x, given a positive relationship


def test_predict_proba_before_fit_raises():
    import pytest

    model = LogisticRegression()
    with pytest.raises(RuntimeError):
        model.predict_proba(np.array([0.0]))


def test_log_loss_near_zero_for_confident_correct_predictions():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.array([0.99, 0.01, 0.99, 0.01])
    assert log_loss(y, p) < 0.02


def test_log_loss_high_for_confident_wrong_predictions():
    y = np.array([1.0, 0.0])
    p = np.array([0.01, 0.99])
    assert log_loss(y, p) > 4.0


def test_log_loss_matches_manual_calculation_for_uninformative_predictions():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    assert math.isclose(log_loss(y, p), -math.log(0.5), rel_tol=1e-6)


def test_log_loss_does_not_blow_up_at_exact_zero_or_one():
    y = np.array([1.0, 0.0])
    p = np.array([1.0, 0.0])  # would be -inf without clipping
    loss = log_loss(y, p)
    assert math.isfinite(loss)
    assert loss < 1e-6
