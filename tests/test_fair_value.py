import math

from poly15m.pricing.fair_value import compute_fair_value, normal_cdf


def test_normal_cdf_at_zero_is_half():
    assert math.isclose(normal_cdf(0.0), 0.5)


def test_normal_cdf_symmetric():
    assert math.isclose(normal_cdf(-1.5), 1.0 - normal_cdf(1.5))


def test_normal_cdf_saturates_for_large_deviation():
    assert normal_cdf(10.0) > 0.999999
    assert normal_cdf(-10.0) < 0.000001


def test_compute_fair_value_none_passthrough():
    assert compute_fair_value(None) is None


def test_compute_fair_value_probabilities_sum_to_one():
    fv = compute_fair_value(0.8)
    assert fv is not None
    assert math.isclose(fv.p_up + fv.p_down, 1.0)
    assert fv.deviation == 0.8
    assert fv.p_up > 0.5  # positive deviation -> favors Up
