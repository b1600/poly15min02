import numpy as np

from poly15m.calibration.fit import fit_calibration, load_calibration_dataset
from poly15m.db import Database
from poly15m.pricing.fair_value import normal_cdf


def build_calibration_db(tmp_path, n, true_prob_fn, seed=0, name="cal.db"):
    db_path = tmp_path / name
    db = Database(db_path)
    rng = np.random.default_rng(seed)
    deviations = rng.uniform(-3.0, 3.0, size=n)
    for i, dev in enumerate(deviations):
        condition_id = f"cond{i}"
        db.upsert_market(
            {
                "condition_id": condition_id,
                "slug": f"slug{i}",
                "question_id": None,
                "token_id_up": f"up{i}",
                "token_id_down": f"down{i}",
                "window_open_ts": float(i),
                "window_close_ts": float(i) + 900.0,
                "discovered_ts": float(i),
                "raw_json": "{}",
            }
        )
        outcome = "up" if rng.uniform() < true_prob_fn(dev) else "down"
        db.set_market_resolution(condition_id, outcome, float(i))
        db.insert_fair_value_log(
            condition_id, float(i), None, None, None, None, float(dev), None, None, None, None, None, None, None, None
        )
    db.close()
    return db_path


def test_load_calibration_dataset_excludes_unresolved_and_null_deviation(tmp_path):
    db_path = tmp_path / "partial.db"
    db = Database(db_path)
    db.upsert_market(
        {
            "condition_id": "resolved", "slug": "s1", "question_id": None,
            "token_id_up": "u1", "token_id_down": "d1",
            "window_open_ts": 0.0, "window_close_ts": 900.0, "discovered_ts": 0.0, "raw_json": "{}",
        }
    )
    db.set_market_resolution("resolved", "up", 1.0)
    db.insert_fair_value_log("resolved", 1.0, None, None, None, None, 1.5, None, None, None, None, None, None, None, None)
    # deviation NULL -- must be excluded
    db.insert_fair_value_log("resolved", 2.0, None, None, None, None, None, None, None, None, None, None, None, None, None)

    db.upsert_market(
        {
            "condition_id": "unresolved", "slug": "s2", "question_id": None,
            "token_id_up": "u2", "token_id_down": "d2",
            "window_open_ts": 0.0, "window_close_ts": 900.0, "discovered_ts": 0.0, "raw_json": "{}",
        }
    )
    db.insert_fair_value_log("unresolved", 3.0, None, None, None, None, 2.0, None, None, None, None, None, None, None, None)
    db.close()

    dataset = load_calibration_dataset(db_path)
    assert len(dataset.deviation) == 1
    assert dataset.deviation[0] == 1.5
    assert dataset.outcome[0] == 1.0


def test_fit_calibration_none_when_insufficient_data(tmp_path):
    db_path = build_calibration_db(tmp_path, n=10, true_prob_fn=normal_cdf)
    dataset = load_calibration_dataset(db_path)
    assert fit_calibration(dataset) is None


def test_fit_calibration_none_when_many_rows_but_too_few_windows(tmp_path):
    # exactly the failure mode found live: hundreds of per-tick rows from a
    # single resolved window are not "a few hundred windows" -- must not
    # report a misleadingly confident (or degenerate) comparison
    db_path = tmp_path / "one_window.db"
    db = Database(db_path)
    db.upsert_market(
        {
            "condition_id": "cond0", "slug": "s", "question_id": None,
            "token_id_up": "u", "token_id_down": "d",
            "window_open_ts": 0.0, "window_close_ts": 900.0, "discovered_ts": 0.0, "raw_json": "{}",
        }
    )
    db.set_market_resolution("cond0", "down", 900.0)
    for i in range(600):
        db.insert_fair_value_log(
            "cond0", float(i), None, None, None, None, -2.0, None, None, None, None, None, None, None, None
        )
    db.close()

    dataset = load_calibration_dataset(db_path)
    assert len(dataset.deviation) == 600  # plenty of rows...
    assert fit_calibration(dataset) is None  # ...but only one window


def test_fit_calibration_matches_analytic_when_analytic_model_is_correct(tmp_path):
    db_path = build_calibration_db(tmp_path, n=800, true_prob_fn=normal_cdf, seed=1)
    dataset = load_calibration_dataset(db_path)
    report = fit_calibration(dataset)

    assert report is not None
    # analytic model IS the true model here -- calibration shouldn't do
    # much better, and definitely shouldn't do much worse
    assert abs(report.improvement) < 0.05


def test_fit_calibration_improves_when_analytic_model_is_biased(tmp_path):
    # true relationship is Phi(deviation) but shifted, so the raw analytic
    # model (unshifted) is systematically miscalibrated -- the logistic
    # correction should recover the shift and beat it out-of-sample
    def shifted_true_prob(dev):
        return normal_cdf(dev - 1.5)

    db_path = build_calibration_db(tmp_path, n=1500, true_prob_fn=shifted_true_prob, seed=2)
    dataset = load_calibration_dataset(db_path)
    report = fit_calibration(dataset)

    assert report is not None
    assert report.calibrated_log_loss < report.analytic_log_loss
    assert report.improvement > 0.02
    intercept, slope = report.coef_
    assert intercept < -0.5  # recovers the negative shift
