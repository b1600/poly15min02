"""Calibration (Implementation_Plan.md Phase 6, item 24): fit the logistic
correction promised back in Phase 2 item 11, and compare it against the
analytic model (`P(Up) = Phi(deviation)`) out-of-sample.

The training set is every (deviation, actual outcome) pair across every
resolved window recorded in `fair_value_log` joined to
`markets.resolved_outcome` -- which, since Phase 6, `paper_trade.py` logs
on every decision tick (see its module docstring), so a single paper-mode
run builds this dataset as a side effect rather than requiring a
separate one.

Split is time-ordered (train on the earlier fraction, test on the later
one), not random -- this is a time series; a random split would leak
future information into training and overstate how well calibration
generalizes.

Needs real accumulated data to mean anything: `Implementation_Plan.md`
itself frames this as "once a few hundred windows of data exist." Fewer
resolved windows than that (or than `min_test_rows` below) and the
comparison is fit to noise, not signal.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..pricing.fair_value import normal_cdf
from .logistic import LogisticRegression, log_loss


@dataclass
class CalibrationDataset:
    ts: np.ndarray
    deviation: np.ndarray
    outcome: np.ndarray  # 1.0 = up, 0.0 = down
    condition_id: np.ndarray  # object array; rows from the same window are correlated, not i.i.d.


@dataclass
class CalibrationReport:
    n_train: int
    n_test: int
    n_windows_train: int
    n_windows_test: int
    analytic_log_loss: float
    calibrated_log_loss: float
    coef_: np.ndarray  # [intercept, slope on deviation]
    improvement: float  # analytic_log_loss - calibrated_log_loss; positive = calibration helps


def load_calibration_dataset(db_path: str | Path) -> CalibrationDataset:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT f.ts, f.deviation, m.resolved_outcome, f.condition_id
            FROM fair_value_log f
            JOIN markets m ON m.condition_id = f.condition_id
            WHERE f.deviation IS NOT NULL AND m.resolved_outcome IS NOT NULL
            ORDER BY f.ts
            """
        ).fetchall()
    finally:
        conn.close()
    ts = np.array([r[0] for r in rows], dtype=float)
    deviation = np.array([r[1] for r in rows], dtype=float)
    outcome = np.array([1.0 if r[2] == "up" else 0.0 for r in rows], dtype=float)
    condition_id = np.array([r[3] for r in rows], dtype=object)
    return CalibrationDataset(ts=ts, deviation=deviation, outcome=outcome, condition_id=condition_id)


def fit_calibration(
    dataset: CalibrationDataset,
    train_frac: float = 0.7,
    min_test_rows: int = 20,
    min_train_rows: int = 30,
    min_windows: int = 20,
) -> CalibrationReport | None:
    """Returns None (rather than a misleadingly confident report) unless
    there's enough data along *both* axes that matter: enough rows to fit
    on, and -- more importantly -- enough distinct resolved windows.
    Per-tick rows within one window are highly correlated (same price path,
    same eventual outcome), so hundreds of rows from a handful of windows
    is not the "few hundred windows" Implementation_Plan.md item 24 has in
    mind; a single trending window can trivially score a 0.0 log-loss on
    both models without that meaning anything about calibration quality."""
    n = len(dataset.ts)
    split = int(n * train_frac)
    x_train, y_train = dataset.deviation[:split], dataset.outcome[:split]
    x_test, y_test = dataset.deviation[split:], dataset.outcome[split:]
    windows_train = len(set(dataset.condition_id[:split]))
    windows_test = len(set(dataset.condition_id[split:]))

    if (
        len(x_train) < min_train_rows
        or len(x_test) < min_test_rows
        or windows_train < min_windows
        or windows_test < min_windows
    ):
        return None

    analytic_pred = np.array([normal_cdf(float(d)) for d in x_test])
    analytic_loss = log_loss(y_test, analytic_pred)

    model = LogisticRegression().fit(x_train, y_train)
    calibrated_pred = model.predict_proba(x_test)
    calibrated_loss = log_loss(y_test, calibrated_pred)

    return CalibrationReport(
        n_train=len(x_train),
        n_test=len(x_test),
        n_windows_train=windows_train,
        n_windows_test=windows_test,
        analytic_log_loss=analytic_loss,
        calibrated_log_loss=calibrated_loss,
        coef_=model.coef_,
        improvement=analytic_loss - calibrated_loss,
    )
