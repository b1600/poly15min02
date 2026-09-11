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

from ..pricing.fair_value import DEFAULT_DEVIATION_CLIP, calibration_features, normal_cdf
from .logistic import LogisticRegression, log_loss


@dataclass
class CalibrationDataset:
    ts: np.ndarray
    deviation: np.ndarray
    t_remaining: np.ndarray
    outcome: np.ndarray  # 1.0 = up, 0.0 = down
    condition_id: np.ndarray  # object array; rows from the same window are correlated, not i.i.d.


# Kept symmetric under p -> 1-p: the strategy buys both tokens, so every
# band needs its mirror scored at the same resolution.
#
# The cheap end is deliberately finer than it looks like it needs to be.
# A single [0.10,0.20) bucket averaged a well-calibrated [0.10,0.15) with
# a [0.15,0.20) that overstated by +0.033 (z=2.4) on recorded data, and
# reported the mean of the two as harmless. Bands are the unit of
# detection here; a band wide enough to contain both a good and a bad
# region cannot report the bad one.
DEFAULT_RELIABILITY_EDGES = (
    0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50,
    0.65, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 1.0,
)


@dataclass
class ReliabilityBucket:
    """Predicted vs actually-realised P(up) for one probability band."""

    lo: float
    hi: float
    n: int
    n_windows: int
    predicted: float
    realised: float

    @property
    def overstatement(self) -> float:
        """Positive = the model claims more probability than reality
        delivered, i.e. it would overpay for this contract."""
        return self.predicted - self.realised

    @property
    def std_error(self) -> float:
        """Standard error of `realised`, clustered on windows.

        Ticks are emphatically not independent: every tick inside one
        15-minute window shares that window's single outcome, so a band
        holding 9,000 ticks from 65 windows carries 65 observations of
        information, not 9,000. Using the tick count here would shrink the
        error bars by ~12x and make ordinary sampling noise look like
        damning evidence of miscalibration.
        """
        var = max(self.realised * (1.0 - self.realised), 1e-6)
        return float(np.sqrt(var / max(self.n_windows, 1)))

    @property
    def z(self) -> float:
        """Overstatement in standard errors. Positive and large = real."""
        return self.overstatement / self.std_error if self.std_error > 0 else 0.0

    # -- both sides of the book -------------------------------------
    # `overstatement` above is signed on the P(up) scale, which only ever
    # describes the *up* token. The strategy buys either token, and
    # P(down) = 1 - P(up), so a band that understates P(up) by d is a band
    # that overstates P(down) by exactly d -- an equally overpaid trade,
    # just on the other side. Scoring only the positive tail leaves half
    # the book unchecked. The properties below score whichever side this
    # band actually overprices.

    @property
    def overstated_side(self) -> str:
        """The token this band overprices. Exactly one of the two is."""
        return "up" if self.predicted >= self.realised else "down"

    @property
    def traded_overstatement(self) -> float:
        """`overstatement` for the overstated side, so always >= 0."""
        return abs(self.overstatement)

    @property
    def traded_z(self) -> float:
        """`traded_overstatement` in clustered standard errors.

        `std_error` is invariant under p -> 1-p (p(1-p) is symmetric), so
        the down side inherits the up side's error bar unchanged.
        """
        return abs(self.z)

    @property
    def overstated_side_predicted(self) -> float:
        """What the model claims the overstated side is worth -- i.e. the
        most the strategy would pay for it before any edge requirement."""
        return self.predicted if self.overstated_side == "up" else 1.0 - self.predicted

    @property
    def relative_overstatement(self) -> float:
        """Overstatement as a fraction of the overstated side's price.
        Reported, not gated on: whether a trade is +EV turns on the
        absolute overstatement against the required edge, not this. It is
        still the honest way to read a cheap band -- +0.038 on a token the
        model prices at 0.074 is a 51% error, not a rounding one."""
        p = self.overstated_side_predicted
        return self.traded_overstatement / p if p > 0 else float("inf")


def reliability(
    y: np.ndarray,
    p: np.ndarray,
    condition_id: np.ndarray,
    edges: tuple[float, ...] = DEFAULT_RELIABILITY_EDGES,
    min_n: int = 200,
    min_windows: int = 5,
) -> list[ReliabilityBucket]:
    """Aggregate log-loss hides *where* a model is wrong. A fit can improve
    average log-loss while being badly miscalibrated in exactly the price
    band it goes on to trade, and an overstated probability is an overpaid
    trade."""
    out = []
    for lo, hi in zip(edges, edges[1:]):
        m = (p >= lo) & (p < hi)
        n_windows = len(set(condition_id[m]))
        if m.sum() < min_n or n_windows < min_windows:
            continue
        out.append(
            ReliabilityBucket(lo, hi, int(m.sum()), n_windows, float(p[m].mean()), float(y[m].mean()))
        )
    return out


@dataclass
class CalibrationReport:
    n_train: int
    n_test: int
    n_windows_train: int
    n_windows_test: int
    analytic_log_loss: float
    calibrated_log_loss: float
    coef_: np.ndarray  # [intercept, dev, dev*log_t, log_t]
    improvement: float  # analytic_log_loss - calibrated_log_loss; positive = calibration helps
    train_end_ts: float  # last timestamp trained on; anything <= this is in-sample
    reliability: list[ReliabilityBucket]  # out-of-sample, calibrated model
    analytic_reliability: list[ReliabilityBucket]  # same bands, analytic model
    deviation_clip: float

    def failing_bands(self, max_overstatement: float, min_z: float = 2.0) -> list[ReliabilityBucket]:
        """Bands where the fit overprices *either* token both *materially*
        (beyond `max_overstatement`, so it would actually cost money) and
        *significantly* (beyond `min_z` clustered standard errors, so it is
        not just this sample's base rate wandering).

        Both tests are needed. Magnitude alone rejects every model on a
        small sample -- the held-out base rate here sits 3.4pp below the
        training one purely by chance, which shows up as a uniform positive
        overstatement in every band. Significance alone would reject
        economically irrelevant errors once the dataset grows.

        Tested on `traded_overstatement`, not `overstatement`: the signed
        version only ever catches an overpriced *up* token, and the
        strategy buys both. See `ReliabilityBucket.overstated_side`.
        """
        return [
            b
            for b in self.reliability
            if b.traded_overstatement > max_overstatement and b.traded_z > min_z
        ]


def load_calibration_dataset(db_path: str | Path) -> CalibrationDataset:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT f.ts, f.deviation, f.t_remaining, m.resolved_outcome, f.condition_id
            FROM fair_value_log f
            JOIN markets m ON m.condition_id = f.condition_id
            WHERE f.deviation IS NOT NULL AND m.resolved_outcome IS NOT NULL
              AND f.t_remaining IS NOT NULL AND f.t_remaining > 0
            ORDER BY f.ts
            """
        ).fetchall()
    finally:
        conn.close()
    ts = np.array([r[0] for r in rows], dtype=float)
    deviation = np.array([r[1] for r in rows], dtype=float)
    t_remaining = np.array([r[2] for r in rows], dtype=float)
    outcome = np.array([1.0 if r[3] == "up" else 0.0 for r in rows], dtype=float)
    condition_id = np.array([r[4] for r in rows], dtype=object)
    return CalibrationDataset(
        ts=ts, deviation=deviation, t_remaining=t_remaining, outcome=outcome, condition_id=condition_id
    )


def fit_calibration(
    dataset: CalibrationDataset,
    train_frac: float = 0.7,
    min_test_rows: int = 20,
    min_train_rows: int = 30,
    min_windows: int = 20,
    deviation_clip: float = DEFAULT_DEVIATION_CLIP,
    iterations: int = 20_000,
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

    def design(idx: slice) -> np.ndarray:
        return np.array(
            [
                calibration_features(float(d), float(t), deviation_clip)
                for d, t in zip(dataset.deviation[idx], dataset.t_remaining[idx])
            ],
            dtype=float,
        )

    analytic_pred = np.array([normal_cdf(float(d)) for d in x_test])
    analytic_loss = log_loss(y_test, analytic_pred)

    # 2,000 gradient steps leaves this materially short of convergence on
    # a multi-feature design; the extra iterations are cheap next to the
    # cost of shipping an under-fit calibration.
    model = LogisticRegression(iterations=iterations).fit(design(slice(None, split)), y_train)
    calibrated_pred = model.predict_proba(design(slice(split, None)))
    calibrated_loss = log_loss(y_test, calibrated_pred)

    cid_test = dataset.condition_id[split:]
    return CalibrationReport(
        n_train=len(x_train),
        n_test=len(x_test),
        n_windows_train=windows_train,
        n_windows_test=windows_test,
        analytic_log_loss=analytic_loss,
        calibrated_log_loss=calibrated_loss,
        coef_=model.coef_,
        improvement=analytic_loss - calibrated_loss,
        reliability=reliability(y_test, calibrated_pred, cid_test),
        analytic_reliability=reliability(y_test, analytic_pred, cid_test),
        deviation_clip=deviation_clip,
        train_end_ts=float(dataset.ts[split - 1]),
    )
