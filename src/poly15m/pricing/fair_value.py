"""Fair-value model (Implementation_Plan.md Phase 2, item 10; calibrated
correction from Phase 6, item 24).

The analytic model is P(Up) = Phi(deviation): the probability BTC ends
above the window's opening price given current distance, recent
volatility, and time left, under a driftless-Brownian-motion assumption on
the price level. It's deliberately not ML: no training dependency, and
it's fast enough to reprice on every tick, which is the actual edge this
strategy is chasing (repricing faster than stale resting orders).

Phi is, however, *systematically overconfident* on this project's recorded
data. Phi(x) ~= sigmoid(1.702x), an implied logit slope of ~1.70, whereas
fitting a logistic on 295 recorded windows gives a slope near 0.54 -- the
analytic model reads roughly 3x more signal into `deviation` than is
actually there. Overconfident fair value inflates `net_edge`, so the
strategy takes trades that have no real edge and pays taker fees to do it.

`Calibration` therefore holds a fitted (intercept, slope) applied on the
logit scale, and carries its own provenance -- crucially `train_end_ts`,
so a backtest can tell which of its windows the calibration was fit on and
is therefore scoring in-sample. Passing `calibration=None` keeps the pure
analytic model, which is also the automatic fallback when no calibration
file has been fitted yet.

`compute_fair_value` returns None when `deviation` is None (e.g. not
enough realized-vol history yet, or the window has already closed) --
callers should treat "no fair value yet" as "don't trade this tick", not
default to a specific probability.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

# Phi(x) ~= sigmoid(LOGIT_SCALE * x); the analytic model's implied slope,
# and the reference a fitted slope should be compared against.
ANALYTIC_LOGIT_SLOPE = 1.702


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _sigmoid(z: float) -> float:
    # branch on sign so neither exp() overflows for large |z|
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


DEFAULT_DEVIATION_CLIP = 10.0


def calibration_features(deviation: float, t_remaining: float, clip: float) -> list[float]:
    """Feature row for the calibrated model, shared by fitting and inference.

    Two things matter here, both learned the hard way:

    `deviation` is (spot - open) / (sigma * sqrt(t_remaining)), so it
    explodes as t_remaining -> 0 -- the 99th percentile is ~135. Left
    unclipped those outliers dominate the fitter's feature standardisation
    and drag the fitted slope down by ~2x, which reads as "the model is
    wildly overconfident" when it is mostly a numerical artifact. Clip
    first.

    The slope on `deviation` is genuinely not constant: fitting per
    time-bucket gives ~0.62 with under 30s left and ~1.21 with 10+ minutes
    left. `deviation` is already time-normalised, so in theory it should
    be constant -- that it isn't says the driftless-Brownian assumption
    degrades near expiry (sigma is an EWMA estimate, and the Binance spot
    the model sees is not the feed that resolves the market). The
    deviation*log(t) interaction lets one model span both regimes instead
    of splitting the difference and being wrong at each end.
    """
    d = min(max(deviation, -clip), clip)
    log_t = math.log(max(t_remaining, 1.0))
    return [d, d * log_t, log_t]


@dataclass(frozen=True)
class Calibration:
    """Fitted correction: P(Up) = sigmoid(coef . [1, features(deviation, t)]).

    `train_end_ts` is the timestamp of the last row the fit trained on.
    Any backtest window at or before it is being scored *in-sample*; the
    backtest reports this rather than silently flattering itself.
    """

    coef: tuple[float, ...]  # [intercept, dev, dev*log_t, log_t]
    deviation_clip: float = DEFAULT_DEVIATION_CLIP
    fitted_ts: float | None = None
    n_train_rows: int | None = None
    n_train_windows: int | None = None
    train_end_ts: float | None = None

    def __post_init__(self) -> None:
        expected = len(calibration_features(0.0, 1.0, self.deviation_clip)) + 1
        if len(self.coef) != expected:
            raise ValueError(f"expected {expected} coefficients, got {len(self.coef)}")

    def p_up(self, deviation: float, t_remaining: float) -> float:
        feats = calibration_features(deviation, t_remaining, self.deviation_clip)
        z = self.coef[0] + sum(c * f for c, f in zip(self.coef[1:], feats))
        return _sigmoid(z)

    def effective_slope(self, t_remaining: float) -> float:
        """Slope on `deviation` at `t_remaining`, for comparison against
        the analytic model's constant implied slope of ~1.702."""
        return self.coef[1] + self.coef[2] * math.log(max(t_remaining, 1.0))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration | None":
        path = Path(path)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        if "coef" in known:
            known["coef"] = tuple(known["coef"])
        return cls(**known)


@dataclass
class FairValue:
    p_up: float
    p_down: float
    deviation: float


def compute_fair_value(
    deviation: float | None,
    t_remaining: float | None = None,
    calibration: Calibration | None = None,
) -> FairValue | None:
    """`t_remaining` is required by the calibrated model (the slope on
    deviation varies with it) and ignored by the analytic one. Passing a
    calibration without a t_remaining is a programming error rather than a
    reason to silently fall back to a different model."""
    if deviation is None:
        return None
    if calibration is not None:
        if t_remaining is None:
            raise ValueError("t_remaining is required when a calibration is active")
        p_up = calibration.p_up(deviation, t_remaining)
    else:
        p_up = normal_cdf(deviation)
    return FairValue(p_up=p_up, p_down=1.0 - p_up, deviation=deviation)


def load_calibration(cfg) -> "Calibration | None":
    """The active fair-value calibration, or None to use raw Phi(deviation).

    Returns None when calibration is switched off *or* when no fit has been
    persisted yet, so an un-calibrated deployment degrades to the analytic
    model rather than failing to start.
    """
    if not cfg.use_calibrated_fair_value:
        return None
    return Calibration.load(cfg.calibration_path)
