"""Minimal logistic regression, fit by gradient descent with a small L2
penalty (numerical stability against the near-perfect-separation cases a
strong single feature like `deviation` can produce, not a serious attempt
at optimal regularization strength).

Implemented directly on numpy rather than adding scikit-learn: the project
already avoided a similar heavier dependency for the analytic fair-value
model (`pricing/fair_value.py` uses `math.erf` instead of scipy), and one
feature (or a handful) with a few hundred to a few thousand rows doesn't
need more than this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


@dataclass
class LogisticRegression:
    learning_rate: float = 0.5
    iterations: int = 2000
    l2: float = 1e-3
    coef_: np.ndarray | None = None  # coef_[0] is the intercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, d = X.shape

        # standardize features for stable/comparable gradient steps; fold
        # the scaling back into the final coefficients so predict() takes
        # raw inputs
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma == 0] = 1.0
        Xs = (X - mu) / sigma

        Xb = np.hstack([np.ones((n, 1)), Xs])
        w = np.zeros(d + 1)
        for _ in range(self.iterations):
            p = _sigmoid(Xb @ w)
            grad = Xb.T @ (p - y) / n
            grad[1:] += self.l2 * w[1:]  # don't regularize the intercept
            w -= self.learning_rate * grad

        # unfold standardization: w0 + sum(w_i * (x_i - mu_i)/sigma_i)
        # = (w0 - sum(w_i*mu_i/sigma_i)) + sum((w_i/sigma_i) * x_i)
        raw_coef = w[1:] / sigma
        raw_intercept = w[0] - np.sum(w[1:] * mu / sigma)
        self.coef_ = np.concatenate([[raw_intercept], raw_coef])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("fit() must be called before predict_proba()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        Xb = np.hstack([np.ones((n, 1)), X])
        return _sigmoid(Xb @ self.coef_)


def log_loss(y_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.clip(np.asarray(p_pred, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p_pred) + (1.0 - y_true) * np.log(1.0 - p_pred)))
