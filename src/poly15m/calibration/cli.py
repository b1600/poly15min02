"""Fit the logistic calibration correction from `settings.db_path`'s
recorded `fair_value_log`/`markets` data and report how it compares to the
raw analytic model out-of-sample (Implementation_Plan.md Phase 6, item 24).
"""

from __future__ import annotations

import argparse
import time

from ..config import settings
from ..logging_setup import setup_logging
from ..pricing.fair_value import ANALYTIC_LOGIT_SLOPE, Calibration
from .fit import fit_calibration, load_calibration_dataset

# Max tolerated overstatement of *either* token in any probability band
# before a fit is considered unsafe to trade, regardless of its average
# log-loss.
#
# A trade's EV per share is `required_edge - overstatement`: the strategy
# only buys at `fair_value - min_edge_to_trade - fees - buffers`, so an
# overstatement smaller than the edge it demands still clears. That makes
# `min_edge_to_trade` the only meaningful yardstick for this threshold --
# and it must be a *fraction* of it, not all of it. The previous flat 0.05
# was exactly `min_edge_to_trade`, so a fit sitting just inside the gate
# had 100% of its edge eaten by model error and was waved through as safe.
# Half leaves the edge half intact.
EDGE_MARGIN_FRACTION = 0.5
MAX_OVERSTATEMENT = EDGE_MARGIN_FRACTION * settings.min_edge_to_trade
# Overstatement must also clear this many window-clustered standard errors
# before it counts as evidence rather than sampling noise.
MIN_Z = 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"persist the fitted coefficients to {settings.calibration_path} "
        "so paper/live/backtest runs use them instead of raw Phi(deviation)",
    )
    args = parser.parse_args()

    setup_logging(
        settings.log_level,
        settings.log_json,
        telegram_bot_token=settings.telegram_bot_token,
        telegram_chat_id=settings.telegram_chat_id,
    )

    dataset = load_calibration_dataset(settings.db_path)
    n = len(dataset.deviation)
    print(f"Loaded {n} (deviation, outcome) sample(s) from {settings.db_path}.")

    report = fit_calibration(dataset)
    if report is None:
        print(
            "Not enough resolved-window data to calibrate yet "
            "(Implementation_Plan.md: \"once a few hundred windows of data exist\")."
        )
        print("Run paper_trade.py for longer, then try again.")
        return

    print(f"\nTrain rows: {report.n_train} ({report.n_windows_train} windows)   "
          f"Test rows: {report.n_test} ({report.n_windows_test} windows)")
    print(f"Analytic model   (Phi(deviation))   log-loss: {report.analytic_log_loss:.4f}")
    print(f"Calibrated model (time-varying)     log-loss: {report.calibrated_log_loss:.4f}")
    verdict = "beats" if report.improvement > 0 else "does not beat"
    print(f"Calibration {verdict} the analytic model out-of-sample (delta={report.improvement:+.4f}).")

    calibration = Calibration(
        coef=tuple(float(c) for c in report.coef_),
        deviation_clip=report.deviation_clip,
        fitted_ts=time.time(),
        n_train_rows=report.n_train,
        n_train_windows=report.n_windows_train,
        train_end_ts=report.train_end_ts,
    )

    # The analytic model's implied slope is constant at ~1.702; the fitted
    # one varies with time left, which is the whole point of the refit.
    print("\nEffective slope on `deviation` (analytic is a flat 1.702):")
    for t in (15.0, 60.0, 300.0, 900.0):
        eff = calibration.effective_slope(t)
        print(f"  {t:>5.0f}s left: {eff:.3f}   ({ANALYTIC_LOGIT_SLOPE / eff:.2f}x overconfident)")

    def show(label, buckets):
        print(f"\n{label} -- predicted vs actually realised P(up), out-of-sample:")
        print(f"  {'band':>13} {'ticks':>8} {'windows':>8} {'pred':>7} {'real':>7} "
              f"{'side':>5} {'over':>7} {'rel':>6} {'z':>6}")
        for b in buckets:
            bad = b.traded_overstatement > MAX_OVERSTATEMENT and b.traded_z > MIN_Z
            print(
                f"  [{b.lo:.2f},{b.hi:.2f}) {b.n:>8,} {b.n_windows:>8} {b.predicted:>7.3f} "
                f"{b.realised:>7.3f} {b.overstated_side:>5} {b.traded_overstatement:>+7.3f} "
                f"{b.relative_overstatement:>5.0%} {b.traded_z:>6.2f}"
                f"{'  <-- overpays' if bad else ''}"
            )

    show("ANALYTIC", report.analytic_reliability)
    show("CALIBRATED", report.reliability)
    analytic_bad = [
        b
        for b in report.analytic_reliability
        if b.traded_overstatement > MAX_OVERSTATEMENT and b.traded_z > MIN_Z
    ]
    failing = report.failing_bands(MAX_OVERSTATEMENT, MIN_Z)
    print(f"\nBands overpricing a token materially (>{MAX_OVERSTATEMENT:.3f}) and significantly "
          f"(z>{MIN_Z:.1f}):  analytic {len(analytic_bad)}, calibrated {len(failing)}.")
    print("('side' is the token the band overprices -- understating P(up) by d overstates "
          "P(down) by d, and the strategy buys both.)")
    print("(z is in window-clustered standard errors -- ticks inside one window are not "
          "independent observations.)")

    if not args.write:
        print("\n(dry run -- pass --write to persist these coefficients.)")
        return
    if report.improvement <= 0:
        print("\nRefusing to write: this fit does not beat the analytic model out-of-sample.")
        return
    if failing:
        worst = max(failing, key=lambda b: b.traded_overstatement)
        print(
            f"\nRefusing to write: overprices the '{worst.overstated_side}' token by "
            f"{worst.traded_overstatement:+.3f} ({worst.relative_overstatement:.0%} of its "
            f"{worst.overstated_side_predicted:.3f} fair value, z={worst.traded_z:.2f}) in band "
            f"[{worst.lo:.2f},{worst.hi:.2f}), despite better log-loss. That error would consume "
            f"{worst.traded_overstatement / settings.min_edge_to_trade:.0%} of the "
            f"{settings.min_edge_to_trade:.3f} edge the strategy demands before trading. "
            "An overstated probability is an overpaid trade."
        )
        return

    calibration.save(settings.calibration_path)
    print(f"\nWrote {settings.calibration_path}.")
    print(
        "Backtests covering windows at or before the training cutoff are "
        "scoring this calibration IN-SAMPLE -- treat their PnL as optimistic."
    )


if __name__ == "__main__":
    main()
