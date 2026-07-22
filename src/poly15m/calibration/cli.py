"""Fit the logistic calibration correction from `settings.db_path`'s
recorded `fair_value_log`/`markets` data and report how it compares to the
raw analytic model out-of-sample (Implementation_Plan.md Phase 6, item 24).
"""

from __future__ import annotations

from ..config import settings
from ..logging_setup import setup_logging
from .fit import fit_calibration, load_calibration_dataset


def main() -> None:
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

    intercept, slope = report.coef_
    print(f"\nTrain rows: {report.n_train} ({report.n_windows_train} windows)   Test rows: {report.n_test} ({report.n_windows_test} windows)")
    print(f"Analytic model   (Phi(deviation))            log-loss: {report.analytic_log_loss:.4f}")
    print(f"Calibrated model (sigmoid({intercept:.3f} + {slope:.3f}*deviation))  log-loss: {report.calibrated_log_loss:.4f}")
    verdict = "beats" if report.improvement > 0 else "does not beat"
    print(f"\nCalibration {verdict} the analytic model out-of-sample (delta={report.improvement:+.4f}).")


if __name__ == "__main__":
    main()
