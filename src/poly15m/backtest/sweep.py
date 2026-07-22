"""Parameter sweep (Implementation_Plan.md Phase 6, item 23: "Sweep
parameters: buffers, Kelly fraction, cancel thresholds").

Each combination gets a fully isolated backtest -- a fresh PositionManager,
RiskGate, PaperExecutor and output Database per run, so results from one
combination can't leak into another via shared mutable state.

Note on scope: `PaperExecutor` is a taker-style "walk the book" simulator
(Phase 3), not a resting-order simulator, so there's no literal "cancel
threshold" for it to sweep -- that concept (`reprice_threshold`) only
applies to the live post-only executor (Phase 4), which isn't something
this backtester replays (simulating whether a resting limit order would
have been filled by historical order flow is a materially different, much
harder problem than replaying taker fills). This sweeps every parameter
that *does* affect paper-simulated outcomes: Kelly fraction, edge
threshold, the uncertainty buffer, slippage buffer, and the arb margin.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Settings
from .engine import run_backtest

DEFAULT_GRID: dict[str, list[Any]] = {
    "kelly_fraction": [0.10, 0.15, 0.20, 0.25],
    "min_edge_to_trade": [0.01, 0.02, 0.05],
    "uncertainty_buffer_base": [0.02, 0.05, 0.08],
}


@dataclass
class SweepRow:
    overrides: dict[str, Any]
    realized_pnl: float
    fees_paid: float
    num_windows: int
    kill_switch_active: bool


def run_sweep(
    db_path: str | Path,
    condition_ids: list[str] | None,
    base_settings: Settings,
    param_grid: dict[str, list[Any]],
) -> list[SweepRow]:
    keys = list(param_grid.keys())
    rows: list[SweepRow] = []
    for combo in itertools.product(*(param_grid[k] for k in keys)):
        overrides = dict(zip(keys, combo))
        settings = base_settings.model_copy(update=overrides)
        _engine, result = run_backtest(db_path, condition_ids, settings)
        rows.append(
            SweepRow(
                overrides=overrides,
                realized_pnl=result.realized_pnl,
                fees_paid=result.fees_paid,
                num_windows=result.num_windows_replayed,
                kill_switch_active=result.kill_switch_active,
            )
        )
    return rows


def format_sweep_results(rows: list[SweepRow]) -> str:
    ranked = sorted(rows, key=lambda r: r.realized_pnl, reverse=True)
    lines = []
    for r in ranked:
        overrides_str = ", ".join(f"{k}={v}" for k, v in r.overrides.items())
        flag = " [KILL SWITCH]" if r.kill_switch_active else ""
        lines.append(
            f"pnl={r.realized_pnl:+.2f}  fees={r.fees_paid:.2f}  windows={r.num_windows}  "
            f"{overrides_str}{flag}"
        )
    return "\n".join(lines)


def main() -> None:
    from ..config import settings as default_settings
    from .data_loader import list_backtestable_markets

    condition_ids = list_backtestable_markets(default_settings.db_path)
    if not condition_ids:
        print(f"No backtestable markets found in {default_settings.db_path}.")
        return
    print(f"Sweeping {len(condition_ids)} recorded window(s) from {default_settings.db_path}...\n")
    rows = run_sweep(default_settings.db_path, condition_ids, default_settings, DEFAULT_GRID)
    print(format_sweep_results(rows))


if __name__ == "__main__":
    main()
