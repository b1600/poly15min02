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
from typing import Any, Callable

from ..config import Settings
from .engine import run_backtest

DEFAULT_GRID: dict[str, list[Any]] = {
    "kelly_fraction": [0.02, 0.05, 0.10],
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
    kill_switch_trigger_count: int


def run_sweep(
    db_path: str | Path,
    condition_ids: list[str] | None,
    base_settings: Settings,
    param_grid: dict[str, list[Any]],
    on_row: Callable[[int, int, SweepRow], None] | None = None,
    on_window: Callable[[int, int, int, int], None] | None = None,
) -> list[SweepRow]:
    """`on_row(index, total, row)` fires as each combo finishes -- a full
    grid replays every window once per combo, which can take minutes to
    hours, so callers (e.g. the CLI) need per-combo progress rather than
    silence until every combo is done. `on_window(combo_index, num_combos,
    windows_done, windows_total)` fires as each window resolves *within*
    a combo, for the same reason at finer granularity -- a single combo
    can itself take minutes with no output otherwise."""
    keys = list(param_grid.keys())
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    rows: list[SweepRow] = []
    for i, combo in enumerate(combos, start=1):
        overrides = dict(zip(keys, combo))
        settings = base_settings.model_copy(update=overrides)
        window_cb = (
            (lambda done, total, i=i: on_window(i, len(combos), done, total)) if on_window is not None else None
        )
        _engine, result = run_backtest(db_path, condition_ids, settings, on_window_resolved=window_cb)
        row = SweepRow(
            overrides=overrides,
            realized_pnl=result.realized_pnl,
            fees_paid=result.fees_paid,
            num_windows=result.num_windows_replayed,
            kill_switch_active=result.kill_switch_active,
            kill_switch_trigger_count=result.kill_switch_trigger_count,
        )
        rows.append(row)
        if on_row is not None:
            on_row(i, len(combos), row)
    return rows


def _kill_switch_suffix(r: SweepRow) -> str:
    # kill_switch_active now reflects only whether the switch is *still*
    # tripped at the very end of the replay (BacktestEngine resets it each
    # simulated day -- see engine.py) -- trigger_count is the meaningful
    # signal for how often a combo blew its daily loss limit across the
    # whole dataset.
    if r.kill_switch_trigger_count == 0:
        return ""
    active = " [KILL SWITCH ACTIVE AT END]" if r.kill_switch_active else ""
    return f"  kill_switch_trips={r.kill_switch_trigger_count}{active}"


def format_sweep_results(rows: list[SweepRow]) -> str:
    ranked = sorted(rows, key=lambda r: r.realized_pnl, reverse=True)
    lines = []
    for r in ranked:
        overrides_str = ", ".join(f"{k}={v}" for k, v in r.overrides.items())
        lines.append(
            f"pnl={r.realized_pnl:+.2f}  fees={r.fees_paid:.2f}  windows={r.num_windows}  "
            f"{overrides_str}{_kill_switch_suffix(r)}"
        )
    return "\n".join(lines)


def main() -> None:
    import time

    from ..config import settings as default_settings
    from .data_loader import list_backtestable_markets

    condition_ids = list_backtestable_markets(default_settings.db_path)
    if not condition_ids:
        print(f"No backtestable markets found in {default_settings.db_path}.")
        return
    num_combos = 1
    for values in DEFAULT_GRID.values():
        num_combos *= len(values)
    print(
        f"Sweeping {len(condition_ids)} recorded window(s) from {default_settings.db_path} "
        f"across {num_combos} parameter combination(s)...\n"
    )

    start = time.monotonic()
    last_progress_print = start

    def on_row(i: int, total: int, row: SweepRow) -> None:
        elapsed = time.monotonic() - start
        overrides_str = ", ".join(f"{k}={v}" for k, v in row.overrides.items())
        print(
            f"[{i}/{total}, {elapsed:.0f}s elapsed] pnl={row.realized_pnl:+.2f}  "
            f"fees={row.fees_paid:.2f}  windows={row.num_windows}  {overrides_str}{_kill_switch_suffix(row)}",
            flush=True,
        )

    def on_window(combo_i: int, num_combos: int, windows_done: int, windows_total: int) -> None:
        nonlocal last_progress_print
        now = time.monotonic()
        # throttled to ~1 line per 15s -- fine-grained enough to prove
        # forward motion without flooding the terminal with one line per
        # window (there can be hundreds per combo)
        if now - last_progress_print < 15.0 and windows_done < windows_total:
            return
        last_progress_print = now
        print(
            f"  combo {combo_i}/{num_combos}: {windows_done}/{windows_total} windows "
            f"({now - start:.0f}s elapsed)",
            flush=True,
        )

    rows = run_sweep(
        default_settings.db_path,
        condition_ids,
        default_settings,
        DEFAULT_GRID,
        on_row=on_row,
        on_window=on_window,
    )
    print("\nRanked by realized PnL:")
    print(format_sweep_results(rows))


if __name__ == "__main__":
    main()
