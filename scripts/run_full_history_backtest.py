"""Replays every backtestable window in var/poly15m.db through the current
strategy code, using corrected open_price anchoring and official-settlement
outcomes, and prints/saves a summary with a t-stat over per-window PnL.

Takes ~40-45 minutes on the full recorded history (~2,168 windows as of
2026-09-22) after the _RollingBars performance fix. Run it directly in your
own terminal (not through the Claude Code session) so it isn't vulnerable to
the session's sandbox suspending during idle gaps, which kills background
jobs launched from inside a Claude Code turn regardless of nohup/caffeinate.

Usage:
    .venv/bin/python scripts/run_full_history_backtest.py

Or, to let it survive a closed terminal too:
    nohup .venv/bin/python scripts/run_full_history_backtest.py \\
        > var/full_history_backtest.log 2>&1 &

Result is printed to stdout and saved to var/full_history_backtest_result.json.
"""

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.ERROR)

from poly15m.config import Settings
from poly15m.backtest.data_loader import list_backtestable_markets
from poly15m.backtest.engine import run_backtest

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "var" / "poly15m.db"
RESULT_PATH = REPO_ROOT / "var" / "full_history_backtest_result.json"


def main() -> None:
    ids = list_backtestable_markets(str(DB_PATH))
    print(f"scope: {len(ids)} windows, full recorded history "
          f"(corrected open_price + official outcomes)", flush=True)

    s = Settings()
    print(f"settings: kelly_fraction={s.kelly_fraction} min_edge_to_trade={s.min_edge_to_trade} "
          f"uncertainty_buffer_base={s.uncertainty_buffer_base} "
          f"use_calibrated_fair_value={s.use_calibrated_fair_value}", flush=True)

    t0 = time.time()
    engine, result = run_backtest(str(DB_PATH), ids, s)
    elapsed = time.time() - t0
    print(f"elapsed: {elapsed:.1f}s", flush=True)

    # Per-window PnL computed directly from fills + resolved_outcome --
    # immune to PaperExecutor.realized_pnl being a cumulative counter,
    # same ground-truth method used throughout this investigation.
    rows = engine.db._conn.execute("""
        select f.condition_id, m.window_open_ts,
               sum(case when f.outcome = m.resolved_outcome then f.size else 0 end)
                 - sum(f.price*f.size + f.fee) pnl
        from fills f join markets m on m.condition_id = f.condition_id
        where m.resolved_outcome is not null
        group by 1, 2
    """).fetchall()
    pnls = [r[2] for r in rows]
    n = len(pnls)

    out = {
        "scope_windows": len(ids),
        "windows_replayed": result.num_windows_replayed,
        "traded_windows": n,
        "realized_pnl": result.realized_pnl,
        "fees_paid": result.fees_paid,
        "abandoned_windows": engine.executor.abandoned_windows,
        "abandoned_cost_basis": engine.executor.abandoned_cost_basis,
        "kill_switch_trips": result.kill_switch_trigger_count,
        "kill_switch_active_at_end": result.kill_switch_active,
        "elapsed_s": elapsed,
    }

    if n >= 2:
        mean = sum(pnls) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in pnls) / (n - 1))
        se = sd / math.sqrt(n)
        t = mean / se if se > 0 else float("inf")
        out.update({"n": n, "mean_pnl_per_window": mean, "sd": sd, "t_stat": t, "sum_pnl": sum(pnls)})

    by_day: dict[str, list[float]] = {}
    for _cid, open_ts, pnl in rows:
        d = datetime.fromtimestamp(open_ts, timezone.utc).date().isoformat()
        by_day.setdefault(d, []).append(pnl)
    out["daily"] = {d: {"n": len(v), "sum": sum(v)} for d, v in sorted(by_day.items())}

    print(json.dumps(out, indent=2), flush=True)
    RESULT_PATH.write_text(json.dumps(out, indent=2))
    print(f"RESULT_WRITTEN: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
