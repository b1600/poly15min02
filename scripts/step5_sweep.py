#!/usr/bin/env python3
"""Step 5 of the 20260928 plan: test candidate fixes on data they weren't
built from.

Each variant replays the whole recorded history (Aug 3 onward) once with a
single setting changed; per-window PnL is then split into

  train: windows opening Aug 3 - Sep 22 2026 (UTC)
  test:  windows opening Sep 23 2026 onward

The variant is picked on train only, by the rule in `report` below, which
was written down before any variant was run (see
20260928/20260928_step5_fixes.txt). Test numbers for the other variants are
printed for information but do not change the choice.

Per-window PnL = sum(size if outcome == official resolved_outcome)
- sum(price*size + fee), from the replay's own fills, over every window in
scope (untraded windows count as 0 so variants are paired window by window).

Takes ~40 min per variant alone on the full history. On an 8 GB machine run
at most ~3-4 in parallel (more swaps heavily and is far slower overall):

    .venv/bin/python scripts/step5_sweep.py run base
    .venv/bin/python scripts/step5_sweep.py run maxT480
    ...
    .venv/bin/python scripts/step5_sweep.py report
"""

import argparse
import json
import logging
import math
import sqlite3
import statistics as st
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.ERROR)

from poly15m.backtest.data_loader import list_backtestable_markets
from poly15m.backtest.engine import run_backtest
from poly15m.config import Settings
from poly15m.db import Database

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "var" / "poly15m.db"
OUT_DIR = REPO_ROOT / "var" / "step5"

TRAIN_START = datetime(2026, 8, 3, tzinfo=timezone.utc).timestamp()
TEST_START = datetime(2026, 9, 23, tzinfo=timezone.utc).timestamp()

# sigma thresholds are the Aug 3 - Sep 22 fair_value_log quartiles
# (q25 = 1.50, median = 2.42), i.e. taken from train inputs, not outcomes.
VARIANTS: dict[str, dict] = {
    "base": {},
    "maxT480": {"max_entry_t_remaining": 480.0},
    "maxT600": {"max_entry_t_remaining": 600.0},
    "volmin1.5": {"min_sigma_to_trade": 1.5},
    "volmin2.4": {"min_sigma_to_trade": 2.4},
    "floor1.5": {"sigma_floor": 1.5},
    "floor2.4": {"sigma_floor": 2.4},
}


def run(name: str) -> None:
    overrides = VARIANTS[name]
    s = Settings().model_copy(update=overrides)
    conn = sqlite3.connect(str(DB_PATH))
    open_ts = dict(conn.execute("select condition_id, window_open_ts from markets"))
    outcome = dict(conn.execute("select condition_id, resolved_outcome from markets"))
    conn.close()
    ids = [c for c in list_backtestable_markets(str(DB_PATH)) if open_ts[c] >= TRAIN_START]
    print(f"[{name}] {overrides} over {len(ids)} windows", flush=True)

    t0 = time.time()

    def progress(n: int, total: int) -> None:
        if n % 250 == 0 or n == total:
            print(f"[{name}] {n}/{total} {time.time() - t0:.0f}s", flush=True)

    # Replay output (fills, ~2.4M fair_value_log rows) goes to a file, not
    # ":memory:": seven in-memory replays at once pushed an 8 GB machine
    # into heavy swap and ran ~7x slower.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_db = OUT_DIR / f"{name}.db"
    for p in OUT_DIR.glob(f"{name}.db*"):
        p.unlink()
    engine, _ = run_backtest(str(DB_PATH), ids, s, output_db=Database(out_db), on_window_resolved=progress)
    pnl = {c: 0.0 for c in ids}
    n_fills = {c: 0 for c in ids}
    for cid, oc, p, sz, fee in engine.db._conn.execute(
        "select condition_id, outcome, price, size, fee from fills"
    ):
        if cid in pnl and outcome.get(cid) is not None:
            pnl[cid] += (sz if oc == outcome[cid] else 0.0) - p * sz - fee
            n_fills[cid] += 1
    rows = [
        {"cid": c, "open_ts": open_ts[c], "resolved": outcome.get(c) is not None, "pnl": pnl[c], "fills": n_fills[c]}
        for c in ids
    ]
    (OUT_DIR / f"{name}.json").write_text(
        json.dumps({"variant": name, "overrides": overrides, "elapsed_s": time.time() - t0, "windows": rows})
    )
    print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)


def _stats(x: list[float]) -> tuple[float, float, float]:
    """sum, mean, t over a list of per-window values."""
    if len(x) < 2:
        return sum(x), float("nan"), float("nan")
    sd = st.stdev(x)
    return sum(x), st.mean(x), (st.mean(x) / sd * math.sqrt(len(x))) if sd > 0 else float("nan")


def report() -> None:
    res = {}
    for name in VARIANTS:
        p = OUT_DIR / f"{name}.json"
        if p.exists():
            res[name] = {w["cid"]: w for w in json.loads(p.read_text())["windows"] if w["resolved"]}
    if "base" not in res:
        sys.exit("base variant has not been run yet")
    common = set.intersection(*(set(r) for r in res.values()))
    base = res["base"]

    def split(r, test):
        return [c for c in common if (r[c]["open_ts"] >= TEST_START) == test]

    train_ids = sorted(split(base, False))
    test_ids = sorted(split(base, True))
    print(f"windows: train {len(train_ids)}, test {len(test_ids)} (resolved, present in every variant run)")
    print(f"variants run: {', '.join(res)}\n")
    hdr = f"{'variant':10} {'traded':>6} {'PnL$':>8} {'/win':>7} {'t':>6} {'vs base':>8} {'paired t':>8}"
    summary = {}
    for label, ids in (("TRAIN Aug 3 - Sep 22", train_ids), ("TEST Sep 23 onward", test_ids)):
        print(f"=== {label} ===")
        print(hdr)
        for name, r in res.items():
            x = [r[c]["pnl"] for c in ids]
            d = [r[c]["pnl"] - base[c]["pnl"] for c in ids]
            tot, mean, t = _stats(x)
            dtot, _, dt_ = _stats(d)
            traded = sum(1 for c in ids if r[c]["fills"] > 0)
            summary[(label[:4], name)] = (tot, t, dtot, dt_)
            print(f"{name:10} {traded:6} {tot:+8.2f} {mean:+7.3f} {t:6.2f} {dtot:+8.2f} {dt_:8.2f}")
        print()

    # --- pre-registered selection rule (see module docstring) ---
    print("=== PRE-REGISTERED DECISION ===")
    cands = [n for n in res if n != "base"]
    best = max(cands, key=lambda n: summary[("TRAI", n)][0]) if cands else None
    if best is None or not summary[("TRAI", best)][3] >= 2.0:
        why = f"best on train is {best} with paired t {summary[('TRAI', best)][3]:.2f} < 2" if best else "no variants"
        print(f"No fix selected ({why}).")
        return
    tot, t, dtot, dt_ = summary[("TEST", best)]
    print(f"Selected on train: {best} (train paired t {summary[('TRAI', best)][3]:.2f} vs base)")
    verdict = "PASS" if tot > 0 and dt_ >= 1.0 else "FAIL"
    print(f"Test: PnL {tot:+.2f} (t {t:.2f}), vs base {dtot:+.2f} (paired t {dt_:.2f}) -> {verdict}")
    print("PASS requires test PnL > 0 AND paired t vs base >= 1.0")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("variant", choices=list(VARIANTS))
    sub.add_parser("report")
    a = ap.parse_args()
    run(a.variant) if a.cmd == "run" else report()


if __name__ == "__main__":
    main()
