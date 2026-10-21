#!/usr/bin/env python3
"""Does the Chainlink settlement price predict 15m outcomes beyond the market
price? Plan and pass bars: 20261021/20261021_chainlink_basis.txt (written
before this was run).

build    writes var/basis/windows.csv (open/close per feed, for the Step 0 gate)
         and var/basis/dataset.csv (one row per kept checkpoint).
analyze  runs Steps 0-2 and appends the results to the plan file.

    .venv/bin/python scripts/chainlink_basis.py build
    .venv/bin/python scripts/chainlink_basis.py analyze
"""

import argparse
import bisect
import csv
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from market_calibration import _boot_windows, _fit, _ll, _logit, _predict

REPO_ROOT = Path(__file__).resolve().parents[1]
DB = REPO_ROOT / "var" / "poly15m.db"
OUT_DIR = REPO_ROOT / "var" / "basis"
WINDOWS = OUT_DIR / "windows.csv"
DATASET = OUT_DIR / "dataset.csv"
REPORT = REPO_ROOT / "20261021" / "20261021_chainlink_basis.txt"

_utc = lambda *a: datetime(*a, tzinfo=timezone.utc).timestamp()  # noqa: E731
START = _utc(2026, 9, 29, 1, 30)
TEST_START = _utc(2026, 10, 9)
END_CLOSE = _utc(2026, 10, 15, 15, 0)

CHECKPOINTS = [180, 120, 90, 60, 30, 15]
GROUPS = {"mid": (90, 180), "late": (15, 60)}
MAX_TICK_AGE = 5.0
MAX_BOOK_AGE = 10.0
SIGMA_LOOKBACK = 300
SIGMA_FLOOR = 1.0
Z_CLIP = 6.0

GATE_MIN_AGREEMENT = 0.97
N_BOOT = 1000
CI_LO, CI_HI = 1.25, 98.75  # 97.5% CI: two groups, Bonferroni
MARGINS = [0.02, 0.04, 0.06, 0.08]
TRADE_MAX_BOOK_AGE = 5.0
FEE_RATE = 0.07

FIELDS = ["cid", "open_ts", "period", "checkpoint", "sigma", "z_cl", "z_bn",
          "up_bid", "up_ask", "down_bid", "down_ask", "book_age", "up_won"]
WFIELDS = ["cid", "open_ts", "period", "official", "cl_open", "cl_close", "bn_open", "bn_close"]


def _at(ts_list, vals, t):
    """Latest value at or before t, if no older than MAX_TICK_AGE."""
    i = bisect.bisect_right(ts_list, t) - 1
    if i < 0 or t - ts_list[i] > MAX_TICK_AGE:
        return None
    return vals[i]


def _ticks(con, source, lo, hi):
    rows = con.execute(
        "select event_ts, price from price_ticks where source=? and event_ts between ? and ? order by event_ts",
        (source, lo, hi),
    ).fetchall()
    return [r[0] for r in rows], [r[1] for r in rows]


def build() -> None:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    markets = con.execute(
        """select condition_id, window_open_ts, window_close_ts, resolved_outcome, token_id_up, token_id_down
           from markets where window_open_ts >= ? and window_close_ts <= ?
             and resolved_outcome in ('up', 'down') order by window_open_ts""",
        (START, END_CLOSE),
    ).fetchall()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_rows = n_skip = 0
    with WINDOWS.open("w", newline="") as wf, DATASET.open("w", newline="") as df:
        ww, dw = csv.writer(wf), csv.writer(df)
        ww.writerow(WFIELDS)
        dw.writerow(FIELDS)
        for i, (cid, open_ts, close_ts, outcome, tok_up, tok_dn) in enumerate(markets):
            period = "test" if open_ts >= TEST_START else "train"
            cl_t, cl_v = _ticks(con, "chainlink_twap60", open_ts - 10, close_ts)
            bn_t, bn_v = _ticks(con, "binance", open_ts - 10, close_ts)
            cl_open, cl_close = _at(cl_t, cl_v, open_ts), _at(cl_t, cl_v, close_ts)
            bn_open, bn_close = _at(bn_t, bn_v, open_ts), _at(bn_t, bn_v, close_ts)
            ww.writerow([cid, open_ts, period, outcome, cl_open, cl_close, bn_open, bn_close])
            if cl_open is None or bn_open is None:
                n_skip += len(CHECKPOINTS)
                continue

            books = {}
            for tok in (tok_up, tok_dn):
                rows = con.execute(
                    """select coalesce(event_ts, recv_ts) t, best_bid, best_ask from book_snapshots
                       where token_id=? and recv_ts between ? and ? order by t""",
                    (tok, close_ts - max(CHECKPOINTS) - 60, close_ts + 5),
                ).fetchall()
                books[tok] = ([r[0] for r in rows], rows)

            for cp in CHECKPOINTS:
                t = close_ts - cp
                cl, bn = _at(cl_t, cl_v, t), _at(bn_t, bn_v, t)
                quotes, ages = [], []
                for tok in (tok_up, tok_dn):
                    bts, rows = books[tok]
                    k = bisect.bisect_right(bts, t) - 1
                    if k < 0 or rows[k][1] is None or rows[k][2] is None:
                        break
                    quotes.append((rows[k][1], rows[k][2]))
                    ages.append(t - bts[k])
                if cl is None or bn is None or len(quotes) < 2 or max(ages) > MAX_BOOK_AGE:
                    n_skip += 1
                    continue
                # Binance sampled every second by last value over the lookback
                grid = []
                for s in range(SIGMA_LOOKBACK, -1, -1):
                    j = bisect.bisect_right(bn_t, t - s) - 1
                    if j >= 0:
                        grid.append(bn_v[j])
                diffs = [b - a for a, b in zip(grid, grid[1:])]
                if len(diffs) < SIGMA_LOOKBACK // 2:
                    n_skip += 1
                    continue
                mu = sum(diffs) / len(diffs)
                sd = math.sqrt(sum((x - mu) ** 2 for x in diffs) / (len(diffs) - 1))
                sigma = max(sd, SIGMA_FLOOR)
                scale = sigma * math.sqrt(cp)
                clip = lambda z: max(-Z_CLIP, min(Z_CLIP, z))  # noqa: E731
                (ub, ua), (db_, da) = quotes
                dw.writerow([cid, open_ts, period, cp, round(sigma, 4),
                             round(clip((cl - cl_open) / scale), 5), round(clip((bn - bn_open) / scale), 5),
                             ub, ua, db_, da, round(max(ages), 3), 1 if outcome == "up" else 0])
                n_rows += 1
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(markets)} windows, {n_rows} rows", file=sys.stderr, flush=True)
    print(f"{len(markets)} windows; wrote {n_rows} checkpoint rows ({n_skip} skipped) -> {DATASET}")


# ---------------------------------------------------------------- analyze
def _group(cp: int) -> str:
    return next(g for g, (lo, hi) in GROUPS.items() if lo <= cp <= hi)


def _load():
    import numpy as np

    with DATASET.open() as fh:
        rows = list(csv.DictReader(fh))
    col = lambda k, f=float: np.array([f(r[k]) for r in rows])  # noqa: E731
    d = {k: col(k) for k in ("sigma", "z_cl", "z_bn", "up_bid", "up_ask", "down_bid", "down_ask", "book_age")}
    d["cp"] = col("checkpoint", int)
    d["y"] = col("up_won")
    d["cid"] = np.array([r["cid"] for r in rows])
    d["test"] = np.array([r["period"] == "test" for r in rows])
    d["group"] = np.array([_group(c) for c in d["cp"]])
    d["mid"] = (d["up_bid"] + d["up_ask"]) / 2
    return d


FEATURES = {"M0": [], "M1": ["z_cl"], "M2": ["z_bn"]}


def _X(d, idx, model):
    import numpy as np

    return np.column_stack([_logit(d["mid"][idx])] + [d[f][idx] for f in FEATURES[model]])


def analyze() -> None:
    import numpy as np

    rng = np.random.default_rng(20261021)
    out: list[str] = []
    P = lambda s="": (print(s), out.append(s))  # noqa: E731

    # ---------------- Step 0
    P("=== STEP 0. DATA VALIDITY GATE (train windows only) ===")
    with WINDOWS.open() as fh:
        wins = [r for r in csv.DictReader(fh) if r["period"] == "train"]
    for feed in ("cl", "bn"):
        ok = [r for r in wins if r[f"{feed}_open"] and r[f"{feed}_close"]]
        agree = sum((float(r[f"{feed}_close"]) >= float(r[f"{feed}_open"])) == (r["official"] == "up") for r in ok)
        P(f"{'Chainlink' if feed == 'cl' else 'Binance':9} proxy vs official: {agree}/{len(ok)} = {agree / len(ok):.2%}"
          f"  ({len(wins) - len(ok)} windows missing a tick)")
        if feed == "cl":
            gate = agree / len(ok)
    gate_ok = gate >= GATE_MIN_AGREEMENT
    P(f"gate (>= {GATE_MIN_AGREEMENT:.0%}): {'PASS' if gate_ok else 'FAIL'}")
    if not gate_ok:
        P("Gate failed: Steps 1-2 not run, per the plan.")
        _append(out)
        return

    d = _load()
    tr, te = ~d["test"], d["test"]
    P(f"\nrows: train {tr.sum()} ({len(np.unique(d['cid'][tr]))} windows), "
      f"test {te.sum()} ({len(np.unique(d['cid'][te]))} windows)")

    # ---------------- Step 1
    P("\n=== STEP 1. INFORMATION (fit on train, log loss on test, lower is better) ===")
    fits, passed = {}, []
    for g in GROUPS:
        itr = np.flatnonzero(tr & (d["group"] == g))
        ite = np.flatnonzero(te & (d["group"] == g))
        w = {m: _fit(_X(d, itr, m), d["y"][itr]) for m in FEATURES}
        fits[g] = w
        ll = {m: _ll(_predict(w[m], _X(d, ite, m)), d["y"][ite]) for m in FEATURES}
        P(f"\n{g:4}  train n={len(itr)}  test n={len(ite)} ({len(np.unique(d['cid'][ite]))} windows)")
        P(f"      train coefs  M1: a={w['M1'][0]:+.3f} b_mid={w['M1'][1]:.3f} c_zcl={w['M1'][2]:+.3f}"
          f"   M2: a={w['M2'][0]:+.3f} b_mid={w['M2'][1]:.3f} c_zbn={w['M2'][2]:+.3f}")
        P("      test log loss: " + "  ".join(f"{m} {v.mean():.4f}" for m, v in ll.items())
          + f"  (raw mid {_ll(np.clip(d['mid'][ite], 0.01, 0.99), d['y'][ite]).mean():.4f})")
        cids = d["cid"][ite]
        for m in ("M1", "M2"):
            diff = ll[m] - ll["M0"]
            boots = np.array([diff[_boot_windows(cids, rng)].mean() for _ in range(N_BOOT)])
            lo, hi = np.percentile(boots, CI_LO), np.percentile(boots, CI_HI)
            line = f"      {m} - M0: {diff.mean():+.4f}  97.5% CI [{lo:+.4f}, {hi:+.4f}]"
            if m == "M1":
                ok = hi < 0 and w["M1"][2] > 0
                line += f"  -> INFORMATION {'PASS' if ok else 'FAIL'}"
                if ok:
                    passed.append(g)
            else:
                line += "  (reference)"
            P(line)

    # ---------------- Step 2
    P("\n=== STEP 2. TRADING RULE (1 share at the ask, first qualifying checkpoint per window) ===")
    fee = lambda p: FEE_RATE * p * (1 - p)  # noqa: E731

    def simulate(q, mask, m):
        order = np.lexsort((-d["cp"], d["cid"]))
        pnl, taken = {}, set()
        for i in order:
            c = d["cid"][i]
            if not mask[i] or c in taken or d["book_age"][i] > TRADE_MAX_BOOK_AGE:
                continue
            ua, da = d["up_ask"][i], d["down_ask"][i]
            e_up, e_dn = q[i] - ua - fee(ua), (1 - q[i]) - da - fee(da)
            if max(e_up, e_dn) <= m:
                continue
            pnl[c] = (d["y"][i] - ua - fee(ua)) if e_up >= e_dn else ((1 - d["y"][i]) - da - fee(da))
            taken.add(c)
        x = np.array(list(pnl.values()))
        t = x.mean() / x.std(ddof=1) * math.sqrt(len(x)) if len(x) > 1 and x.std(ddof=1) > 0 else float("nan")
        return len(x), x.sum() if len(x) else 0.0, x.mean() if len(x) else float("nan"), t

    for model in ("M1", "M0"):
        for g in GROUPS:
            q = np.full(len(d["y"]), np.nan)
            i = np.flatnonzero(d["group"] == g)
            q[i] = _predict(fits[g][model], _X(d, i, model))
            in_g = d["group"] == g
            P(f"\n[{model} {g}]  {'m':>5} | {'train n':>7} {'PnL':>8} {'/trade':>7} {'t':>6} |")
            best = None
            for m in MARGINS:
                n, s, mu, t = simulate(q, tr & in_g, m)
                P(f"{'':12}{m:5.2f} | {n:7} {s:+8.2f} {mu:+7.4f} {t:6.2f} |")
                if best is None or s > best[1]:
                    best = (m, s)
            m = best[0]
            n, s, mu, t = simulate(q, te & in_g, m)
            if model == "M1" and g in passed:
                verdict = "PASS" if s > 0 and t >= 2 else "PROMISING" if s > 0 and t >= 1 else "FAIL"
            else:
                verdict = "info only (group failed Step 1)" if model == "M1" else "reference"
            P(f"   chosen m={m:.2f} on train -> TEST n={n} PnL {s:+.2f} per trade {mu:+.4f} t={t:.2f} -> {verdict}")

    _append(out)


def _append(out: list[str]) -> None:
    with REPORT.open("a") as fh:
        fh.write(f"\n-----------------------RESULTS {datetime.now().strftime('%d %b %Y %H:%M')}\n")
        fh.write("\n".join(out) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["build", "analyze"])
    a = ap.parse_args()
    build() if a.cmd == "build" else analyze()


if __name__ == "__main__":
    main()
