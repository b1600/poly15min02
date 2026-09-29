#!/usr/bin/env python3
"""Reconcile the backtester with live paper trading over a date range.

Replays every window in [--start, --end) through the backtester (same code
path and settings as live, see backtest/engine.py) into a scratch DB, then
compares it window by window with the live paper fills and fair_value_log
recorded in the same DB:

  1. headline PnL / t-stat / ROI, live vs backtest
  2. window overlap: traded by both, live only, backtest only; same side?
     entry time and price differences
  3. fills per window, size, entry price, time remaining at first entry
  4. model inputs (spot, sigma, deviation, p_up_fair, mid) live vs backtest
  5. coverage for windows only one side traded
  6. per-day PnL
  7. per-window detail

If live and backtest agree, losses are the strategy's, and the backtester
can be trusted to test fixes. If the backtest is much better than live, the
simulator (fills, latency, book state) is too optimistic.

Window PnL for both = sum(size if outcome == official resolved_outcome)
- sum(price*size + fee), so neither depends on PaperExecutor's cumulative
realized_pnl counter. Settings come from .env; check
use_calibrated_fair_value matches the live run's "starting" log line.

First written for the Sep 23-28 2026 reconciliation
(20260928/20260928_reconbacktest.txt): ~330 s for 485 windows.

Usage:
    .venv/bin/python scripts/reconcile_backtest.py --start 2026-09-23
    .venv/bin/python scripts/reconcile_backtest.py --start 2026-09-23 --end 2026-09-28 \\
        --out 20260928/recon_raw.txt
    # re-run only the comparison on an existing replay:
    .venv/bin/python scripts/reconcile_backtest.py --start 2026-09-23 --bt-db var/recon_bt.db --skip-replay
"""

import argparse
import bisect
import collections
import datetime as dt
import logging
import math
import sqlite3
import statistics as st
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.ERROR)

from poly15m.config import Settings
from poly15m.backtest.engine import run_backtest
from poly15m.db import Database

REPO_ROOT = Path(__file__).resolve().parents[1]


def _utc(day: str) -> float:
    return dt.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=dt.UTC).timestamp()


def replay(live_db: Path, bt_db: Path, start: float, end: float) -> None:
    conn = sqlite3.connect(str(live_db))
    ids = [r[0] for r in conn.execute(
        """SELECT m.condition_id FROM markets m
           WHERE m.open_price IS NOT NULL AND m.window_open_ts >= ? AND m.window_open_ts < ?
             AND EXISTS (SELECT 1 FROM book_snapshots b WHERE b.condition_id = m.condition_id)
           ORDER BY m.window_open_ts""", (start, end))]
    conn.close()
    s = Settings()
    print(f"replaying {len(ids)} windows -> {bt_db}", file=sys.stderr, flush=True)
    print(f"settings: kelly_fraction={s.kelly_fraction} min_edge_to_trade={s.min_edge_to_trade} "
          f"uncertainty_buffer_base={s.uncertainty_buffer_base} "
          f"use_calibrated_fair_value={s.use_calibrated_fair_value}", file=sys.stderr, flush=True)
    for p in bt_db.parent.glob(bt_db.name + "*"):
        p.unlink()
    out = Database(bt_db)
    t0 = time.time()

    def progress(n: int, total: int) -> None:
        if n % 50 == 0 or n == total:
            print(f"  {n}/{total} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    _engine, res = run_backtest(live_db, ids, s, output_db=out, on_window_resolved=progress)
    out.flush()
    print(f"replayed {res.num_windows_replayed} windows in {time.time() - t0:.0f}s, "
          f"realized {res.realized_pnl:+.2f}", file=sys.stderr, flush=True)


def reconcile(live_db: Path, bt_db: Path, start: float, end: float, P) -> None:
    L = sqlite3.connect(str(live_db)); B = sqlite3.connect(str(bt_db))
    mk = {r[0]: dict(open=r[1], close=r[2], out=r[3], slug=r[4]) for r in L.execute(
        "select condition_id, window_open_ts, window_close_ts, resolved_outcome, slug from markets "
        "where window_open_ts>=? and window_open_ts<? and open_price is not null", (start, end))}
    bt_ids = set(r[0] for r in B.execute("select condition_id from markets"))
    mk = {k: v for k, v in mk.items() if k in bt_ids}

    def fills(conn):
        d = collections.defaultdict(list)
        for cid, oc, p, s, fee, ts in conn.execute(
                "select condition_id,outcome,price,size,fee,ts from fills where mode='paper' and ts>=? order by ts",
                (start - 900,)):
            if cid in mk:
                d[cid].append((ts, oc, p, s, fee))
        return d

    LF, BF = fills(L), fills(B)

    def pnl(cid, fl):
        out = mk[cid]["out"]
        if out is None:
            return None
        return sum((s if oc == out else 0) - p * s - fee for ts, oc, p, s, fee in fl)

    def day(cid): return dt.datetime.fromtimestamp(mk[cid]["open"], dt.UTC).strftime("%m-%d")
    def tstat(x): return (st.mean(x) / st.stdev(x) * math.sqrt(len(x))) if len(x) > 1 and st.stdev(x) > 0 else float("nan")
    def fmt(t): return dt.datetime.fromtimestamp(t, dt.UTC).strftime("%m-%d %H:%M:%S")

    ids = sorted(mk, key=lambda c: mk[c]["open"])
    P(f"Windows in scope: {len(ids)}  ({fmt(mk[ids[0]]['open'])} -> {fmt(mk[ids[-1]]['close'])} UTC), unresolved: {sum(mk[c]['out'] is None for c in ids)}")
    P()
    # ---------- 1. headline
    P("=== 1. HEADLINE: live vs backtest over the same windows ===")
    P(f"{'period':10} {'src':5} {'traded':>6} {'fills':>6} {'cost$':>8} {'fees$':>7} {'PnL$':>8} {'PnL/trd win':>11} {'t':>6} {'ROI':>6}")
    for period in ("ALL",):
        for name, F in (("live", LF), ("bt", BF)):
            cs = [c for c in ids if period == "ALL" and F.get(c) and mk[c]["out"]]
            pl = [pnl(c, F[c]) for c in cs]
            cost = sum(p*s+fee for c in cs for _, _, p, s, fee in F[c]); fees = sum(f[4] for c in cs for f in F[c])
            n = sum(len(F[c]) for c in cs)
            P(f"{period:10} {name:5} {len(cs):6} {n:6} {cost:8.2f} {fees:7.2f} {sum(pl):+8.2f} {st.mean(pl) if pl else 0:+11.3f} {tstat(pl):6.2f} {sum(pl)/cost*100 if cost else 0:+5.1f}%")
    P()
    # ---------- 2. overlap
    P("=== 2. WINDOW OVERLAP ===")
    cat = collections.defaultdict(list)
    for c in ids:
        l, b = bool(LF.get(c)), bool(BF.get(c))
        cat[("both" if l and b else "live_only" if l else "bt_only" if b else "neither")].append(c)
    for k in ("both", "live_only", "bt_only", "neither"):
        cs = cat[k]
        lp = [pnl(c, LF[c]) for c in cs if LF.get(c) and mk[c]["out"]]
        bp = [pnl(c, BF[c]) for c in cs if BF.get(c) and mk[c]["out"]]
        P(f"{k:10} windows={len(cs):4}  live PnL={sum(lp):+8.2f} (n={len(lp)})  bt PnL={sum(bp):+8.2f} (n={len(bp)})")
    both = cat["both"]
    same = [c for c in both if LF[c][0][1] == BF[c][0][1]]
    P(f"\nOf {len(both)} windows both traded: first entry on SAME side in {len(same)} ({len(same)/max(1,len(both))*100:.0f}%), OPPOSITE side in {len(both)-len(same)}")
    dts = [LF[c][0][0] - BF[c][0][0] for c in same]
    dps = [LF[c][0][2] - BF[c][0][2] for c in same]
    if dts:
        P(f"Same-side first entries: live entry time minus bt entry time: median {st.median(dts):+.1f}s, p10 {sorted(dts)[len(dts)//10]:+.1f}s, p90 {sorted(dts)[len(dts)*9//10]:+.1f}s")
        P(f"                         live price minus bt price:           median {st.median(dps):+.3f}, mean {st.mean(dps):+.4f}")
        P(f"                         |time diff| <= 2s: {sum(abs(x)<=2 for x in dts)}/{len(dts)}, identical price: {sum(abs(x)<1e-9 for x in dps)}/{len(dps)}")
    dd = [pnl(c, LF[c]) - pnl(c, BF[c]) for c in both if mk[c]["out"]]
    P(f"PnL(live) - PnL(bt) on shared windows: total {sum(dd):+.2f}, mean {st.mean(dd):+.3f}")
    opp = [c for c in both if c not in same and mk[c]["out"]]
    P(f"Opposite-side windows: live PnL {sum(pnl(c,LF[c]) for c in opp):+.2f}, bt PnL {sum(pnl(c,BF[c]) for c in opp):+.2f}")
    P()
    # ---------- 3. fill counts / sizing
    P("=== 3. FILLS PER TRADED WINDOW & SIZE ===")
    for name, F in (("live", LF), ("bt", BF)):
        cs = [c for c in ids if F.get(c)]
        sz = [s*p for c in cs for _, _, p, s, _ in F[c]]
        P(f"{name:5} fills/window mean {st.mean(len(F[c]) for c in cs):.2f}   $ per fill median {st.median(sz):.2f}   avg entry price {st.mean(f[2] for c in cs for f in F[c]):.3f}")
        tr = [mk[c]["close"] - F[c][0][0] for c in cs]
        P(f"      first-entry time remaining: >8min {sum(t>480 for t in tr)/len(tr)*100:.0f}%, median {st.median(tr):.0f}s")
    P()
    # ---------- 4. fair value agreement
    P("=== 4. MODEL INPUTS: fair_value_log live vs bt (nearest row within 1s, same window) ===")
    def fv(conn):
        d = collections.defaultdict(list)
        for cid, ts, spot, sig, dev, p, mid in conn.execute("select condition_id,ts,spot,sigma,deviation,p_up_fair,market_mid_up from fair_value_log where ts>=?", (start,)):
            if cid in mk: d[cid].append((ts, spot, sig, dev, p, mid))
        for v in d.values(): v.sort()
        return d
    LV, BV = fv(L), fv(B)
    diffs = collections.defaultdict(list); nm = 0; nl = 0
    for c, rows in LV.items():
        br = BV.get(c)
        if not br: continue
        bts = [r[0] for r in br]
        for r in rows[::5]:
            nl += 1
            i = bisect.bisect_left(bts, r[0])
            j = min((k for k in (i-1, i) if 0 <= k < len(bts)), key=lambda k: abs(bts[k]-r[0]), default=None)
            if j is None or abs(bts[j]-r[0]) > 1: continue
            nm += 1; b = br[j]
            for k, name in ((1,"spot"),(2,"sigma"),(3,"deviation"),(4,"p_up_fair"),(5,"market_mid_up")):
                if r[k] is not None and b[k] is not None: diffs[name].append(r[k]-b[k])
    P(f"live rows sampled {nl}, matched to a bt row within 1s: {nm}")
    for k, v in diffs.items():
        a = sorted(abs(x) for x in v)
        P(f"  {k:14} |live-bt| median {a[len(a)//2]:.5f}  p90 {a[len(a)*9//10]:.5f}  p99 {a[len(a)*99//100]:.5f}  mean signed {st.mean(v):+.5f}")
    P(f"  fair_value_log rows: live {sum(map(len,LV.values()))}, bt {sum(map(len,BV.values()))}")
    P()
    # ---------- 5. live gaps
    P("=== 5. WINDOWS WHERE ONLY ONE SIDE TRADED: coverage check ===")
    for k in ("live_only", "bt_only"):
        cs = cat[k]
        haslv = sum(1 for c in cs if LV.get(c)); hasbv = sum(1 for c in cs if BV.get(c))
        P(f"{k:10} n={len(cs)}: live had fair_value rows in {haslv}, bt had fair_value rows in {hasbv}")
    P()
    # ---------- 6. per day
    P("=== 6. PER DAY ===")
    P(f"{'day':6} {'live n':>6} {'live PnL':>9} {'bt n':>5} {'bt PnL':>8} {'both':>5} {'same side':>9}")
    for d in sorted(set(day(c) for c in ids)):
        cs = [c for c in ids if day(c) == d]
        l = [c for c in cs if LF.get(c) and mk[c]["out"]]; b = [c for c in cs if BF.get(c) and mk[c]["out"]]
        bo = [c for c in cs if c in both]; sa = [c for c in bo if c in same]
        P(f"{d:6} {len(l):6} {sum(pnl(c,LF[c]) for c in l):+9.2f} {len(b):5} {sum(pnl(c,BF[c]) for c in b):+8.2f} {len(bo):5} {len(sa):9}")
    P()
    # ---------- 7. per window table
    P("=== 7. PER-WINDOW DETAIL (L=live, B=backtest; first fill: side@price t_rem; n=fills) ===")
    for c in ids:
        l, b = LF.get(c), BF.get(c)
        if not l and not b: continue
        def s(F):
            if not F: return f"{'-':28}"
            f = F[0]; return f"{f[1]:4}@{f[2]:.2f} t={mk[c]['close']-f[0]:4.0f} n={len(F)} {pnl(c,F) if mk[c]['out'] else float('nan'):+6.2f}"
        flag = "" if (l and b and l[0][1] == b[0][1]) else ("  OPP" if l and b else "  L-only" if l else "  B-only")
        P(f"{fmt(mk[c]['open'])} out={str(mk[c]['out']):5} L: {s(l):32} B: {s(b):32}{flag}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", required=True, help="first window open date, UTC, YYYY-MM-DD")
    ap.add_argument("--end", help="exclusive end date, UTC, YYYY-MM-DD (default: all recorded)")
    ap.add_argument("--db", default=str(REPO_ROOT / "var" / "poly15m.db"), help="live DB (server copy)")
    ap.add_argument("--bt-db", default=str(REPO_ROOT / "var" / "recon_bt.db"), help="scratch DB for the replay")
    ap.add_argument("--skip-replay", action="store_true", help="reuse an existing --bt-db")
    ap.add_argument("--out", help="also write the report to this file")
    a = ap.parse_args()
    start = _utc(a.start)
    end = _utc(a.end) if a.end else float("inf")
    live_db, bt_db = Path(a.db), Path(a.bt_db)
    if not a.skip_replay:
        replay(live_db, bt_db, start, end)
    lines: list[str] = []

    def P(*args):
        line = " ".join(str(x) for x in args)
        print(line)
        lines.append(line)

    reconcile(live_db, bt_db, start, end, P)
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
