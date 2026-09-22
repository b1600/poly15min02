#!/usr/bin/env python3
"""Pre-registered test check from 20260911_todo.txt step 4.

Target: 660 resolved markets with fills, t-stat on per-market PnL.
    t >= 2      -> edge is real, start the live conversation
    t < 1       -> it isn't, stop
    1 <= t < 2  -> no decision, keep running

Per-market PnL is computed directly from fills + markets.resolved_outcome
(payout - cost - fee per fill, summed per condition_id). This is immune to
the paper_pnl_log.realized_pnl cumulative-counter restart bug: that column
lives in memory (PaperExecutor.realized_pnl) and resets to 0 on process
restart, corrupting the delta for whichever market resolves right after a
restart. As of 2026-09-21 the dry-run process has restarted unexplained
(no matching deploy commit) 6 times since the 2026-09-12 clean-start point,
so diffing the cumulative column is no longer reliable -- compute from
ground truth instead.

ONLY officially-settled windows count toward the pre-registered n.

`resolved_outcome` used to be graded by a Binance close-vs-open proxy
anchored to an open price sampled at market-discovery time. That anchor
ran ~7.5s late (mean absolute error ~$10 vs a ~$99 mean absolute 15m
move), which mislabelled the ~7% of windows moving less than the error --
and because the proxy shares `open_price` with the fair-value model that
picks the trades, the mislabels landed in the strategy's favour rather
than at random. On the Sep 12-22 2026 sample that turned a true +0.55%
ROI into a reported +23.6%, and this very check into a t-stat of +4.93
against a true +0.13. A proxy-graded sample cannot answer the question
this script exists to answer, so rows without an official settlement are
reported and excluded rather than scored.

Usage: .venv/bin/python scripts/check_pretest.py [--restart-ts TS] [--db PATH]
"""
import argparse
import math
import sqlite3
import sys
from datetime import datetime, timezone

# Config deploy commit (38982ba, "fix strategy after continuous kill switch"),
# 2026-09-11T22:13:27+07:00 -> UTC epoch. This is when the clean 7-day/n=660
# test window starts.
DEFAULT_RESTART_TS = 1789139607.0
TARGET_N = 660
OFFICIAL_SOURCE = "polymarket_official"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="var/poly15m.db")
    ap.add_argument("--restart-ts", type=float, default=DEFAULT_RESTART_TS)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    has_source = any(r[1] == "resolved_source" for r in cur.execute("PRAGMA table_info(markets)"))
    source_col = "resolved_source" if has_source else "NULL"
    cur.execute(
        f"""
        SELECT condition_id, resolved_outcome, resolved_ts, {source_col}
        FROM markets
        WHERE resolved_ts >= ? AND resolved_outcome IS NOT NULL
          AND EXISTS (SELECT 1 FROM fills f WHERE f.condition_id = markets.condition_id)
        ORDER BY resolved_ts ASC
        """,
        (args.restart_ts,),
    )
    all_markets = cur.fetchall()
    if not all_markets:
        print("No resolved markets with fills since restart_ts; nothing to test yet.")
        return 0

    markets = [m for m in all_markets if m[3] == OFFICIAL_SOURCE]
    ungraded = [m for m in all_markets if m[3] != OFFICIAL_SOURCE]

    if ungraded:
        print(
            f"EXCLUDED: {len(ungraded)} of {len(all_markets)} resolved markets were not graded "
            f"against official settlement."
        )
        by_source: dict[str, int] = {}
        for m in ungraded:
            by_source[m[3] or "(none)"] = by_source.get(m[3] or "(none)", 0) + 1
        for source, count in sorted(by_source.items(), key=lambda kv: -kv[1]):
            print(f"  resolved_source={source}: {count}")
        print(
            "  These were graded by the Binance close-vs-open proxy, whose errors correlate\n"
            "  with the trades it grades (see this file's docstring). They cannot be scored."
        )
        print()

    if not markets:
        print(f"0 / {TARGET_N} officially-settled markets with fills -- pre-registered decision CANNOT be made.")
        print("Run poly15m-paper-trade with official-settlement grading to accumulate a scorable sample.")
        return 0

    cur.execute("SELECT condition_id, outcome, price, size, fee FROM fills")
    fills_by_market: dict[str, list[tuple[str, float, float, float]]] = {}
    for cid, outcome, price, size, fee in cur.fetchall():
        fills_by_market.setdefault(cid, []).append((outcome, price, size, fee))

    pnls = []
    for cid, resolved_outcome, resolved_ts, _source in markets:
        pnl = 0.0
        for outcome, price, size, fee in fills_by_market[cid]:
            payout = size if outcome == resolved_outcome else 0.0
            pnl += payout - price * size - fee
        pnls.append(pnl)

    n = len(pnls)
    first_ts = markets[0][2]
    last_ts = markets[-1][2]
    first_dt = datetime.fromtimestamp(first_ts, timezone.utc)
    last_dt = datetime.fromtimestamp(last_ts, timezone.utc)
    elapsed_days = (last_ts - args.restart_ts) / 86400.0

    print(f"officially-settled markets with fills since restart: {n} / {TARGET_N}")
    print(f"window: {first_dt.isoformat()} .. {last_dt.isoformat()} ({elapsed_days:.2f} days)")

    if n < 2:
        print("n < 2, can't compute a t-stat yet.")
        return 0

    mean = sum(pnls) / n
    var = sum((x - mean) ** 2 for x in pnls) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else float("inf")

    print(f"mean/market: {mean:+.4f}   sd: {sd:.4f}   t-stat: {t:+.4f}")
    print(f"worst market: {min(pnls):+.2f}   best market: {max(pnls):+.2f}   sum: {sum(pnls):+.2f}")

    if n < TARGET_N:
        rate = n / elapsed_days if elapsed_days > 0 else 0
        remaining_days = (TARGET_N - n) / rate if rate > 0 else float("inf")
        print(f"\nBelow target n={TARGET_N} -- pre-registered decision not due yet.")
        print(f"pace so far: {rate:.1f} markets/day -> ~{remaining_days:.1f} more days to reach {TARGET_N}")
        print("(This is informational only -- the todo's own rule is to decide at n=660, not before.)")
    else:
        print(f"\nn >= {TARGET_N} -- pre-registered decision is due.")
        if t >= 2:
            verdict = "t >= 2: edge is real -> start the live conversation."
        elif t < 1:
            verdict = "t < 1: edge isn't there -> stop."
        else:
            verdict = "1 <= t < 2: no decision -> keep running."
        print(verdict)

    return 0


if __name__ == "__main__":
    sys.exit(main())
