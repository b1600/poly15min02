#!/usr/bin/env python3
"""Stage 1 shakeout check: is the rebuilt pipeline sound enough to start
the Stage 2 measurement run?

This deliberately does NOT look at PnL. A ~3-day shakeout cannot detect an
edge -- at the observed per-window sd (~$5.27) and rate (~63 traded
windows/day) the smallest edge distinguishable at t=2 after 3 days is
roughly 16% ROI, so any PnL number here is noise. Reading it as a verdict
is exactly the mistake that made the Sep 12-22 2026 run look profitable.

What it checks instead is that the two things that invalidated that run
are actually fixed, and that the data being accumulated is scorable:

  1. every window's open price is anchored at the window open, not at
     market-discovery time
  2. every graded window was graded against Polymarket's own settlement,
     not the Binance proxy
  3. settlements are actually arriving (nothing stuck or abandoned)
  4. the proxy still disagrees with settlement at a believable rate.
     With anchoring verified (check 1) the first shakeout still saw
     ~6% (19/321, Sep 23-26 2026): Binance vs the settlement feed
     transiently diverges by tens of dollars (up to ~$86 seen), which
     flips small- and mid-move windows. So ~6% is the baseline, not a
     regression; anchoring regressions are caught directly by check 1.
     Only a rate well above baseline, or a flat 0% over many windows
     (grading not independent of the proxy), is worth a look
  5. the bot is still trading at a useful rate, so Stage 2's n accrues

Scope defaults to the first window recorded with the fixed anchoring, so
it measures the post-fix run only.

Usage: .venv/bin/python scripts/check_shakeout.py [--db PATH] [--since TS]
Exit code 0 = clear to start Stage 2, 1 = problems to fix first.
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timezone

ANCHORED_SOURCE = "binance_at_open"
OFFICIAL_SOURCE = "polymarket_official"

MIN_SHAKEOUT_DAYS = 3.0
STAGE2_N = 1900
# Post-fix baseline is ~6% (Binance-vs-settlement-feed basis, see check 4
# in the docstring); the 0.10 ceiling leaves room for sampling noise at
# n~300 while still flagging a real jump.
DIVERGENCE_MAX = 0.10
# Zero disagreements is only suspicious once there are enough windows
# that the ~6% basis would certainly have shown up.
ZERO_DIVERGENCE_MIN_N = 200

failures: list[str] = []
warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def warn(msg: str) -> None:
    warnings.append(msg)
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  [FAIL] {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="var/poly15m.db")
    ap.add_argument(
        "--since",
        type=float,
        default=None,
        help="unix ts to scope from (default: first window with fixed anchoring)",
    )
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    columns = {r[1] for r in cur.execute("PRAGMA table_info(markets)")}
    missing = {"open_price_ts", "resolved_source", "proxy_outcome"} - columns
    if missing:
        print(f"Database predates the provenance columns ({', '.join(sorted(missing))}).")
        print("Start poly15m-paper-trade once to migrate, then re-run this check.")
        return 1

    since = args.since
    if since is None:
        row = cur.execute(
            "SELECT MIN(window_open_ts) FROM markets WHERE open_price_source = ?", (ANCHORED_SOURCE,)
        ).fetchone()
        since = row[0] if row and row[0] is not None else None

    if since is None:
        print("No windows recorded with the fixed open-price anchoring yet.")
        print("Start poly15m-paper-trade and let it run ~3 days, then re-run this check.")
        return 1

    now = cur.execute("SELECT MAX(window_close_ts) FROM markets").fetchone()[0] or since
    elapsed_days = (now - since) / 86400.0
    print(
        f"Stage 1 shakeout -- scope: {datetime.fromtimestamp(since, timezone.utc).isoformat()} "
        f"onward ({elapsed_days:.2f} days)\n"
    )

    # -- 1. open-price anchoring ------------------------------------
    print("1. Open-price anchoring")
    rows = cur.execute(
        "SELECT open_price_source, COUNT(*) FROM markets WHERE window_open_ts >= ? "
        "AND open_price IS NOT NULL GROUP BY 1",
        (since,),
    ).fetchall()
    total_priced = sum(c for _, c in rows)
    bad = {s: c for s, c in rows if s != ANCHORED_SOURCE}
    if not total_priced:
        fail("no windows with an open price yet")
    elif bad:
        fail(f"{sum(bad.values())} of {total_priced} windows not anchored at open: {bad}")
    else:
        ok(f"all {total_priced} priced windows anchored at window open")

    lag = cur.execute(
        "SELECT MAX(window_open_ts - open_price_ts), AVG(window_open_ts - open_price_ts) "
        "FROM markets WHERE window_open_ts >= ? AND open_price_ts IS NOT NULL",
        (since,),
    ).fetchone()
    if lag and lag[0] is not None:
        if lag[0] > 5.0:
            fail(f"worst anchor lag {lag[0]:.2f}s exceeds the 5s tolerance")
        else:
            ok(f"anchor lag: mean {lag[1]:.3f}s, worst {lag[0]:.3f}s")

    unanchored, total = cur.execute(
        "SELECT SUM(CASE WHEN open_price IS NULL THEN 1 ELSE 0 END), COUNT(*) "
        "FROM markets WHERE window_open_ts >= ?",
        (since,),
    ).fetchone()
    rate = (unanchored or 0) / total if total else 0
    msg = f"{unanchored or 0} of {total} windows unanchored and therefore untraded ({rate:.1%})"
    if rate > 0.10:
        fail(msg + " -- check Binance feed continuity")
    elif rate > 0.02:
        warn(msg)
    else:
        ok(msg)

    # -- 2. settlement provenance -----------------------------------
    print("\n2. Settlement provenance")
    rows = cur.execute(
        "SELECT resolved_source, COUNT(*) FROM markets WHERE window_open_ts >= ? "
        "AND resolved_outcome IS NOT NULL GROUP BY 1",
        (since,),
    ).fetchall()
    total_graded = sum(c for _, c in rows)
    bad = {s: c for s, c in rows if s != OFFICIAL_SOURCE}
    if not total_graded:
        fail("no windows graded yet")
    elif bad:
        fail(f"{sum(bad.values())} of {total_graded} windows not graded on official settlement: {bad}")
    else:
        ok(f"all {total_graded} graded windows used official settlement")

    # -- 3. settlement delivery -------------------------------------
    print("\n3. Settlement delivery")
    # Only windows closed long enough ago that settlement should have landed.
    stuck, closed = cur.execute(
        "SELECT SUM(CASE WHEN resolved_outcome IS NULL THEN 1 ELSE 0 END), COUNT(*) "
        "FROM markets WHERE window_open_ts >= ? AND window_close_ts <= ?",
        (since, now - 3600.0),
    ).fetchone()
    stuck, closed = stuck or 0, closed or 0
    rate = stuck / closed if closed else 0
    msg = f"{stuck} of {closed} long-closed windows never settled ({rate:.1%})"
    if rate > 0.01:
        fail(msg + " -- settlement lookup may be failing; check resolution_timeout logs")
    elif stuck:
        warn(msg)
    else:
        ok(f"every one of {closed} long-closed windows settled")

    # -- 4. proxy divergence ----------------------------------------
    print("\n4. Proxy vs official divergence")
    diverged, comparable = cur.execute(
        "SELECT SUM(CASE WHEN proxy_outcome != resolved_outcome THEN 1 ELSE 0 END), COUNT(*) "
        "FROM markets WHERE window_open_ts >= ? AND resolved_outcome IS NOT NULL "
        "AND proxy_outcome IS NOT NULL",
        (since,),
    ).fetchone()
    diverged, comparable = diverged or 0, comparable or 0
    if comparable < 50:
        warn(f"only {comparable} comparable windows -- too few to judge the divergence rate yet")
    else:
        rate = diverged / comparable
        msg = (
            f"proxy disagreed with settlement on {diverged}/{comparable} windows ({rate:.1%}); "
            f"~6% baseline post-fix"
        )
        if diverged == 0 and comparable >= ZERO_DIVERGENCE_MIN_N:
            warn(msg + " -- zero over this many windows; confirm grading is genuinely independent of the proxy")
        elif rate > DIVERGENCE_MAX:
            warn(msg + " -- well above baseline; check the Binance feed and the proxy close price")
        else:
            ok(msg)

    # -- 5. trading rate --------------------------------------------
    print("\n5. Trading rate (Stage 2 sample accrual)")
    traded = cur.execute(
        "SELECT COUNT(DISTINCT m.condition_id) FROM markets m "
        "WHERE m.window_open_ts >= ? AND EXISTS (SELECT 1 FROM fills f WHERE f.condition_id = m.condition_id)",
        (since,),
    ).fetchone()[0]
    per_day = traded / elapsed_days if elapsed_days > 0 else 0
    if per_day < 20:
        fail(f"only {per_day:.1f} traded windows/day ({traded} total) -- Stage 2 would take far too long")
    elif per_day < 45:
        warn(f"{per_day:.1f} traded windows/day ({traded} total), below the ~63/day baseline")
    else:
        ok(f"{per_day:.1f} traded windows/day ({traded} total)")
    if per_day > 0:
        remaining = max(STAGE2_N - traded, 0)
        print(
            f"         -> Stage 2 (n={STAGE2_N:,}, ~5% ROI floor) needs ~{remaining / per_day:.0f} "
            f"more days at this rate"
        )

    # -- verdict ----------------------------------------------------
    print("\n" + "=" * 62)
    if elapsed_days < MIN_SHAKEOUT_DAYS:
        print(f"Shakeout is {elapsed_days:.2f} days in; let it reach {MIN_SHAKEOUT_DAYS:.0f} before judging.")
    if failures:
        print(f"NOT READY for Stage 2 -- {len(failures)} blocking issue(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    if warnings:
        print(f"Clear to start Stage 2, with {len(warnings)} thing(s) worth a look:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("All checks passed.")
    if elapsed_days >= MIN_SHAKEOUT_DAYS:
        print("\nStage 2: pre-register the decision at n=1,900 traded windows and do not")
        print("peek at the t-stat before then.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
