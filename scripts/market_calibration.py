#!/usr/bin/env python3
"""Is the Polymarket BTC 15m market miscalibrated, and does the fair-value
model add anything beyond the market's own price? (Point 2 of the step 5
follow-ups, 20260928/20260928_step5_fixes.txt.)

build    samples every window at fixed times-to-close and writes
         var/calib/dataset.csv: model state from a backtest replay's
         fair_value_log (var/step5/base.db, correct open prices) and best
         bid/ask per token from the recorded book_snapshots.
analyze  runs the analysis. Everything is fitted on train (windows opening
         Aug 3 - Sep 22) and only evaluated on test (Sep 23 onward); see
         20260928/20260929_market_calibration.txt for the plan written
         before test was looked at.

    .venv/bin/python scripts/market_calibration.py build
    .venv/bin/python scripts/market_calibration.py analyze
"""

import argparse
import bisect
import csv
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_DB = REPO_ROOT / "var" / "poly15m.db"
REPLAY_DB = REPO_ROOT / "var" / "step5" / "base.db"
OUT = REPO_ROOT / "var" / "calib" / "dataset.csv"

TRAIN_START = datetime(2026, 8, 3, tzinfo=timezone.utc).timestamp()
TEST_START = datetime(2026, 9, 23, tzinfo=timezone.utc).timestamp()
CHECKPOINTS = [840, 720, 600, 480, 360, 240, 180, 120, 60, 30]  # seconds left
MAX_FV_GAP = 5.0  # a checkpoint needs a fair_value_log row within this many seconds
MAX_BOOK_AGE = 30.0  # and a book snapshot for each token no older than this

FIELDS = [
    "cid", "open_ts", "period", "checkpoint", "t_rem", "spot", "open", "sigma", "dev", "p_model",
    "up_bid", "up_ask", "down_bid", "down_ask", "book_age", "up_won",
]


def build() -> None:
    live = sqlite3.connect(str(LIVE_DB))
    rep = sqlite3.connect(str(REPLAY_DB))
    markets = live.execute(
        """select condition_id, window_open_ts, window_close_ts, open_price, resolved_outcome,
                  token_id_up, token_id_down
           from markets where window_open_ts >= ? and open_price is not null
             and resolved_outcome in ('up', 'down') order by window_open_ts""",
        (TRAIN_START,),
    ).fetchall()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_rows = n_skip = 0
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for i, (cid, open_ts, close_ts, open_px, outcome, tok_up, tok_dn) in enumerate(markets):
            fv = rep.execute(
                "select ts, spot, sigma, deviation, p_up_fair from fair_value_log where condition_id=? order by ts",
                (cid,),
            ).fetchall()
            if not fv:
                continue
            fv_ts = [r[0] for r in fv]
            books: dict[str, tuple[list[float], list[tuple]]] = {}
            for tok in (tok_up, tok_dn):
                rows = live.execute(
                    """select coalesce(event_ts, recv_ts) t, best_bid, best_ask from book_snapshots
                       where condition_id=? and token_id=? order by t""",
                    (cid, tok),
                ).fetchall()
                books[tok] = ([r[0] for r in rows], rows)
            for cp in CHECKPOINTS:
                target = close_ts - cp
                j = bisect.bisect_right(fv_ts, target) - 1
                if j < 0 or target - fv_ts[j] > MAX_FV_GAP:
                    n_skip += 1
                    continue
                ts, spot, sigma, dev, p_model = fv[j]
                quotes, ages = [], []
                for tok in (tok_up, tok_dn):
                    bts, rows = books[tok]
                    k = bisect.bisect_right(bts, ts) - 1
                    if k < 0:
                        break
                    quotes.append((rows[k][1], rows[k][2]))
                    ages.append(ts - bts[k])
                if len(quotes) < 2 or max(ages) > MAX_BOOK_AGE:
                    n_skip += 1
                    continue
                (ub, ua), (db_, da) = quotes
                w.writerow([
                    cid, open_ts, "test" if open_ts >= TEST_START else "train", cp, round(close_ts - ts, 3),
                    spot, open_px, sigma, dev, p_model, ub, ua, db_, da, round(max(ages), 3),
                    1 if outcome == "up" else 0,
                ])
                n_rows += 1
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(markets)} windows, {n_rows} rows", file=sys.stderr, flush=True)
    print(f"wrote {n_rows} rows ({n_skip} checkpoints skipped: no model row or stale book) -> {OUT}")


# ---------------------------------------------------------------- analyze
GROUPS = {"early": (600, 840), "mid": (240, 480), "late": (30, 180)}
MARGINS = [0.02, 0.04, 0.06, 0.08]
TRADE_MAX_BOOK_AGE = 5.0
FEE_RATE = 0.07
N_BOOT = 200


def _group(cp: int) -> str:
    return next(g for g, (lo, hi) in GROUPS.items() if lo <= cp <= hi)


def _logit(p):
    import numpy as np

    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def _fit(X, y, ridge: float = 1e-6):
    """Logistic regression by Newton's method; X gets an intercept column."""
    import numpy as np

    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + ridge * w
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w


def _predict(w, X):
    import numpy as np

    X = np.column_stack([np.ones(len(X)), X])
    return 1 / (1 + np.exp(-X @ w))


def _ll(p, y):
    import numpy as np

    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _load():
    import numpy as np

    rows = []
    with OUT.open() as fh:
        for r in csv.DictReader(fh):
            if not all(r[k] for k in ("up_bid", "up_ask", "down_bid", "down_ask", "p_model")):
                continue
            rows.append(r)
    col = lambda k, f=float: np.array([f(r[k]) for r in rows])  # noqa: E731
    d = {k: col(k) for k in ("t_rem", "sigma", "p_model", "up_bid", "up_ask", "down_bid", "down_ask", "book_age")}
    d["cp"] = col("checkpoint", int)
    d["y"] = col("up_won")
    d["cid"] = np.array([r["cid"] for r in rows])
    d["test"] = np.array([r["period"] == "test" for r in rows])
    d["group"] = np.array([_group(c) for c in d["cp"]])
    d["mid"] = (d["up_bid"] + d["up_ask"]) / 2
    return d


def _features(d, idx, with_model: bool):
    import numpy as np

    cols = [_logit(d["mid"][idx])]
    if with_model:
        cols.append(_logit(d["p_model"][idx]))
    return np.column_stack(cols)


def _boot_windows(cids, rng):
    """Row indices of a window-level bootstrap resample."""
    import numpy as np

    uniq = np.unique(cids)
    by = {c: np.flatnonzero(cids == c) for c in uniq}
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([by[c] for c in pick])


def _ci(x):
    import numpy as np

    return np.percentile(x, 2.5), np.percentile(x, 97.5)


def analyze() -> None:
    import numpy as np

    rng = np.random.default_rng(20260929)
    d = _load()
    out: list[str] = []
    P = lambda s="": (print(s), out.append(s))  # noqa: E731
    tr, te = ~d["test"], d["test"]
    P(f"rows: train {tr.sum()} ({len(np.unique(d['cid'][tr]))} windows), "
      f"test {te.sum()} ({len(np.unique(d['cid'][te]))} windows)")

    # ---------------- A. market calibration
    P("\n=== A. MARKET CALIBRATION: realized Up rate vs mid_up ===")
    for g in GROUPS:
        for label, mask in (("train", tr), ("test", te)):
            idx = np.flatnonzero(mask & (d["group"] == g))
            X, y = _features(d, idx, False), d["y"][idx]
            a, b = _fit(X, y)
            boots = []
            for _ in range(N_BOOT):
                bi = _boot_windows(d["cid"][idx], rng)
                boots.append(_fit(_features(d, idx[bi], False), y[bi]))
            boots = np.array(boots)
            (alo, ahi), (blo, bhi) = _ci(boots[:, 0]), _ci(boots[:, 1])
            P(f"\n{g:5} {label:5} n={len(idx):5}  fit y~a+b*logit(mid): a={a:+.3f} [{alo:+.3f},{ahi:+.3f}]  "
              f"b={b:.3f} [{blo:.3f},{bhi:.3f}]")
            P(f"      {'mid bin':>9} {'n':>5} {'mean mid':>8} {'up rate':>7} {'diff':>6}")
            for lo in np.arange(0, 1, 0.1):
                m = (d["mid"][idx] >= lo) & (d["mid"][idx] < lo + 0.1 + (1e-9 if lo >= 0.9 else 0))
                if m.sum() < 20:
                    continue
                mm, ur = d["mid"][idx][m].mean(), y[m].mean()
                P(f"      {lo:.1f}-{lo + 0.1:.1f} {m.sum():5} {mm:8.3f} {ur:7.3f} {ur - mm:+6.3f}")

    # ---------------- B. information
    P("\n=== B. DOES THE MODEL ADD INFORMATION? (fit on train, scored on test; log loss, lower is better) ===")
    models = {}
    for g in GROUPS:
        itr = np.flatnonzero(tr & (d["group"] == g))
        ite = np.flatnonzero(te & (d["group"] == g))
        y_tr, y_te = d["y"][itr], d["y"][ite]
        w0 = _fit(_features(d, itr, False), y_tr)
        w1 = _fit(_features(d, itr, True), y_tr)
        # 5-fold window-grouped CV on train to choose between M0 and M1
        uniq = np.unique(d["cid"][itr])
        fold_of = {c: i % 5 for i, c in enumerate(rng.permutation(uniq))}
        folds = np.array([fold_of[c] for c in d["cid"][itr]])
        cv = {0: [], 1: []}
        for k in range(5):
            fit_i, val_i = itr[folds != k], itr[folds == k]
            for m, wm in ((0, False), (1, True)):
                w = _fit(_features(d, fit_i, wm), d["y"][fit_i])
                cv[m].append(_ll(_predict(w, _features(d, val_i, wm)), d["y"][val_i]))
        cv0, cv1 = np.concatenate(cv[0]).mean(), np.concatenate(cv[1]).mean()
        use_model = cv1 < cv0
        models[g] = (w1, True) if use_model else (w0, False)

        ll = {
            "raw mid": _ll(np.clip(d["mid"][ite], 0.01, 0.99), y_te),
            "raw model": _ll(np.clip(d["p_model"][ite], 0.01, 0.99), y_te),
            "M0 market": _ll(_predict(w0, _features(d, ite, False)), y_te),
            "M1 mkt+model": _ll(_predict(w1, _features(d, ite, True)), y_te),
        }
        diff = ll["M1 mkt+model"] - ll["M0 market"]
        cids = d["cid"][ite]
        boots = [diff[_boot_windows(cids, rng)].mean() for _ in range(N_BOOT)]
        lo, hi = _ci(np.array(boots))
        P(f"\n{g:5} train M1 coefs: a={w1[0]:+.3f} b_mid={w1[1]:.3f} c_model={w1[2]:+.3f}   "
          f"train CV log loss M0 {cv0:.4f} M1 {cv1:.4f} -> trade with {'M1' if use_model else 'M0'}")
        P("      test log loss: " + "  ".join(f"{k} {v.mean():.4f}" for k, v in ll.items()))
        verdict = "model adds information" if hi < 0 else "no evidence the model adds information"
        P(f"      M1 - M0 on test: {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")

    # ---------------- C. trading
    P("\n=== C. TRADING RULE (1 share at the ask, first qualifying checkpoint per window) ===")

    def q_for(which: str):
        q = np.empty(len(d["y"]))
        for g in GROUPS:
            i = np.flatnonzero(d["group"] == g)
            if which == "chosen":
                w, wm = models[g]
            else:
                w, wm = _fit(_features(d, np.flatnonzero(tr & (d["group"] == g)), False),
                             d["y"][tr & (d["group"] == g)]), False
            q[i] = _predict(w, _features(d, i, wm))
        return q

    def simulate(q, mask, m):
        fee = lambda p: FEE_RATE * p * (1 - p)  # noqa: E731
        order = np.lexsort((-d["cp"], d["cid"]))  # by window, then earliest checkpoint first
        pnl, taken = {}, set()
        for i in order:
            c = d["cid"][i]
            if not mask[i] or c in taken or d["book_age"][i] > TRADE_MAX_BOOK_AGE:
                continue
            ua, da = d["up_ask"][i], d["down_ask"][i]
            e_up = q[i] - ua - fee(ua)
            e_dn = (1 - q[i]) - da - fee(da)
            if max(e_up, e_dn) <= m:
                continue
            if e_up >= e_dn:
                pnl[c] = d["y"][i] - ua - fee(ua)
            else:
                pnl[c] = (1 - d["y"][i]) - da - fee(da)
            taken.add(c)
        x = np.array(list(pnl.values()))
        t = x.mean() / x.std(ddof=1) * math.sqrt(len(x)) if len(x) > 1 and x.std(ddof=1) > 0 else float("nan")
        return len(x), x.sum(), (x.mean() if len(x) else float("nan")), t

    for which in ("chosen", "M0 only"):
        q = q_for(which)
        P(f"\n[{which}]  {'m':>5} | {'train n':>7} {'PnL':>8} {'/trade':>7} {'t':>6} |")
        best = None
        for m in MARGINS:
            n, s, mu, t = simulate(q, tr, m)
            P(f"{'':12}{m:5.2f} | {n:7} {s:+8.2f} {mu:+7.4f} {t:6.2f} |")
            if best is None or s > best[1]:
                best = (m, s)
        m = best[0]
        n, s, mu, t = simulate(q, te, m)
        if which == "chosen":
            verdict = "PASS" if s > 0 and t >= 2 else "PROMISING" if s > 0 and t >= 1 else "FAIL"
        else:
            verdict = "info only"
        P(f"   chosen m={m:.2f} on train -> TEST n={n} PnL {s:+.2f} per trade {mu:+.4f} t={t:.2f} -> {verdict}")

    report = REPO_ROOT / "20260928" / "20260929_market_calibration.txt"
    with report.open("a") as fh:
        fh.write(f"\n-----------------------RESULTS {datetime.now().strftime('%d %b %Y %H:%M')}\n")
        fh.write("\n".join(out) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["build", "analyze"])
    a = ap.parse_args()
    build() if a.cmd == "build" else analyze()


if __name__ == "__main__":
    main()
