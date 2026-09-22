"""Re-anchor historical `open_price` values from recorded Binance ticks.

Windows recorded before the open-price fix carry
`open_price_source = 'binance_at_discovery'`: a sample taken at
market-discovery time rather than at `window_open_ts`. With discovery on
a 15s poll that landed ~7.5s late on average (mean absolute error ~$10
against BTC), which is small next to the ~$99 mean absolute 15m move but
decisive for the ~7% of windows that travel less than the error.

Every Binance trade was recorded from day one, so the correct anchor is
still recoverable offline: the last tick at or before `window_open_ts`.
This rewrites `open_price` in place for those windows and tags them
`binance_at_open_backfill` so re-anchored rows stay distinguishable from
live-anchored ones.

What this does NOT fix: `resolved_outcome` on historical rows was graded
by the Binance proxy, which shares `open_price` with the fair-value model
and so cannot be trusted. Re-anchoring makes the *inputs* replayable;
grading those windows still needs official settlement from Gamma. Rows
whose outcome came from the proxy are reported here but left alone.

Dry-run by default; pass --apply to write.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

from .config import settings

logger = logging.getLogger(__name__)

BACKFILL_SOURCE = "binance_at_open_backfill"


def backfill_open_prices(
    db_path: Path | str, max_anchor_lag_seconds: float, apply: bool = False, limit: int | None = None
) -> dict[str, float | int]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    query = """
        SELECT condition_id, window_open_ts, open_price, open_price_source
        FROM markets
        WHERE window_open_ts IS NOT NULL
          AND (open_price_source IS NULL OR open_price_source != ?)
        ORDER BY window_open_ts
    """
    params: list = [BACKFILL_SOURCE]
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(query, params).fetchall()
    stats = {
        "examined": len(rows),
        "rewritten": 0,
        "unchanged": 0,
        "no_anchor": 0,
        "stale_anchor": 0,
        "abs_error_sum": 0.0,
        "max_abs_error": 0.0,
        "flipped_sign": 0,
    }

    for row in rows:
        anchor = conn.execute(
            "SELECT event_ts, price FROM price_ticks WHERE source = 'binance' AND event_ts <= ? "
            "ORDER BY event_ts DESC LIMIT 1",
            (row["window_open_ts"],),
        ).fetchone()

        if anchor is None:
            stats["no_anchor"] += 1
            continue
        if row["window_open_ts"] - anchor["event_ts"] > max_anchor_lag_seconds:
            stats["stale_anchor"] += 1
            continue

        corrected = anchor["price"]
        if row["open_price"] is not None:
            error = abs(corrected - row["open_price"])
            stats["abs_error_sum"] += error
            stats["max_abs_error"] = max(stats["max_abs_error"], error)
            if error < 1e-9:
                stats["unchanged"] += 1

        if apply:
            conn.execute(
                "UPDATE markets SET open_price = ?, open_price_source = ?, open_price_ts = ? "
                "WHERE condition_id = ?",
                (corrected, BACKFILL_SOURCE, anchor["event_ts"], row["condition_id"]),
            )
        stats["rewritten"] += 1

    if apply:
        conn.commit()
    conn.close()

    graded = stats["rewritten"] - stats["unchanged"]
    stats["mean_abs_error"] = round(stats["abs_error_sum"] / graded, 4) if graded else 0.0
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(settings.db_path), help="path to poly15m.db")
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    parser.add_argument("--limit", type=int, default=None, help="only examine the first N windows")
    parser.add_argument(
        "--max-anchor-lag",
        type=float,
        default=settings.open_price_max_anchor_lag_seconds,
        help="max seconds the anchoring tick may predate window open",
    )
    args = parser.parse_args()

    stats = backfill_open_prices(args.db, args.max_anchor_lag, apply=args.apply, limit=args.limit)

    mode = "APPLIED" if args.apply else "DRY RUN (nothing written; pass --apply to write)"
    print(f"open_price backfill -- {mode}")
    print(f"  windows examined     : {stats['examined']}")
    print(f"  re-anchored          : {stats['rewritten']}")
    print(f"    already correct    : {stats['unchanged']}")
    print(f"  no tick at/before open: {stats['no_anchor']}")
    print(f"  anchor too stale     : {stats['stale_anchor']}")
    print(f"  mean abs correction  : ${stats['mean_abs_error']}")
    print(f"  max abs correction   : ${round(stats['max_abs_error'], 2)}")
    print()
    print("NOTE: historical `resolved_outcome` was graded by the Binance proxy and is")
    print("      NOT corrected here. Re-anchoring fixes strategy inputs for replay;")
    print("      trustworthy PnL still needs official settlement.")


if __name__ == "__main__":
    main()
