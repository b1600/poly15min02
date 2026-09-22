"""Re-anchoring historical open prices from recorded ticks."""

from poly15m.backfill import BACKFILL_SOURCE, backfill_open_prices
from poly15m.db import Database


def build(db_path, open_ts=1000.0, discovery_price=100040.0):
    db = Database(db_path)
    db.upsert_market(
        {
            "condition_id": "cond1",
            "slug": "slug-1",
            "question_id": None,
            "token_id_up": "up",
            "token_id_down": "down",
            "window_open_ts": open_ts,
            "window_close_ts": open_ts + 900.0,
            "discovered_ts": open_ts + 7.5,
            "raw_json": "{}",
        }
    )
    # the buggy historical value: sampled 7.5s after open
    db.set_market_open_price("cond1", discovery_price, "binance_at_discovery")
    db.insert_tick("binance", "BTCUSDT", 100000.0, 1.0, open_ts - 0.4)
    db.insert_tick("binance", "BTCUSDT", discovery_price, 1.0, open_ts + 7.5)
    db.close()


def read(db_path):
    db = Database(db_path)
    row = db._conn.execute(
        "SELECT open_price, open_price_source, open_price_ts FROM markets WHERE condition_id='cond1'"
    ).fetchone()
    db.close()
    return row


def test_dry_run_reports_without_writing(tmp_path):
    db_path = tmp_path / "f.db"
    build(db_path)

    stats = backfill_open_prices(db_path, max_anchor_lag_seconds=5.0, apply=False)

    assert stats["rewritten"] == 1
    assert stats["mean_abs_error"] == 40.0
    assert read(db_path)[0] == 100040.0  # untouched


def test_apply_rewrites_to_the_tick_at_window_open(tmp_path):
    db_path = tmp_path / "f.db"
    build(db_path)

    backfill_open_prices(db_path, max_anchor_lag_seconds=5.0, apply=True)

    price, source, anchor_ts = read(db_path)
    assert price == 100000.0
    assert source == BACKFILL_SOURCE
    assert anchor_ts == 999.6


def test_skips_windows_with_no_usable_anchor(tmp_path):
    db_path = tmp_path / "f.db"
    db = Database(db_path)
    db.upsert_market(
        {
            "condition_id": "cond1",
            "slug": "slug-1",
            "question_id": None,
            "token_id_up": "up",
            "token_id_down": "down",
            "window_open_ts": 1000.0,
            "window_close_ts": 1900.0,
            "discovered_ts": 1007.5,
            "raw_json": "{}",
        }
    )
    db.set_market_open_price("cond1", 100040.0, "binance_at_discovery")
    db.insert_tick("binance", "BTCUSDT", 99000.0, 1.0, 900.0)  # 100s before open
    db.close()

    stats = backfill_open_prices(db_path, max_anchor_lag_seconds=5.0, apply=True)

    assert stats["stale_anchor"] == 1
    assert stats["rewritten"] == 0
    assert read(db_path)[0] == 100040.0  # left alone rather than guessed


def test_is_idempotent(tmp_path):
    db_path = tmp_path / "f.db"
    build(db_path)

    backfill_open_prices(db_path, max_anchor_lag_seconds=5.0, apply=True)
    second = backfill_open_prices(db_path, max_anchor_lag_seconds=5.0, apply=True)

    assert second["examined"] == 0  # already-backfilled rows are skipped
    assert read(db_path)[0] == 100000.0


def test_migration_adds_columns_to_a_preexisting_database(tmp_path):
    """Databases created before the provenance columns existed must open
    cleanly and keep their data -- `_SCHEMA` alone can't add columns to an
    existing table."""
    import sqlite3

    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE markets (
            condition_id TEXT PRIMARY KEY,
            slug TEXT UNIQUE NOT NULL,
            question_id TEXT,
            token_id_up TEXT NOT NULL,
            token_id_down TEXT NOT NULL,
            window_open_ts REAL NOT NULL,
            window_close_ts REAL NOT NULL,
            open_price REAL,
            open_price_source TEXT,
            resolved_outcome TEXT,
            resolved_ts REAL,
            discovered_ts REAL NOT NULL,
            raw_json TEXT
        );
        INSERT INTO markets VALUES
            ('c1','s1',NULL,'up','down',1000.0,1900.0,100040.0,'binance_at_discovery','up',1900.0,1007.5,'{}');
        """
    )
    conn.commit()
    conn.close()

    db = Database(db_path)  # must migrate, not raise

    columns = {r[1] for r in db._conn.execute("PRAGMA table_info(markets)")}
    assert {"open_price_ts", "resolved_source", "proxy_outcome"} <= columns

    row = db._conn.execute(
        "SELECT open_price, resolved_outcome, open_price_ts, resolved_source FROM markets WHERE condition_id='c1'"
    ).fetchone()
    assert row[0] == 100040.0 and row[1] == "up"  # existing data preserved
    assert row[2] is None and row[3] is None  # new columns read back NULL

    db.close()
    Database(db_path).close()  # re-opening is a no-op, not a duplicate-column error
