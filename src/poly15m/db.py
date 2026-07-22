"""SQLite persistence.

Every Binance tick, CLOB book snapshot, CLOB trade print, market lifecycle
event, order, and fill gets written here from day one. This is the
recording layer the Phase 6 backtester replays -- Polymarket does not
expose historical order-book data, so if it isn't captured live it can
never be recovered.

Runs on a single sqlite3 connection used only from the asyncio event loop
thread (never handed to a worker thread), so no locking is needed between
callers. Writes are staged with `execute()` and flushed on a timer by
`run_flush_loop` rather than committed on every insert, which keeps
high-frequency book updates cheap.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
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

CREATE TABLE IF NOT EXISTS price_ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    symbol TEXT NOT NULL,
    price REAL NOT NULL,
    qty REAL,
    is_buyer_maker INTEGER,
    event_ts REAL NOT NULL,
    recv_ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_price_ticks_ts ON price_ticks (source, event_ts);

CREATE TABLE IF NOT EXISTS book_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id TEXT NOT NULL,
    token_id TEXT NOT NULL,
    best_bid REAL,
    best_ask REAL,
    bids_json TEXT NOT NULL,
    asks_json TEXT NOT NULL,
    event_ts REAL,
    recv_ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_book_snapshots_token ON book_snapshots (token_id, recv_ts);

CREATE TABLE IF NOT EXISTS clob_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id TEXT NOT NULL,
    token_id TEXT NOT NULL,
    trade_id TEXT,
    price REAL NOT NULL,
    size REAL NOT NULL,
    side TEXT,
    event_ts REAL,
    recv_ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clob_trades_token ON clob_trades (token_id, recv_ts);

CREATE TABLE IF NOT EXISTS lifecycle_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id TEXT NOT NULL,
    event TEXT NOT NULL,
    ts REAL NOT NULL,
    meta_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_lifecycle_condition ON lifecycle_events (condition_id, ts);

CREATE TABLE IF NOT EXISTS fair_value_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id TEXT NOT NULL,
    ts REAL NOT NULL,
    spot REAL,
    open_price REAL,
    t_remaining REAL,
    sigma REAL,
    deviation REAL,
    momentum_1m REAL,
    momentum_3m REAL,
    momentum_5m REAL,
    book_imbalance_up REAL,
    aggressive_flow_up REAL,
    p_up_fair REAL,
    market_mid_up REAL,
    divergence REAL
);
CREATE INDEX IF NOT EXISTS idx_fair_value_log_condition ON fair_value_log (condition_id, ts);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL DEFAULT 'paper',
    client_order_id TEXT,
    exchange_order_id TEXT,
    condition_id TEXT,
    token_id TEXT,
    side TEXT,
    price REAL,
    size REAL,
    status TEXT,
    ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL DEFAULT 'paper',
    order_ref TEXT,
    condition_id TEXT,
    token_id TEXT,
    outcome TEXT,
    price REAL,
    size REAL,
    fee REAL,
    ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_pnl_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    condition_id TEXT,
    event TEXT NOT NULL,
    realized_pnl REAL NOT NULL,
    fees_paid REAL NOT NULL,
    up_position REAL,
    down_position REAL,
    up_cost_basis REAL,
    down_cost_basis REAL
);
CREATE INDEX IF NOT EXISTS idx_paper_pnl_log_ts ON paper_pnl_log (ts);
"""


class Database:
    def __init__(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._dirty = False
        logger.info("db_opened", extra={"path": str(path)})

    # -- lifecycle -------------------------------------------------
    async def run_flush_loop(self, interval: float = 1.0) -> None:
        while True:
            await asyncio.sleep(interval)
            self.flush()

    def flush(self) -> None:
        if self._dirty:
            self._conn.commit()
            self._dirty = False

    def close(self) -> None:
        self.flush()
        self._conn.close()

    # -- writes ------------------------------------------------------
    def upsert_market(self, market: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO markets (
                condition_id, slug, question_id, token_id_up, token_id_down,
                window_open_ts, window_close_ts, discovered_ts, raw_json
            ) VALUES (:condition_id, :slug, :question_id, :token_id_up, :token_id_down,
                      :window_open_ts, :window_close_ts, :discovered_ts, :raw_json)
            ON CONFLICT(condition_id) DO NOTHING
            """,
            market,
        )
        self._dirty = True

    def set_market_open_price(self, condition_id: str, price: float, source: str) -> None:
        self._conn.execute(
            "UPDATE markets SET open_price = ?, open_price_source = ? WHERE condition_id = ?",
            (price, source, condition_id),
        )
        self._dirty = True

    def set_market_resolution(self, condition_id: str, outcome: str, ts: float) -> None:
        self._conn.execute(
            "UPDATE markets SET resolved_outcome = ?, resolved_ts = ? WHERE condition_id = ?",
            (outcome, ts, condition_id),
        )
        self._dirty = True

    def insert_tick(
        self,
        source: str,
        symbol: str,
        price: float,
        qty: float | None,
        event_ts: float,
        is_buyer_maker: bool | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO price_ticks (source, symbol, price, qty, is_buyer_maker, event_ts, recv_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (source, symbol, price, qty, is_buyer_maker, event_ts, time.time()),
        )
        self._dirty = True

    def insert_book_snapshot(
        self,
        condition_id: str,
        token_id: str,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        event_ts: float | None,
    ) -> None:
        best_bid = max((p for p, _ in bids), default=None)
        best_ask = min((p for p, _ in asks), default=None)
        self._conn.execute(
            """INSERT INTO book_snapshots
               (condition_id, token_id, best_bid, best_ask, bids_json, asks_json, event_ts, recv_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                condition_id,
                token_id,
                best_bid,
                best_ask,
                json.dumps(bids),
                json.dumps(asks),
                event_ts,
                time.time(),
            ),
        )
        self._dirty = True

    def insert_clob_trade(
        self,
        condition_id: str,
        token_id: str,
        price: float,
        size: float,
        side: str | None,
        trade_id: str | None,
        event_ts: float | None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO clob_trades (condition_id, token_id, trade_id, price, size, side, event_ts, recv_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (condition_id, token_id, trade_id, price, size, side, event_ts, time.time()),
        )
        self._dirty = True

    def insert_fair_value_log(
        self,
        condition_id: str,
        ts: float,
        spot: float | None,
        open_price: float | None,
        t_remaining: float | None,
        sigma: float | None,
        deviation: float | None,
        momentum_1m: float | None,
        momentum_3m: float | None,
        momentum_5m: float | None,
        book_imbalance_up: float | None,
        aggressive_flow_up: float | None,
        p_up_fair: float | None,
        market_mid_up: float | None,
        divergence: float | None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO fair_value_log (
                   condition_id, ts, spot, open_price, t_remaining, sigma, deviation,
                   momentum_1m, momentum_3m, momentum_5m, book_imbalance_up, aggressive_flow_up,
                   p_up_fair, market_mid_up, divergence
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                condition_id,
                ts,
                spot,
                open_price,
                t_remaining,
                sigma,
                deviation,
                momentum_1m,
                momentum_3m,
                momentum_5m,
                book_imbalance_up,
                aggressive_flow_up,
                p_up_fair,
                market_mid_up,
                divergence,
            ),
        )
        self._dirty = True

    def insert_lifecycle_event(
        self, condition_id: str, event: str, ts: float, meta: dict[str, Any] | None = None
    ) -> None:
        self._conn.execute(
            "INSERT INTO lifecycle_events (condition_id, event, ts, meta_json) VALUES (?, ?, ?, ?)",
            (condition_id, event, ts, json.dumps(meta) if meta else None),
        )
        self._dirty = True

    def insert_order(
        self,
        condition_id: str,
        token_id: str,
        side: str,
        price: float,
        size: float,
        status: str,
        ts: float,
        mode: str = "paper",
        client_order_id: str | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO orders (mode, client_order_id, condition_id, token_id, side, price, size, status, ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (mode, client_order_id, condition_id, token_id, side, price, size, status, ts),
        )
        self._dirty = True

    def insert_fill(
        self,
        condition_id: str,
        token_id: str,
        outcome: str,
        price: float,
        size: float,
        fee: float,
        ts: float,
        mode: str = "paper",
        order_ref: str | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO fills (mode, order_ref, condition_id, token_id, outcome, price, size, fee, ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (mode, order_ref, condition_id, token_id, outcome, price, size, fee, ts),
        )
        self._dirty = True

    def insert_paper_pnl_log(
        self,
        ts: float,
        condition_id: str | None,
        event: str,
        realized_pnl: float,
        fees_paid: float,
        up_position: float | None,
        down_position: float | None,
        up_cost_basis: float | None,
        down_cost_basis: float | None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO paper_pnl_log (
                   ts, condition_id, event, realized_pnl, fees_paid,
                   up_position, down_position, up_cost_basis, down_cost_basis
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, condition_id, event, realized_pnl, fees_paid, up_position, down_position, up_cost_basis, down_cost_basis),
        )
        self._dirty = True
