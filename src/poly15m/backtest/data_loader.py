"""Reads the Phase 1 SQLite recording back out for replay.

Every row already carries a real timestamp -- `event_ts` where the source
provided one, `recv_ts` (our own receive time) as a fallback -- so replay
just needs to walk rows in that order. Nothing here interprets the data;
that's `engine.py`'s job (it turns these rows back into the same
WebSocket-shaped messages `BinanceFeed`/`ClobFeed` already know how to
parse, so replay drives the exact live code path instead of a
reimplementation of it).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MarketRow:
    condition_id: str
    slug: str
    token_id_up: str
    token_id_down: str
    open_ts: float
    close_ts: float
    open_price: float | None
    resolved_outcome: str | None


@dataclass
class BinanceTick:
    ts: float
    price: float
    qty: float | None
    is_buyer_maker: bool | None


@dataclass
class BookEvent:
    ts: float
    token_id: str
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]


@dataclass
class TradeEvent:
    ts: float
    token_id: str
    price: float
    size: float
    side: str | None


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def list_backtestable_markets(db_path: str | Path) -> list[str]:
    """condition_ids with an open_price and at least one recorded book
    snapshot -- the minimum needed to run a decision loop over."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT m.condition_id FROM markets m
            WHERE m.open_price IS NOT NULL
              AND EXISTS (SELECT 1 FROM book_snapshots b WHERE b.condition_id = m.condition_id)
            ORDER BY m.window_open_ts
            """
        ).fetchall()
        return [r["condition_id"] for r in rows]
    finally:
        conn.close()


def load_market(db_path: str | Path, condition_id: str) -> MarketRow | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM markets WHERE condition_id = ?", (condition_id,)
        ).fetchone()
        if row is None:
            return None
        return MarketRow(
            condition_id=row["condition_id"],
            slug=row["slug"],
            token_id_up=row["token_id_up"],
            token_id_down=row["token_id_down"],
            open_ts=row["window_open_ts"],
            close_ts=row["window_close_ts"],
            open_price=row["open_price"],
            resolved_outcome=row["resolved_outcome"],
        )
    finally:
        conn.close()


def load_binance_ticks(db_path: str | Path, start_ts: float, end_ts: float) -> list[BinanceTick]:
    """The Binance trade stream is global (not per-market), so this is
    queried once per backtest range and shared across every market in it --
    matching how a continuously-running live bot's `BinanceFeed` buffer
    would actually accumulate, rather than resetting per window."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT price, qty, is_buyer_maker, event_ts FROM price_ticks
            WHERE source = 'binance' AND event_ts BETWEEN ? AND ?
            ORDER BY event_ts
            """,
            (start_ts, end_ts),
        ).fetchall()
        return [
            BinanceTick(ts=r["event_ts"], price=r["price"], qty=r["qty"], is_buyer_maker=bool(r["is_buyer_maker"]))
            for r in rows
        ]
    finally:
        conn.close()


def load_book_events(db_path: str | Path, condition_id: str) -> list[BookEvent]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT token_id, bids_json, asks_json, COALESCE(event_ts, recv_ts) AS ts
            FROM book_snapshots WHERE condition_id = ?
            ORDER BY ts
            """,
            (condition_id,),
        ).fetchall()
        events = []
        for r in rows:
            bids = [(float(p), float(s)) for p, s in json.loads(r["bids_json"])]
            asks = [(float(p), float(s)) for p, s in json.loads(r["asks_json"])]
            events.append(BookEvent(ts=r["ts"], token_id=r["token_id"], bids=bids, asks=asks))
        return events
    finally:
        conn.close()


def load_trade_events(db_path: str | Path, condition_id: str) -> list[TradeEvent]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT token_id, price, size, side, COALESCE(event_ts, recv_ts) AS ts
            FROM clob_trades WHERE condition_id = ?
            ORDER BY ts
            """,
            (condition_id,),
        ).fetchall()
        return [TradeEvent(ts=r["ts"], token_id=r["token_id"], price=r["price"], size=r["size"], side=r["side"]) for r in rows]
    finally:
        conn.close()
