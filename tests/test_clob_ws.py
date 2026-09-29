import asyncio
import json
from pathlib import Path

import websockets

from poly15m.config import Settings
from poly15m.data import clob_ws
from poly15m.data.clob_ws import ClobFeed
from poly15m.db import Database


def _book(asset_id: str, bid: str, ask: str) -> dict:
    return {
        "event_type": "book",
        "market": "0xcond",
        "asset_id": asset_id,
        "timestamp": "1790000000000",
        "bids": [{"price": bid, "size": "10"}],
        "asks": [{"price": ask, "size": "10"}],
    }


async def _wait_for(pred, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not pred():
        assert asyncio.get_running_loop().time() < deadline, "condition not met in time"
        await asyncio.sleep(0.01)


async def test_applies_books_and_resubscribes_on_market_change(tmp_path):
    subscriptions: list[list[str]] = []

    async def fake_clob(ws):
        sub = json.loads(await ws.recv())
        subscriptions.append(sub["assets_ids"])
        await ws.send(json.dumps([_book(a, "0.40", "0.42") for a in sub["assets_ids"]]))
        await ws.wait_closed()

    async with websockets.serve(fake_clob, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        feed = ClobFeed(Settings(clob_ws_base=f"ws://127.0.0.1:{port}"), Database(tmp_path / "t.db"))
        feed.subscribe("0xcond", ["up1", "down1"])
        task = asyncio.create_task(feed.run())
        try:
            await _wait_for(lambda: feed.books["up1"].best_bid() == 0.40)
            feed.subscribe("0xcond2", ["up2", "down2"])
            await _wait_for(lambda: "up2" in feed.books and feed.books["up2"].best_ask() == 0.42)
        finally:
            task.cancel()

    assert subscriptions == [["down1", "up1"], ["down2", "up2"]]


async def test_backoff_resets_after_a_long_lived_connection(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(clob_ws, "_BACKOFF_RESET_UPTIME_S", 0.2)
    connections = 0

    async def fake_clob(ws):
        nonlocal connections
        connections += 1
        await ws.recv()
        # two quick drops grow the backoff, then one after a long uptime resets it
        if connections == 3:
            await asyncio.sleep(0.3)
        await ws.close(code=1011)

    def retries() -> list[float]:
        return [r.retry_in for r in caplog.records if r.getMessage() == "clob_ws_disconnected"]

    async with websockets.serve(fake_clob, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        feed = ClobFeed(Settings(clob_ws_base=f"ws://127.0.0.1:{port}"), Database(tmp_path / "t.db"))
        feed.subscribe("0xcond", ["up1", "down1"])
        task = asyncio.create_task(feed.run())
        try:
            await _wait_for(lambda: len(retries()) >= 3, timeout=6.0)
        finally:
            task.cancel()

    assert retries()[:3] == [1.0, 2.0, 1.0]


FIXTURE = Path(__file__).parent / "fixtures" / "clob_market_frames.json"


def _price_change(asset_id: str, price: str, size: str, side: str, best_bid: str, best_ask: str, ts_ms: int) -> str:
    return json.dumps({
        "event_type": "price_change",
        "market": "0xcond",
        "timestamp": str(ts_ms),
        "price_changes": [
            {"asset_id": asset_id, "price": price, "size": size, "side": side,
             "best_bid": best_bid, "best_ask": best_ask},
        ],
    })


def test_live_frames_keep_book_in_step_with_exchange_best_prices(tmp_path):
    """Every price_changes entry carries the exchange's best bid/ask after it;
    replaying real frames, the book must agree after every message. This
    fixture includes a trade that consumes an ask level without a
    price_change removing it."""
    cap = json.loads(FIXTURE.read_text())
    feed = ClobFeed(Settings(), Database(tmp_path / "t.db"))
    feed.subscribe(cap["condition_id"], cap["assets"])
    checked = 0
    for raw in cap["frames"]:
        feed._handle_message(raw)
        payload = json.loads(raw)
        for item in payload if isinstance(payload, list) else [payload]:
            if item.get("event_type") != "price_change":
                continue
            for change in item["price_changes"]:
                book = feed.books[change["asset_id"]]
                assert (book.best_bid(), book.best_ask()) == (float(change["best_bid"]), float(change["best_ask"]))
                checked += 1
    assert checked > 500


def test_price_change_snapshots_are_throttled_but_keep_the_last_state(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(clob_ws.time, "time", lambda: clock[0])
    db = Database(tmp_path / "t.db")
    feed = ClobFeed(Settings(clob_book_persist_interval_seconds=0.5), db)
    feed.subscribe("0xcond", ["up1", "down1"])
    feed._handle_message(json.dumps([_book("up1", "0.40", "0.42")]))

    # a burst inside one interval: only the book snapshot is written so far
    for i, bid in enumerate(("0.405", "0.41", "0.415")):
        clock[0] += 0.1
        feed._handle_message(_price_change("up1", bid, "5", "BUY", bid, "0.42", 1_790_000_000_100 + i))
    db.flush()
    rows = db._conn.execute("select best_bid, event_ts from book_snapshots where token_id='up1' order by id").fetchall()
    assert rows == [(0.40, 1_790_000_000.0)]

    # the next frame after the interval flushes the burst's final state, stamped with its own time
    clock[0] += 0.5
    feed._handle_message(json.dumps({"event_type": "last_trade_price", "market": "0xcond", "asset_id": "down1",
                                     "price": "0.6", "size": "1", "side": "BUY", "timestamp": "1790000000900"}))
    db.flush()
    rows = db._conn.execute("select best_bid, event_ts from book_snapshots where token_id='up1' order by id").fetchall()
    assert rows == [(0.40, 1_790_000_000.0), (0.415, 1_790_000_000.102)]
