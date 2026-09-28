import asyncio
import json

import pytest
import websockets

from poly15m.config import Settings
from poly15m.data.settlement_twap_ws import SOURCE, SettlementTwapFeed, _AuthRejected
from poly15m.db import Database

CREDS = dict(polymarket_api_key="k", polymarket_api_secret="s", polymarket_api_passphrase="p")


def _ticks(db: Database) -> list[tuple[float, float]]:
    db.flush()
    return db._conn.execute(
        "SELECT event_ts, price FROM price_ticks WHERE source = ? ORDER BY event_ts", (SOURCE,)
    ).fetchall()


def _update(ts_ms: int, value: str) -> dict:
    return {
        "v": 1,
        "channel": "price.crypto.twap",
        "seq": 2,
        "ts": ts_ms,
        "payload": {"symbol": "btcusd", "value": float(value), "full_accuracy_value": value, "timestamp": ts_ms},
    }


def _snapshot(points: list[tuple[int, str]]) -> dict:
    return {
        "v": 1,
        "channel": "price.crypto.twap",
        "seq": 1,
        "ts": points[-1][0],
        "snapshot": True,
        "payload": {
            "symbol": "btcusd",
            "data": [{"timestamp": t, "value": float(v), "full_accuracy_value": v} for t, v in points],
        },
    }


def test_records_snapshot_then_updates_in_seconds_with_full_precision(tmp_path):
    db = Database(tmp_path / "t.db")
    feed = SettlementTwapFeed(Settings(**CREDS), db)

    feed._handle_item(_snapshot([(1_790_000_000_000, "64123.50000001"), (1_790_000_001_000, "64125.1")]))
    feed._handle_item(_update(1_790_000_002_000, "64126.123456789"))

    assert _ticks(db) == [
        (1_790_000_000.0, 64123.50000001),
        (1_790_000_001.0, 64125.1),
        (1_790_000_002.0, 64126.123456789),
    ]
    assert feed.last_value == 64126.123456789


def test_reconnect_snapshot_does_not_duplicate_points_already_stored(tmp_path):
    db = Database(tmp_path / "t.db")
    feed = SettlementTwapFeed(Settings(**CREDS), db)
    feed._handle_item(_update(1_790_000_000_000, "1"))
    feed._handle_item(_update(1_790_000_001_000, "2"))

    # a reconnect replays the last two minutes, overlapping what we have
    feed._handle_item(_snapshot([(1_790_000_000_000, "1"), (1_790_000_001_000, "2"), (1_790_000_002_000, "3")]))

    assert [t for t, _ in _ticks(db)] == [1_790_000_000.0, 1_790_000_001.0, 1_790_000_002.0]


def test_dedup_survives_a_process_restart(tmp_path):
    path = tmp_path / "t.db"
    db = Database(path)
    SettlementTwapFeed(Settings(**CREDS), db)._handle_item(_update(1_790_000_005_000, "5"))
    db.close()

    db = Database(path)
    feed = SettlementTwapFeed(Settings(**CREDS), db)
    feed._handle_item(_snapshot([(1_790_000_004_000, "4"), (1_790_000_005_000, "5"), (1_790_000_006_000, "6")]))

    assert [t for t, _ in _ticks(db)] == [1_790_000_005.0, 1_790_000_006.0]


def test_accepts_sdk_style_envelope(tmp_path):
    db = Database(tmp_path / "t.db")
    feed = SettlementTwapFeed(Settings(**CREDS), db)
    feed._handle_item(
        {
            "topic": "prices.crypto.twap",
            "type": "update",
            "timestamp": 1_790_000_000_000,
            "payload": {"symbol": "btcusd", "timestamp": 1_790_000_000_000, "value": "64000.5", "windowSeconds": 60},
        }
    )
    assert _ticks(db) == [(1_790_000_000.0, 64000.5)]


def test_ignores_other_channels_and_acks(tmp_path):
    db = Database(tmp_path / "t.db")
    feed = SettlementTwapFeed(Settings(**CREDS), db)
    spot = _update(1_790_000_000_000, "1")
    spot["channel"] = "price.crypto"  # Pyth spot, not the settlement TWAP
    feed._handle_item(spot)
    feed._handle_item({"op": "subscribed", "channel": "price.crypto.twap", "rid": "s1"})
    feed._handle_item({"op": "pong"})
    assert _ticks(db) == []


def test_invalid_credentials_raise_so_run_backs_off(tmp_path):
    feed = SettlementTwapFeed(Settings(**CREDS), Database(tmp_path / "t.db"))
    with pytest.raises(_AuthRejected):
        feed._handle_item({"op": "error", "code": "auth_invalid", "rid": "a1"})


async def test_run_is_a_no_op_without_credentials(tmp_path):
    feed = SettlementTwapFeed(Settings(), Database(tmp_path / "t.db"))
    await asyncio.wait_for(feed.run(), timeout=1.0)  # returns instead of connecting


async def test_authenticates_then_subscribes_then_records(tmp_path):
    received: list[dict] = []

    async def fake_polybolt(ws):
        auth = json.loads(await ws.recv())
        received.append(auth)
        await ws.send(json.dumps({"op": "authed", "rid": auth["rid"]}))
        sub = json.loads(await ws.recv())
        received.append(sub)
        await ws.send(json.dumps({"op": "subscribed", "channel": "price.crypto.twap", "rid": sub["rid"]}))
        await ws.send(json.dumps(_snapshot([(1_790_000_000_000, "64000.1")])))
        await ws.send(json.dumps(_update(1_790_000_001_000, "64000.2")))
        await asyncio.sleep(1)

    async with websockets.serve(fake_polybolt, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        db = Database(tmp_path / "t.db")
        settings = Settings(**CREDS, polybolt_ws_base=f"ws://127.0.0.1:{port}", settlement_twap_stale_seconds=0.5)
        feed = SettlementTwapFeed(settings, db)
        # returns once the fake server goes quiet past the stale timeout
        await asyncio.wait_for(feed._connect_and_listen(), timeout=3.0)

    assert received[0]["op"] == "auth"
    assert received[0]["auth"] == {"apiKey": "k", "secret": "s", "passphrase": "p"}
    assert received[1]["subscriptions"] == [
        {"channel": "price.crypto.twap", "filter": {"symbol": "btcusd", "window_seconds": 60}}
    ]
    assert _ticks(db) == [(1_790_000_000.0, 64000.1), (1_790_000_001.0, 64000.2)]
