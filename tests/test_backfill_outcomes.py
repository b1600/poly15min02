"""One-shot official-outcome backfill for windows graded by the old
Binance proxy before official-settlement grading existed."""

import asyncio

from poly15m.backfill_outcomes import backfill_outcomes
from poly15m.db import Database


def build(db_path, condition_id="cond1", slug="slug-1", proxy_outcome="up", close_ts=1000.0, source=None):
    db = Database(db_path)
    db.upsert_market(
        {
            "condition_id": condition_id,
            "slug": slug,
            "question_id": None,
            "token_id_up": "up",
            "token_id_down": "down",
            "window_open_ts": close_ts - 900.0,
            "window_close_ts": close_ts,
            "discovered_ts": close_ts - 900.0,
            "raw_json": "{}",
        }
    )
    if proxy_outcome is not None:
        # mimics the pre-fix code path: set_market_resolution called with no
        # source, so resolved_source/proxy_outcome are left NULL by the old
        # writer even though the column now exists.
        db._conn.execute(
            "UPDATE markets SET resolved_outcome = ?, resolved_ts = ?, resolved_source = ? WHERE condition_id = ?",
            (proxy_outcome, close_ts, source, condition_id),
        )
        db._dirty = True
    db.close()


def read(db_path, condition_id="cond1"):
    db = Database(db_path)
    row = db._conn.execute(
        "SELECT resolved_outcome, resolved_source, proxy_outcome FROM markets WHERE condition_id = ?",
        (condition_id,),
    ).fetchone()
    db.close()
    return row


def event(condition_id, prices='["1", "0"]', closed=True):
    return {"conditionId": condition_id, "closed": closed, "outcomes": '["Up", "Down"]', "outcomePrices": prices}


def fake_fetch(mapping):
    async def fetch(slugs):
        return {slug: mapping[slug] for slug in slugs if slug in mapping}

    return fetch


def test_dry_run_reports_without_writing(tmp_path):
    db_path = tmp_path / "f.db"
    build(db_path, proxy_outcome="down")  # old proxy was wrong

    stats = asyncio.run(
        backfill_outcomes(db_path, apply=False, fetch_events=fake_fetch({"slug-1": event("cond1", '["1", "0"]')}))
    )

    assert stats == {
        "examined": 1,
        "settled": 1,
        "diverged_from_proxy": 1,
        "still_open_on_gamma": 0,
        "not_found_on_gamma": 0,
    }
    row = read(db_path)
    assert row[0] == "down"  # untouched
    assert row[1] is None


def test_apply_overwrites_outcome_and_preserves_old_proxy(tmp_path):
    db_path = tmp_path / "f.db"
    build(db_path, proxy_outcome="down")  # the bug: proxy said down

    asyncio.run(
        backfill_outcomes(db_path, apply=True, fetch_events=fake_fetch({"slug-1": event("cond1", '["1", "0"]')}))
    )

    outcome, source, proxy = read(db_path)
    assert outcome == "up"  # official settlement wins
    assert source == "polymarket_official"
    assert proxy == "down"  # old (wrong) proxy preserved for auditability


def test_skips_rows_already_officially_graded(tmp_path):
    db_path = tmp_path / "f.db"
    build(db_path, proxy_outcome="up", source="polymarket_official")

    stats = asyncio.run(backfill_outcomes(db_path, apply=True, fetch_events=fake_fetch({})))

    assert stats["examined"] == 0  # never even queried


def test_leaves_still_open_windows_alone(tmp_path):
    db_path = tmp_path / "f.db"
    build(db_path, proxy_outcome="up")

    stats = asyncio.run(
        backfill_outcomes(
            db_path, apply=True, fetch_events=fake_fetch({"slug-1": event("cond1", '["0.5", "0.5"]', closed=False)})
        )
    )

    assert stats["still_open_on_gamma"] == 1
    assert stats["settled"] == 0
    row = read(db_path)
    assert row[0] == "up" and row[1] is None  # untouched, not guessed


def test_counts_events_gamma_never_returned(tmp_path):
    db_path = tmp_path / "f.db"
    build(db_path, proxy_outcome="up")

    stats = asyncio.run(backfill_outcomes(db_path, apply=True, fetch_events=fake_fetch({})))

    assert stats["not_found_on_gamma"] == 1
    assert stats["settled"] == 0


def test_is_idempotent(tmp_path):
    db_path = tmp_path / "f.db"
    build(db_path, proxy_outcome="down")
    fetch = fake_fetch({"slug-1": event("cond1", '["1", "0"]')})

    asyncio.run(backfill_outcomes(db_path, apply=True, fetch_events=fetch))
    second = asyncio.run(backfill_outcomes(db_path, apply=True, fetch_events=fetch))

    assert second["examined"] == 0
    row = read(db_path)
    assert row[0] == "up" and row[2] == "down"


def test_batches_every_pending_slug_into_one_fetch_call(tmp_path):
    db_path = tmp_path / "f.db"
    build(db_path, condition_id="cond1", slug="slug-1", proxy_outcome="up", close_ts=1000.0)
    build(db_path, condition_id="cond2", slug="slug-2", proxy_outcome="down", close_ts=2000.0)

    calls = []

    async def fetch(slugs):
        calls.append(sorted(slugs))
        return {
            "slug-1": event("cond1", '["1", "0"]'),
            "slug-2": event("cond2", '["0", "1"]'),
        }

    stats = asyncio.run(backfill_outcomes(db_path, apply=True, fetch_events=fetch))

    assert len(calls) == 1  # one fetch covering both pending windows
    assert calls[0] == ["slug-1", "slug-2"]
    assert stats["settled"] == 2
    assert stats["diverged_from_proxy"] == 0


def test_leaves_still_pending_windows_untouched(tmp_path):
    """A window still trading (window_close_ts in the future) must not be
    queried at all -- it hasn't closed yet."""
    db_path = tmp_path / "f.db"
    future = 4102444800.0  # 2100-01-01, safely in the future
    build(db_path, close_ts=future, proxy_outcome=None)

    calls = []

    async def fetch(slugs):
        calls.append(slugs)
        return {}

    stats = asyncio.run(backfill_outcomes(db_path, apply=True, fetch_events=fetch))

    assert stats["examined"] == 0
    assert calls == []
