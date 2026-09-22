"""Official settlement parsing and the poll/timeout state machine."""

import asyncio

import pytest

from poly15m.config import Settings
from poly15m.data.resolution import ResolutionPoller, parse_official_outcome


def market(closed=True, prices='["1", "0"]', outcomes='["Up", "Down"]', **extra):
    return {"conditionId": "cond1", "closed": closed, "outcomes": outcomes, "outcomePrices": prices, **extra}


# -- parser ----------------------------------------------------------

def test_parses_settled_up_and_down():
    assert parse_official_outcome(market(prices='["1", "0"]')) == "up"
    assert parse_official_outcome(market(prices='["0", "1"]')) == "down"


def test_accepts_real_lists_as_well_as_json_strings():
    assert parse_official_outcome(market(prices=["1", "0"], outcomes=["Up", "Down"])) == "up"


def test_open_market_is_not_settled():
    """Live mid prices must never be read as a settlement."""
    assert parse_official_outcome(market(closed=False, prices='["0.505", "0.495"]')) is None


def test_closed_but_indecisive_is_not_settled():
    """`closed` alone is not enough -- a 1/0 payout is the real signal."""
    assert parse_official_outcome(market(prices='["0.5", "0.5"]')) is None
    assert parse_official_outcome(market(prices='["0.97", "0.03"]')) is None


def test_malformed_payloads_return_none_rather_than_guessing():
    assert parse_official_outcome(market(prices="not json")) is None
    assert parse_official_outcome(market(prices='["1"]')) is None  # length mismatch
    assert parse_official_outcome({"conditionId": "c"}) is None


def test_yes_no_outcome_naming_maps_to_up_down():
    assert parse_official_outcome(market(outcomes='["Yes", "No"]', prices='["1", "0"]')) == "up"
    assert parse_official_outcome(market(outcomes='["Yes", "No"]', prices='["0", "1"]')) == "down"


# -- poller ----------------------------------------------------------

def make_poller(fetch, timeout=3600.0):
    resolved, abandoned = [], []
    poller = ResolutionPoller(
        Settings(resolution_timeout_seconds=timeout),
        on_resolved=lambda cid, outcome, ts: resolved.append((cid, outcome)),
        on_abandoned=abandoned.append,
        fetch=fetch,
    )
    return poller, resolved, abandoned


def test_resolves_once_settlement_appears():
    state = {"settled": False}

    async def fetch(ids):
        return {"cond1": market(closed=state["settled"], prices='["0", "1"]')}

    poller, resolved, abandoned = make_poller(fetch)
    poller.enqueue("cond1", now=0.0)

    asyncio.run(poller.poll_once(now=10.0))
    assert resolved == [] and "cond1" in poller.pending  # not settled yet

    state["settled"] = True
    asyncio.run(poller.poll_once(now=20.0))
    assert resolved == [("cond1", "down")]
    assert not poller.pending
    assert abandoned == []


def test_abandons_after_timeout_instead_of_guessing():
    async def fetch(ids):
        return {}  # settlement never shows up

    poller, resolved, abandoned = make_poller(fetch, timeout=60.0)
    poller.enqueue("cond1", now=0.0)

    asyncio.run(poller.poll_once(now=30.0))
    assert abandoned == [] and "cond1" in poller.pending

    asyncio.run(poller.poll_once(now=61.0))
    assert abandoned == ["cond1"]
    assert resolved == []
    assert not poller.pending


def test_fetch_failure_is_survivable_and_retried():
    calls = {"n": 0}

    async def fetch(ids):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("gamma down")
        return {"cond1": market(prices='["1", "0"]')}

    poller, resolved, _ = make_poller(fetch)
    poller.enqueue("cond1", now=0.0)

    asyncio.run(poller.poll_once(now=10.0))  # must not propagate
    assert "cond1" in poller.pending

    asyncio.run(poller.poll_once(now=20.0))
    assert resolved == [("cond1", "up")]


def test_enqueue_is_idempotent():
    async def fetch(ids):
        return {}

    poller, _, _ = make_poller(fetch)
    poller.enqueue("cond1", now=0.0)
    poller.enqueue("cond1", now=500.0)  # must not extend the deadline
    assert poller.pending == {"cond1": 3600.0}


# -- gamma lookup ----------------------------------------------------
#
# Lookup goes through /events?slug=, not /markets?condition_id(s)=. Both
# condition-id routes return HTTP 200 while failing to filter (empty list,
# or a default page of 20 unrelated markets), so a regression here would
# look like "nothing ever settles" rather than an error.

class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, list(params or [])))
        return FakeResponse(self._payload)


def event(slug, condition_id, prices='["1", "0"]', closed=True):
    return {
        "slug": slug,
        "markets": [
            {
                "conditionId": condition_id,
                "closed": closed,
                "outcomes": '["Up", "Down"]',
                "outcomePrices": prices,
            }
        ],
    }


def poller_with_slugs(mapping):
    from poly15m.db import Database

    db = Database(":memory:")
    for cid, slug in mapping.items():
        db.upsert_market(
            {
                "condition_id": cid,
                "slug": slug,
                "question_id": None,
                "token_id_up": "u",
                "token_id_down": "d",
                "window_open_ts": 0.0,
                "window_close_ts": 0.0,
                "discovered_ts": 0.0,
                "raw_json": "{}",
            }
        )
    return ResolutionPoller(Settings(), lambda *a: None, lambda *a: None, db=db)


def test_queries_events_by_slug():
    poller = poller_with_slugs({"cond1": "btc-updown-15m-111"})
    session = FakeSession([event("btc-updown-15m-111", "cond1")])

    found = asyncio.run(poller._fetch_from_gamma(session, ["cond1"]))

    url, params = session.calls[0]
    assert url.endswith("/events")
    assert params == [("slug", "btc-updown-15m-111")]
    assert parse_official_outcome(found["cond1"]) == "up"


def test_batches_multiple_pending_windows_into_one_request():
    poller = poller_with_slugs({"cond1": "slug-1", "cond2": "slug-2"})
    session = FakeSession([event("slug-1", "cond1"), event("slug-2", "cond2", prices='["0", "1"]')])

    found = asyncio.run(poller._fetch_from_gamma(session, ["cond1", "cond2"]))

    assert len(session.calls) == 1
    assert sorted(session.calls[0][1]) == [("slug", "slug-1"), ("slug", "slug-2")]
    assert parse_official_outcome(found["cond1"]) == "up"
    assert parse_official_outcome(found["cond2"]) == "down"


def test_ignores_events_we_did_not_ask_about():
    """A default page of unrelated markets must never grade a window."""
    poller = poller_with_slugs({"cond1": "slug-1"})
    session = FakeSession([event("other-slug", "someone-elses-condition")])

    found = asyncio.run(poller._fetch_from_gamma(session, ["cond1"]))

    assert found == {}


def test_skips_condition_ids_with_no_known_slug():
    poller = poller_with_slugs({})
    session = FakeSession([])

    found = asyncio.run(poller._fetch_from_gamma(session, ["cond-unknown"]))

    assert found == {}
    assert session.calls == []  # nothing to ask about, so no request at all
