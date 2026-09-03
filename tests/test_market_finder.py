from poly15m.data.market_finder import _select_market

PREFIX = "btc-updown-15m-"
WINDOW = 900  # 15m


def _event(slug: str, end_date: str, event_start_time: str | None = None) -> dict:
    return {"slug": slug, "endDate": end_date, "eventStartTime": event_start_time}


def test_picks_the_in_window_market_over_upcoming_ones():
    now = 1_800_000_000.0
    events = [
        # closed just before `now`
        _event(f"{PREFIX}{int(now - WINDOW - 1)}", "2027-01-01T00:14:59Z"),
        # currently in-window: opened 100s ago, closes in 800s
        _event(f"{PREFIX}{int(now - 100)}", "2027-01-01T00:13:20Z"),
        # far-future pre-listed window
        _event(f"{PREFIX}{int(now + 20_000)}", "2027-01-01T06:00:00Z"),
    ]
    # patch endDate to be derivable from now for the in-window/expired cases
    events[0]["endDate"] = _iso(now - 1)
    events[1]["endDate"] = _iso(now + 800)
    events[2]["endDate"] = _iso(now + 20_000 + WINDOW)

    chosen = _select_market(events, PREFIX, WINDOW, now)
    assert chosen["slug"] == events[1]["slug"]


def test_falls_back_to_soonest_upcoming_when_no_market_is_in_window():
    now = 1_800_000_000.0
    events = [
        _event(f"{PREFIX}{int(now + 5_000)}", _iso(now + 5_000 + WINDOW)),
        _event(f"{PREFIX}{int(now + 500)}", _iso(now + 500 + WINDOW)),  # soonest
    ]
    chosen = _select_market(events, PREFIX, WINDOW, now)
    assert chosen["slug"] == events[1]["slug"]


def test_ignores_events_outside_the_slug_prefix():
    now = 1_800_000_000.0
    events = [
        {"slug": "eth-updown-15m-123", "endDate": _iso(now + 800), "eventStartTime": None},
    ]
    assert _select_market(events, PREFIX, WINDOW, now) is None


def test_derives_open_ts_from_end_date_when_event_start_time_is_missing():
    now = 1_800_000_000.0
    close_ts = now + 800
    events = [_event(f"{PREFIX}{int(now - 100)}", _iso(close_ts), event_start_time=None)]
    chosen = _select_market(events, PREFIX, WINDOW, now)
    assert chosen["_open_ts"] == close_ts - WINDOW


def _iso(epoch: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
