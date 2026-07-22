from poly15m.data.clock import WindowClock


def test_emits_each_milestone_once_in_order():
    clock = WindowClock(condition_id="c1", open_ts=1000.0, close_ts=1900.0)  # 15m window

    assert clock.poll(now=999.0) == []
    assert clock.poll(now=1000.0) == ["window_open"]
    assert clock.poll(now=1000.5) == []  # no duplicate

    assert clock.poll(now=1600.0) == ["t_minus_5m"]  # 300s remaining
    assert clock.poll(now=1780.0) == ["t_minus_2m"]  # 120s remaining
    assert clock.poll(now=1870.0) == ["t_minus_30s"]  # 30s remaining
    assert clock.poll(now=1900.0) == ["resolved"]
    assert clock.poll(now=1950.0) == []  # already emitted


def test_late_first_poll_emits_all_crossed_milestones_at_once():
    clock = WindowClock(condition_id="c1", open_ts=1000.0, close_ts=1900.0)
    events = clock.poll(now=1900.0)
    assert events == ["window_open", "t_minus_5m", "t_minus_2m", "t_minus_30s", "resolved"]


def test_time_remaining_and_is_open():
    clock = WindowClock(condition_id="c1", open_ts=1000.0, close_ts=1900.0)
    assert clock.time_remaining(now=1000.0) == 900.0
    assert clock.time_remaining(now=2000.0) == 0.0
    assert clock.is_open(now=999.0) is False
    assert clock.is_open(now=1500.0) is True
    assert clock.is_open(now=1900.0) is False
