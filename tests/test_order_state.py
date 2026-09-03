from poly15m.execution.order_state import OrderStateMachine


def make_osm_with_order(price=0.5, size=20.0):
    osm = OrderStateMachine()
    osm.add_local_order("c1", "cond1", "tok_up", "up", "BUY", price, size, "directional_kelly", ts=1.0)
    return osm


def test_add_local_order_starts_pending():
    osm = make_osm_with_order()
    record = osm.orders["c1"]
    assert record.status == "pending"
    assert record.exchange_order_id is None
    assert record.remaining_size == 20.0


def test_confirm_exchange_id_opens_order_and_enables_lookup():
    osm = make_osm_with_order()
    osm.confirm_exchange_id("c1", "ex1", ts=2.0)
    record = osm.orders["c1"]
    assert record.status == "open"
    assert record.exchange_order_id == "ex1"
    assert osm._find("ex1") is record


def test_apply_fill_partial_then_full():
    osm = make_osm_with_order()
    osm.confirm_exchange_id("c1", "ex1", ts=2.0)

    osm.apply_fill("ex1", 8.0, ts=3.0)
    record = osm.orders["c1"]
    assert record.status == "partially_filled"
    assert record.filled_size == 8.0
    assert record.remaining_size == 12.0

    osm.apply_fill("ex1", 12.0, ts=4.0)
    assert record.status == "filled"
    assert record.remaining_size == 0.0


def test_apply_fill_unknown_order_returns_none():
    osm = make_osm_with_order()
    assert osm.apply_fill("nonexistent", 5.0) is None


def test_mark_cancelled():
    osm = make_osm_with_order()
    osm.confirm_exchange_id("c1", "ex1", ts=2.0)
    record = osm.mark_cancelled("ex1", ts=3.0)
    assert record.status == "cancelled"
    assert osm.open_orders() == []


def test_open_orders_filters_by_status_and_condition():
    osm = OrderStateMachine()
    osm.add_local_order("c1", "cond1", "tok_up", "up", "BUY", 0.5, 20.0, "directional_kelly", ts=1.0)
    osm.add_local_order("c2", "cond2", "tok_up", "up", "BUY", 0.5, 20.0, "directional_kelly", ts=1.0)
    osm.confirm_exchange_id("c2", "ex2", ts=2.0)
    osm.apply_fill("ex2", 20.0, ts=3.0)  # fully filled -> not "open" anymore

    assert [r.client_order_id for r in osm.open_orders()] == ["c1"]
    assert osm.open_orders(condition_id="cond2") == []


def test_orders_for_token():
    osm = OrderStateMachine()
    osm.add_local_order("c1", "cond1", "tok_up", "up", "BUY", 0.5, 20.0, "directional_kelly", ts=1.0)
    osm.add_local_order("c2", "cond1", "tok_down", "down", "BUY", 0.5, 20.0, "directional_kelly", ts=1.0)
    assert [r.client_order_id for r in osm.orders_for_token("tok_up")] == ["c1"]


def test_reconcile_flags_locally_open_order_missing_remotely():
    osm = make_osm_with_order()
    osm.confirm_exchange_id("c1", "ex1", ts=2.0)

    result = osm.reconcile(remote_orders=[])

    assert len(result.presumed_closed_externally) == 1
    assert result.presumed_closed_externally[0].client_order_id == "c1"
    assert result.unknown_remote_orders == []


def test_reconcile_does_not_flag_order_present_remotely():
    osm = make_osm_with_order()
    osm.confirm_exchange_id("c1", "ex1", ts=2.0)

    result = osm.reconcile(remote_orders=[{"id": "ex1", "status": "LIVE"}])

    assert result.presumed_closed_externally == []


def test_reconcile_ignores_unconfirmed_local_orders():
    osm = make_osm_with_order()  # never confirmed -- exchange_order_id is still None

    result = osm.reconcile(remote_orders=[])

    assert result.presumed_closed_externally == []


def test_reconcile_surfaces_unknown_remote_orders():
    osm = make_osm_with_order()
    osm.confirm_exchange_id("c1", "ex1", ts=2.0)

    result = osm.reconcile(remote_orders=[{"id": "ex1"}, {"id": "ex-mystery"}])

    assert [o["id"] for o in result.unknown_remote_orders] == ["ex-mystery"]
