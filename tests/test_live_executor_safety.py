"""Safety-gate tests for the live executor. No real credentials, no network
calls, no real funds anywhere near this file -- see the note on
`test_key` below for why constructing a ClobClient with it is inert."""

import pytest

from poly15m.config import Settings
from poly15m.execution.executor import LiveExecutor, _missing_credentials
from poly15m.execution.order_state import OrderStateMachine

# A syntactically valid but arbitrary, unfunded, non-secret private key --
# eth_account derives an address from it purely locally (no network call),
# so this is safe to use for testing that ClobClient construction wires
# the right parameters through, without ever touching a real account.
THROWAWAY_TEST_KEY = "0x" + "11" * 32


def base_creds_kwargs():
    return dict(
        polymarket_private_key=THROWAWAY_TEST_KEY,
        polymarket_api_key="test-key",
        polymarket_api_secret="test-secret",
        polymarket_api_passphrase="test-passphrase",
    )


def test_missing_credentials_lists_all_when_none_set():
    settings = Settings(live_trading_enabled=True)
    missing = _missing_credentials(settings)
    assert set(missing) == {
        "polymarket_private_key",
        "polymarket_api_key",
        "polymarket_api_secret",
        "polymarket_api_passphrase",
    }


def test_missing_credentials_empty_when_all_set():
    settings = Settings(live_trading_enabled=True, **base_creds_kwargs())
    assert _missing_credentials(settings) == []


def test_refuses_construction_when_live_trading_disabled():
    settings = Settings(live_trading_enabled=False, **base_creds_kwargs())
    with pytest.raises(RuntimeError, match="live_trading_enabled"):
        LiveExecutor(settings, OrderStateMachine())


def test_refuses_construction_when_credentials_missing():
    settings = Settings(live_trading_enabled=True)
    with pytest.raises(RuntimeError, match="credentials"):
        LiveExecutor(settings, OrderStateMachine())


def test_constructs_when_gates_are_satisfied():
    settings = Settings(live_trading_enabled=True, **base_creds_kwargs())
    executor = LiveExecutor(settings, OrderStateMachine())
    assert executor.client.get_address() is not None


def test_cancel_everything_calls_cancel_all_and_marks_local_orders_cancelled():
    settings = Settings(live_trading_enabled=True, **base_creds_kwargs())
    executor = LiveExecutor(settings, OrderStateMachine())
    calls = []
    executor.client.cancel_all = lambda: calls.append(True)  # stub out the real network call

    executor.order_state.add_local_order("c1", "cond1", "tok_up", "up", "BUY", 0.5, 10.0, "directional_kelly", ts=1.0)
    executor.order_state.confirm_exchange_id("c1", "ex1", ts=1.0)

    executor.cancel_everything()

    assert calls == [True]
    assert executor.order_state.orders["c1"].status == "cancelled"


def test_cancel_everything_does_not_raise_when_client_call_fails():
    settings = Settings(live_trading_enabled=True, **base_creds_kwargs())
    executor = LiveExecutor(settings, OrderStateMachine())

    def boom():
        raise RuntimeError("network down")

    executor.client.cancel_all = boom
    executor.cancel_everything()  # must not raise
