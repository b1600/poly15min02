"""Live execution engine (Implementation_Plan.md Phase 4, item 17), using
`py-clob-client` to place real orders with real funds.

Every method here that touches `self.client` is a real, irreversible
action. The constructor refuses to build a client at all unless
`settings.live_trading_enabled` is explicitly True *and* real credentials
are configured -- there is no code path in this project that flips that
flag or supplies credentials on its own; both come only from the operator's
own `.env`.

Order style: post-only limit orders, split across `order_price_levels`
price levels stepping down from `intent.limit_price` by the market's tick
size (Implementation_Plan.md item 17) -- for a BUY this keeps every level
at or below the limit, so post-only can never reject for crossing the
spread. `reprice_if_needed` is the stale-order protection: once fair value
has drifted past `reprice_threshold` from a resting order's price, cancel
it rather than risk being picked off by a trader who repriced faster than
we did.

Unverified: `py-clob-client`'s REST methods (create_order, post_order,
cancel, get_orders) are used exactly per their source in
`py_clob_client/client.py`, but this class has never been exercised
against a live account -- there are no funded credentials in this project
to test with. Treat it as reviewed-but-unverified code, not
live-validated, unlike the rest of this codebase.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..config import Settings
from ..positions.manager import TradeIntent
from .order_state import OrderRecord, OrderStateMachine, ReconcileResult

logger = logging.getLogger(__name__)


def _missing_credentials(settings: Settings) -> list[str]:
    required = {
        "polymarket_private_key": settings.polymarket_private_key,
        "polymarket_api_key": settings.polymarket_api_key,
        "polymarket_api_secret": settings.polymarket_api_secret,
        "polymarket_api_passphrase": settings.polymarket_api_passphrase,
    }
    return [name for name, value in required.items() if not value]


class LiveExecutor:
    def __init__(self, settings: Settings, order_state: OrderStateMachine):
        if not settings.live_trading_enabled:
            raise RuntimeError(
                "live_trading_enabled is False -- refusing to construct a live order client"
            )
        missing = _missing_credentials(settings)
        if missing:
            raise RuntimeError(f"missing required live-trading credentials: {', '.join(missing)}")

        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds

        creds = ApiCreds(
            api_key=settings.polymarket_api_key,
            api_secret=settings.polymarket_api_secret,
            api_passphrase=settings.polymarket_api_passphrase,
        )
        self.client = ClobClient(
            host=settings.clob_rest_base,
            chain_id=settings.polymarket_chain_id,
            key=settings.polymarket_private_key,
            creds=creds,
            signature_type=settings.polymarket_signature_type,
            funder=settings.polymarket_funder_address,
        )
        self.settings = settings
        self.order_state = order_state
        logger.warning("live_executor_constructed", extra={"address": self.client.get_address()})

    def submit_split_order(self, intent: TradeIntent, tick_size: float) -> list[OrderRecord]:
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        levels = max(1, self.settings.order_price_levels)
        per_level_size = intent.size / levels
        spacing = tick_size * self.settings.order_level_tick_multiplier

        records: list[OrderRecord] = []
        for i in range(levels):
            price = round(intent.limit_price - i * spacing, 4)
            if price <= 0:
                break

            client_order_id = f"{intent.condition_id[:10]}-{intent.outcome}-{int(time.time() * 1000)}-{i}"
            record = self.order_state.add_local_order(
                client_order_id,
                intent.condition_id,
                intent.token_id,
                intent.outcome,
                "BUY",
                price,
                per_level_size,
                intent.reason,
            )
            records.append(record)

            try:
                order_args = OrderArgs(token_id=intent.token_id, price=price, size=per_level_size, side=BUY)
                signed_order = self.client.create_order(order_args)
                response = self.client.post_order(signed_order, orderType=OrderType.GTC, post_only=True)
                exchange_order_id = (
                    response.get("orderID") or response.get("orderId") or response.get("id")
                    if isinstance(response, dict)
                    else None
                )
                if exchange_order_id:
                    self.order_state.confirm_exchange_id(client_order_id, exchange_order_id)
                else:
                    logger.warning("post_order_no_id_in_response", extra={"response": response})
            except Exception:
                logger.exception("post_order_failed", extra={"client_order_id": client_order_id})
                self.order_state.mark_rejected(client_order_id)

        return records

    def cancel_order(self, exchange_order_id: str) -> None:
        try:
            self.client.cancel(exchange_order_id)
            self.order_state.mark_cancelled(exchange_order_id)
            logger.info("order_cancelled", extra={"order_id": exchange_order_id})
        except Exception:
            logger.exception("cancel_failed", extra={"order_id": exchange_order_id})

    def cancel_all_for_condition(self, condition_id: str) -> None:
        for record in self.order_state.open_orders(condition_id):
            if record.exchange_order_id:
                self.cancel_order(record.exchange_order_id)

    def cancel_everything(self) -> None:
        """Kill-switch action: cancel every resting order for this account,
        including any we have no local record of -- uses the exchange's own
        cancel-all rather than iterating local state, so it isn't limited
        by what OrderStateMachine happens to know about."""
        try:
            self.client.cancel_all()
            logger.critical("cancel_everything_sent")
        except Exception:
            logger.exception("cancel_everything_failed")
        for record in self.order_state.open_orders():
            self.order_state.mark_cancelled(record.exchange_order_id or record.client_order_id)

    def reprice_if_needed(self, condition_id: str, fair_value_by_outcome: dict[str, float]) -> None:
        """Cancel resting orders whose price has drifted more than
        `reprice_threshold` from the current fair value for that outcome --
        the caller is expected to replace them on a subsequent decision
        cycle via the normal `PositionManager` -> `submit_split_order` path."""
        for record in self.order_state.open_orders(condition_id):
            if record.exchange_order_id is None:
                continue
            fair_p = fair_value_by_outcome.get(record.outcome)
            if fair_p is None:
                continue
            if abs(fair_p - record.price) > self.settings.reprice_threshold:
                logger.info(
                    "reprice_stale_order",
                    extra={
                        "order_id": record.exchange_order_id,
                        "order_price": record.price,
                        "fair_value": fair_p,
                    },
                )
                self.cancel_order(record.exchange_order_id)

    def reconcile(self, condition_id: str | None = None) -> ReconcileResult:
        from py_clob_client.clob_types import OpenOrderParams

        params = OpenOrderParams(market=condition_id) if condition_id else None
        remote_orders: list[dict[str, Any]] = self.client.get_orders(params)
        return self.order_state.reconcile(remote_orders)
