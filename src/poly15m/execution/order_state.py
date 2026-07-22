"""Local order state machine (Implementation_Plan.md Phase 4, item 18).

Tracks our own view of every order we've placed, keyed by a client-side id
we generate up front (so we can record intent before the exchange has
even acknowledged it), with a side table mapping the exchange-assigned
order id back to that client id once known -- fills and status updates
arriving over the user-channel WebSocket reference the exchange id, not
ours.

`reconcile()` compares this local state against a fresh REST snapshot
(`ClobClient.get_orders()`), which is what makes a crash-safe restart
possible (Phase 5): orders open remotely that we have no local record of
get surfaced so the caller can adopt them; orders we think are still open
but that vanished from the remote snapshot are presumed filled or
cancelled externally.

Note: the exact field names in a `get_orders()` response are asserted here
per Polymarket's publicly documented CLOB REST shape (id, status, market,
asset_id, side, price, original_size, size_matched) but have not been
verified against a live account -- this project has no funded credentials
to test against. Treat `reconcile()` as unverified until checked against a
real response.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

OPEN_STATUSES = frozenset({"pending", "open", "partially_filled"})


@dataclass
class OrderRecord:
    client_order_id: str
    condition_id: str
    token_id: str
    outcome: str  # "up" | "down"
    side: str  # "BUY" | "SELL"
    price: float
    size: float
    reason: str  # from positions.manager.TradeIntent.reason, for logging/debugging
    created_ts: float
    exchange_order_id: str | None = None
    filled_size: float = 0.0
    status: str = "pending"
    updated_ts: float = field(default=0.0)

    def __post_init__(self) -> None:
        if not self.updated_ts:
            self.updated_ts = self.created_ts

    @property
    def remaining_size(self) -> float:
        return max(0.0, self.size - self.filled_size)


@dataclass
class ReconcileResult:
    presumed_closed_externally: list[OrderRecord]  # locally open, missing from the remote snapshot
    unknown_remote_orders: list[dict[str, Any]]  # open remotely, no local record (e.g. after a restart)


class OrderStateMachine:
    def __init__(self):
        self.orders: dict[str, OrderRecord] = {}  # keyed by client_order_id
        self._exchange_to_client: dict[str, str] = {}

    def add_local_order(
        self,
        client_order_id: str,
        condition_id: str,
        token_id: str,
        outcome: str,
        side: str,
        price: float,
        size: float,
        reason: str,
        ts: float | None = None,
    ) -> OrderRecord:
        record = OrderRecord(
            client_order_id=client_order_id,
            condition_id=condition_id,
            token_id=token_id,
            outcome=outcome,
            side=side,
            price=price,
            size=size,
            reason=reason,
            created_ts=ts if ts is not None else time.time(),
        )
        self.orders[client_order_id] = record
        return record

    def confirm_exchange_id(self, client_order_id: str, exchange_order_id: str, ts: float | None = None) -> None:
        record = self.orders.get(client_order_id)
        if record is None:
            logger.warning("confirm_exchange_id_unknown_client_order", extra={"client_order_id": client_order_id})
            return
        record.exchange_order_id = exchange_order_id
        record.status = "open"
        record.updated_ts = ts if ts is not None else time.time()
        self._exchange_to_client[exchange_order_id] = client_order_id

    def apply_fill(self, order_id: str, fill_size: float, ts: float | None = None) -> OrderRecord | None:
        record = self._find(order_id)
        if record is None:
            logger.warning("fill_for_unknown_order", extra={"order_id": order_id, "fill_size": fill_size})
            return None
        record.filled_size += fill_size
        record.updated_ts = ts if ts is not None else time.time()
        record.status = "filled" if record.remaining_size <= 1e-9 else "partially_filled"
        return record

    def mark_cancelled(self, order_id: str, ts: float | None = None) -> OrderRecord | None:
        return self._set_status(order_id, "cancelled", ts)

    def mark_rejected(self, order_id: str, ts: float | None = None) -> OrderRecord | None:
        return self._set_status(order_id, "rejected", ts)

    def _set_status(self, order_id: str, status: str, ts: float | None) -> OrderRecord | None:
        record = self._find(order_id)
        if record is None:
            return None
        record.status = status
        record.updated_ts = ts if ts is not None else time.time()
        return record

    def _find(self, order_id: str) -> OrderRecord | None:
        if order_id in self.orders:
            return self.orders[order_id]
        client_id = self._exchange_to_client.get(order_id)
        return self.orders.get(client_id) if client_id else None

    def open_orders(self, condition_id: str | None = None) -> list[OrderRecord]:
        return [
            r
            for r in self.orders.values()
            if r.status in OPEN_STATUSES and (condition_id is None or r.condition_id == condition_id)
        ]

    def orders_for_token(self, token_id: str) -> list[OrderRecord]:
        return [r for r in self.orders.values() if r.token_id == token_id]

    def reconcile(self, remote_orders: list[dict[str, Any]]) -> ReconcileResult:
        remote_ids = {o.get("id") for o in remote_orders if o.get("id")}
        local_open = self.open_orders()

        presumed_closed = []
        for record in local_open:
            if record.exchange_order_id is None:
                continue  # never confirmed by the exchange -- not a reconciliation concern
            if record.exchange_order_id not in remote_ids:
                presumed_closed.append(record)

        known_exchange_ids = set(self._exchange_to_client)
        unknown_remote = [o for o in remote_orders if o.get("id") and o.get("id") not in known_exchange_ids]

        for record in presumed_closed:
            logger.warning(
                "order_presumed_closed_externally",
                extra={"client_order_id": record.client_order_id, "exchange_order_id": record.exchange_order_id},
            )
        for remote in unknown_remote:
            logger.warning("unknown_remote_order", extra={"order_id": remote.get("id")})

        return ReconcileResult(presumed_closed_externally=presumed_closed, unknown_remote_orders=unknown_remote)
