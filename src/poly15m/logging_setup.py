"""Structured (JSON) logging setup.

Every component logs through the stdlib `logging` module; in JSON mode each
record is emitted as one line of JSON so logs are greppable/parseable
alongside the SQLite recording (SQLite holds the structured tick/order/fill
data used for backtesting, logging is for operational visibility).
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        extra = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str)


class TelegramHandler(logging.Handler):
    """Mirrors formatted log lines to a Telegram chat via the Bot API.

    Runs a background thread that batches whatever's queued and posts it
    every `batch_interval` seconds. The rest of this codebase is asyncio-based,
    so a blocking HTTP call inside `emit` would stall the event loop on every
    log line -- batching also keeps well clear of Telegram's rate limits.
    """

    _API_BASE = "https://api.telegram.org"
    _MAX_MESSAGE_LEN = 4000  # Telegram's hard limit is 4096; leave margin

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        level: int = logging.NOTSET,
        batch_interval: float = 2.0,
    ) -> None:
        super().__init__(level)
        self._url = f"{self._API_BASE}/bot{bot_token}/sendMessage"
        self._chat_id = chat_id
        self._batch_interval = batch_interval
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="telegram-log-forwarder", daemon=True
        )
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._queue.put_nowait(self.format(record))
        except Exception:
            self.handleError(record)

    def _run(self) -> None:
        while True:
            line = self._queue.get()
            if line is None:
                return
            batch = [line]
            deadline = time.monotonic() + self._batch_interval
            while (remaining := deadline - time.monotonic()) > 0:
                try:
                    nxt = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is None:
                    self._send("\n".join(batch))
                    return
                batch.append(nxt)
            self._send("\n".join(batch))

    def _send(self, text: str) -> None:
        for start in range(0, len(text), self._MAX_MESSAGE_LEN):
            chunk = text[start : start + self._MAX_MESSAGE_LEN]
            payload = json.dumps({"chat_id": self._chat_id, "text": chunk}).encode("utf-8")
            req = urllib.request.Request(
                self._url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                urllib.request.urlopen(req, timeout=10).close()
            except urllib.error.URLError:
                pass  # best-effort: never let Telegram delivery issues affect the bot itself

    def close(self) -> None:
        self._queue.put_nowait(None)
        self._thread.join(timeout=self._batch_interval + 5)
        super().close()


def setup_logging(
    level: str = "INFO",
    json_output: bool = True,
    telegram_bot_token: str | None = None,
    telegram_chat_id: str | None = None,
) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    if json_output:
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    if telegram_bot_token and telegram_chat_id:
        telegram_handler = TelegramHandler(telegram_bot_token, telegram_chat_id)
        telegram_handler.setFormatter(formatter)
        root.addHandler(telegram_handler)

    # noisy third-party loggers
    logging.getLogger("websockets").setLevel(max(logging.WARNING, root.level))
