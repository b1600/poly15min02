"""Regression guard: the Telegram forwarder thread must survive any send
failure, not just urllib.error.URLError. A RemoteDisconnected/ConnectionReset
during getresponse() used to escape the old except-URLError-only clause,
killing the daemon thread silently -- the bot kept trading, Telegram just
stopped forever until the process was restarted.
"""

import http.client
import time
from unittest.mock import patch

from poly15m.logging_setup import TelegramHandler


def test_forwarder_survives_non_urlerror_send_failure():
    calls = {"n": 0}

    def fake_urlopen(req, timeout=10):
        calls["n"] += 1
        if calls["n"] == 1:
            raise http.client.RemoteDisconnected("Remote end closed connection")
        return patch.mock_open()()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        handler = TelegramHandler("token", "chat", batch_interval=0.01)
        try:
            handler._send("first batch, will raise RemoteDisconnected")
            assert handler._thread.is_alive()
            handler._send("second batch, should still be sent")
        finally:
            handler.close()

    assert calls["n"] == 2
    assert not handler._thread.is_alive()


def test_run_loop_keeps_going_after_unexpected_exception():
    sent = []

    def fake_send(text):
        sent.append(text)
        if len(sent) == 1:
            raise RuntimeError("boom")

    with patch("urllib.request.urlopen"):
        handler = TelegramHandler("token", "chat", batch_interval=0.01)
        with patch.object(handler, "_send", side_effect=fake_send):
            handler.emit(_record("line one"))
            time.sleep(0.05)  # let the first batch flush (and raise) before the next
            handler.emit(_record("line two"))
            handler.close()

    assert sent == ["line one", "line two"]


def _record(msg: str):
    import logging

    return logging.LogRecord("test", logging.INFO, __file__, 0, msg, (), None)
