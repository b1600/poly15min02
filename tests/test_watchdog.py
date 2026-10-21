import asyncio
from types import SimpleNamespace

from poly15m.watchdog import feed_watchdog_loop, stale_feeds


def test_stale_feeds_flags_only_silent_ones():
    feeds = {"binance": SimpleNamespace(last_msg_ts=995.0), "clob": SimpleNamespace(last_msg_ts=800.0)}
    assert stale_feeds(feeds, started_at=700.0, threshold=120.0, now=1000.0) == {"clob": 200.0}


def test_feed_that_never_sent_counts_from_start():
    feeds = {"chainlink": SimpleNamespace(last_msg_ts=None)}
    assert stale_feeds(feeds, started_at=950.0, threshold=120.0, now=1000.0) == {}
    assert stale_feeds(feeds, started_at=850.0, threshold=120.0, now=1000.0) == {"chainlink": 150.0}


def test_loop_calls_on_stale_once_and_returns():
    clock = [1000.0]
    binance = SimpleNamespace(last_msg_ts=1000.0)
    tripped = []

    async def advance():
        # binance keeps ticking for a while, then goes silent (the Oct 15 2026 stall)
        for _ in range(30):
            await asyncio.sleep(0)
            clock[0] += 10.0
            if clock[0] < 1100.0:
                binance.last_msg_ts = clock[0]

    async def main():
        loop = asyncio.create_task(
            feed_watchdog_loop({"binance": binance}, 120.0, tripped.append, check_interval=0, clock=lambda: clock[0])
        )
        await advance()
        await asyncio.wait_for(loop, 1.0)

    asyncio.run(main())
    assert len(tripped) == 1
    assert tripped[0]["binance"] > 120.0
