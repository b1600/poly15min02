"""Replay every backtestable window in `settings.db_path` and print a
summary. This is the "did net_edge survive contact with historical data"
report -- the more rigorous companion to paper_trade.py's live milestone
(item 15), reusing the same recorded data Phase 1 onward has been writing.
"""

from __future__ import annotations

from ..config import settings
from ..logging_setup import setup_logging
from .data_loader import list_backtestable_markets
from .engine import run_backtest


def main() -> None:
    setup_logging(
        settings.log_level,
        settings.log_json,
        telegram_bot_token=settings.telegram_bot_token,
        telegram_chat_id=settings.telegram_chat_id,
    )

    condition_ids = list_backtestable_markets(settings.db_path)
    if not condition_ids:
        print(f"No backtestable markets found in {settings.db_path}.")
        print("Run record.py, paper_signals.py, or paper_trade.py for a while first to accumulate data.")
        return

    print(f"Replaying {len(condition_ids)} recorded window(s) from {settings.db_path}...")
    _engine, result = run_backtest(settings.db_path, condition_ids, settings)

    print(f"\nWindows replayed: {result.num_windows_replayed}")
    print(f"Realized PnL:     {result.realized_pnl:+.2f}")
    print(f"Fees paid:        {result.fees_paid:.2f}")
    print(f"Net after fees:   {result.realized_pnl:+.2f}")
    print(f"Daily PnL (last day tracked): {result.daily_pnl:+.2f}")
    print(f"Kill switch triggered: {result.kill_switch_active}")


if __name__ == "__main__":
    main()
