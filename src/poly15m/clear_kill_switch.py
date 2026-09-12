"""Operator tool: clear a kill-switch hard halt (Implementation_Plan.md
Phase 5, item 21 follow-up).

`RiskGate` hard-halts once its kill switch has tripped on
`kill_switch_hard_halt_trips` distinct days inside a trailing
`kill_switch_hard_halt_window_days`-day window, and that halt survives a
process restart by design (see risk/limits.py) -- a restart used to be
exactly how the halt got silently bypassed. This is the one sanctioned way
past it: it requires a reason, writes an auditable
`kill_switch_operator_cleared` marker to the same event log the halt reads
its trip history from, and takes effect on the *next* process start (the
running process, if any, keeps whatever state it already has in memory --
restart it after clearing).

It does not touch `daily_loss_limit` or any of the sizing settings that
caused the trips in the first place -- if those haven't changed, clearing
the halt just buys the next trip a little more runway before it recurs.
"""

from __future__ import annotations

import argparse
import time

from .config import settings
from .db import Database
from .logging_setup import setup_logging
from .risk.limits import RiskGate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reason",
        required=True,
        help="why it's safe to resume -- persisted alongside the clear marker for the record",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the interactive confirmation prompt",
    )
    args = parser.parse_args()

    settings.ensure_dirs()
    setup_logging(
        settings.log_level,
        settings.log_json,
        telegram_bot_token=settings.telegram_bot_token,
        telegram_chat_id=settings.telegram_chat_id,
    )
    db = Database(settings.db_path)
    gate = RiskGate(settings, db)

    if not gate.hard_halted:
        print(f"No active hard halt at {settings.db_path} -- nothing to clear.")
        return

    print(f"Hard halt is active at {settings.db_path}.")
    print(f"Reason to clear: {args.reason!r}")
    if not args.yes:
        confirm = input("Type 'clear' to proceed: ")
        if confirm.strip() != "clear":
            print("Aborted.")
            return

    gate.clear_hard_halt(args.reason, ts=time.time())
    db.flush()  # inserts are batch-committed elsewhere (run_flush_loop); this is a one-shot CLI, not the trader
    print("Cleared. Restart the trader process for this to take effect.")


if __name__ == "__main__":
    main()
