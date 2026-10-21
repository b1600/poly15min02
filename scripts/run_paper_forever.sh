#!/bin/sh
# Keeps the paper bot running: restarts it whenever it exits, including when
# the stale-feed watchdog (src/poly15m/watchdog.py) exits it on purpose.
# Ctrl+C / SIGTERM to the bot exits 0, which stops this loop too.
#
#   tmux new -d -s poly15m 'sh scripts/run_paper_forever.sh'
cd "$(dirname "$0")/.." || exit 1
while true; do
    .venv/bin/poly15m-paper-trade 2>> var/paper_trade.stderr
    code=$?
    if [ "$code" -eq 0 ]; then
        exit 0
    fi
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') paper bot exited with code $code, restarting in 30s" >> var/paper_trade.stderr
    sleep 30
done
