#!/bin/sh
# Run every step 5 variant through a 4-wide queue, then write the report.
# Skips variants that already have var/step5/<name>.json, so it can be
# re-run after an interruption. Refuses to start if a sweep is running.
#
# Takes ~4 h on an 8 GB Mac (4 in parallel at ~2.8 s/window). Start it from
# your own terminal so it isn't tied to a Claude Code session:
#
#     nohup caffeinate -i sh scripts/run_step5_all.sh > var/step5/queue.log 2>&1 &
#
# Report: var/step5/report.txt (also appended to 20260928/20260928_step5_fixes.txt)

cd "$(dirname "$0")/.." || exit 1
if pgrep -f "step5_sweep.py run" > /dev/null; then
    echo "a step5 sweep is already running"; exit 1
fi
mkdir -p var/step5
PARALLEL=${PARALLEL:-4}

todo=""
for v in base maxT480 maxT600 volmin1.5 volmin2.4 floor1.5 floor2.4; do
    [ -f "var/step5/$v.json" ] || todo="$todo $v"
done
echo "$(date '+%F %T') running:$todo"

printf "%s\n" $todo | xargs -P "$PARALLEL" -I{} sh -c \
    '.venv/bin/python scripts/step5_sweep.py run {} > var/step5/{}.log 2>&1; echo "$(date "+%F %T") finished {} (exit $?)"'

.venv/bin/python scripts/step5_sweep.py report > var/step5/report.txt 2>&1
{
    echo
    echo "-----------------------RESULTS $(date '+%d %b %Y %H:%M')"
    cat var/step5/report.txt
} >> 20260928/20260928_step5_fixes.txt
echo "$(date '+%F %T') report written"
