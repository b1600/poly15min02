# Poly15min02

Everything's already built and installed from our work together — you just need to run it. Here's the sequence, safest first.

## 1. Environment is already set up

The venv exists at `.venv` with everything installed (confirmed working — `poly15m` package imports fine, all 7 console scripts are present). I also just created `.env` from the template — it's all commented out, meaning every setting uses its safe default. Nothing to fill in yet.

If you're ever doing this on a fresh machine, that setup was:

```
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"
```

(I used `uv` because this machine's system Python has a broken `pyexpat` linkage that breaks the normal venv/pip path.)

## 2. First run: `poly15m-record` — pure data recording, nothing else

This is the lowest-risk way to confirm the whole pipeline works: it connects to Binance + Polymarket, discovers the current 15-minute BTC market, and records everything to SQLite. It makes no decisions and places no orders — real or simulated.

```
.venv/bin/poly15m-record
```

Let it run for a minute or two, then Ctrl+C. You should see log lines like `market_rollover`, `binance_connected`, `clob_ws_connected`, `lifecycle_event`. If you see those with no errors, the network/data layer works.

Check it actually wrote something:

```
sqlite3 var/poly15m.db "select condition_id, slug, open_price from markets;"
sqlite3 var/poly15m.db "select count(*) from price_ticks;"
```

## 3. Main tool: `poly15m-paper-trade` — the one to actually run

This is a superset of step 2 (records the same data) plus fair-value modeling, trade decisions, and simulated fills — all with fake money, zero financial risk. This is what you should run for real.

```
.venv/bin/poly15m-paper-trade
```

Watch for these log events as it runs:

- `market_rollover` — found the current 15-min window
- `trade_decision` — it decided to buy Up or Down (reason tells you why: `matched_arb`, `directional_kelly`, or `temporal_hedge`)
- `window_resolved` — a window finished; shows outcome, pnl, and realized_pnl_total
- `pnl_status` — periodic summary (every 60s), including `kill_switch_active` (should stay false)

Stop it anytime with Ctrl+C — it shuts down cleanly (cancels nothing since paper mode has no real orders, just flushes the DB).

## 4. This needs to run for a while to mean anything

A few minutes just proves it works. The actual question — "is this edge real after costs?" — needs 1–2+ days of continuous data (that's Phase 3's explicit milestone). Practically, that means running it in the background across a longer stretch:

```
nohup .venv/bin/poly15m-paper-trade > paper_trade.log 2>&1 &
```

Or, equivalently, in a detached tmux session (lets you reattach later to watch it live):

```
tmux new -d -s poly15m '.venv/bin/poly15m-paper-trade > paper_trade.log 2>&1'
```

Reattach anytime with `tmux attach -t poly15m` (detach again with `Ctrl+B` then `D`). Stop it with `tmux kill-session -t poly15m`, or reattach and hit `Ctrl+C`.

(screen/a systemd unit also work if you want it to survive a terminal close/reboot). Just don't delete `var/poly15m.db` between runs — that's the whole point, it's accumulating your track record.

## 5. Check the results

```
sqlite3 var/poly15m.db "select event, ts, realized_pnl, fees_paid from paper_pnl_log order by ts desc limit 10;"
sqlite3 var/poly15m.db "select condition_id, resolved_outcome from markets where resolved_outcome is not null;"
```

Positive `realized_pnl` after `fees_paid` across many windows is the signal to actually trust the edge.

## 6. Once you have real data: backtest and calibrate

```
.venv/bin/poly15m-backtest      # replays everything recorded so far through the exact same strategy code
.venv/bin/poly15m-sweep         # tries a grid of Kelly fraction / edge threshold / buffer settings
.venv/bin/poly15m-calibrate     # fits a logistic correction, compares it to the analytic model
```

`poly15m-calibrate` will likely tell you there isn't enough data yet — it deliberately refuses to report anything until you have a real number of distinct resolved windows (not just rows), not just one trending window's worth. That's intentional, not a bug.

## 7. Live trading — do not run this yet

`poly15m-live-trade` exists but will refuse to start right now (no credentials configured), and I'd strongly recommend not touching it until the paper-trading milestone above actually shows sustained positive PnL. When/if you get there, it needs real setup: a funded Polymarket wallet, API credentials derived from it, `POLY15M_LIVE_TRADING_ENABLED=true` in `.env`, and a read of `execution/executor.py`'s and `execution/user_ws.py`'s docstrings first — both are correct against the documented API but have never been exercised against a real account. That's a separate, deliberate conversation whenever you're ready for it — not part of today's setup.

---

Suggested order for right now: run step 2 for a couple minutes to confirm things connect, then kick off step 3/4 in the background and let it run. Want me to start it now and check back on it periodically?
