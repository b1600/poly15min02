# Implementation Plan — 15m BTC Up/Down Polymarket Bot (Python)

Based on `Strategy_v1.md`. Phased so there is always something testable before adding complexity.

---

## Phase 0 — Project Setup

1. **Project scaffold**: `pyproject.toml` with deps — `py-clob-client` (Polymarket CLOB SDK), `websockets`, `aiohttp`, `pydantic` (config/models), `numpy`/`pandas`, `python-dotenv`.
2. **Config module**: single `config.py` / `.env` for API keys (Polymarket private key + API creds; Binance is public), risk limits, fee assumptions, Kelly fraction, kill-switch thresholds.
3. **Structured logging + SQLite persistence** for every tick, order, fill, and market resolution from day one — this becomes the backtest dataset.

## Phase 1 — Data Layer (read-only)

4. **Binance feed** (`data/binance_ws.py`): async WebSocket client for BTCUSDT trade/bookTicker stream. Maintain rolling price buffer (last ~30 min at 1s resolution) for momentum/volatility features.
5. **Polymarket market discovery** (`data/market_finder.py`): poll Gamma API to find the current active "BTC Up or Down 15m" market, its condition ID, token IDs (Up/Down), open time, and the recorded opening price. Handle rollover to the next window automatically.
6. **Polymarket CLOB feed** (`data/clob_ws.py`): subscribe to the market channel WebSocket for order book (bids/asks/depth) and trade prints on both Up and Down tokens. Maintain in-memory book state.
7. **Clock/window manager**: track time-remaining-to-resolution; emit lifecycle events (window_open, T-5m, T-2m, T-30s, resolved).
8. **Milestone**: run for a few hours, verify recorded data — opening price matches Polymarket's, book state is consistent, no feed gaps.

## Phase 2 — Signal Engine & Pricing Model

9. **Feature computation** (`signals/features.py`), recomputed on each tick:
   - Deviation: `(spot − open_price) / (σ · √t_remaining)` with σ from recent realized vol
   - Momentum: 1m/3m/5m returns
   - Realized volatility (e.g., EWMA of 1s returns)
   - Book imbalance and aggressive trade flow on the CLOB
10. **Fair value model** (`pricing/fair_value.py`): start analytic, not ML — `P(Up) = Φ(deviation)` (probability BTC ends above open given current distance, vol, and time left, driftless Brownian assumption). Fast, no training dependency, and exactly the "reprice faster than stale orders" edge.
11. **Model calibration hooks**: log fair value vs. actual resolution every window; later fit a logistic-regression correction on top once a few hundred windows of data exist.
12. **Milestone (paper signals)**: run live, log `fair_value vs. market mid` divergences. Confirm dislocations actually appear and how long they persist before writing any execution code.

## Phase 3 — Edge Calculation & Paper Trading

13. **Executable edge** (`pricing/edge.py`): walk the book to compute expected VWAP fill for a target size, then
    `net_edge = fair_value − vwap_fill − fees − slippage_buffer − uncertainty_buffer`.
    Uncertainty buffer scales with vol and shrinks as `t_remaining → 0` — but widen it in the final minute (resolution-feed divergence risk vs. Binance).
14. **Paper trading simulator** (`sim/paper.py`): same interfaces as the live executor; simulate fills against the recorded book with latency + partial-fill assumptions. Run the full loop end-to-end in paper mode.
15. **Milestone**: paper PnL over ≥1–2 days of live data, including fee/slippage assumptions. Only proceed if positive after costs.

## Phase 4 — Position Structure & Execution (live, tiny size)

16. **Position manager** (`positions/manager.py`): implement the recommended hybrid —
    - Track Up/Down inventory per market; keep most of it matched (Up+Down pairs lock in `1 − cost` if bought under $1 combined)
    - Directional imbalance sized by fractional Kelly (10–25%) on net edge
    - Temporal arb: when one leg is cheap, buy it and register a standing hedge target for the other leg
17. **Execution engine** (`execution/executor.py`) using `py-clob-client`:
    - Post-only limit orders, split across 2–3 price levels
    - Cancel/replace when fair value moves more than a threshold (stale-order protection — don't become the bot that gets picked off)
    - Partial-fill handling: re-evaluate hedge targets on every fill event
18. **Order state machine**: local order tracker reconciled against exchange state (user channel WebSocket for own fills).
19. **Milestone**: live with minimum size ($5–20 per market), one market at a time.

## Phase 5 — Risk Management (before scaling up)

20. **Risk module** (`risk/limits.py`), enforced as a hard gate in front of the executor:
    - Max notional per 15m market
    - Max net directional exposure
    - Max inventory imbalance `|Up − Down|`
    - Daily loss limit → kill switch (cancel all orders, flatten if possible, halt process, alert)
21. **End-of-window handling**: rules for the final 2 minutes — stop opening new directional risk; near-resolution buys only when deviation is many σ, tail-risk capped.
22. **Watchdogs**: feed-staleness detection (halt trading if Binance or CLOB data older than N seconds), reconnect logic, crash-safe restart (rebuild inventory from exchange state).

## Phase 6 — Backtest & Iterate

23. **Backtester** replaying the Phase-1 recorded book data through the identical strategy code path (paper simulator reused). Sweep parameters: buffers, Kelly fraction, cancel thresholds.
24. **Calibration**: fit the logistic correction from step 11; compare against the analytic model out-of-sample.
25. **Then extend**: multiple concurrent windows, 5m-vs-15m cross-market z-score signals, ETH/SOL correlated exposure cap.

---

## Key Ordering Rationale

- Data recording comes first: backtesting is impossible without captured order-book history (Polymarket does not provide historical L2).
- Paper trading gates live trading.
- Risk limits gate size increases.
