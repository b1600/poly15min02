# 15m Bitcoin Up/Down Bot Strategy

**Inspired by Polymarket Short-Duration Market Bots**  
*(Based on analysis of @Dan1ro0’s thread - July 2026)*

---

## Analysis of the Source Post

The post by @Dan1ro0 provides a detailed breakdown of how profitable bots trade Polymarket’s “BTC Up or Down” markets (mainly 5m, with strong applicability to 15m).

**Key takeaway**: These bots do **not** primarily make money by predicting whether Bitcoin will go up or down. They profit by exploiting **temporary pricing inefficiencies** between the prediction market’s order book and real-time external BTC prices (stale orders, slow repricing).

The post outlines a complete professional trading pipeline used by bots generating significant monthly PnL.

---

## Core Lessons Learned

- Edge comes from **market microstructure**, not superior directional prediction.
- Bots follow a structured pipeline:  
  `Data → Signal → Fair Value → Executable Edge → Position Structure → Execution → Risk`
- Raw probability gaps must be adjusted for **fees, slippage, partial fills, and model uncertainty**.
- **Position construction** is as important as the signal (Temporal Arbitrage, Hedged Directional, Inventory Management, etc.).
- **Execution quality** and **strict risk management** often determine long-term survival.
- 15m markets reprice more slowly than 5m markets → larger and more persistent dislocations.

---

## Recommended Strategy: 15m BTC Up/Down Bot

### Goal
Build an automated system that systematically captures edge from pricing inefficiencies in Polymarket’s 15-minute BTC Up/Down markets by combining real-time external price data with disciplined order book execution and risk controls.

### Market Mechanics
- Each market has a clear **opening price** (BTC price at the start of the 15-minute window).
- Traders buy **Up** (BTC closes **above** opening price) or **Down**.
- Shares pay **$1** if correct, **$0** otherwise.
- Resolution uses Polymarket’s designated price feed (typically Chainlink or equivalent).

---

## Bot Architecture

### 1. Data Layer
- Real-time BTC price:
  - **Primary**: Binance spot/futures WebSocket (fastest)
  - **Secondary**: Polymarket resolution feed / Chainlink (for accuracy)
- Full Polymarket CLOB data (best bid/ask, depth, recent trades)
- Time remaining until resolution
- Related markets (concurrent 5m BTC, next 15m window, ETH/SOL 15m)
- Bot’s own orders, fills, and current inventory

### 2. Signal Engine
Convert data into actionable features:
- BTC price deviation from the market’s opening price (normalized by expected volatility × √time remaining)
- Short-term momentum (1m / 3m / 5m returns)
- Recent realized volatility
- Order book imbalance
- Aggressive trade flow
- Cross-market deviations (especially 15m vs 5m)
- Time-decay features

### 3. Pricing Model
Calculate **independent fair value** for Up and Down in the current 15m market.

- Output estimated probability **P(Up)**
- Use statistical models or lightweight ML (logistic regression / gradient boosting)
- Apply Bayesian-style updates when new signals arrive
- Optional: Track z-score between this 15m market and related 5m/15m markets

### 4. Executable Edge Calculation
Only trade when **net edge** exists after costs:

**Net Edge** = Fair Value − Expected VWAP Fill Price − Fees − Slippage Buffer − Model Uncertainty Buffer

### 5. Position Structure (Core Decision Layer)

| Position Structure          | Description                                      | Best Used When                     | Risk Level     | 15m Suitability |
|----------------------------|--------------------------------------------------|------------------------------------|----------------|-----------------|
| **Hedged Directional**     | Mostly matched inventory + small directional bet | Moderate clear signal              | Medium         | Excellent      |
| **Temporal Arbitrage**     | Build one leg now, hedge the other later         | One side becomes cheap first       | Low–Medium     | Very Good      |
| **Inventory Market Making**| Manage positions across multiple 15m windows     | Multiple active markets            | Medium         | Good           |
| **Dynamic Rotation**       | Switch between Up and Down as edge changes       | Strong flipping signals            | Higher         | Good           |
| **Near-Resolution**        | Buy near-certain outcome in final minutes        | Last 1–2 minutes                   | Low (tail risk)| Moderate       |

**Recommended Starting Approach**:  
**Hedged Directional + Temporal Arbitrage** hybrid  
- Keep most of the position matched (protected)  
- Add small directional imbalance based on net edge  
- Opportunistically build the cheaper leg first and hedge later

### 6. Execution Engine
- Use **limit orders** (prefer post-only)
- Split orders across multiple price levels
- Intelligent handling of partial fills (especially arbitrage legs)
- Fast cancellation of stale orders
- Dynamic reservation prices adjusted for inventory imbalance

### 7. Risk Management (Most Critical Layer)
- **Position Sizing**: Fractional Kelly (start with 10–25%)
- Hard Limits:
  - Maximum size per individual 15m market
  - Maximum net directional exposure
  - Maximum inventory imbalance (`|Up shares − Down shares|`)
  - Maximum correlated exposure (BTC + ETH + SOL)
- Daily loss limit + automatic kill switch
- Inventory rebalancing rules

---

## Implementation Recommendations

- **Backtesting is mandatory** — Simulate with historical order book data + realistic latency, partial fills, and slippage.
- **Start simple** — Begin with one 15m market using Hedged Directional + basic temporal arbitrage. Add cross-market logic later.
- **Data Sources**:
  - Binance (signals)
  - Polymarket WebSocket + Chainlink (resolution accuracy)
  - Useful starter repo: `FrondEnt/PolymarketBTC15mAssistant` (real-time data foundation)

---

## Expected Edge Sources Specific to 15m Markets

- Slower repricing of 15m markets compared to 5m after sharp moves
- Temporal arbitrage opportunities (longer window gives more time to build both legs)
- Relative value dislocations between consecutive or overlapping 15m windows
- Persistent liquidity imbalances

---

## Final Core Principle

> Your bot does **not** need to predict where Bitcoin will be in 15 minutes.  
> It only needs to be **faster and more disciplined** than other participants at calculating what “Up” and “Down” are worth *right now*, given current external price, time remaining, liquidity, and all costs — then executing cleanly and managing risk rigorously.

---

*Strategy Version 1.0 - July 2026*# 15m Bitcoin Up/Down Bot Strategy

**Inspired by Polymarket Short-Duration Market Bots**  
*(Based on analysis of @Dan1ro0’s thread - July 2026)*

---

## Analysis of the Source Post

The post by @Dan1ro0 provides a detailed breakdown of how profitable bots trade Polymarket’s “BTC Up or Down” markets (mainly 5m, with strong applicability to 15m).

**Key takeaway**: These bots do **not** primarily make money by predicting whether Bitcoin will go up or down. They profit by exploiting **temporary pricing inefficiencies** between the prediction market’s order book and real-time external BTC prices (stale orders, slow repricing).

The post outlines a complete professional trading pipeline used by bots generating significant monthly PnL.

---

## Core Lessons Learned

- Edge comes from **market microstructure**, not superior directional prediction.
- Bots follow a structured pipeline:  
  `Data → Signal → Fair Value → Executable Edge → Position Structure → Execution → Risk`
- Raw probability gaps must be adjusted for **fees, slippage, partial fills, and model uncertainty**.
- **Position construction** is as important as the signal (Temporal Arbitrage, Hedged Directional, Inventory Management, etc.).
- **Execution quality** and **strict risk management** often determine long-term survival.
- 15m markets reprice more slowly than 5m markets → larger and more persistent dislocations.

---

## Recommended Strategy: 15m BTC Up/Down Bot

### Goal
Build an automated system that systematically captures edge from pricing inefficiencies in Polymarket’s 15-minute BTC Up/Down markets by combining real-time external price data with disciplined order book execution and risk controls.

### Market Mechanics
- Each market has a clear **opening price** (BTC price at the start of the 15-minute window).
- Traders buy **Up** (BTC closes **above** opening price) or **Down**.
- Shares pay **$1** if correct, **$0** otherwise.
- Resolution uses Polymarket’s designated price feed (typically Chainlink or equivalent).

---

## Bot Architecture

### 1. Data Layer
- Real-time BTC price:
  - **Primary**: Binance spot/futures WebSocket (fastest)
  - **Secondary**: Polymarket resolution feed / Chainlink (for accuracy)
- Full Polymarket CLOB data (best bid/ask, depth, recent trades)
- Time remaining until resolution
- Related markets (concurrent 5m BTC, next 15m window, ETH/SOL 15m)
- Bot’s own orders, fills, and current inventory

### 2. Signal Engine
Convert data into actionable features:
- BTC price deviation from the market’s opening price (normalized by expected volatility × √time remaining)
- Short-term momentum (1m / 3m / 5m returns)
- Recent realized volatility
- Order book imbalance
- Aggressive trade flow
- Cross-market deviations (especially 15m vs 5m)
- Time-decay features

### 3. Pricing Model
Calculate **independent fair value** for Up and Down in the current 15m market.

- Output estimated probability **P(Up)**
- Use statistical models or lightweight ML (logistic regression / gradient boosting)
- Apply Bayesian-style updates when new signals arrive
- Optional: Track z-score between this 15m market and related 5m/15m markets

### 4. Executable Edge Calculation
Only trade when **net edge** exists after costs:

**Net Edge** = Fair Value − Expected VWAP Fill Price − Fees − Slippage Buffer − Model Uncertainty Buffer

### 5. Position Structure (Core Decision Layer)

| Position Structure          | Description                                      | Best Used When                     | Risk Level     | 15m Suitability |
|----------------------------|--------------------------------------------------|------------------------------------|----------------|-----------------|
| **Hedged Directional**     | Mostly matched inventory + small directional bet | Moderate clear signal              | Medium         | Excellent      |
| **Temporal Arbitrage**     | Build one leg now, hedge the other later         | One side becomes cheap first       | Low–Medium     | Very Good      |
| **Inventory Market Making**| Manage positions across multiple 15m windows     | Multiple active markets            | Medium         | Good           |
| **Dynamic Rotation**       | Switch between Up and Down as edge changes       | Strong flipping signals            | Higher         | Good           |
| **Near-Resolution**        | Buy near-certain outcome in final minutes        | Last 1–2 minutes                   | Low (tail risk)| Moderate       |

**Recommended Starting Approach**:  
**Hedged Directional + Temporal Arbitrage** hybrid  
- Keep most of the position matched (protected)  
- Add small directional imbalance based on net edge  
- Opportunistically build the cheaper leg first and hedge later

### 6. Execution Engine
- Use **limit orders** (prefer post-only)
- Split orders across multiple price levels
- Intelligent handling of partial fills (especially arbitrage legs)
- Fast cancellation of stale orders
- Dynamic reservation prices adjusted for inventory imbalance

### 7. Risk Management (Most Critical Layer)
- **Position Sizing**: Fractional Kelly (start with 10–25%)
- Hard Limits:
  - Maximum size per individual 15m market
  - Maximum net directional exposure
  - Maximum inventory imbalance (`|Up shares − Down shares|`)
  - Maximum correlated exposure (BTC + ETH + SOL)
- Daily loss limit + automatic kill switch
- Inventory rebalancing rules

---

## Implementation Recommendations

- **Backtesting is mandatory** — Simulate with historical order book data + realistic latency, partial fills, and slippage.
- **Start simple** — Begin with one 15m market using Hedged Directional + basic temporal arbitrage. Add cross-market logic later.
- **Data Sources**:
  - Binance (signals)
  - Polymarket WebSocket + Chainlink (resolution accuracy)
  - Useful starter repo: `FrondEnt/PolymarketBTC15mAssistant` (real-time data foundation)

---

## Expected Edge Sources Specific to 15m Markets

- Slower repricing of 15m markets compared to 5m after sharp moves
- Temporal arbitrage opportunities (longer window gives more time to build both legs)
- Relative value dislocations between consecutive or overlapping 15m windows
- Persistent liquidity imbalances

---

## Final Core Principle

> Your bot does **not** need to predict where Bitcoin will be in 15 minutes.  
> It only needs to be **faster and more disciplined** than other participants at calculating what “Up” and “Down” are worth *right now*, given current external price, time remaining, liquidity, and all costs — then executing cleanly and managing risk rigorously.

---

*Strategy Version 1.0 - July 2026*