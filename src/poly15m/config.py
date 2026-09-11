"""Central configuration.

All settings are loaded from environment variables / a `.env` file (see
`.env.example`). Nothing here is required for Phase 0/1 (data recording is
read-only against public endpoints); the Polymarket credential fields only
become mandatory once the execution engine (Phase 4) is wired up.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="POLY15M_",
        extra="ignore",
    )

    # --- runtime -----------------------------------------------------
    environment: str = "development"
    log_level: str = "INFO"
    log_json: bool = True
    db_path: Path = REPO_ROOT / "var" / "poly15m.db"

    # --- optional: mirror all log output to a Telegram chat -----------
    # both must be set to enable forwarding; leave unset to disable
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    # --- Binance (public data feed) -----------------------------------
    binance_ws_base: str = "wss://stream.binance.com:9443"
    binance_symbol: str = "btcusdt"
    # rolling in-memory price buffer window, at ~1s resolution
    binance_buffer_seconds: int = 1800

    # --- Polymarket public data ----------------------------------------
    gamma_api_base: str = "https://gamma-api.polymarket.com"
    clob_rest_base: str = "https://clob.polymarket.com"
    clob_ws_base: str = "wss://ws-subscriptions-clob.polymarket.com/ws"
    # Gamma event slugs for this series look like "btc-updown-15m-<unix ts>"
    market_slug_prefix: str = "btc-updown-15m-"
    # Gamma series slug used to resolve the numeric series_id for scoped discovery
    market_series_slug: str = "btc-up-or-down-15m"
    market_window_seconds: int = 15 * 60
    market_poll_interval_seconds: float = 15.0

    # --- Phase 2 signal engine / fair value ----------------------------
    vol_lookback_seconds: int = 300
    vol_bar_seconds: float = 1.0
    vol_halflife_seconds: float = 60.0
    book_imbalance_depth: int = 10
    flow_lookback_seconds: float = 60.0
    clob_trade_buffer_seconds: float = 600.0
    fair_value_log_interval_seconds: float = 1.0
    divergence_alert_threshold: float = 0.05

    # --- Phase 3 executable edge / paper trading -----------------------
    slippage_buffer_bps: float = 50.0  # extra latency/adverse-selection buffer beyond the walked VWAP
    uncertainty_buffer_base: float = 0.05  # probability-scale buffer at full time-remaining, reference vol
    reference_sigma: float = 1.0  # $ per sqrt(second); scales uncertainty_buffer_base with current vol
    uncertainty_final_minute_seconds: float = 60.0
    uncertainty_final_minute_extra: float = 0.05  # added on top, ramping in over the final minute
    sim_fill_ratio: float = 0.9  # haircut on walked size, modeling competing order flow
    min_edge_to_trade: float = 0.05  # required net_edge (probability units) before paper-trading it
    # Shares per simulated order. This is *also* the size the book is
    # walked for when pricing an edge, so it must sit above the Kelly
    # stake it is meant to bound -- otherwise it silently becomes the
    # sizer and `kelly_fraction` stops doing anything (see
    # positions/manager.kelly_headroom). It must also track
    # `max_notional_per_market`: walking the book for far more size than
    # the cap will ever let us buy prices a fill we would never take, and
    # the resulting pessimistic VWAP silently suppresses trades.
    paper_trade_size: float = 20.0
    paper_max_position_per_market: float = 100.0  # shares per side, per market
    paper_min_order_size: float = 5.0  # matches Polymarket's live orderMinSize; also stops cap-tail dust orders

    # --- Phase 4 position manager ---------------------------------------
    bankroll: float = 1000.0  # capital base for Kelly sizing -- must match real funded capital before going live
    matched_arb_min_margin: float = 0.01  # min locked-in profit (probability units) to take a matched-pair trade

    # --- Phase 4 live execution (all unused/inert until explicitly enabled) --
    live_trading_enabled: bool = False  # hard gate -- live_trade.py refuses to place real orders unless True
    polymarket_chain_id: int = 137  # Polygon mainnet
    order_price_levels: int = 3  # split resting limit orders across this many price levels
    order_level_tick_multiplier: float = 1.0  # spacing between levels, in multiples of the market tick size
    reprice_threshold: float = 0.02  # cancel/replace a resting order once fair value drifts this far from its price
    order_poll_interval_seconds: float = 5.0  # local order-state reconciliation cadence

    # --- Phase 5 risk management -----------------------------------------
    # end-of-window handling: stop opening new directional risk in the
    # final `end_of_window_seconds`, except a tail-capped near-certain bet
    end_of_window_seconds: float = 120.0
    near_resolution_deviation_threshold: float = 3.0  # "many sigma"
    near_resolution_max_size: float = 10.0  # shares -- tail-risk cap for that exception
    polymarket_data_api_base: str = "https://data-api.polymarket.com"

    # --- Polymarket trading credentials (unused until Phase 4) --------
    polymarket_private_key: str | None = None
    polymarket_api_key: str | None = None
    polymarket_api_secret: str | None = None
    polymarket_api_passphrase: str | None = None
    polymarket_funder_address: str | None = None
    polymarket_signature_type: int = 1

    # --- fees (verified 2026-09-03 against Polymarket's published
    # schedule: help.polymarket.com/en/articles/13364478-trading-fees).
    # The taker fee is a function of price, not a flat rate:
    #     fee = shares * taker_fee_rate * p * (1 - p)
    # Crypto is the priciest category at 0.07 -- $1.75 per 100 shares at
    # 50c, tapering toward 0 at both extremes. Makers pay nothing.
    # See pricing/fees.py; do not reintroduce a flat-bps approximation.
    taker_fee_rate: float = 0.07
    maker_fee_bps: float = 0.0

    # --- fair-value calibration (Phase 6, item 24) --------------------
    # Coefficients are *fitted*, not configured: run `poly15m-calibrate
    # --write` to fit from recorded data and persist to calibration_path.
    # When the file is absent, fair value falls back to the raw analytic
    # Phi(deviation) -- so this is safe to leave enabled from day one.
    use_calibrated_fair_value: bool = True
    calibration_path: Path = REPO_ROOT / "var" / "calibration.json"

    # --- risk limits (Phase 5 gate; declared now so they live in one
    # place from day one) -----------------------------------------------
    # Sizing and the daily loss limit are ONE decision, not two. Setting
    # them independently is what produced the 2026-09-11 failure: eleven
    # `daily_loss_limit_breached` kill switches in nine days, three of
    # them on 2026-09-11 alone, one after only two trades.
    #
    # Measured over 387 resolved markets (2026-09-03..11): per-market PnL
    # mean +$0.91, sd $15.02, worst single market -$18.87, and 20% of
    # markets lose more than $15. Against a $25 limit that is a budget
    # 1.3 trades deep -- a two-strike rule, not a daily loss limit. The
    # projected 96-window day was mu +$88 / sigma $147, so the stop sat
    # 0.17 sigma below zero and tripped on ~70% of days *while the
    # strategy was profitable*.
    #
    # The rule: the limit must be many multiples of the worst plausible
    # single-market loss, which `max_notional_per_market` sets directly.
    # At a $6 cap the worst market is -$6.73 and the limit is ~15 trades
    # deep, tripping ~5% of days. Note the exchange's 5-share
    # `paper_min_order_size` floors how small this can go at all: even
    # with every order at the minimum, day sigma is $35.7 and the
    # smallest 5%-false-alarm limit is $54. A $25/day limit is not
    # reachable by any configuration -- if you want one, the bankroll has
    # to be ~$2,200, not $1,000.
    #
    # Fractional Kelly. Deliberately small: sized so the Kelly stake lands
    # *under* max_notional_per_market rather than being clipped by it, and
    # because full Kelly on a fair-value model measured at ~3x overconfident
    # would be ruinous. Raising this without raising the caps re-creates the
    # inert-Kelly bug -- kelly_headroom() will tell you. Conversely,
    # cutting the cap without cutting this re-creates it from the other
    # side: these two were divided by the same 3.33 so headroom is
    # unchanged at every price (0.24/0.35/0.57 at p=0.3/0.5/0.7).
    kelly_fraction: float = 0.015
    max_notional_per_market: float = 6.0
    # Both are *share* counts, not dollars. They exist to stop lopsided
    # directional books, not to bound capital -- max_notional_per_market
    # does that, and still binds first at any realistic price. They were
    # low enough to clip every Kelly-sized order to a flat 20 shares,
    # which is what made kelly_fraction inert; sized here to sit above a
    # typical stake so the notional cap is the constraint that actually
    # governs risk.
    max_net_directional_exposure: float = 60.0
    max_inventory_imbalance: float = 25.0
    daily_loss_limit: float = 100.0
    feed_staleness_seconds: float = 5.0

    # --- kill-switch behaviour -------------------------------------------
    # Re-arm the kill switch at the UTC day boundary instead of latching
    # forever. The module docstring in risk/limits.py used to argue that a
    # self-clearing loss limit is a foot-gun, and a *bare* one is. What
    # actually happened is worse: the switch latched, the operator
    # restarted the process, and the limit provided no protection at all
    # while still truncating every trading day. Auto re-arm plus the
    # escalation below is the honest version of what was already
    # happening manually -- with a hard halt that a restart cannot clear.
    kill_switch_auto_rearm: bool = True
    # Hard halt (no auto re-arm; requires a human) once the switch trips
    # on this many distinct UTC days inside the trailing window. At a ~5%
    # per-day false-alarm rate, 2 trips in 3 days is p ~ 0.007 -- that is
    # a broken model, not a losing streak.
    kill_switch_hard_halt_trips: int = 2
    kill_switch_hard_halt_window_days: int = 3

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
