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
    min_edge_to_trade: float = 0.02  # required net_edge (probability units) before paper-trading it
    paper_trade_size: float = 20.0  # shares per simulated order
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

    # --- fees (placeholders -- verify against the live market's
    # feeSchedule before trading; Polymarket's crypto-market fee schedule
    # is a function of price, not a flat rate) -------------------------
    taker_fee_bps: float = 200.0
    maker_fee_bps: float = 0.0

    # --- risk limits (Phase 5 gate; declared now so they live in one
    # place from day one) -----------------------------------------------
    kelly_fraction: float = 0.15
    max_notional_per_market: float = 20.0
    max_net_directional_exposure: float = 50.0
    max_inventory_imbalance: float = 20.0
    daily_loss_limit: float = 25.0
    feed_staleness_seconds: float = 5.0

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
