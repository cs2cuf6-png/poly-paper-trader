"""Configuration for the Polymarket paper-trading bot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class TraderConfig:
    # Paper bankroll & risk management
    initial_bankroll: float = 1000.0
    kelly_fraction: float = 0.25         # 1/4 Kelly to control variance
    max_position_pct: float = 0.05       # Max 5% of equity per single trade
    max_event_pct: float = 0.12          # Max 12% of equity per event
    min_stake_usd: float = 5.0           # Ignore dust trades below $5

    # Entry & orderbook quality filters
    min_edge: float = 0.06               # Require >= 6% edge over executable ask
    max_spread: float = 0.06             # Max effective spread (effective_fill - best_bid) <= 6c
    max_price_impact: float = 0.03       # Max VWAP slippage above best_ask <= 3c
    min_bid_price: float = 0.03          # Reject dead/collapsed bid books (< 3c)
    taker_fee_rate: float = 0.01         # 1.0% conservative taker fee / markup
    slippage_buffer: float = 0.005       # Conservative +0.5c fill penalty
    min_fill_ratio: float = 0.80         # Require >= 80% of target size available
    min_price: float = 0.04              # Avoid extreme lottery-ticket noise
    max_price: float = 0.94              # Avoid capped upside with high tail risk

    # Strategy-specific settings
    negrisk_min_margin: float = 0.025    # Sum of YES asks must be < 0.975
    weather_min_days_ahead: int = 1      # Skip same-day (lead_days=0) half-resolved markets
    weather_max_days_ahead: int = 3      # Forecast skill drops sharply past 3 days
    max_events_per_scan: int = 120

    # Storage
    db_path: Path = Path("data/paper_trader.db")

    # Optional free LLM rule-checker & parallel ensemble
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    nvidia_api_key: str = ""
    nvidia_model: str = "mistralai/mistral-large-4-1t-a52b"
    nvidia_native_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    llm_enabled: bool = True

    # Early take-profit exit & Telegram alerts
    early_exit_edge_buffer: float = 0.03  # Sell if best_bid >= model_prob + 3c
    early_exit_near_certain: float = 0.96 # Or if best_bid >= 96c
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    @classmethod
    def from_env(cls, db_path: str | Path | None = None) -> "TraderConfig":
        resolved_db = Path(db_path or os.getenv("DB_PATH", "data/paper_trader.db"))
        return cls(
            initial_bankroll=_env_float("INITIAL_BANKROLL", 1000.0),
            kelly_fraction=_env_float("KELLY_FRACTION", 0.25),
            max_position_pct=_env_float("MAX_POSITION_PCT", 0.05),
            max_event_pct=_env_float("MAX_EVENT_PCT", 0.12),
            min_stake_usd=_env_float("MIN_STAKE_USD", 5.0),
            min_edge=_env_float("MIN_EDGE", 0.06),
            max_spread=_env_float("MAX_SPREAD", 0.06),
            max_price_impact=_env_float("MAX_PRICE_IMPACT", 0.03),
            min_bid_price=_env_float("MIN_BID_PRICE", 0.03),
            taker_fee_rate=_env_float("TAKER_FEE_RATE", 0.01),
            slippage_buffer=_env_float("SLIPPAGE_BUFFER", 0.005),
            negrisk_min_margin=_env_float("NEGRISK_MIN_MARGIN", 0.025),
            weather_min_days_ahead=_env_int("WEATHER_MIN_DAYS_AHEAD", 1),
            weather_max_days_ahead=_env_int("WEATHER_MAX_DAYS_AHEAD", 3),
            max_events_per_scan=_env_int("MAX_EVENTS_PER_SCAN", 120),
            db_path=resolved_db,
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip(),
            nvidia_api_key=os.getenv("NVIDIA_API_KEY", "").strip(),
            nvidia_model=os.getenv("NVIDIA_MODEL", "mistralai/mistral-large-4-1t-a52b").strip(),
            nvidia_native_model=os.getenv("NVIDIA_NATIVE_MODEL", "nvidia/nemotron-3-ultra-550b-a55b").strip(),
            llm_enabled=os.getenv("LLM_ENABLED", "1").strip().lower() not in {"0", "false", "no"},
            early_exit_edge_buffer=_env_float("EARLY_EXIT_EDGE_BUFFER", 0.03),
            early_exit_near_certain=_env_float("EARLY_EXIT_NEAR_CERTAIN", 0.96),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
        )
