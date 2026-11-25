"""Core data structures for markets, orderbooks, signals, and paper trades."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class OrderLevel:
    price: float
    size: float  # Number of shares available at this price


@dataclass(frozen=True)
class OrderBook:
    token_id: str
    bids: tuple[OrderLevel, ...]  # Sorted descending by price
    asks: tuple[OrderLevel, ...]  # Sorted ascending by price

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return self.best_ask if self.best_ask is not None else self.best_bid

    @property
    def spread(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return max(0.0, self.best_ask - self.best_bid)
        return None


@dataclass(frozen=True)
class FillSimulation:
    executable: bool
    vwap_price: float
    effective_price: float  # vwap_price + slippage_buffer
    filled_usd: float
    filled_shares: float
    best_bid: float | None
    best_ask: float | None
    mid_price: float
    spread: float
    reason: str = ""


@dataclass(frozen=True)
class MarketOutcome:
    market_id: str
    event_id: str
    event_title: str
    question: str
    group_item_title: str
    description: str
    yes_token_id: str
    no_token_id: str
    yes_price: float
    no_price: float
    volume: float
    liquidity: float
    end_date: str
    closed: bool = False
    resolved_yes: float | None = None  # 1.0 if YES won, 0.0 if NO won, None if unresolved


@dataclass(frozen=True)
class EventBundle:
    event_id: str
    slug: str
    title: str
    description: str
    neg_risk: bool
    end_date: str
    markets: tuple[MarketOutcome, ...]


@dataclass(frozen=True)
class TradeSignal:
    strategy: str
    event_id: str
    market_id: str
    token_id: str
    question: str
    outcome_label: str
    side: Literal["YES", "NO"]
    model_prob: float
    market_mid: float
    expected_edge: float
    description: str = ""
    notes: str = ""
    fixed_stake_usd: float | None = None  # Optional override (e.g. arbitrage legs)


@dataclass(frozen=True)
class LLMVerdict:
    approved: bool
    provider: str
    reason: str
    risk_flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PerformanceSummary:
    initial_bankroll: float
    cash_usd: float
    open_exposure_usd: float
    realized_pnl_usd: float
    total_trades: int
    open_trades: int
    closed_trades: int
    wins: int
    losses: int
    win_rate: float | None
    bot_brier: float | None
    market_brier: float | None
    brier_skill_score: float | None
    roi_pct: float
    by_strategy: dict[str, dict[str, float | int | None]]
