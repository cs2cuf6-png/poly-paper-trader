"""Strategy #2: Structural NegRisk Arbitrage Scanner.

In a mutually exclusive multi-outcome event (negRisk=True), exactly one outcome
resolves to $1.00 and all others to $0.00. If the sum of executable YES ask prices
across ALL outcomes is strictly below $1.00 - margin (accounting for orderbook
depth and slippage), buying equal shares of every outcome locks in positive EV
regardless of who wins.
"""

from __future__ import annotations

import urllib.error

from ..config import TraderConfig
from ..models import EventBundle, FillSimulation, MarketOutcome, TradeSignal
from ..polymarket import PolymarketClient, simulate_buy_fill


def scan_negrisk_arbitrage(
    client: PolymarketClient,
    events: list[EventBundle],
    config: TraderConfig,
    bundle_budget_usd: float = 40.0,
) -> list[TradeSignal]:
    """Scan negRisk events for full-set YES arbitrage under $1.00."""
    signals: list[TradeSignal] = []

    for event in events:
        if not event.neg_risk:
            continue
        active_markets = [m for m in event.markets if not m.closed]
        if len(active_markets) < 2 or len(active_markets) > 25:
            continue

        # Fast pre-filter using Gamma indicative prices before hitting CLOB API
        indicative_sum = sum(m.yes_price for m in active_markets)
        if indicative_sum <= 0.10 or indicative_sum >= 1.02:
            continue

        legs: list[tuple[MarketOutcome, FillSimulation]] = []
        valid_bundle = True
        for market in active_markets:
            try:
                book = client.fetch_orderbook(market.yes_token_id)
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
                valid_bundle = False
                break

            est_leg_usd = max(config.min_stake_usd, bundle_budget_usd * max(market.yes_price, 0.05))
            sim = simulate_buy_fill(
                book=book,
                target_usd=est_leg_usd,
                slippage_buffer=config.slippage_buffer,
                max_spread=config.max_spread,
                min_fill_ratio=config.min_fill_ratio,
            )
            if not sim.executable:
                valid_bundle = False
                break
            legs.append((market, sim))

        if not valid_bundle or len(legs) != len(active_markets):
            continue

        total_ask = sum(sim.effective_price for _, sim in legs)
        margin = 1.0 - total_ask
        if margin < config.negrisk_min_margin:
            continue

        # Equal share count across all legs guarantees $1.00 payout per share
        shares_per_leg = bundle_budget_usd / total_ask
        for market, sim in legs:
            # Normalized implied fair prob so Brier Score remains meaningful
            implied_prob = sim.effective_price / total_ask
            leg_stake = shares_per_leg * sim.effective_price
            signals.append(
                TradeSignal(
                    strategy="negrisk_arb",
                    event_id=event.event_id,
                    market_id=market.market_id,
                    token_id=market.yes_token_id,
                    question=market.question,
                    outcome_label=market.group_item_title or market.question,
                    side="YES",
                    model_prob=implied_prob,
                    market_mid=sim.mid_price,
                    expected_edge=margin / len(legs),
                    description=market.description,
                    notes=f"NegRisk full-set arb: sum_ask={total_ask:.4f}, locked_margin={margin:.4f}",
                    fixed_stake_usd=leg_stake,
                )
            )

    return signals
