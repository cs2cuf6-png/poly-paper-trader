"""Category #2: Crypto Price-Target Log-Normal Model (BTC, ETH, SOL, XRP).

Fetches live spot prices from free public APIs (Coinbase / CoinGecko, no key)
and computes risk-neutral log-normal probabilities P(S_T >= K) or P(S_T <= K)
for Polymarket crypto price-target markets.
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from ..config import TraderConfig
from ..models import EventBundle, MarketOutcome, TradeSignal
from ..polymarket import PolymarketClient, simulate_buy_fill

USER_AGENT = "poly-paper-trader/0.1"

# Annualized realized volatility priors
ASSETS: dict[str, tuple[tuple[str, ...], str, float]] = {
    "BTC": (("bitcoin", "btc"), "BTC-USD", 0.55),
    "ETH": (("ethereum", "eth"), "ETH-USD", 0.68),
    "SOL": (("solana", "sol"), "SOL-USD", 0.85),
    "XRP": (("xrp", "ripple"), "XRP-USD", 0.80),
}


def fetch_spot_price(pair: str, timeout: float = 8.0) -> float | None:
    """Fetch live spot price from Coinbase public API (no key required)."""
    url = f"https://api.coinbase.com/v2/prices/{pair}/spot"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return float(data["data"]["amount"])
    except (urllib.error.URLError, TimeoutError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def lognormal_touch_prob(spot: float, strike: float, days_ahead: float, vol_annual: float, direction: str) -> float:
    """Estimate terminal probability P(S_T >= strike) or P(S_T <= strike) under log-normal GBM."""
    if spot <= 0 or strike <= 0 or days_ahead <= 0:
        return 0.5
    t = max(days_ahead / 365.0, 1.0 / 365.0)
    sigma_t = vol_annual * math.sqrt(t)
    d2 = (math.log(spot / strike) - 0.5 * (vol_annual ** 2) * t) / sigma_t
    p_above = _norm_cdf(d2)
    p = p_above if direction == "above" else (1.0 - p_above)
    return max(0.02, min(0.98, p))


def parse_crypto_target(question: str) -> tuple[str, float, str] | None:
    """Parse (asset, strike_usd, direction) from a Polymarket crypto price question."""
    lower = question.lower()
    asset_found: str | None = None
    for sym, (aliases, _, _) in ASSETS.items():
        if any(re.search(rf"\b{re.escape(a)}\b", lower) for a in aliases):
            asset_found = sym
            break
    if asset_found is None:
        return None

    direction = None
    if re.search(r"\b(above|over|exceed|reach|hit|higher than|>=|>)\b", lower):
        direction = "above"
    elif re.search(r"\b(below|under|dip|drop|lower than|<=|<)\b", lower):
        direction = "below"
    if direction is None:
        return None

    price_match = re.search(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*([kKmM])?", question)
    if not price_match:
        return None
    raw_num = float(price_match.group(1).replace(",", ""))
    suffix = (price_match.group(2) or "").lower()
    if suffix == "k":
        raw_num *= 1000.0
    elif suffix == "m":
        raw_num *= 1_000_000.0
    if raw_num <= 0:
        return None
    return asset_found, raw_num, direction


def _days_until(end_date_str: str, now: datetime) -> float | None:
    if not end_date_str:
        return None
    try:
        cleaned = end_date_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        days = (dt - now).total_seconds() / 86400.0
        return days if 0.25 <= days <= 45.0 else None
    except ValueError:
        return None


def scan_crypto_markets(
    client: PolymarketClient,
    events: list[EventBundle],
    config: TraderConfig,
    now: datetime | None = None,
    spot_fetcher: Any = fetch_spot_price,
) -> list[TradeSignal]:
    """Scan Polymarket crypto price-target markets using live spot + log-normal model."""
    ref_now = now or datetime.now(timezone.utc)
    signals: list[TradeSignal] = []
    spot_cache: dict[str, float | None] = {}

    for event in events:
        for market in event.markets:
            if market.closed:
                continue
            parsed = parse_crypto_target(market.question)
            if parsed is None:
                continue
            asset, strike, direction = parsed
            days = _days_until(market.end_date or event.end_date, ref_now)
            if days is None:
                continue

            _, pair, vol = ASSETS[asset]
            if pair not in spot_cache:
                spot_cache[pair] = spot_fetcher(pair)
            spot = spot_cache[pair]
            if spot is None:
                continue

            prob_yes = lognormal_touch_prob(spot, strike, days, vol, direction)
            notes = f"Crypto {asset} spot=${spot:,.0f} vs ${strike:,.0f} ({direction}, {days:.1f}d, vol={vol:.0%})"

            for side, tok, p_model in (("YES", market.yes_token_id, prob_yes), ("NO", market.no_token_id, 1.0 - prob_yes)):
                try:
                    book = client.fetch_orderbook(tok)
                except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
                    continue
                sim = simulate_buy_fill(
                    book=book,
                    target_usd=max(config.min_stake_usd, 25.0),
                    slippage_buffer=config.slippage_buffer,
                    max_spread=config.max_spread,
                    min_fill_ratio=config.min_fill_ratio,
                    taker_fee_rate=config.taker_fee_rate,
                    max_price_impact=config.max_price_impact,
                    min_bid_price=config.min_bid_price,
                )
                if not sim.executable or not (config.min_price <= sim.effective_price <= config.max_price):
                    continue
                edge = p_model - sim.effective_price
                if edge >= config.min_edge:
                    signals.append(
                        TradeSignal(
                            strategy="crypto_lognormal",
                            event_id=market.event_id,
                            market_id=market.market_id,
                            token_id=tok,
                            question=market.question,
                            outcome_label=market.group_item_title or market.question,
                            side=side,
                            model_prob=p_model,
                            market_mid=sim.mid_price,
                            expected_edge=edge,
                            description=market.description,
                            notes=notes,
                        )
                    )

    return signals
