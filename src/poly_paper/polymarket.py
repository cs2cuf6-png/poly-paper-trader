"""Polymarket Gamma & CLOB API client with realistic orderbook fill simulation."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .models import EventBundle, FillSimulation, MarketOutcome, OrderBook, OrderLevel

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
USER_AGENT = "poly-paper-trader/0.1 (+https://github.com/local/poly-paper-trader)"


def _http_get_json(url: str, timeout: float = 15.0) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def _parse_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_market(raw: dict[str, Any], event_id: str = "", event_title: str = "") -> MarketOutcome | None:
    """Parse a raw Polymarket Gamma market dict into MarketOutcome."""
    outcomes = [str(x).strip() for x in _parse_json_list(raw.get("outcomes"))]
    prices = [_to_float(x, -1.0) for x in _parse_json_list(raw.get("outcomePrices"))]
    token_ids = [str(x).strip() for x in _parse_json_list(raw.get("clobTokenIds"))]

    if len(outcomes) < 2 or len(token_ids) < 2:
        return None

    # Locate Yes/No indices (usually [0, 1], but never assume)
    yes_idx = 0
    no_idx = 1
    for i, label in enumerate(outcomes):
        lower = label.lower()
        if lower == "yes":
            yes_idx = i
        elif lower == "no":
            no_idx = i

    yes_price = prices[yes_idx] if yes_idx < len(prices) and prices[yes_idx] >= 0 else 0.5
    no_price = prices[no_idx] if no_idx < len(prices) and prices[no_idx] >= 0 else max(0.0, 1.0 - yes_price)

    closed = bool(raw.get("closed", False))
    resolved_yes: float | None = None
    if closed:
        if yes_price >= 0.99:
            resolved_yes = 1.0
        elif yes_price <= 0.01:
            resolved_yes = 0.0

    return MarketOutcome(
        market_id=str(raw.get("id", "")),
        event_id=event_id or str(raw.get("eventId", "")),
        event_title=event_title or str(raw.get("question", "")),
        question=str(raw.get("question", "")),
        group_item_title=str(raw.get("groupItemTitle", "") or ""),
        description=str(raw.get("description", "") or ""),
        yes_token_id=token_ids[yes_idx],
        no_token_id=token_ids[no_idx],
        yes_price=yes_price,
        no_price=no_price,
        volume=_to_float(raw.get("volumeNum", raw.get("volume", 0.0))),
        liquidity=_to_float(raw.get("liquidityNum", raw.get("liquidity", 0.0))),
        end_date=str(raw.get("endDate", "") or ""),
        closed=closed,
        resolved_yes=resolved_yes,
    )


def parse_event(raw: dict[str, Any]) -> EventBundle | None:
    event_id = str(raw.get("id", ""))
    title = str(raw.get("title", "") or "")
    slug = str(raw.get("slug", "") or "")
    description = str(raw.get("description", "") or "")
    end_date = str(raw.get("endDate", "") or "")
    neg_risk = bool(raw.get("negRisk", False) or raw.get("enableNegRisk", False))

    raw_markets = raw.get("markets") or []
    markets: list[MarketOutcome] = []
    for rm in raw_markets:
        if not isinstance(rm, dict):
            continue
        if rm.get("active") is False:
            continue
        parsed = parse_market(rm, event_id=event_id, event_title=title)
        if parsed is not None:
            markets.append(parsed)

    if not markets:
        return None

    return EventBundle(
        event_id=event_id,
        slug=slug,
        title=title,
        description=description,
        neg_risk=neg_risk,
        end_date=end_date,
        markets=tuple(markets),
    )


def simulate_buy_fill(
    book: OrderBook,
    target_usd: float,
    slippage_buffer: float = 0.005,
    max_spread: float = 0.06,
    min_fill_ratio: float = 0.80,
    taker_fee_rate: float = 0.01,
    max_price_impact: float = 0.03,
    min_bid_price: float = 0.03,
) -> FillSimulation:
    """Walk the real CLOB ask book with fees, slippage, and effective-spread checks."""
    best_bid = book.best_bid
    best_ask = book.best_ask
    mid_price = book.mid_price if book.mid_price is not None else 0.5
    spread = book.spread if book.spread is not None else 1.0

    if not book.asks or best_ask is None:
        return FillSimulation(
            executable=False,
            vwap_price=1.0,
            effective_price=1.0,
            filled_usd=0.0,
            filled_shares=0.0,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=spread,
            reason="Empty ask orderbook",
        )

    if best_bid is None or best_bid < min_bid_price:
        return FillSimulation(
            executable=False,
            vwap_price=best_ask,
            effective_price=min(0.995, best_ask * (1.0 + taker_fee_rate) + slippage_buffer),
            filled_usd=0.0,
            filled_shares=0.0,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=spread,
            reason=f"Dead or missing bid ({best_bid} < {min_bid_price:.3f})",
        )

    if spread > max_spread:
        return FillSimulation(
            executable=False,
            vwap_price=best_ask,
            effective_price=min(0.995, best_ask * (1.0 + taker_fee_rate) + slippage_buffer),
            filled_usd=0.0,
            filled_shares=0.0,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=spread,
            reason=f"Top-of-book spread too wide ({spread:.3f} > {max_spread:.3f})",
        )

    remaining_usd = max(0.0, target_usd)
    filled_usd = 0.0
    filled_shares = 0.0

    for level in book.asks:
        if remaining_usd <= 1e-9:
            break
        if level.price <= 0.0 or level.size <= 0.0:
            continue
        level_usd = level.price * level.size
        if level_usd >= remaining_usd:
            shares = remaining_usd / level.price
            filled_shares += shares
            filled_usd += remaining_usd
            remaining_usd = 0.0
        else:
            filled_shares += level.size
            filled_usd += level_usd
            remaining_usd -= level_usd

    if filled_shares <= 0.0 or filled_usd < target_usd * min_fill_ratio:
        return FillSimulation(
            executable=False,
            vwap_price=best_ask,
            effective_price=min(0.995, best_ask * (1.0 + taker_fee_rate) + slippage_buffer),
            filled_usd=filled_usd,
            filled_shares=filled_shares,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=spread,
            reason=f"Insufficient ask depth (${filled_usd:.2f} < ${target_usd * min_fill_ratio:.2f})",
        )

    vwap = filled_usd / filled_shares
    price_impact = vwap - best_ask
    if price_impact > max_price_impact:
        return FillSimulation(
            executable=False,
            vwap_price=vwap,
            effective_price=min(0.995, vwap * (1.0 + taker_fee_rate) + slippage_buffer),
            filled_usd=0.0,
            filled_shares=0.0,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=spread,
            reason=f"VWAP price impact too high ({price_impact:.3f} > {max_price_impact:.3f})",
        )

    effective_price = min(0.995, vwap * (1.0 + taker_fee_rate) + slippage_buffer)
    effective_spread = effective_price - best_bid
    if effective_spread > max_spread + 0.02:
        return FillSimulation(
            executable=False,
            vwap_price=vwap,
            effective_price=effective_price,
            filled_usd=0.0,
            filled_shares=0.0,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=effective_spread,
            reason=f"Effective spread after fees/depth too wide ({effective_spread:.3f})",
        )

    effective_shares = filled_usd / effective_price

    return FillSimulation(
        executable=True,
        vwap_price=vwap,
        effective_price=effective_price,
        filled_usd=filled_usd,
        filled_shares=effective_shares,
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid_price,
        spread=effective_spread,
        reason="OK",
    )


def simulate_sell_fill(
    book: OrderBook,
    shares_to_sell: float,
    slippage_buffer: float = 0.005,
    max_spread: float = 0.06,
    min_fill_ratio: float = 0.90,
    taker_fee_rate: float = 0.01,
) -> FillSimulation:
    """Walk the real CLOB bid book to compute an honest VWAP sell execution with fees."""
    best_bid = book.best_bid
    best_ask = book.best_ask
    mid_price = book.mid_price if book.mid_price is not None else 0.5
    spread = book.spread if book.spread is not None else 1.0

    if not book.bids or best_bid is None or shares_to_sell <= 0:
        return FillSimulation(
            executable=False,
            vwap_price=0.0,
            effective_price=0.0,
            filled_usd=0.0,
            filled_shares=0.0,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=spread,
            reason="Empty bid orderbook",
        )

    if spread > max_spread:
        return FillSimulation(
            executable=False,
            vwap_price=best_bid,
            effective_price=max(0.005, best_bid * (1.0 - taker_fee_rate) - slippage_buffer),
            filled_usd=0.0,
            filled_shares=0.0,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=spread,
            reason=f"Spread too wide ({spread:.3f})",
        )

    remaining_shares = shares_to_sell
    filled_shares = 0.0
    filled_usd = 0.0

    for level in book.bids:
        if remaining_shares <= 1e-9:
            break
        if level.price <= 0.0 or level.size <= 0.0:
            continue
        take = min(remaining_shares, level.size)
        filled_shares += take
        filled_usd += take * level.price
        remaining_shares -= take

    if filled_shares < shares_to_sell * min_fill_ratio or filled_shares <= 0.0:
        return FillSimulation(
            executable=False,
            vwap_price=best_bid,
            effective_price=max(0.005, best_bid * (1.0 - taker_fee_rate) - slippage_buffer),
            filled_usd=filled_usd,
            filled_shares=filled_shares,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=spread,
            reason="Insufficient bid depth",
        )

    vwap = filled_usd / filled_shares
    effective_price = max(0.005, vwap * (1.0 - taker_fee_rate) - slippage_buffer)
    effective_usd = filled_shares * effective_price

    return FillSimulation(
        executable=True,
        vwap_price=vwap,
        effective_price=effective_price,
        filled_usd=effective_usd,
        filled_shares=filled_shares,
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid_price,
        spread=spread,
        reason="OK",
    )


class PolymarketClient:
    """Thin, dependency-free client for Polymarket Gamma & CLOB APIs."""

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout
        self._book_cache: dict[str, OrderBook] = {}

    def fetch_active_events(
        self,
        limit: int = 100,
        tag_slug: str | None = None,
        order: str = "volume24hr",
    ) -> list[EventBundle]:
        events: list[EventBundle] = []
        offset = 0
        page_size = min(50, limit)

        while len(events) < limit:
            params: dict[str, str | int] = {
                "active": "true",
                "closed": "false",
                "limit": page_size,
                "offset": offset,
                "order": order,
                "ascending": "false",
            }
            if tag_slug:
                params["tag_slug"] = tag_slug

            url = f"{GAMMA_BASE_URL}/events?{urllib.parse.urlencode(params)}"
            data = _http_get_json(url, timeout=self.timeout)
            if not isinstance(data, list) or not data:
                break

            for raw_event in data:
                if not isinstance(raw_event, dict):
                    continue
                parsed = parse_event(raw_event)
                if parsed is not None:
                    events.append(parsed)
                    if len(events) >= limit:
                        break

            if len(data) < page_size:
                break
            offset += page_size

        return events

    def fetch_orderbook(self, token_id: str, use_cache: bool = True) -> OrderBook:
        if use_cache and token_id in self._book_cache:
            return self._book_cache[token_id]

        url = f"{CLOB_BASE_URL}/book?{urllib.parse.urlencode({'token_id': token_id})}"
        data = _http_get_json(url, timeout=self.timeout)

        bids = sorted(
            (
                OrderLevel(price=_to_float(x.get("price")), size=_to_float(x.get("size")))
                for x in (data.get("bids") or [])
                if _to_float(x.get("price")) > 0 and _to_float(x.get("size")) > 0
            ),
            key=lambda lvl: lvl.price,
            reverse=True,
        )
        asks = sorted(
            (
                OrderLevel(price=_to_float(x.get("price")), size=_to_float(x.get("size")))
                for x in (data.get("asks") or [])
                if _to_float(x.get("price")) > 0 and _to_float(x.get("size")) > 0
            ),
            key=lambda lvl: lvl.price,
        )

        book = OrderBook(token_id=token_id, bids=tuple(bids), asks=tuple(asks))
        self._book_cache[token_id] = book
        return book

    def fetch_market(self, market_id: str) -> MarketOutcome | None:
        url = f"{GAMMA_BASE_URL}/markets/{urllib.parse.quote(market_id)}"
        try:
            data = _http_get_json(url, timeout=self.timeout)
        except urllib.error.HTTPError:
            return None
        if not isinstance(data, dict):
            return None
        return parse_market(data)
