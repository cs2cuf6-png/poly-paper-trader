"""Strategy #1: Daily Weather Ensemble Model (Open-Meteo GFS + ECMWF + ICON).

Why this niche has the highest verifiable EV on a free/cron stack:
- Polymarket resolves daily high-temperature buckets against airport stations.
- Open-Meteo Ensemble API (100% free, no API key) provides ~120 physical weather
  model trajectories (GFS + ECMWF + ICON), giving an objective probability
  distribution rather than LLM guesswork.
- Markets resolve every 24h, yielding fast statistical validation (Brier Score).
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from ..config import TraderConfig
from ..models import EventBundle, MarketOutcome, TradeSignal
from ..polymarket import PolymarketClient, simulate_buy_fill

OPEN_METEO_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
USER_AGENT = "poly-paper-trader/0.1"


@dataclass(frozen=True)
class WeatherStation:
    city_key: str
    aliases: tuple[str, ...]
    icao: str
    lat: float
    lon: float
    default_unit: str  # "F" or "C"
    timezone_name: str


# Resolution stations commonly used on Polymarket temperature markets
WEATHER_STATIONS: tuple[WeatherStation, ...] = (
    WeatherStation("nyc", ("nyc", "new york", "new york city", "laguardia"), "KLGA", 40.7769, -73.8740, "F", "America/New_York"),
    WeatherStation("london", ("london",), "EGLC", 51.5053, 0.0553, "C", "Europe/London"),
    WeatherStation("seoul", ("seoul",), "RKSI", 37.4602, 126.4407, "C", "Asia/Seoul"),
    WeatherStation("chicago", ("chicago",), "KORD", 41.9742, -87.9073, "F", "America/Chicago"),
    WeatherStation("dallas", ("dallas",), "KDAL", 32.8471, -96.8518, "F", "America/Chicago"),
    WeatherStation("miami", ("miami",), "KMIA", 25.7959, -80.2870, "F", "America/New_York"),
    WeatherStation("atlanta", ("atlanta",), "KATL", 33.6407, -84.4277, "F", "America/New_York"),
    WeatherStation("seattle", ("seattle",), "KSEA", 47.4502, -122.3088, "F", "America/Los_Angeles"),
    WeatherStation("toronto", ("toronto",), "CYYZ", 43.6777, -79.6248, "C", "America/Toronto"),
    WeatherStation("buenos_aires", ("buenos aires",), "SAEZ", -34.8222, -58.5358, "C", "America/Argentina/Buenos_Aires"),
    WeatherStation("ankara", ("ankara",), "LTAC", 40.1281, 32.9951, "C", "Europe/Istanbul"),
    WeatherStation("paris", ("paris",), "LFPG", 49.0097, 2.5479, "C", "Europe/Paris"),
)

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


@dataclass(frozen=True)
class TempBucket:
    low: float | None   # Continuous lower bound (None = -inf)
    high: float | None  # Continuous upper bound (None = +inf)
    unit: str           # "F" or "C"


def detect_station(text: str) -> WeatherStation | None:
    lower = text.lower()
    for station in WEATHER_STATIONS:
        for alias in station.aliases:
            if re.search(rf"\b{re.escape(alias)}\b", lower):
                return station
    return None


def parse_target_date(text: str, today: date | None = None) -> date | None:
    ref = today or datetime.now(timezone.utc).date()

    iso_match = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text)
    if iso_match:
        try:
            return date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))
        except ValueError:
            return None

    m_pattern = r"\b(" + "|".join(MONTHS.keys()) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(20\d{2}))?\b"
    match = re.search(m_pattern, text, flags=re.IGNORECASE)
    if not match:
        return None

    month = MONTHS[match.group(1).lower()]
    day = int(match.group(2))
    year = int(match.group(3)) if match.group(3) else ref.year

    try:
        candidate = date(year, month, day)
    except ValueError:
        return None

    # Handle Dec -> Jan year rollover when year is omitted
    if not match.group(3):
        if (candidate - ref).days < -180:
            candidate = date(year + 1, month, day)
        elif (candidate - ref).days > 180:
            candidate = date(year - 1, month, day)
    return candidate


def parse_temp_bucket(label: str, default_unit: str = "F") -> TempBucket | None:
    """Parse Polymarket temperature bucket text into continuous [low, high) bounds.

    Polymarket resolves to rounded whole degrees, so integer bucket [50, 51] maps
    to continuous interval [49.5, 51.5).
    """
    text = label.replace("°", " ").replace("–", "-").replace("—", "-").strip()
    unit_match = re.search(r"\b([FC])\b", text, flags=re.IGNORECASE)
    unit = unit_match.group(1).upper() if unit_match else default_unit

    # Range: "50-51 F" or "50 to 51"
    range_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:-|to)\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if range_match:
        lo = float(range_match.group(1))
        hi = float(range_match.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return TempBucket(low=lo - 0.5, high=hi + 0.5, unit=unit)

    # Upper tail: "58 or higher", ">= 58", "58+", "above 57"
    if re.search(r"(or higher|or above|or more|>=|\+)", text, flags=re.IGNORECASE):
        num = re.search(r"(-?\d+(?:\.\d+)?)", text)
        if num:
            return TempBucket(low=float(num.group(1)) - 0.5, high=None, unit=unit)
    if re.search(r"\b(above|over|greater than|>)\b", text, flags=re.IGNORECASE):
        num = re.search(r"(-?\d+(?:\.\d+)?)", text)
        if num:
            return TempBucket(low=float(num.group(1)) + 0.5, high=None, unit=unit)

    # Lower tail: "49 or below", "49 or less", "<= 49", "below 50", "under 50"
    if re.search(r"(or below|or lower|or less|<=)", text, flags=re.IGNORECASE):
        num = re.search(r"(-?\d+(?:\.\d+)?)", text)
        if num:
            return TempBucket(low=None, high=float(num.group(1)) + 0.5, unit=unit)
    if re.search(r"\b(below|under|less than|<)\b", text, flags=re.IGNORECASE):
        num = re.search(r"(-?\d+(?:\.\d+)?)", text)
        if num:
            return TempBucket(low=None, high=float(num.group(1)) - 0.5, unit=unit)

    # Single whole-degree bucket: "12 C", "54 F"
    single = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*[FC]?", text, flags=re.IGNORECASE)
    if single:
        val = float(single.group(1))
        return TempBucket(low=val - 0.5, high=val + 0.5, unit=unit)

    return None


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def bucket_probabilities_from_ensemble(
    member_highs: list[float],
    buckets: list[TempBucket],
    lead_days: int = 1,
    unit: str = "F",
) -> list[float]:
    """Compute smoothed, normalized probabilities across mutually exclusive buckets.

    Uses Gaussian kernel density estimation over ensemble members so tail buckets
    receive calibrated non-zero probability rather than brittle raw histograms.
    """
    if not member_highs or not buckets:
        return []

    # Empirical forecast-error kernel bandwidth (scales gently with lead time)
    base_sigma = 0.85 if unit == "F" else 0.48
    sigma = base_sigma * (1.0 + 0.25 * max(0, lead_days - 1))

    raw_probs: list[float] = []
    n = float(len(member_highs))
    for bucket in buckets:
        prob_sum = 0.0
        for m in member_highs:
            hi_cdf = 1.0 if bucket.high is None else _normal_cdf(bucket.high, m, sigma)
            lo_cdf = 0.0 if bucket.low is None else _normal_cdf(bucket.low, m, sigma)
            prob_sum += max(0.0, hi_cdf - lo_cdf)
        raw_probs.append(prob_sum / n)

    total = sum(raw_probs)
    if total <= 1e-12:
        return [1.0 / len(buckets)] * len(buckets)
    return [p / total for p in raw_probs]


def fetch_ensemble_daily_highs(
    station: WeatherStation,
    target_date: date,
    unit: str,
    timeout: float = 15.0,
) -> list[float]:
    """Fetch daily max temperature across GFS, ECMWF, and ICON ensemble members."""
    temp_unit = "fahrenheit" if unit.upper() == "F" else "celsius"
    params = {
        "latitude": f"{station.lat:.4f}",
        "longitude": f"{station.lon:.4f}",
        "hourly": "temperature_2m",
        "models": "gfs_seamless,ecmwf_ifs025,icon_seamless",
        "temperature_unit": temp_unit,
        "timezone": station.timezone_name,
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
    }
    url = f"{OPEN_METEO_ENSEMBLE_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))

    hourly = data.get("hourly") or {}
    times: list[str] = hourly.get("time") or []
    date_prefix = target_date.isoformat()
    target_indices = [i for i, t in enumerate(times) if str(t).startswith(date_prefix)]
    if not target_indices:
        return []

    member_highs: list[float] = []
    for key, values in hourly.items():
        if not key.startswith("temperature_2m") or not isinstance(values, list):
            continue
        day_vals = [float(values[i]) for i in target_indices if i < len(values) and values[i] is not None]
        if day_vals:
            member_highs.append(max(day_vals))
    return member_highs


def fetch_metar_observed_temp(icao: str, unit: str = "F", timeout: float = 8.0) -> float | None:
    """Fetch latest airport METAR temperature from NOAA Aviation Weather (free, no key)."""
    url = f"https://aviationweather.gov/api/data/metar?ids={urllib.parse.quote(icao)}&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not isinstance(data, list) or not data:
            return None
        temp_c = data[0].get("temp")
        if temp_c is None:
            return None
        c_val = float(temp_c)
        return (c_val * 9.0 / 5.0 + 32.0) if unit.upper() == "F" else c_val
    except (urllib.error.URLError, TimeoutError, ValueError, TypeError, json.JSONDecodeError):
        return None


def apply_observed_temp_floor(
    buckets: list[TempBucket],
    probs: list[float],
    observed_temp: float,
) -> list[float]:
    """Zero out daily-high buckets strictly below an already-observed temperature."""
    adjusted: list[float] = []
    for bucket, p in zip(buckets, probs):
        if bucket.high is not None and observed_temp >= bucket.high:
            adjusted.append(0.0)
        else:
            adjusted.append(p)
    total = sum(adjusted)
    if total <= 1e-12:
        return probs
    return [p / total for p in adjusted]


def _is_weather_temp_event(event: EventBundle) -> bool:
    title = event.title.lower()
    return "temperature" in title and ("highest" in title or "high" in title)


def scan_weather_markets(
    client: PolymarketClient,
    events: list[EventBundle],
    config: TraderConfig,
    today: date | None = None,
    ensemble_fetcher: Any = fetch_ensemble_daily_highs,
) -> list[TradeSignal]:
    """Scan Polymarket daily temperature events against Open-Meteo weather ensembles."""
    ref_date = today or datetime.now(timezone.utc).date()
    signals: list[TradeSignal] = []

    for event in events:
        if not _is_weather_temp_event(event):
            continue
        station = detect_station(event.title + " " + event.description)
        if station is None:
            continue
        target_date = parse_target_date(event.title, today=ref_date)
        if target_date is None:
            continue

        lead_days = (target_date - ref_date).days
        if lead_days < config.weather_min_days_ahead or lead_days > config.weather_max_days_ahead:
            continue

        parsed_pairs: list[tuple[MarketOutcome, TempBucket]] = []
        unit = station.default_unit
        for market in event.markets:
            if market.closed:
                continue
            label = market.group_item_title or market.question
            bucket = parse_temp_bucket(label, default_unit=station.default_unit)
            if bucket is not None:
                unit = bucket.unit
                parsed_pairs.append((market, bucket))

        if len(parsed_pairs) < 2:
            continue

        try:
            member_highs = ensemble_fetcher(station, target_date, unit)
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
            continue

        if len(member_highs) < 10:
            continue

        buckets = [b for _, b in parsed_pairs]
        probs = bucket_probabilities_from_ensemble(member_highs, buckets, lead_days=lead_days, unit=unit)
        if lead_days == 0:
            obs_temp = fetch_metar_observed_temp(station.icao, unit=unit)
            if obs_temp is not None:
                probs = apply_observed_temp_floor(buckets, probs, obs_temp)
        mean_high = sum(member_highs) / len(member_highs)

        for (market, _bucket), prob_yes in zip(parsed_pairs, probs):
            # Evaluate YES side
            _evaluate_side(
                client=client,
                market=market,
                side="YES",
                token_id=market.yes_token_id,
                model_prob=prob_yes,
                config=config,
                notes=(
                    f"Weather {station.city_key.upper()}({station.icao}) {target_date.isoformat()}: "
                    f"n={len(member_highs)} ensemble mean={mean_high:.1f}{unit}"
                ),
                signals=signals,
            )
            # Evaluate NO side
            _evaluate_side(
                client=client,
                market=market,
                side="NO",
                token_id=market.no_token_id,
                model_prob=1.0 - prob_yes,
                config=config,
                notes=(
                    f"Weather {station.city_key.upper()}({station.icao}) {target_date.isoformat()}: "
                    f"n={len(member_highs)} ensemble mean={mean_high:.1f}{unit}"
                ),
                signals=signals,
            )

    return signals


def _evaluate_side(
    client: PolymarketClient,
    market: MarketOutcome,
    side: str,
    token_id: str,
    model_prob: float,
    config: TraderConfig,
    notes: str,
    signals: list[TradeSignal],
) -> None:
    try:
        book = client.fetch_orderbook(token_id)
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
        return

    probe = simulate_buy_fill(
        book=book,
        target_usd=max(config.min_stake_usd, 25.0),
        slippage_buffer=config.slippage_buffer,
        max_spread=config.max_spread,
        min_fill_ratio=config.min_fill_ratio,
        taker_fee_rate=config.taker_fee_rate,
        max_price_impact=config.max_price_impact,
        min_bid_price=config.min_bid_price,
    )
    if not probe.executable:
        return
    if not (config.min_price <= probe.effective_price <= config.max_price):
        return

    edge = model_prob - probe.effective_price
    if edge < config.min_edge:
        return

    signals.append(
        TradeSignal(
            strategy="weather_ensemble",
            event_id=market.event_id,
            market_id=market.market_id,
            token_id=token_id,
            question=market.question,
            outcome_label=market.group_item_title or market.question,
            side="YES" if side == "YES" else "NO",
            model_prob=model_prob,
            market_mid=probe.mid_price,
            expected_edge=edge,
            description=market.description,
            notes=notes,
        )
    )
