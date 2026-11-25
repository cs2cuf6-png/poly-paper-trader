"""High-EV strategies for Polymarket paper trading."""

from .negrisk_arb import scan_negrisk_arbitrage
from .science import scan_science_markets
from .weather import scan_weather_markets

__all__ = ["scan_weather_markets", "scan_negrisk_arbitrage", "scan_science_markets"]
