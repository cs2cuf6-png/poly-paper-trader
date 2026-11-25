"""Unit tests for Polymarket paper-trading engine, orderbook sim, weather & NegRisk strategies."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from poly_paper.config import TraderConfig
from poly_paper.dashboard import render_dashboard_html
from poly_paper.engine import PaperTradingEngine, fractional_kelly_stake
from poly_paper.models import EventBundle, MarketOutcome, OrderBook, OrderLevel, TradeSignal
from poly_paper.polymarket import PolymarketClient, parse_market, simulate_buy_fill, simulate_sell_fill
from poly_paper.strategies.negrisk_arb import scan_negrisk_arbitrage
from poly_paper.strategies.science import (
    ScientificEvidence,
    calibrate_with_market_shrinkage,
    extract_search_query,
    is_science_event,
    scan_science_markets,
)
from poly_paper.strategies.weather import (
    apply_observed_temp_floor,
    bucket_probabilities_from_ensemble,
    detect_station,
    parse_target_date,
    parse_temp_bucket,
    scan_weather_markets,
)


class _StubClient(PolymarketClient):
    def __init__(
        self,
        books: dict[str, OrderBook] | None = None,
        markets: dict[str, MarketOutcome] | None = None,
    ) -> None:
        super().__init__()
        self._books = books or {}
        self._markets = markets or {}

    def fetch_orderbook(self, token_id: str, use_cache: bool = True) -> OrderBook:
        return self._books[token_id]

    def fetch_market(self, market_id: str) -> MarketOutcome | None:
        return self._markets.get(market_id)


class TestOrderbookSimulation(unittest.TestCase):
    def test_vwap_walk_and_slippage(self) -> None:
        book = OrderBook(
            token_id="tok",
            bids=(OrderLevel(0.48, 100.0),),
            asks=(OrderLevel(0.50, 20.0), OrderLevel(0.52, 50.0)),
        )
        # Target $20.40: fills 20 shares @ 0.50 ($10.00) + 20 shares @ 0.52 ($10.40) -> VWAP = 0.51
        sim = simulate_buy_fill(book, target_usd=20.40, slippage_buffer=0.005, max_spread=0.05)
        self.assertTrue(sim.executable)
        self.assertAlmostEqual(sim.vwap_price, 0.51, places=5)
        self.assertAlmostEqual(sim.effective_price, 0.515, places=5)

    def test_rejects_wide_spread(self) -> None:
        book = OrderBook(
            token_id="tok",
            bids=(OrderLevel(0.30, 100.0),),
            asks=(OrderLevel(0.45, 100.0),),
        )
        sim = simulate_buy_fill(book, target_usd=10.0, max_spread=0.08)
        self.assertFalse(sim.executable)
        self.assertIn("Spread too wide", sim.reason)

    def test_parse_market_json_strings(self) -> None:
        raw = {
            "id": "123",
            "question": "Will it rain?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.62", "0.38"]',
            "clobTokenIds": '["tok_y", "tok_n"]',
            "closed": False,
        }
        m = parse_market(raw, event_id="ev1", event_title="Rain")
        self.assertIsNotNone(m)
        assert m is not None
        self.assertEqual(m.yes_token_id, "tok_y")
        self.assertAlmostEqual(m.yes_price, 0.62)


class TestWeatherStrategy(unittest.TestCase):
    def test_station_date_and_bucket_parsing(self) -> None:
        st = detect_station("Highest temperature in NYC on November 26?")
        self.assertIsNotNone(st)
        assert st is not None
        self.assertEqual(st.icao, "KLGA")

        d = parse_target_date("Highest temperature in NYC on November 26?", today=date(2026, 11, 25))
        self.assertEqual(d, date(2026, 11, 26))

        b_low = parse_temp_bucket("49°F or below", "F")
        b_mid = parse_temp_bucket("50-51°F", "F")
        b_high = parse_temp_bucket("52°F or higher", "F")
        b_single = parse_temp_bucket("12°C", "C")
        self.assertEqual((b_low.low, b_low.high), (None, 49.5))
        self.assertEqual((b_mid.low, b_mid.high), (49.5, 51.5))
        self.assertEqual((b_high.low, b_high.high), (51.5, None))
        self.assertEqual((b_single.low, b_single.high), (11.5, 12.5))

    def test_ensemble_probabilities_sum_to_one(self) -> None:
        buckets = [
            parse_temp_bucket("49°F or below", "F"),
            parse_temp_bucket("50-51°F", "F"),
            parse_temp_bucket("52°F or higher", "F"),
        ]
        members = [50.4, 50.6, 50.2, 50.8, 50.5, 51.0, 49.8, 50.3, 50.7, 50.1]
        probs = bucket_probabilities_from_ensemble(members, buckets, lead_days=1, unit="F")  # type: ignore[arg-type]
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertGreater(probs[1], 0.60)

        # If observed METAR temperature is already 50.0F, bucket <=49F (high=49.5) is impossible
        floored = apply_observed_temp_floor(buckets, probs, observed_temp=50.0)  # type: ignore[arg-type]
        self.assertEqual(floored[0], 0.0)
        self.assertAlmostEqual(sum(floored), 1.0, places=6)

    def test_sell_fill_and_early_take_profit(self) -> None:
        book = OrderBook(
            token_id="tok",
            bids=(OrderLevel(0.85, 500.0),),
            asks=(OrderLevel(0.87, 500.0),),
        )
        sell_sim = simulate_sell_fill(book, shares_to_sell=50.0, slippage_buffer=0.005)
        self.assertTrue(sell_sim.executable)
        self.assertAlmostEqual(sell_sim.effective_price, 0.845, places=5)

        with tempfile.TemporaryDirectory() as tmp:
            cfg = TraderConfig(db_path=Path(tmp) / "tp.db", llm_enabled=False, slippage_buffer=0.005)
            engine = PaperTradingEngine(cfg)
            sig = TradeSignal("weather_ensemble", "ev1", "m1", "tok1", "Q?", "50-51F", "YES", 0.75, 0.45, 0.29)
            buy_book = OrderBook("tok1", (OrderLevel(0.44, 500),), (OrderLevel(0.46, 500),))
            tp_book = OrderBook("tok1", (OrderLevel(0.82, 500),), (OrderLevel(0.84, 500),))
            client = _StubClient(books={"tok1": buy_book})
            opened = engine.execute_signals([sig], client)
            self.assertEqual(len(opened), 1)

            # Price rallies to 0.82 >= 0.75 + 0.03 -> trigger early TP exit
            client._books["tok1"] = tp_book
            exited = engine.check_early_exits(client)
            self.assertEqual(len(exited), 1)
            self.assertGreater(exited[0]["pnl_usd"], 0.0)


class TestNegRiskArbitrage(unittest.TestCase):
    def test_detects_full_set_under_one_dollar(self) -> None:
        event = EventBundle(
            event_id="ev-arb",
            slug="arb",
            title="Who wins?",
            description="Mutually exclusive",
            neg_risk=True,
            end_date="2026-12-01",
            markets=(
                MarketOutcome("m1", "ev-arb", "Who wins?", "A?", "A", "", "a_y", "a_n", 0.30, 0.70, 5000, 1000, "2026-12-01"),
                MarketOutcome("m2", "ev-arb", "Who wins?", "B?", "B", "", "b_y", "b_n", 0.30, 0.70, 5000, 1000, "2026-12-01"),
                MarketOutcome("m3", "ev-arb", "Who wins?", "C?", "C", "", "c_y", "c_n", 0.30, 0.70, 5000, 1000, "2026-12-01"),
            ),
        )
        books = {
            "a_y": OrderBook("a_y", (OrderLevel(0.29, 200),), (OrderLevel(0.30, 200),)),
            "b_y": OrderBook("b_y", (OrderLevel(0.29, 200),), (OrderLevel(0.30, 200),)),
            "c_y": OrderBook("c_y", (OrderLevel(0.29, 200),), (OrderLevel(0.30, 200),)),
        }
        cfg = TraderConfig(slippage_buffer=0.005, negrisk_min_margin=0.025)
        signals = scan_negrisk_arbitrage(_StubClient(books=books), [event], cfg, bundle_budget_usd=30.0)
        self.assertEqual(len(signals), 3)
        self.assertAlmostEqual(sum(s.fixed_stake_usd or 0.0 for s in signals), 30.0, places=4)


class TestPaperEngineAndBrier(unittest.TestCase):
    def test_kelly_open_resolve_and_brier_score(self) -> None:
        stake = fractional_kelly_stake(0.70, 0.50, equity_usd=1000.0, kelly_fraction=0.25, max_position_pct=0.05)
        self.assertAlmostEqual(stake, 50.0, places=4)  # 0.25 * 0.40 = 0.10, capped at 0.05 -> $50

        with tempfile.TemporaryDirectory() as tmp:
            cfg = TraderConfig(db_path=Path(tmp) / "test.db", llm_enabled=False, slippage_buffer=0.005)
            engine = PaperTradingEngine(cfg)

            sig = TradeSignal(
                strategy="weather_ensemble",
                event_id="ev1",
                market_id="m1",
                token_id="tok1",
                question="Temp 50-51F?",
                outcome_label="50-51F",
                side="YES",
                model_prob=0.80,
                market_mid=0.45,
                expected_edge=0.34,
            )
            client = _StubClient(
                books={"tok1": OrderBook("tok1", (OrderLevel(0.44, 300),), (OrderLevel(0.46, 300),))},
                markets={
                    "m1": MarketOutcome(
                        "m1", "ev1", "Temp", "Temp 50-51F?", "50-51F", "",
                        "tok1", "tok1n", 1.0, 0.0, 1000, 0, "2026-11-26",
                        closed=True, resolved_yes=1.0,
                    )
                },
            )
            opened = engine.execute_signals([sig], client)
            self.assertEqual(len(opened), 1)
            self.assertLess(engine.get_cash(), 1000.0)

            settled = engine.resolve_open_trades(client)
            self.assertEqual(len(settled), 1)
            self.assertGreater(settled[0]["pnl_usd"], 0.0)

            summary = engine.summarize()
            self.assertEqual(summary.closed_trades, 1)
            self.assertIsNotNone(summary.bot_brier)
            self.assertIsNotNone(summary.market_brier)
            assert summary.bot_brier is not None and summary.market_brier is not None
            # Bot (0.80 -> 1.0 => 0.04) should beat Market (0.45 -> 1.0 => 0.3025)
            self.assertLess(summary.bot_brier, summary.market_brier)
            self.assertGreater(summary.brier_skill_score or 0.0, 0.80)

            html_doc = render_dashboard_html(engine)
            self.assertIn("Polymarket High-EV Paper Trader", html_doc)
            self.assertIn("BEATING MARKET", html_doc)
            self.assertEqual(len(engine.list_trades(status="CLOSED")), 1)

            engine.reset_portfolio(initial_bankroll=500.0)
            self.assertEqual(engine.summarize().total_trades, 0)
            self.assertAlmostEqual(engine.get_cash(), 500.0)


class TestScienceStrategy(unittest.TestCase):
    def test_science_detection_shrinkage_and_signal(self) -> None:
        event = EventBundle(
            event_id="ev-fda",
            slug="fda-approve-crispr-drug",
            title="Will the FDA approve CRISPR drug X in 2026?",
            description="Resolves YES if FDA approves NDA for drug X.",
            neg_risk=False,
            end_date="2026-12-31",
            markets=(
                MarketOutcome(
                    "m-fda", "ev-fda", "FDA CRISPR", "Will the FDA approve CRISPR drug X in 2026?",
                    "Yes", "Resolves YES if FDA approves.", "fy", "fn", 0.40, 0.60, 25000, 5000, "2026-12-31"
                ),
            ),
        )
        self.assertTrue(is_science_event(event))
        self.assertIn("CRISPR", extract_search_query(event.title))

        # Shrinkage pulls raw 0.80 toward market 0.40 -> 0.65*0.80 + 0.35*0.40 = 0.66
        cal = calibrate_with_market_shrinkage(0.80, 0.40, "high")
        self.assertAlmostEqual(cal, 0.66, places=5)

        books = {
            "fy": OrderBook("fy", (OrderLevel(0.39, 200),), (OrderLevel(0.41, 200),)),
            "fn": OrderBook("fn", (OrderLevel(0.59, 200),), (OrderLevel(0.61, 200),)),
        }
        cfg = TraderConfig(slippage_buffer=0.005, min_edge=0.06)
        fake_ev = ScientificEvidence("CRISPR", ("NCT001 [PHASE3 | COMPLETED]",), ("NDA123: AP",), ())
        signals = scan_science_markets(
            client=_StubClient(books=books),
            events=[event],
            config=cfg,
            evidence_fetcher=lambda _m: fake_ev,
            llm_estimator=lambda _m, _e, _mid, _c: (0.80, "high", "Phase 3 completed with positive readout"),
        )
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].strategy, "science_research")
        self.assertEqual(signals[0].side, "YES")
        self.assertGreater(signals[0].expected_edge, 0.20)


if __name__ == "__main__":
    unittest.main()
