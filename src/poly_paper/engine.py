"""SQLite paper-trading engine, Fractional Kelly sizer, resolution tracker, and Brier analytics."""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import TraderConfig
from .llm_analyst import LLMAnalyst
from .models import PerformanceSummary, TradeSignal
from .polymarket import PolymarketClient, simulate_buy_fill, simulate_sell_fill


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fractional_kelly_stake(
    model_prob: float,
    price: float,
    equity_usd: float,
    kelly_fraction: float,
    max_position_pct: float,
) -> float:
    """Compute position size in USD for a binary contract priced at `price`."""
    if not (0.0 < price < 1.0) or not (0.0 < model_prob < 1.0):
        return 0.0
    if model_prob <= price:
        return 0.0
    full_kelly = (model_prob - price) / (1.0 - price)
    frac = min(max(0.0, full_kelly * kelly_fraction), max_position_pct)
    return equity_usd * frac


class PaperTradingEngine:
    """Persistent SQLite paper-trading engine with honest execution & Brier tracking."""

    def __init__(self, config: TraderConfig) -> None:
        self.config = config
        self.db_path: Path = config.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    initial_bankroll REAL NOT NULL,
                    cash_usd REAL NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    token_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    outcome_label TEXT NOT NULL,
                    side TEXT NOT NULL,
                    model_prob REAL NOT NULL,
                    market_mid REAL NOT NULL,
                    fill_price REAL NOT NULL,
                    shares REAL NOT NULL,
                    stake_usd REAL NOT NULL,
                    expected_edge REAL NOT NULL,
                    opened_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    resolved_outcome REAL,
                    exit_price REAL,
                    pnl_usd REAL,
                    closed_at TEXT,
                    llm_check TEXT,
                    notes TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
                CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_id, side, status);

                CREATE TABLE IF NOT EXISTS scan_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ran_at TEXT NOT NULL,
                    events_scanned INTEGER NOT NULL,
                    signals_found INTEGER NOT NULL,
                    trades_opened INTEGER NOT NULL,
                    resolved_count INTEGER NOT NULL,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS top10_picks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL DEFAULT 'weather',
                    strategy TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    token_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    outcome_label TEXT NOT NULL,
                    side TEXT NOT NULL,
                    model_prob REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    shares REAL NOT NULL,
                    stake_usd REAL NOT NULL DEFAULT 100.0,
                    expected_edge REAL NOT NULL,
                    current_mid REAL,
                    current_bid REAL,
                    unrealized_pnl REAL DEFAULT 0.0,
                    opened_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    resolved_outcome REAL,
                    pnl_usd REAL,
                    closed_at TEXT,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS top10_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pick_id INTEGER NOT NULL,
                    captured_at TEXT NOT NULL,
                    market_mid REAL NOT NULL,
                    best_bid REAL,
                    best_ask REAL,
                    unrealized_pnl REAL NOT NULL
                );
                """
            )
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(top10_picks)").fetchall()}
            if "category" not in cols:
                conn.execute("ALTER TABLE top10_picks ADD COLUMN category TEXT NOT NULL DEFAULT 'weather'")
            row = conn.execute("SELECT id FROM portfolio_state WHERE id = 1").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO portfolio_state (id, initial_bankroll, cash_usd, created_at) VALUES (1, ?, ?, ?)",
                    (self.config.initial_bankroll, self.config.initial_bankroll, utc_now_iso()),
                )

    def get_cash(self) -> float:
        with self._connect() as conn:
            row = conn.execute("SELECT cash_usd FROM portfolio_state WHERE id = 1").fetchone()
            return float(row["cash_usd"]) if row else self.config.initial_bankroll

    def get_open_exposure(self, event_id: str | None = None) -> float:
        with self._connect() as conn:
            if event_id is None:
                row = conn.execute(
                    "SELECT COALESCE(SUM(stake_usd), 0.0) AS total FROM trades WHERE status = 'OPEN'"
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COALESCE(SUM(stake_usd), 0.0) AS total FROM trades WHERE status = 'OPEN' AND event_id = ?",
                    (event_id,),
                ).fetchone()
            return float(row["total"]) if row else 0.0

    def has_open_position(self, market_id: str, side: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM trades WHERE market_id = ? AND side = ? AND status = 'OPEN' LIMIT 1",
                (market_id, side),
            ).fetchone()
            return row is not None

    def execute_signals(
        self,
        signals: list[TradeSignal],
        client: PolymarketClient,
        analyst: LLMAnalyst | None = None,
    ) -> list[dict[str, Any]]:
        """Validate, size, simulate CLOB fill, and persist paper trades."""
        opened: list[dict[str, Any]] = []
        # Prioritize highest expected edge first
        ranked = sorted(signals, key=lambda s: s.expected_edge, reverse=True)

        for signal in ranked:
            if self.has_open_position(signal.market_id, signal.side):
                continue

            cash = self.get_cash()
            open_exposure = self.get_open_exposure()
            equity = cash + open_exposure
            event_exposure = self.get_open_exposure(event_id=signal.event_id)
            remaining_event_cap = max(0.0, equity * self.config.max_event_pct - event_exposure)

            if signal.fixed_stake_usd is not None:
                target_usd = min(signal.fixed_stake_usd, cash, remaining_event_cap)
            else:
                # Provisional price from mid + half-spread estimate before walking book
                est_price = min(0.98, max(0.02, signal.market_mid))
                kelly_usd = fractional_kelly_stake(
                    model_prob=signal.model_prob,
                    price=est_price,
                    equity_usd=equity,
                    kelly_fraction=self.config.kelly_fraction,
                    max_position_pct=self.config.max_position_pct,
                )
                target_usd = min(kelly_usd, cash, remaining_event_cap)

            if target_usd < self.config.min_stake_usd:
                continue

            try:
                book = client.fetch_orderbook(signal.token_id)
            except Exception:
                continue

            sim = simulate_buy_fill(
                book=book,
                target_usd=target_usd,
                slippage_buffer=self.config.slippage_buffer,
                max_spread=self.config.max_spread,
                min_fill_ratio=self.config.min_fill_ratio,
                taker_fee_rate=self.config.taker_fee_rate,
                max_price_impact=self.config.max_price_impact,
                min_bid_price=self.config.min_bid_price,
            )
            if not sim.executable:
                continue

            real_edge = signal.model_prob - sim.effective_price
            if signal.strategy != "negrisk_arb" and real_edge < self.config.min_edge:
                continue

            # Re-verify Kelly stake at the actual executable fill price
            if signal.fixed_stake_usd is None:
                final_kelly = fractional_kelly_stake(
                    model_prob=signal.model_prob,
                    price=sim.effective_price,
                    equity_usd=equity,
                    kelly_fraction=self.config.kelly_fraction,
                    max_position_pct=self.config.max_position_pct,
                )
                stake_usd = min(final_kelly, sim.filled_usd, cash, remaining_event_cap)
            else:
                stake_usd = min(sim.filled_usd, cash, remaining_event_cap)

            if stake_usd < self.config.min_stake_usd:
                continue

            llm_status = "none"
            if analyst is not None:
                verdict = analyst.audit_signal(signal)
                llm_status = f"{verdict.provider}:{'OK' if verdict.approved else 'REJECT'} ({verdict.reason})"
                if not verdict.approved:
                    continue

            shares = stake_usd / sim.effective_price
            opened_at = utc_now_iso()

            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO trades (
                        strategy, event_id, market_id, token_id, question, outcome_label,
                        side, model_prob, market_mid, fill_price, shares, stake_usd,
                        expected_edge, opened_at, status, llm_check, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
                    """,
                    (
                        signal.strategy,
                        signal.event_id,
                        signal.market_id,
                        signal.token_id,
                        signal.question,
                        signal.outcome_label,
                        signal.side,
                        signal.model_prob,
                        sim.mid_price,
                        sim.effective_price,
                        shares,
                        stake_usd,
                        real_edge,
                        opened_at,
                        llm_status,
                        signal.notes,
                    ),
                )
                conn.execute(
                    "UPDATE portfolio_state SET cash_usd = cash_usd - ? WHERE id = 1",
                    (stake_usd,),
                )

            opened.append(
                {
                    "strategy": signal.strategy,
                    "question": signal.question,
                    "outcome_label": signal.outcome_label,
                    "side": signal.side,
                    "fill_price": sim.effective_price,
                    "model_prob": signal.model_prob,
                    "edge": real_edge,
                    "stake_usd": stake_usd,
                    "shares": shares,
                }
            )

        return opened

    def resolve_open_trades(self, client: PolymarketClient) -> list[dict[str, Any]]:
        """Check Gamma API for any open paper trades whose market has resolved."""
        with self._connect() as conn:
            open_rows = conn.execute("SELECT * FROM trades WHERE status = 'OPEN'").fetchall()

        settled: list[dict[str, Any]] = []
        market_cache: dict[str, Any] = {}

        for row in open_rows:
            market_id = str(row["market_id"])
            if market_id not in market_cache:
                market_cache[market_id] = client.fetch_market(market_id)
            market = market_cache[market_id]
            if market is None or market.resolved_yes is None:
                continue

            outcome_for_side = market.resolved_yes if row["side"] == "YES" else (1.0 - market.resolved_yes)
            payout_usd = float(row["shares"]) * outcome_for_side
            pnl_usd = payout_usd - float(row["stake_usd"])
            closed_at = utc_now_iso()

            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE trades
                    SET status = 'CLOSED',
                        resolved_outcome = ?,
                        exit_price = ?,
                        pnl_usd = ?,
                        closed_at = ?
                    WHERE id = ?
                    """,
                    (outcome_for_side, outcome_for_side, pnl_usd, closed_at, row["id"]),
                )
                conn.execute(
                    "UPDATE portfolio_state SET cash_usd = cash_usd + ? WHERE id = 1",
                    (payout_usd,),
                )

            settled.append(
                {
                    "id": row["id"],
                    "strategy": row["strategy"],
                    "question": row["question"],
                    "side": row["side"],
                    "outcome": outcome_for_side,
                    "pnl_usd": pnl_usd,
                }
            )

        return settled

    def check_early_exits(self, client: PolymarketClient) -> list[dict[str, Any]]:
        """Take profit early via real CLOB bid book when market price overshoots fair prob."""
        with self._connect() as conn:
            open_rows = conn.execute("SELECT * FROM trades WHERE status = 'OPEN'").fetchall()

        exited: list[dict[str, Any]] = []
        for row in open_rows:
            if row["strategy"] == "negrisk_arb":
                continue
            try:
                book = client.fetch_orderbook(str(row["token_id"]))
            except Exception:
                continue

            sim = simulate_sell_fill(
                book=book,
                shares_to_sell=float(row["shares"]),
                slippage_buffer=self.config.slippage_buffer,
                max_spread=self.config.max_spread,
                taker_fee_rate=self.config.taker_fee_rate,
            )
            if not sim.executable:
                continue

            model_prob = float(row["model_prob"])
            should_exit = (
                sim.effective_price >= model_prob + self.config.early_exit_edge_buffer
                or sim.effective_price >= self.config.early_exit_near_certain
            )
            if not should_exit:
                continue

            payout_usd = float(row["shares"]) * sim.effective_price
            pnl_usd = payout_usd - float(row["stake_usd"])
            closed_at = utc_now_iso()

            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE trades
                    SET status = 'CLOSED',
                        exit_price = ?,
                        pnl_usd = ?,
                        closed_at = ?,
                        notes = COALESCE(notes, '') || ' | Early TP exit'
                    WHERE id = ?
                    """,
                    (sim.effective_price, pnl_usd, closed_at, row["id"]),
                )
                conn.execute(
                    "UPDATE portfolio_state SET cash_usd = cash_usd + ? WHERE id = 1",
                    (payout_usd,),
                )

            exited.append(
                {
                    "id": row["id"],
                    "strategy": row["strategy"],
                    "question": row["question"],
                    "side": row["side"],
                    "outcome": sim.effective_price,
                    "pnl_usd": pnl_usd,
                }
            )

        return exited

    def log_scan_run(
        self,
        events_scanned: int,
        signals_found: int,
        trades_opened: int,
        resolved_count: int,
        notes: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scan_runs (ran_at, events_scanned, signals_found, trades_opened, resolved_count, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (utc_now_iso(), events_scanned, signals_found, trades_opened, resolved_count, notes),
            )

    def list_trades(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent trades ordered newest first."""
        with self._connect() as conn:
            if status in {"OPEN", "CLOSED"}:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE status = ? ORDER BY id DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trades ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def list_scan_runs(self, limit: int = 15) -> list[dict[str, Any]]:
        """Return recent scan run logs ordered newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scan_runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def reset_portfolio(self, initial_bankroll: float | None = None) -> None:
        """Clear all paper trades and reset bankroll to initial amount."""
        bankroll = initial_bankroll if initial_bankroll is not None else self.config.initial_bankroll
        with self._connect() as conn:
            conn.execute("DELETE FROM trades")
            conn.execute("DELETE FROM scan_runs")
            conn.execute("DELETE FROM top10_snapshots")
            conn.execute("DELETE FROM top10_picks")
            conn.execute("DELETE FROM portfolio_state")
            conn.execute(
                "INSERT INTO portfolio_state (id, initial_bankroll, cash_usd, created_at) VALUES (1, ?, ?, ?)",
                (bankroll, bankroll, utc_now_iso()),
            )

    @staticmethod
    def strategy_to_category(strategy: str) -> str:
        mapping = {
            "weather_ensemble": "weather",
            "crypto_lognormal": "crypto",
            "science_research": "science",
            "economics_research": "economics",
            "tech_research": "tech",
            "negrisk_arb": "arbitrage",
        }
        return mapping.get(strategy, strategy)

    def update_top10_portfolio(
        self,
        signals: list[TradeSignal],
        client: PolymarketClient,
        stake_per_market: float = 100.0,
        max_slots: int = 15,
        max_per_category: int = 3,
    ) -> dict[str, int]:
        """Maintain a separate Top-15 ($100/market) portfolio diversified across Top-5 categories."""
        now = utc_now_iso()
        resolved_count = 0
        updated_count = 0

        with self._connect() as conn:
            open_picks = conn.execute("SELECT * FROM top10_picks WHERE status = 'OPEN' ORDER BY id ASC").fetchall()

        for pick in open_picks:
            market = client.fetch_market(str(pick["market_id"]))
            if market is not None and market.resolved_yes is not None:
                outcome_for_side = market.resolved_yes if pick["side"] == "YES" else (1.0 - market.resolved_yes)
                payout = float(pick["shares"]) * outcome_for_side
                pnl = payout - float(pick["stake_usd"])
                with self._connect() as conn:
                    conn.execute(
                        """
                        UPDATE top10_picks
                        SET status = 'CLOSED', resolved_outcome = ?, pnl_usd = ?,
                            unrealized_pnl = ?, last_updated = ?, closed_at = ?
                        WHERE id = ?
                        """,
                        (outcome_for_side, pnl, pnl, now, now, pick["id"]),
                    )
                resolved_count += 1
                continue

            try:
                book = client.fetch_orderbook(str(pick["token_id"]), use_cache=False)
            except Exception:
                continue

            mid = book.mid_price if book.mid_price is not None else float(pick["entry_price"])
            bid = book.best_bid
            ask = book.best_ask
            mark_price = bid if bid is not None else mid
            unrealized = float(pick["shares"]) * mark_price - float(pick["stake_usd"])

            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE top10_picks
                    SET current_mid = ?, current_bid = ?, unrealized_pnl = ?, last_updated = ?
                    WHERE id = ?
                    """,
                    (mid, bid, unrealized, now, pick["id"]),
                )
                conn.execute(
                    """
                    INSERT INTO top10_snapshots (pick_id, captured_at, market_mid, best_bid, best_ask, unrealized_pnl)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (pick["id"], now, mid, bid, ask, unrealized),
                )
            updated_count += 1

        with self._connect() as conn:
            active_rows = conn.execute("SELECT market_id, category FROM top10_picks WHERE status = 'OPEN'").fetchall()
            open_market_ids = {str(r["market_id"]) for r in active_rows}
            open_slots_used = len(open_market_ids)
            cat_counts: dict[str, int] = {}
            for r in active_rows:
                cat = str(r["category"] or "weather")
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

        opened_count = 0
        ranked = sorted(signals, key=lambda s: s.expected_edge, reverse=True)
        for sig in ranked:
            if open_slots_used >= max_slots:
                break
            if sig.market_id in open_market_ids or sig.strategy == "negrisk_arb":
                continue
            cat = self.strategy_to_category(sig.strategy)
            if cat_counts.get(cat, 0) >= max_per_category:
                continue

            try:
                book = client.fetch_orderbook(sig.token_id)
            except Exception:
                continue

            sim = simulate_buy_fill(
                book=book,
                target_usd=stake_per_market,
                slippage_buffer=self.config.slippage_buffer,
                max_spread=self.config.max_spread,
                min_fill_ratio=0.80,
                taker_fee_rate=self.config.taker_fee_rate,
                max_price_impact=self.config.max_price_impact,
                min_bid_price=self.config.min_bid_price,
            )
            if not sim.executable:
                continue

            real_edge = sig.model_prob - sim.effective_price
            if real_edge < self.config.min_edge:
                continue

            shares = stake_per_market / sim.effective_price
            mark_price = sim.best_bid if sim.best_bid is not None else sim.mid_price
            unrealized = shares * mark_price - stake_per_market

            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO top10_picks (
                        category, strategy, event_id, market_id, token_id, question, outcome_label, side,
                        model_prob, entry_price, shares, stake_usd, expected_edge,
                        current_mid, current_bid, unrealized_pnl, opened_at, last_updated, status, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
                    """,
                    (
                        cat, sig.strategy, sig.event_id, sig.market_id, sig.token_id, sig.question,
                        sig.outcome_label, sig.side, sig.model_prob, sim.effective_price,
                        shares, stake_per_market, real_edge, sim.mid_price, sim.best_bid,
                        unrealized, now, now, sig.notes,
                    ),
                )
                pick_id = cur.lastrowid
                conn.execute(
                    """
                    INSERT INTO top10_snapshots (pick_id, captured_at, market_mid, best_bid, best_ask, unrealized_pnl)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (pick_id, now, sim.mid_price, sim.best_bid, sim.best_ask, unrealized),
                )
            open_market_ids.add(sig.market_id)
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            open_slots_used += 1
            opened_count += 1

        return {"opened": opened_count, "updated": updated_count, "resolved": resolved_count}

    def list_top10(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.*,
                       (SELECT COUNT(*) FROM top10_snapshots s WHERE s.pick_id = p.id) AS snapshot_count
                FROM top10_picks p
                ORDER BY CASE WHEN p.status = 'OPEN' THEN 0 ELSE 1 END, p.expected_edge DESC, p.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def format_top10_report(self) -> str:
        picks = self.list_top10(limit=30)
        open_picks = [p for p in picks if p["status"] == "OPEN"]
        closed_picks = [p for p in picks if p["status"] == "CLOSED"]
        total_unrealized = sum(float(p["unrealized_pnl"] or 0.0) for p in open_picks)
        total_realized = sum(float(p["pnl_usd"] or 0.0) for p in closed_picks)

        by_cat: dict[str, list[dict[str, Any]]] = {}
        for p in picks:
            cat = str(p.get("category") or self.strategy_to_category(str(p["strategy"])))
            by_cat.setdefault(cat, []).append(p)

        lines = [
            "# Top 15 High-Conviction Markets Across Top 5 Categories ($100 / Market)",
            "",
            f"- Active Slots: **{len(open_picks)} / 15** (${len(open_picks) * 100:,.2f} deployed, max 3 per category)",
            f"- Open Unrealized PnL: **${total_unrealized:+,.2f}**",
            f"- Closed Realized PnL: **${total_realized:+,.2f}** ({len(closed_picks)} settled)",
            "",
            "## Top 5 Categories Breakdown",
            "",
            "| Category | Open / 3 | Closed | Unrealized PnL | Realized PnL |",
            "|---|---:|---:|---:|---:|",
        ]
        for cat in ("weather", "crypto", "science", "economics", "tech"):
            c_picks = by_cat.get(cat, [])
            c_open = [p for p in c_picks if p["status"] == "OPEN"]
            c_closed = [p for p in c_picks if p["status"] == "CLOSED"]
            c_unr = sum(float(p["unrealized_pnl"] or 0.0) for p in c_open)
            c_real = sum(float(p["pnl_usd"] or 0.0) for p in c_closed)
            lines.append(f"| `{cat}` | {len(c_open)} / 3 | {len(c_closed)} | ${c_unr:+.2f} | ${c_real:+.2f} |")

        lines.extend(
            [
                "",
                "## Tracked Markets ($100 Each)",
                "",
                "| # | Status | Category | Side | Market | Entry | Now (Bid) | Model | Edge | PnL ($100) | Snapshots |",
                "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for idx, p in enumerate(picks, start=1):
            cat = str(p.get("category") or self.strategy_to_category(str(p["strategy"])))
            bid_str = f"${float(p['current_bid']):.3f}" if p["current_bid"] is not None else "—"
            pnl_val = float(p["pnl_usd"]) if p["status"] == "CLOSED" and p["pnl_usd"] is not None else float(p["unrealized_pnl"] or 0.0)
            lines.append(
                f"| {idx} | {p['status']} | `{cat}` | **{p['side']}** | "
                f"{p['question']} ({p['outcome_label']}) | ${float(p['entry_price']):.3f} | {bid_str} | "
                f"{float(p['model_prob']) * 100:.1f}% | +{float(p['expected_edge']) * 100:.1f}% | "
                f"${pnl_val:+.2f} | {p['snapshot_count']} |"
            )
        if not picks:
            lines.append("| — | — | — | — | Пока нет отобранных рынков | — | — | — | — | — | 0 |")
        return "\n".join(lines) + "\n"

    def summarize(self) -> PerformanceSummary:
        with self._connect() as conn:
            state = conn.execute("SELECT initial_bankroll, cash_usd FROM portfolio_state WHERE id = 1").fetchone()
            rows = conn.execute("SELECT * FROM trades ORDER BY id ASC").fetchall()

        initial_bankroll = float(state["initial_bankroll"]) if state else self.config.initial_bankroll
        cash_usd = float(state["cash_usd"]) if state else self.config.initial_bankroll

        open_rows = [r for r in rows if r["status"] == "OPEN"]
        closed_rows = [r for r in rows if r["status"] == "CLOSED"]
        resolved_rows = [r for r in closed_rows if r["resolved_outcome"] is not None]

        open_exposure = sum(float(r["stake_usd"]) for r in open_rows)
        realized_pnl = sum(float(r["pnl_usd"] or 0.0) for r in closed_rows)
        wins = sum(1 for r in closed_rows if float(r["pnl_usd"] or 0.0) > 0)
        losses = len(closed_rows) - wins
        win_rate = (wins / len(closed_rows)) if closed_rows else None

        bot_brier = None
        market_brier = None
        bss = None
        if resolved_rows:
            bot_brier = sum(
                (float(r["model_prob"]) - float(r["resolved_outcome"])) ** 2 for r in resolved_rows
            ) / len(resolved_rows)
            market_brier = sum(
                (float(r["market_mid"]) - float(r["resolved_outcome"])) ** 2 for r in resolved_rows
            ) / len(resolved_rows)
            if market_brier > 1e-12:
                bss = 1.0 - (bot_brier / market_brier)

        by_strategy: dict[str, dict[str, float | int | None]] = {}
        for strat in sorted({str(r["strategy"]) for r in rows}):
            s_all = [r for r in rows if r["strategy"] == strat]
            s_closed = [r for r in closed_rows if r["strategy"] == strat]
            s_resolved = [r for r in resolved_rows if r["strategy"] == strat]
            s_pnl = sum(float(r["pnl_usd"] or 0.0) for r in s_closed)
            s_bot_brier = (
                sum((float(r["model_prob"]) - float(r["resolved_outcome"])) ** 2 for r in s_resolved) / len(s_resolved)
                if s_resolved
                else None
            )
            s_mkt_brier = (
                sum((float(r["market_mid"]) - float(r["resolved_outcome"])) ** 2 for r in s_resolved) / len(s_resolved)
                if s_resolved
                else None
            )
            by_strategy[strat] = {
                "total": len(s_all),
                "open": sum(1 for r in s_all if r["status"] == "OPEN"),
                "closed": len(s_closed),
                "pnl_usd": s_pnl,
                "bot_brier": s_bot_brier,
                "market_brier": s_mkt_brier,
            }

        roi_pct = ((realized_pnl / initial_bankroll) * 100.0) if initial_bankroll > 0 else 0.0

        return PerformanceSummary(
            initial_bankroll=initial_bankroll,
            cash_usd=cash_usd,
            open_exposure_usd=open_exposure,
            realized_pnl_usd=realized_pnl,
            total_trades=len(rows),
            open_trades=len(open_rows),
            closed_trades=len(closed_rows),
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            bot_brier=bot_brier,
            market_brier=market_brier,
            brier_skill_score=bss,
            roi_pct=roi_pct,
            by_strategy=by_strategy,
        )

    def format_report(self) -> str:
        s = self.summarize()
        lines = [
            "=== POLYMARKET PAPER TRADER REPORT ===",
            f"Initial Bankroll : ${s.initial_bankroll:,.2f}",
            f"Cash Available   : ${s.cash_usd:,.2f}",
            f"Open Exposure    : ${s.open_exposure_usd:,.2f}",
            f"Realized PnL     : ${s.realized_pnl_usd:+,.2f} ({s.roi_pct:+.2f}%)",
            f"Trades           : {s.total_trades} total ({s.open_trades} open, {s.closed_trades} closed)",
        ]
        if s.win_rate is not None:
            lines.append(f"Win Rate         : {s.win_rate * 100:.1f}% ({s.wins}W / {s.losses}L)")
        if s.bot_brier is not None and s.market_brier is not None:
            verdict = "BEATING MARKET" if s.bot_brier < s.market_brier else "TRAILING MARKET"
            bss_str = f"{s.brier_skill_score:+.3f}" if s.brier_skill_score is not None else "n/a"
            lines.append(
                f"Brier Score      : Bot={s.bot_brier:.4f} vs Market={s.market_brier:.4f} "
                f"(BSS={bss_str} -> {verdict})"
            )
        else:
            lines.append("Brier Score      : Waiting for resolved trades (need 50+ for statistical significance)")

        if s.by_strategy:
            lines.append("")
            lines.append("By Strategy:")
            for name, st in s.by_strategy.items():
                brier_info = ""
                if st["bot_brier"] is not None and st["market_brier"] is not None:
                    brier_info = f" | Brier {st['bot_brier']:.4f} vs {st['market_brier']:.4f}"
                lines.append(
                    f"  - {name}: {st['total']} trades ({st['open']} open, {st['closed']} closed) "
                    f"| PnL ${float(st['pnl_usd'] or 0.0):+.2f}{brier_info}"
                )
        return "\n".join(lines)
