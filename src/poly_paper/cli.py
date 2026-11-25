"""CLI entrypoint & interactive menu for Polymarket paper-trading bot."""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from datetime import date
from pathlib import Path

from .config import TraderConfig
from .dashboard import write_dashboard
from .engine import PaperTradingEngine
from .llm_analyst import LLMAnalyst
from .models import EventBundle, MarketOutcome, OrderBook, OrderLevel
from .notifier import TelegramNotifier
from .polymarket import PolymarketClient
from .strategies import scan_negrisk_arbitrage, scan_science_markets, scan_weather_markets
from .strategies.science import gather_scientific_evidence, is_science_event

_USE_COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _banner(title: str) -> None:
    line = "─" * 58
    print(_c(f"┌{line}┐", "36"))
    print(_c(f"│ {title:<56} │", "1;36"))
    print(_c(f"└{line}┘", "36"))


def _dedupe_events(events: list[EventBundle]) -> list[EventBundle]:
    seen: dict[str, EventBundle] = {}
    for ev in events:
        if ev.event_id not in seen:
            seen[ev.event_id] = ev
    return list(seen.values())


def cmd_scan(config: TraderConfig) -> None:
    client = PolymarketClient()
    engine = PaperTradingEngine(config)
    analyst = LLMAnalyst(config)

    print(_c("⏳ Загружаю активные рынки Polymarket...", "36"))
    weather_events = client.fetch_active_events(limit=60, tag_slug="weather")
    science_events = client.fetch_active_events(limit=40, tag_slug="science")
    general_events = client.fetch_active_events(limit=config.max_events_per_scan)
    all_events = _dedupe_events(weather_events + science_events + general_events)

    weather_signals = scan_weather_markets(client, all_events, config)
    arb_signals = scan_negrisk_arbitrage(client, all_events, config)
    science_signals = scan_science_markets(client, all_events, config)
    all_signals = weather_signals + arb_signals + science_signals

    opened = engine.execute_signals(all_signals, client, analyst=analyst)
    engine.log_scan_run(
        events_scanned=len(all_events),
        signals_found=len(all_signals),
        trades_opened=len(opened),
        resolved_count=0,
        notes=f"weather={len(weather_signals)}, negrisk={len(arb_signals)}, science={len(science_signals)}",
    )

    print(
        f"Событий: {_c(str(len(all_events)), '1')} | "
        f"Сигналов: {_c(str(len(all_signals)), '33')} "
        f"(weather={len(weather_signals)}, negrisk={len(arb_signals)}, science={len(science_signals)}) | "
        f"Открыто сделок: {_c(str(len(opened)), '1;32')}"
    )
    for t in opened:
        print(
            f"  {_c('[OPEN]', '32')} {t['strategy']} | {t['side']} {t['outcome_label']} @ ${t['fill_price']:.3f} "
            f"(model={t['model_prob']:.1%}, edge={t['edge']:+.1%}, stake=${t['stake_usd']:.2f})"
        )


def cmd_resolve(config: TraderConfig) -> list[dict]:
    client = PolymarketClient()
    engine = PaperTradingEngine(config)
    settled = engine.resolve_open_trades(client)
    exited = engine.check_early_exits(client)
    all_closed = settled + exited
    print(f"Закрыто сделок: {_c(str(len(all_closed)), '1')} (резолв={len(settled)}, take-profit={len(exited)})")
    for s in all_closed:
        pnl_str = _c(f"${s['pnl_usd']:+.2f}", "32" if s["pnl_usd"] >= 0 else "31")
        print(f"  [CLOSED] #{s['id']} {s['strategy']} {s['side']} | PnL {pnl_str}")
    return all_closed


def cmd_report(config: TraderConfig, output_md: str | None = None) -> None:
    engine = PaperTradingEngine(config)
    report = engine.format_report()
    print()
    print(report)
    if output_md:
        Path(output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(output_md).write_text(f"```text\n{report}\n```\n", encoding="utf-8")


def cmd_trades(config: TraderConfig, status: str | None = None) -> None:
    engine = PaperTradingEngine(config)
    trades = engine.list_trades(status=status, limit=30)
    _banner(f"PAPER TRADES ({status or 'ALL'})")
    if not trades:
        print(_c("Сделок пока нет.", "33"))
        return
    for t in trades:
        st = _c("OPEN", "34") if t["status"] == "OPEN" else _c("CLOSED", "90")
        pnl = ""
        if t["pnl_usd"] is not None:
            pnl_val = float(t["pnl_usd"])
            pnl = " | PnL " + _c(f"${pnl_val:+.2f}", "32" if pnl_val >= 0 else "31")
        print(
            f"#{t['id']:<3} [{st}] {t['strategy']:<17} {t['side']:<3} @ ${float(t['fill_price']):.3f} "
            f"| Model {float(t['model_prob']):.1%} | Stake ${float(t['stake_usd']):.2f}{pnl}"
        )
        print(f"     {t['question']} ({t['outcome_label']})")


def cmd_dashboard(config: TraderConfig, output_html: str = "data/dashboard.html", open_browser: bool = True) -> None:
    engine = PaperTradingEngine(config)
    path = write_dashboard(engine, output_path=output_html)
    print(_c(f"✓ Дашборд сохранён: {path.resolve()}", "32"))
    if open_browser:
        webbrowser.open(path.resolve().as_uri())


def cmd_science(config: TraderConfig, limit: int = 8) -> None:
    """List active science/chemistry/FDA markets with live ClinicalTrials/openFDA/OpenAlex evidence."""
    client = PolymarketClient()
    print(_c("⏳ Ищу научные / химические / FDA рынки и подтягиваю базы данных...", "36"))
    science_events = client.fetch_active_events(limit=40, tag_slug="science")
    general_events = client.fetch_active_events(limit=config.max_events_per_scan)
    all_events = _dedupe_events(science_events + general_events)

    shown = 0
    for event in all_events:
        if shown >= limit:
            break
        if not is_science_event(event):
            continue
        for market in event.markets:
            if shown >= limit:
                break
            if market.closed or not (0.04 <= market.yes_price <= 0.96):
                continue
            shown += 1
            ev = gather_scientific_evidence(market)
            print(f"\n{_c(f'[{shown}]', '1;36')} {_c(market.question, '1')}")
            print(f"    YES={market.yes_price:.2f} | NO={market.no_price:.2f} | Vol=${market.volume:,.0f} | End={market.end_date}")
            print(f"    Evidence ({ev.query}):")
            for line in ev.to_text().splitlines():
                print(f"    {line}")
    if shown == 0:
        print(_c("Активных научных рынков под фильтр сейчас не найдено.", "33"))


def cmd_run(config: TraderConfig, output_md: str | None = None) -> None:
    _banner("POLYMARKET PAPER TRADER — FULL CYCLE")
    closed = cmd_resolve(config)

    client = PolymarketClient()
    engine = PaperTradingEngine(config)
    analyst = LLMAnalyst(config)

    print(_c("⏳ Загружаю активные рынки Polymarket...", "36"))
    weather_events = client.fetch_active_events(limit=60, tag_slug="weather")
    science_events = client.fetch_active_events(limit=40, tag_slug="science")
    general_events = client.fetch_active_events(limit=config.max_events_per_scan)
    all_events = _dedupe_events(weather_events + science_events + general_events)

    weather_signals = scan_weather_markets(client, all_events, config)
    arb_signals = scan_negrisk_arbitrage(client, all_events, config)
    science_signals = scan_science_markets(client, all_events, config)
    all_signals = weather_signals + arb_signals + science_signals

    opened = engine.execute_signals(all_signals, client, analyst=analyst)
    engine.log_scan_run(
        events_scanned=len(all_events),
        signals_found=len(all_signals),
        trades_opened=len(opened),
        resolved_count=len(closed),
        notes=f"weather={len(weather_signals)}, negrisk={len(arb_signals)}, science={len(science_signals)}",
    )
    print(
        f"Событий: {_c(str(len(all_events)), '1')} | "
        f"Сигналов: {_c(str(len(all_signals)), '33')} | "
        f"Открыто: {_c(str(len(opened)), '1;32')}"
    )

    cmd_report(config, output_md=output_md)
    dash = write_dashboard(engine)
    print(_c(f"\n✓ HTML-дашборд обновлён: {dash}", "32"))

    notifier = TelegramNotifier(config)
    if notifier.enabled and (opened or closed):
        s = engine.summarize()
        notifier.notify_cycle(opened, closed, equity=s.cash_usd + s.open_exposure_usd)


def cmd_setup() -> None:
    """Interactive wizard to configure .env API keys and bankroll."""
    _banner("НАСТРОЙКА .env (API-КЛЮЧИ И БАНКРОЛЛ)")
    print("Ключи необязательны: без них работают Погода и NegRisk-арбитраж.")
    print("Для ИИ-оценки научных рынков нужен бесплатный ключ Google AI Studio или NVIDIA NIM.\n")

    gemini = input("GEMINI_API_KEY (Enter — пропустить): ").strip()
    nvidia = input("NVIDIA_API_KEY (Enter — пропустить): ").strip()
    bankroll = input("Начальный бумажный банкролл в $ [1000]: ").strip() or "1000.0"

    env_content = (
        f"GEMINI_API_KEY={gemini}\n"
        f"NVIDIA_API_KEY={nvidia}\n"
        "NVIDIA_MODEL=meta/llama-3.3-70b-instruct\n"
        f"INITIAL_BANKROLL={bankroll}\n"
        "KELLY_FRACTION=0.25\n"
        "MIN_EDGE=0.06\n"
        "SLIPPAGE_BUFFER=0.005\n"
        "DB_PATH=data/paper_trader.db\n"
    )
    Path(".env").write_text(env_content, encoding="utf-8")
    print(_c("\n✓ Файл .env сохранён! Перезапустите скрипт, чтобы применить настройки.", "32"))


def cmd_reset(config: TraderConfig) -> None:
    engine = PaperTradingEngine(config)
    engine.reset_portfolio()
    write_dashboard(engine)
    print(_c(f"✓ Бумажный портфель сброшен к ${config.initial_bankroll:,.2f}.", "32"))


def cmd_demo(db_path: str) -> None:
    """Offline end-to-end demo with synthetic CLOB books & weather ensembles."""
    demo_db = Path(db_path)
    if demo_db.exists():
        demo_db.unlink()

    config = TraderConfig(db_path=demo_db, llm_enabled=False)
    engine = PaperTradingEngine(config)

    nyc_event = EventBundle(
        event_id="ev-nyc-demo",
        slug="highest-temp-nyc-nov-26",
        title="Highest temperature in NYC on November 26?",
        description="Resolves to the highest temperature recorded at LaGuardia Airport (KLGA).",
        neg_risk=True,
        end_date="2026-11-26T23:59:59Z",
        markets=(
            MarketOutcome("m1", "ev-nyc-demo", "NYC", "Highest temp <=49°F?", "49°F or below", "", "t1_y", "t1_n", 0.25, 0.75, 12000, 3000, "2026-11-26"),
            MarketOutcome("m2", "ev-nyc-demo", "NYC", "Highest temp 50-51°F?", "50-51°F", "", "t2_y", "t2_n", 0.35, 0.65, 15000, 4000, "2026-11-26"),
            MarketOutcome("m3", "ev-nyc-demo", "NYC", "Highest temp 52°F or higher?", "52°F or higher", "", "t3_y", "t3_n", 0.40, 0.60, 14000, 3500, "2026-11-26"),
        ),
    )

    class _DemoClient(PolymarketClient):
        def __init__(self) -> None:
            super().__init__()
            self._books = {
                "t1_y": OrderBook("t1_y", (OrderLevel(0.24, 200),), (OrderLevel(0.26, 200),)),
                "t1_n": OrderBook("t1_n", (OrderLevel(0.74, 200),), (OrderLevel(0.76, 200),)),
                "t2_y": OrderBook("t2_y", (OrderLevel(0.34, 200),), (OrderLevel(0.36, 200),)),
                "t2_n": OrderBook("t2_n", (OrderLevel(0.64, 200),), (OrderLevel(0.66, 200),)),
                "t3_y": OrderBook("t3_y", (OrderLevel(0.39, 200),), (OrderLevel(0.41, 200),)),
                "t3_n": OrderBook("t3_n", (OrderLevel(0.59, 200),), (OrderLevel(0.61, 200),)),
            }
            self._resolved = {
                "m1": MarketOutcome("m1", "ev-nyc-demo", "NYC", "Highest temp <=49°F?", "49°F or below", "", "t1_y", "t1_n", 0.0, 1.0, 12000, 0, "2026-11-26", closed=True, resolved_yes=0.0),
                "m2": MarketOutcome("m2", "ev-nyc-demo", "NYC", "Highest temp 50-51°F?", "50-51°F", "", "t2_y", "t2_n", 1.0, 0.0, 15000, 0, "2026-11-26", closed=True, resolved_yes=1.0),
                "m3": MarketOutcome("m3", "ev-nyc-demo", "NYC", "Highest temp 52°F or higher?", "52°F or higher", "", "t3_y", "t3_n", 0.0, 1.0, 14000, 0, "2026-11-26", closed=True, resolved_yes=0.0),
            }

        def fetch_orderbook(self, token_id: str, use_cache: bool = True) -> OrderBook:
            return self._books[token_id]

        def fetch_market(self, market_id: str) -> MarketOutcome | None:
            return self._resolved.get(market_id)

    demo_client = _DemoClient()
    fake_ensemble = [50.2, 50.4, 50.6, 50.8, 50.5, 50.1, 50.9, 50.3, 50.7, 50.5, 51.0, 49.9] * 5
    signals = scan_weather_markets(
        client=demo_client,
        events=[nyc_event],
        config=config,
        today=date(2026, 11, 25),
        ensemble_fetcher=lambda station, target_date, unit: fake_ensemble,
    )
    opened = engine.execute_signals(signals, demo_client)
    print(f"[DEMO] Сигналов найдено: {len(signals)} | Открыто сделок: {len(opened)}")
    settled = engine.resolve_open_trades(demo_client)
    print(f"[DEMO] Закрыто по резолву: {len(settled)}\n")
    print(engine.format_report())


def cmd_menu(config: TraderConfig) -> None:
    """Interactive menu when run without arguments."""
    while True:
        engine = PaperTradingEngine(config)
        s = engine.summarize()
        equity = s.cash_usd + s.open_exposure_usd
        llm_state = (
            _c("Gemini", "32") if config.gemini_api_key
            else (_c("NVIDIA NIM", "32") if config.nvidia_api_key else _c("выкл (только кванты)", "33"))
        )
        print()
        _banner("POLYMARKET HIGH-EV PAPER TRADER")
        print(
            f"Капитал: {_c(f'${equity:,.2f}', '1;32')} | "
            f"Сделок: {s.total_trades} ({s.open_trades} откр.) | "
            f"ИИ: {llm_state}\n"
        )
        print("  1. Запустить полный цикл (проверить резолвы + скан + дашборд)")
        print("  2. Показать научные / химические / FDA рынки + базы данных")
        print("  3. Открыть визуальный HTML-дашборд в браузере")
        print("  4. Показать список сделок и Brier-отчёт")
        print("  5. Настроить API-ключи (.env) и стартовый банкролл")
        print("  6. Запустить офлайн-демо (без интернета)")
        print("  7. Сбросить бумажный портфель")
        print("  0. Выход\n")

        choice = input("Выберите действие [1-7, 0]: ").strip()
        if choice == "1":
            cmd_run(config)
        elif choice == "2":
            cmd_science(config)
        elif choice == "3":
            cmd_dashboard(config, open_browser=True)
        elif choice == "4":
            cmd_trades(config)
            cmd_report(config)
        elif choice == "5":
            cmd_setup()
            config = TraderConfig.from_env(db_path=config.db_path)
        elif choice == "6":
            cmd_demo("data/demo_paper.db")
        elif choice == "7":
            confirm = input("Точно удалить все бумажные сделки? (y/N): ").strip().lower()
            if confirm == "y":
                cmd_reset(config)
        elif choice in {"0", "q", "exit"}:
            break


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Polymarket high-EV paper-trading bot")
    parser.add_argument("--db", default=None, help="Path to SQLite database (default: data/paper_trader.db)")
    sub = parser.add_subparsers(dest="command", required=False)

    sub.add_parser("menu", help="Interactive menu (default when no command is given)")
    sub.add_parser("scan", help="Scan live Polymarket markets and open paper trades")
    sub.add_parser("resolve", help="Check and settle resolved paper trades")

    p_report = sub.add_parser("report", help="Print portfolio & Brier Score report")
    p_report.add_argument("--output-md", default=None, help="Optional path to save markdown report")

    p_trades = sub.add_parser("trades", help="List open/closed paper trades")
    p_trades.add_argument("--status", choices=["OPEN", "CLOSED"], default=None)

    p_dash = sub.add_parser("dashboard", help="Generate standalone HTML dashboard")
    p_dash.add_argument("--output", default="data/dashboard.html", help="Output HTML path")
    p_dash.add_argument("--no-open", action="store_true", help="Do not open browser automatically")

    p_run = sub.add_parser("run", help="Full cycle: resolve -> scan -> report -> dashboard")
    p_run.add_argument("--output-md", default=None, help="Optional path to save markdown report")

    p_science = sub.add_parser("science", help="Inspect active science/chemistry/FDA markets with live DB evidence")
    p_science.add_argument("--limit", type=int, default=8, help="Max markets to display")

    sub.add_parser("setup", help="Interactive wizard to create .env with API keys")
    sub.add_parser("reset", help="Reset paper portfolio and clear trades")

    p_demo = sub.add_parser("demo", help="Run an offline end-to-end simulation")
    p_demo.add_argument("--demo-db", default="data/demo_paper.db", help="Temporary DB for offline demo")

    args = parser.parse_args(argv)
    config = TraderConfig.from_env(db_path=args.db)

    if not args.command or args.command == "menu":
        cmd_menu(config)
    elif args.command == "scan":
        cmd_scan(config)
    elif args.command == "resolve":
        cmd_resolve(config)
    elif args.command == "report":
        cmd_report(config, output_md=args.output_md)
    elif args.command == "trades":
        cmd_trades(config, status=args.status)
    elif args.command == "dashboard":
        cmd_dashboard(config, output_html=args.output, open_browser=not args.no_open)
    elif args.command == "run":
        cmd_run(config, output_md=args.output_md)
    elif args.command == "science":
        cmd_science(config, limit=args.limit)
    elif args.command == "setup":
        cmd_setup()
    elif args.command == "reset":
        cmd_reset(config)
    elif args.command == "demo":
        cmd_demo(args.demo_db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
