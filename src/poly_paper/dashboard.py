"""Standalone dark-mode HTML dashboard generator for Polymarket paper trader."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .engine import PaperTradingEngine
from .models import PerformanceSummary


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _fmt_pnl(val: float | None) -> str:
    if val is None:
        return '<span class="muted">—</span>'
    cls = "pos" if val > 0 else ("neg" if val < 0 else "muted")
    return f'<span class="{cls}">${val:+,.2f}</span>'


def _render_equity_svg(initial_bankroll: float, trades: list[dict[str, Any]]) -> str:
    closed = [t for t in reversed(trades) if t["status"] == "CLOSED" and t["pnl_usd"] is not None]
    points: list[float] = [initial_bankroll]
    running = initial_bankroll
    for t in closed:
        running += float(t["pnl_usd"])
        points.append(running)

    if len(points) < 2:
        return '<div class="muted small" style="padding: 18px 0;">Кривая капитала появится после первых закрытых сделок.</div>'

    w, h, pad = 680, 150, 16
    min_v = min(points)
    max_v = max(points)
    span = max(max_v - min_v, 1.0)

    coords: list[str] = []
    for idx, val in enumerate(points):
        x = pad + (idx / (len(points) - 1)) * (w - 2 * pad)
        y = h - pad - ((val - min_v) / span) * (h - 2 * pad)
        coords.append(f"{x:.1f},{y:.1f}")

    line_color = "#22c55e" if points[-1] >= initial_bankroll else "#ef4444"
    poly = " ".join(coords)
    baseline_y = h - pad - ((initial_bankroll - min_v) / span) * (h - 2 * pad)
    return (
        f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:150px;display:block;">'
        f'<line x1="{pad}" y1="{baseline_y:.1f}" x2="{w - pad}" y2="{baseline_y:.1f}" '
        f'stroke="#334155" stroke-dasharray="4 4" stroke-width="1" />'
        f'<polyline fill="none" stroke="{line_color}" stroke-width="2.5" points="{poly}" />'
        f"</svg>"
    )


def render_dashboard_html(engine: PaperTradingEngine) -> str:
    summary: PerformanceSummary = engine.summarize()
    trades = engine.list_trades(limit=100)
    runs = engine.list_scan_runs(limit=10)
    equity = summary.cash_usd + summary.open_exposure_usd
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if summary.bot_brier is not None and summary.market_brier is not None:
        beating = summary.bot_brier < summary.market_brier
        brier_badge = (
            '<span class="badge badge-green">BEATING MARKET</span>'
            if beating
            else '<span class="badge badge-red">TRAILING MARKET</span>'
        )
        brier_main = f"{summary.bot_brier:.4f}"
        brier_sub = f"Market: {summary.market_brier:.4f} | BSS: {(summary.brier_skill_score or 0.0):+.3f}"
    else:
        brier_badge = '<span class="badge badge-gray">COLLECTING DATA</span>'
        brier_main = "—"
        brier_sub = f"Closed: {summary.closed_trades} / 50+ needed"

    pnl_cls = "pos" if summary.realized_pnl_usd > 0 else ("neg" if summary.realized_pnl_usd < 0 else "")
    win_rate_str = f"{summary.win_rate * 100:.1f}%" if summary.win_rate is not None else "—"

    strategy_cards = []
    strategy_labels = {
        "weather_ensemble": ("Weather Ensemble", "GFS + ECMWF + ICON"),
        "negrisk_arb": ("NegRisk Arbitrage", "Full-set YES < $1.00"),
        "science_research": ("Science / FDA / Chem", "ClinicalTrials + openFDA + OpenAlex"),
    }
    for key, st in summary.by_strategy.items():
        title, subtitle = strategy_labels.get(key, (key, ""))
        brier_txt = "Waiting for resolutions"
        if st["bot_brier"] is not None and st["market_brier"] is not None:
            brier_txt = f"Brier: {st['bot_brier']:.4f} vs Mkt {st['market_brier']:.4f}"
        strategy_cards.append(
            f"""
            <div class="card strat-card">
              <div class="strat-head">
                <div>
                  <div class="strat-title">{_esc(title)}</div>
                  <div class="muted small">{_esc(subtitle)}</div>
                </div>
                <div class="strat-pnl">{_fmt_pnl(float(st['pnl_usd'] or 0.0))}</div>
              </div>
              <div class="strat-meta">
                <span>Trades: <b>{st['total']}</b> ({st['open']} open, {st['closed']} closed)</span>
                <span class="muted">{_esc(brier_txt)}</span>
              </div>
            </div>
            """
        )

    if not strategy_cards:
        strategy_cards.append(
            '<div class="card muted">Сделок пока нет. Запустите <code>python run.py run</code> или <code>python run.py demo</code>.</div>'
        )

    trade_rows = []
    for t in trades:
        side_cls = "badge-green" if t["side"] == "YES" else "badge-amber"
        status_cls = "badge-blue" if t["status"] == "OPEN" else "badge-gray"
        edge_pct = float(t["expected_edge"]) * 100.0
        trade_rows.append(
            f"""
            <tr>
              <td class="muted">#{t['id']}</td>
              <td><span class="badge badge-dark">{_esc(t['strategy'])}</span></td>
              <td>
                <div class="q-title">{_esc(t['question'])}</div>
                <div class="muted small">{_esc(t['outcome_label'])} · {_esc(t['notes'] or '')}</div>
              </td>
              <td><span class="badge {side_cls}">{_esc(t['side'])}</span></td>
              <td>${float(t['fill_price']):.3f}</td>
              <td>{float(t['model_prob']) * 100:.1f}%</td>
              <td class="pos">+{edge_pct:.1f}%</td>
              <td>${float(t['stake_usd']):,.2f}</td>
              <td><span class="badge {status_cls}">{_esc(t['status'])}</span></td>
              <td>{_fmt_pnl(t['pnl_usd'])}</td>
            </tr>
            """
        )

    if not trade_rows:
        trade_rows.append('<tr><td colspan="10" class="muted center">История сделок пуста</td></tr>')

    run_rows = []
    for r in runs:
        run_rows.append(
            f"""
            <tr>
              <td>{_esc(r['ran_at'])}</td>
              <td>{r['events_scanned']}</td>
              <td>{r['signals_found']}</td>
              <td>{r['trades_opened']}</td>
              <td class="muted">{_esc(r['notes'])}</td>
            </tr>
            """
        )
    if not run_rows:
        run_rows.append('<tr><td colspan="5" class="muted center">Сканирований ещё не было</td></tr>')

    equity_svg = _render_equity_svg(summary.initial_bankroll, trades)

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Polymarket High-EV Paper Trader</title>
  <style>
    :root {{
      --bg: #0b0f17;
      --card: #131a26;
      --border: #1f2a3c;
      --text: #e6edf7;
      --muted: #8b9bb4;
      --green: #22c55e;
      --red: #ef4444;
      --blue: #3b82f6;
      --amber: #f59e0b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
      padding: 28px 20px 60px;
    }}
    .container {{ max-width: 1180px; margin: 0 auto; }}
    .header {{
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
      gap: 16px; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border);
    }}
    h1 {{ margin: 0; font-size: 22px; font-weight: 700; letter-spacing: -0.02em; }}
    h2 {{ margin: 28px 0 12px; font-size: 16px; font-weight: 600; color: #cbd5e1; }}
    .grid-4 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; }}
    .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }}
    .card {{
      background: var(--card); border: 1px solid var(--border);
      border-radius: 12px; padding: 16px 18px;
    }}
    .kpi-label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
    .kpi-val {{ font-size: 26px; font-weight: 700; margin: 6px 0 4px; }}
    .kpi-sub {{ font-size: 12px; color: var(--muted); }}
    .pos {{ color: var(--green); font-weight: 600; }}
    .neg {{ color: var(--red); font-weight: 600; }}
    .muted {{ color: var(--muted); }}
    .small {{ font-size: 12px; }}
    .center {{ text-align: center; padding: 24px !important; }}
    .badge {{
      display: inline-block; padding: 2px 8px; border-radius: 999px;
      font-size: 11px; font-weight: 600; letter-spacing: 0.03em;
    }}
    .badge-green {{ background: rgba(34,197,94,0.14); color: #4ade80; }}
    .badge-red {{ background: rgba(239,68,68,0.14); color: #f87171; }}
    .badge-blue {{ background: rgba(59,130,246,0.15); color: #60a5fa; }}
    .badge-amber {{ background: rgba(245,158,11,0.15); color: #fbbf24; }}
    .badge-gray {{ background: rgba(148,163,184,0.15); color: #cbd5e1; }}
    .badge-dark {{ background: #1e293b; color: #94a3b8; }}
    .strat-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }}
    .strat-title {{ font-weight: 600; font-size: 15px; }}
    .strat-pnl {{ font-size: 18px; }}
    .strat-meta {{ display: flex; justify-content: space-between; margin-top: 12px; font-size: 12px; border-top: 1px solid var(--border); padding-top: 10px; }}
    .table-wrap {{ overflow-x: auto; background: var(--card); border: 1px solid var(--border); border-radius: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); vertical-align: top; }}
    th {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); background: #0f1520; }}
    tr:last-child td {{ border-bottom: none; }}
    .q-title {{ font-weight: 500; max-width: 340px; }}
    code {{ background: #1e293b; padding: 2px 6px; border-radius: 5px; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1>Polymarket High-EV Paper Trader</h1>
        <div class="muted small">Weather Ensemble · NegRisk Arbitrage · Science / Chemistry / FDA · Updated {_esc(generated_at)}</div>
      </div>
      <div>{brier_badge}</div>
    </div>

    <div class="grid-4">
      <div class="card">
        <div class="kpi-label">Total Equity</div>
        <div class="kpi-val">${equity:,.2f}</div>
        <div class="kpi-sub">Cash: ${summary.cash_usd:,.2f} · Open: ${summary.open_exposure_usd:,.2f}</div>
      </div>
      <div class="card">
        <div class="kpi-label">Realized PnL</div>
        <div class="kpi-val {pnl_cls}">${summary.realized_pnl_usd:+,.2f}</div>
        <div class="kpi-sub">ROI: {summary.roi_pct:+.2f}% · Start: ${summary.initial_bankroll:,.2f}</div>
      </div>
      <div class="card">
        <div class="kpi-label">Trades & Win Rate</div>
        <div class="kpi-val">{win_rate_str}</div>
        <div class="kpi-sub">{summary.total_trades} total ({summary.open_trades} open, {summary.wins}W / {summary.losses}L)</div>
      </div>
      <div class="card">
        <div class="kpi-label">Brier Score (Lower = Better)</div>
        <div class="kpi-val">{brier_main}</div>
        <div class="kpi-sub">{_esc(brier_sub)}</div>
      </div>
    </div>

    <h2>Equity Curve</h2>
    <div class="card">
      {equity_svg}
    </div>

    <h2>Strategies</h2>
    <div class="grid-3">
      {"".join(strategy_cards)}
    </div>

    <h2>Paper Trades</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ID</th><th>Strategy</th><th>Market</th><th>Side</th>
            <th>Fill</th><th>Model</th><th>Edge</th><th>Stake</th><th>Status</th><th>PnL</th>
          </tr>
        </thead>
        <tbody>
          {"".join(trade_rows)}
        </tbody>
      </table>
    </div>

    <h2>Recent Scans</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Time (UTC)</th><th>Events</th><th>Signals</th><th>Opened</th><th>Notes</th></tr>
        </thead>
        <tbody>
          {"".join(run_rows)}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""


def write_dashboard(engine: PaperTradingEngine, output_path: str | Path = "data/dashboard.html") -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dashboard_html(engine), encoding="utf-8")
    return path
