"""Optional zero-dependency Telegram notifier for trade & resolution alerts."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .config import TraderConfig


class TelegramNotifier:
    """Sends concise markdown alerts to Telegram when token + chat_id are configured."""

    def __init__(self, config: TraderConfig, timeout: float = 10.0) -> None:
        self.token = config.telegram_bot_token
        self.chat_id = config.telegram_chat_id
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text[:3900],
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status == 200
        except (urllib.error.URLError, TimeoutError):
            return False

    def notify_cycle(self, opened: list[dict[str, Any]], settled: list[dict[str, Any]], equity: float) -> bool:
        if not self.enabled or (not opened and not settled):
            return False
        lines = [f"<b>Polymarket Paper Trader</b> · Equity: <b>${equity:,.2f}</b>"]
        for t in opened[:6]:
            lines.append(
                f"🟢 <b>OPEN</b> [{t['strategy']}] {t['side']} {t['outcome_label']} "
                f"@ ${t['fill_price']:.3f} (edge {t['edge']:+.1%}, ${t['stake_usd']:.2f})"
            )
        for s in settled[:6]:
            icon = "✅" if s["pnl_usd"] >= 0 else "❌"
            lines.append(
                f"{icon} <b>CLOSE</b> #{s['id']} {s['strategy']} {s['side']} | PnL <b>${s['pnl_usd']:+.2f}</b>"
            )
        return self.send("\n".join(lines))
