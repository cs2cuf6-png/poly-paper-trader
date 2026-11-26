"""Optional free LLM rule-checker using Google AI Studio (Gemini) or NVIDIA NIM."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from .config import TraderConfig
from .models import LLMVerdict, TradeSignal

SYSTEM_PROMPT = (
    "You are a skeptical Polymarket risk & resolution-rule auditor. "
    "Your ONLY job is to catch traps in market resolution rules, station/unit mismatches, "
    "ambiguous wording, or obvious logical errors before a paper-trading bot enters a trade. "
    "Respond ONLY with valid JSON: "
    '{"approve": true/false, "risk_flags": ["..."], "reason": "short explanation"}'
)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(cleaned[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _build_prompt(signal: TradeSignal) -> str:
    return (
        f"Strategy: {signal.strategy}\n"
        f"Question: {signal.question}\n"
        f"Outcome label: {signal.outcome_label}\n"
        f"Proposed trade: BUY {signal.side}\n"
        f"Model win probability: {signal.model_prob:.3f}\n"
        f"Market mid price: {signal.market_mid:.3f}\n"
        f"Expected edge: {signal.expected_edge:.3f}\n"
        f"Model notes: {signal.notes}\n"
        f"Resolution rules:\n{signal.description[:1500]}\n\n"
        "Should the bot APPROVE this paper trade? Reject ONLY if there is a clear rule trap, "
        "wrong station/date/unit mismatch, or ambiguous resolution criteria."
    )


class LLMAnalyst:
    """Checks trade signals against resolution rules using free Gemini or NVIDIA APIs."""

    def __init__(self, config: TraderConfig, timeout: float = 20.0) -> None:
        self.config = config
        self.timeout = timeout

    def audit_signal(self, signal: TradeSignal) -> LLMVerdict:
        if not self.config.llm_enabled:
            return LLMVerdict(approved=True, provider="disabled", reason="LLM audit disabled")

        prompt = _build_prompt(signal)

        if self.config.gemini_api_key:
            verdict = self._call_gemini(prompt)
            if verdict is not None:
                return verdict

        if self.config.nvidia_api_key:
            verdict = self._call_nvidia(prompt)
            if verdict is not None:
                return verdict

        has_keys = bool(self.config.gemini_api_key or self.config.nvidia_api_key)
        reason = (
            "LLM API call failed or rejected key (passed quant checks only)"
            if has_keys
            else "No LLM API key configured (passed quant checks only)"
        )
        return LLMVerdict(
            approved=True,
            provider="none",
            reason=reason,
        )

    def _call_gemini(self, prompt: str) -> LLMVerdict | None:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.gemini_model}:generateContent?key={self.config.gemini_api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = _extract_json_object(text)
            if not parsed:
                return None
            return LLMVerdict(
                approved=bool(parsed.get("approve", True)),
                provider=f"gemini:{self.config.gemini_model}",
                reason=str(parsed.get("reason", "OK")),
                risk_flags=tuple(str(x) for x in (parsed.get("risk_flags") or [])),
            )
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, TimeoutError):
            return None

    def _call_nvidia(self, prompt: str) -> LLMVerdict | None:
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        payload = {
            "model": self.config.nvidia_model,
            "temperature": 0.1,
            "max_tokens": 800,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.config.nvidia_api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            msg = data["choices"][0]["message"]
            text = msg.get("content") or msg.get("reasoning_content") or ""
            parsed = _extract_json_object(text)
            if not parsed:
                return None
            return LLMVerdict(
                approved=bool(parsed.get("approve", True)),
                provider=f"nvidia:{self.config.nvidia_model}",
                reason=str(parsed.get("reason", "OK")),
                risk_flags=tuple(str(x) for x in (parsed.get("risk_flags") or [])),
            )
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, TimeoutError):
            return None
