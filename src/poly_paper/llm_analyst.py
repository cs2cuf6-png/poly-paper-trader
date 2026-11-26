"""Optional free LLM rule-checker using Google AI Studio (Gemini) or NVIDIA NIM."""

from __future__ import annotations

import concurrent.futures
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
    """Runs Gemini, Mistral, and NVIDIA Nemotron in parallel to audit resolution rules."""

    def __init__(self, config: TraderConfig, timeout: float = 35.0) -> None:
        self.config = config
        self.timeout = timeout

    def audit_signal(self, signal: TradeSignal) -> LLMVerdict:
        if not self.config.llm_enabled:
            return LLMVerdict(approved=True, provider="disabled", reason="LLM audit disabled")

        prompt = _build_prompt(signal)
        tasks: list[tuple[str, Any]] = []
        if self.config.gemini_api_key:
            tasks.append(("gemini", lambda: self._call_gemini(prompt)))
        if self.config.nvidia_api_key:
            if self.config.nvidia_model:
                tasks.append(("mistral", lambda m=self.config.nvidia_model: self._call_nvidia(prompt, m)))
            if self.config.nvidia_native_model and self.config.nvidia_native_model != self.config.nvidia_model:
                tasks.append(("nemotron", lambda m=self.config.nvidia_native_model: self._call_nvidia(prompt, m)))

        if not tasks:
            return LLMVerdict(
                approved=True,
                provider="none",
                reason="No LLM API key configured (passed quant checks only)",
            )

        verdicts: list[LLMVerdict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = [pool.submit(fn) for _, fn in tasks]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    res = fut.result()
                    if res is not None:
                        verdicts.append(res)
                except Exception:
                    continue

        if not verdicts:
            return LLMVerdict(
                approved=True,
                provider="none",
                reason="LLM API call failed or rejected key (passed quant checks only)",
            )

        approves = sum(1 for v in verdicts if v.approved)
        rejects = len(verdicts) - approves
        approved = approves >= rejects
        providers = ",".join(v.provider for v in verdicts)
        flags = tuple(dict.fromkeys(f for v in verdicts for f in v.risk_flags))
        reasons = " | ".join(f"{v.provider}:{v.reason}" for v in verdicts)
        return LLMVerdict(
            approved=approved,
            provider=f"ensemble[{providers}]",
            reason=reasons[:240],
            risk_flags=flags,
        )

    def _call_gemini(self, prompt: str) -> LLMVerdict | None:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.gemini_model}:generateContent"
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
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.config.gemini_api_key,
                },
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

    def _call_nvidia(self, prompt: str, model_name: str | None = None) -> LLMVerdict | None:
        model = model_name or self.config.nvidia_model
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        payload = {
            "model": model,
            "temperature": 0.1,
            "max_tokens": 1400,
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
                provider=f"nvidia:{model}",
                reason=str(parsed.get("reason", "OK")),
                risk_flags=tuple(str(x) for x in (parsed.get("risk_flags") or [])),
            )
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, TimeoutError):
            return None
