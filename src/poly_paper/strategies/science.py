"""Strategy #3: Science, Chemistry, Biotech & FDA Niche Strategy.

Why this niche can have real +EV:
- Polymarket science/pharma/FDA/climate/Nobel markets are often traded by generalists
  who misprice clinical-trial base rates, FDA approval timelines, or recent scientific
  literature.
- Combines 100% free, no-key scientific APIs (ClinicalTrials.gov v2, openFDA, OpenAlex)
  with a calibrated superforecaster prompt on free Gemini / NVIDIA NIM APIs, applying
  Bayesian shrinkage toward market price to prevent LLM overconfidence.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ..config import TraderConfig
from ..models import EventBundle, MarketOutcome, TradeSignal
from ..polymarket import PolymarketClient, simulate_buy_fill

USER_AGENT = "poly-paper-trader/0.1 (science-research)"

SCIENCE_KEYWORDS: tuple[str, ...] = (
    "fda",
    "clinical trial",
    "phase 3",
    "phase iii",
    "phase 2",
    "phase ii",
    "drug",
    "vaccine",
    "biotech",
    "pharma",
    "nobel",
    "chemistry",
    "physics",
    "medicine",
    "crispr",
    "gene therapy",
    "alzheimer",
    "cancer",
    "glp-1",
    "ozempic",
    "wegovy",
    "superconductor",
    "fusion",
    "quantum",
    "nasa",
    "spacex",
    "starship",
    "mars",
    "asteroid",
    "bird flu",
    "h5n1",
    "mpox",
    "who ",
    "cdc ",
    "pandemic",
    "climate",
    "co2",
    "emissions",
    "battery",
    "semiconductor",
    "molecule",
    "protein",
    "alphafold",
)

PHARMA_KEYWORDS: tuple[str, ...] = (
    "fda",
    "clinical trial",
    "phase 3",
    "phase iii",
    "phase 2",
    "phase ii",
    "drug",
    "vaccine",
    "approval",
    "approve",
    "nda",
    "bla",
    "pdufa",
    "therapy",
    "biotech",
    "pharma",
    "glp-1",
    "crispr",
    "alzheimer",
    "cancer",
)

SUPERFORECASTER_SYSTEM = (
    "You are a calibrated scientific superforecaster (chemistry, biotech, FDA, physics, climate). "
    "You are given a Polymarket binary question, its resolution rules, current market price, "
    "and fresh snippets from ClinicalTrials.gov, openFDA, and OpenAlex. "
    "Step 1: Anchor on historical base rates (outside view). "
    "Step 2: Adjust cautiously for concrete evidence (inside view). "
    "Step 3: Never be overconfident; if evidence is weak or ambiguous, stay close to market price "
    "and set confidence='low'. "
    "Respond ONLY with valid JSON: "
    '{"fair_prob": 0.50, "confidence": "low|medium|high", "reasoning": "brief scientific rationale"}'
)


@dataclass(frozen=True)
class ScientificEvidence:
    query: str
    clinical_trials: tuple[str, ...]
    fda_records: tuple[str, ...]
    openalex_papers: tuple[str, ...]
    pubchem_compounds: tuple[str, ...] = ()

    def to_text(self) -> str:
        parts: list[str] = []
        if self.fda_records:
            parts.append("openFDA:\n" + "\n".join(f"  - {x}" for x in self.fda_records))
        if self.clinical_trials:
            parts.append("ClinicalTrials.gov:\n" + "\n".join(f"  - {x}" for x in self.clinical_trials))
        if self.pubchem_compounds:
            parts.append("PubChem:\n" + "\n".join(f"  - {x}" for x in self.pubchem_compounds))
        if self.openalex_papers:
            parts.append("OpenAlex recent papers:\n" + "\n".join(f"  - {x}" for x in self.openalex_papers))
        return "\n".join(parts) if parts else "No matching records found in public scientific databases."


CLINICAL_PHASE_BASE_RATES = (
    "Historical Bio/Pharma Base Rates (BIO Industry Analysis): "
    "Phase I -> Approval ~8%; Phase II -> Approval ~15%; Phase III -> Approval ~50%; "
    "NDA/BLA submitted -> FDA Approval ~85% (oncology ~35-45% from Phase III)."
)


def is_science_event(event: EventBundle) -> bool:
    haystack = f"{event.title} {event.slug} {event.description[:300]}".lower()
    return any(kw in haystack for kw in SCIENCE_KEYWORDS)


def is_pharma_market(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in PHARMA_KEYWORDS)


def extract_search_query(question: str) -> str:
    """Extract concise scientific search keywords from a Polymarket question."""
    cleaned = re.sub(
        r"\b(will|the|be|by|in|on|of|to|for|a|an|before|after|during|202\d|yes|no|polymarket|fda|approve|approved|approval)\b",
        " ",
        question,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[^a-zA-Z0-9\-\s]", " ", cleaned)
    tokens = [t for t in cleaned.split() if len(t) >= 3]
    return " ".join(tokens[:6]) or question[:60]


def _get_json(url: str, timeout: float = 12.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_clinical_trials(query: str, limit: int = 3) -> tuple[str, ...]:
    if not query.strip():
        return ()
    params = urllib.parse.urlencode({"query.term": query, "pageSize": limit})
    url = f"https://clinicaltrials.gov/api/v2/studies?{params}"
    try:
        data = _get_json(url)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return ()

    out: list[str] = []
    for study in (data.get("studies") or [])[:limit]:
        proto = study.get("protocolSection") or {}
        ident = proto.get("identificationModule") or {}
        status = proto.get("statusModule") or {}
        design = proto.get("designModule") or {}
        nct = ident.get("nctId", "")
        title = ident.get("briefTitle", "")[:110]
        overall = status.get("overallStatus", "UNKNOWN")
        phases = ",".join(design.get("phases") or []) or "N/A"
        out.append(f"{nct} [{phases} | {overall}]: {title}")
    return tuple(out)


def fetch_openfda_drugs(query: str, limit: int = 3) -> tuple[str, ...]:
    if not query.strip():
        return ()
    # Quote first meaningful term for openFDA brand/generic search
    first_term = re.sub(r"[^a-zA-Z0-9\-]", "", query.split()[0])
    if len(first_term) < 3:
        return ()
    search_expr = f'openfda.brand_name:"{first_term}"+openfda.generic_name:"{first_term}"'
    url = f"https://api.fda.gov/drug/drugsfda.json?search={urllib.parse.quote(search_expr, safe=':+:\"')}&limit={limit}"
    try:
        data = _get_json(url)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return ()

    out: list[str] = []
    for rec in (data.get("results") or [])[:limit]:
        sponsor = rec.get("sponsor_name", "Unknown")
        app_num = rec.get("application_number", "")
        subs = rec.get("submissions") or []
        latest = subs[0] if subs else {}
        sub_status = latest.get("submission_status", "?")
        sub_date = latest.get("submission_status_date", "?")
        out.append(f"{app_num} ({sponsor}): status={sub_status} date={sub_date}")
    return tuple(out)


def fetch_openalex_papers(query: str, limit: int = 3) -> tuple[str, ...]:
    if not query.strip():
        return ()
    params = urllib.parse.urlencode({"search": query, "per-page": limit, "sort": "publication_date:desc"})
    url = f"https://api.openalex.org/works?{params}"
    try:
        data = _get_json(url)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return ()

    out: list[str] = []
    for work in (data.get("results") or [])[:limit]:
        title = str(work.get("title") or "")[:110]
        year = work.get("publication_year", "?")
        cites = work.get("cited_by_count", 0)
        if title:
            out.append(f"({year}, cites={cites}) {title}")
    return tuple(out)


def fetch_pubchem_compound(query: str) -> tuple[str, ...]:
    """Query PubChem PUG REST API for chemical/drug formula and molecular weight."""
    tokens = [re.sub(r"[^a-zA-Z0-9\-]", "", t) for t in query.split()]
    candidates = [t for t in tokens if len(t) >= 4][:2]
    out: list[str] = []
    for name in candidates:
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{urllib.parse.quote(name)}/property/MolecularFormula,MolecularWeight,IUPACName/JSON"
        )
        try:
            data = _get_json(url, timeout=8.0)
            props = (data.get("PropertyTable") or {}).get("Properties") or []
            if props:
                p = props[0]
                out.append(
                    f"{name}: {p.get('MolecularFormula', '?')}, MW={p.get('MolecularWeight', '?')} g/mol"
                )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            continue
    return tuple(out)


def gather_scientific_evidence(market: MarketOutcome) -> ScientificEvidence:
    query = extract_search_query(market.question)
    pharma = is_pharma_market(market.question + " " + market.description[:300])
    trials = fetch_clinical_trials(query) if pharma else ()
    fda = fetch_openfda_drugs(query) if pharma else ()
    pubchem = fetch_pubchem_compound(query) if pharma else ()
    papers = fetch_openalex_papers(query)
    return ScientificEvidence(
        query=query,
        clinical_trials=trials,
        fda_records=fda,
        openalex_papers=papers,
        pubchem_compounds=pubchem,
    )


def _extract_json(text: str) -> dict[str, Any] | None:
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


def estimate_science_probability_llm(
    market: MarketOutcome,
    evidence: ScientificEvidence,
    market_mid: float,
    config: TraderConfig,
    timeout: float = 20.0,
) -> tuple[float, str, str] | None:
    """Use free Gemini or NVIDIA NIM to estimate calibrated YES probability."""
    if not config.llm_enabled or (not config.gemini_api_key and not config.nvidia_api_key):
        return None

    user_prompt = (
        f"Question: {market.question}\n"
        f"Outcome: {market.group_item_title or 'YES'}\n"
        f"End date: {market.end_date}\n"
        f"Current Polymarket YES mid price: {market_mid:.3f}\n"
        f"{CLINICAL_PHASE_BASE_RATES}\n"
        f"Resolution rules:\n{market.description[:1400]}\n\n"
        f"Live scientific database evidence (query='{evidence.query}'):\n{evidence.to_text()}\n\n"
        "Estimate the true probability of YES resolving to 1.0."
    )

    raw_text: str | None = None
    if config.gemini_api_key:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{config.gemini_model}:generateContent?key={config.gemini_api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": SUPERFORECASTER_SYSTEM}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.15, "responseMimeType": "application/json"},
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, TimeoutError):
            raw_text = None

    if raw_text is None and config.nvidia_api_key:
        payload = {
            "model": config.nvidia_model,
            "temperature": 0.15,
            "max_tokens": 900,
            "messages": [
                {"role": "system", "content": SUPERFORECASTER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            req = urllib.request.Request(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config.nvidia_api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            msg = data["choices"][0]["message"]
            raw_text = msg.get("content") or msg.get("reasoning_content")
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, TimeoutError):
            raw_text = None

    if not raw_text:
        return None

    parsed = _extract_json(raw_text)
    if not parsed:
        return None

    try:
        raw_prob = float(parsed["fair_prob"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0.01 <= raw_prob <= 0.99):
        return None

    confidence = str(parsed.get("confidence", "low")).lower().strip()
    reasoning = str(parsed.get("reasoning", "")).strip()
    return raw_prob, confidence, reasoning


def calibrate_with_market_shrinkage(raw_llm_prob: float, market_mid: float, confidence: str) -> float:
    """Shrink raw LLM estimate toward market price to combat LLM overconfidence."""
    weight = 0.65 if confidence == "high" else (0.45 if confidence == "medium" else 0.15)
    return max(0.01, min(0.99, weight * raw_llm_prob + (1.0 - weight) * market_mid))


def scan_science_markets(
    client: PolymarketClient,
    events: list[EventBundle],
    config: TraderConfig,
    max_markets_to_analyze: int = 8,
    evidence_fetcher: Any = gather_scientific_evidence,
    llm_estimator: Any = estimate_science_probability_llm,
) -> list[TradeSignal]:
    """Scan science/pharma/chemistry/FDA markets using public scientific DBs + calibrated LLM."""
    signals: list[TradeSignal] = []
    analyzed = 0

    for event in events:
        if analyzed >= max_markets_to_analyze:
            break
        if not is_science_event(event):
            continue

        for market in event.markets:
            if analyzed >= max_markets_to_analyze:
                break
            if market.closed:
                continue
            # Focus on markets with real uncertainty (avoid 98c/2c settled noise)
            if not (0.06 <= market.yes_price <= 0.94):
                continue

            analyzed += 1
            try:
                yes_book = client.fetch_orderbook(market.yes_token_id)
                no_book = client.fetch_orderbook(market.no_token_id)
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
                continue

            yes_sim = simulate_buy_fill(
                yes_book,
                target_usd=max(config.min_stake_usd, 25.0),
                slippage_buffer=config.slippage_buffer,
                max_spread=config.max_spread,
                min_fill_ratio=config.min_fill_ratio,
                taker_fee_rate=config.taker_fee_rate,
                max_price_impact=config.max_price_impact,
                min_bid_price=config.min_bid_price,
            )
            no_sim = simulate_buy_fill(
                no_book,
                target_usd=max(config.min_stake_usd, 25.0),
                slippage_buffer=config.slippage_buffer,
                max_spread=config.max_spread,
                min_fill_ratio=config.min_fill_ratio,
                taker_fee_rate=config.taker_fee_rate,
                max_price_impact=config.max_price_impact,
                min_bid_price=config.min_bid_price,
            )
            if not yes_sim.executable and not no_sim.executable:
                continue

            market_mid = yes_sim.mid_price if yes_sim.executable else (1.0 - no_sim.mid_price)
            evidence = evidence_fetcher(market)
            llm_result = llm_estimator(market, evidence, market_mid, config)
            if llm_result is None:
                continue

            raw_prob, confidence, reasoning = llm_result
            if confidence not in {"medium", "high"}:
                continue

            calibrated_yes = calibrate_with_market_shrinkage(raw_prob, market_mid, confidence)
            note = f"Science[{confidence}] raw={raw_prob:.2f} cal={calibrated_yes:.2f}: {reasoning[:160]}"

            if yes_sim.executable and (config.min_price <= yes_sim.effective_price <= config.max_price):
                edge_yes = calibrated_yes - yes_sim.effective_price
                if edge_yes >= config.min_edge:
                    signals.append(
                        TradeSignal(
                            strategy="science_research",
                            event_id=market.event_id,
                            market_id=market.market_id,
                            token_id=market.yes_token_id,
                            question=market.question,
                            outcome_label=market.group_item_title or market.question,
                            side="YES",
                            model_prob=calibrated_yes,
                            market_mid=yes_sim.mid_price,
                            expected_edge=edge_yes,
                            description=market.description,
                            notes=note,
                        )
                    )

            if no_sim.executable and (config.min_price <= no_sim.effective_price <= config.max_price):
                calibrated_no = 1.0 - calibrated_yes
                edge_no = calibrated_no - no_sim.effective_price
                if edge_no >= config.min_edge:
                    signals.append(
                        TradeSignal(
                            strategy="science_research",
                            event_id=market.event_id,
                            market_id=market.market_id,
                            token_id=market.no_token_id,
                            question=market.question,
                            outcome_label=market.group_item_title or market.question,
                            side="NO",
                            model_prob=calibrated_no,
                            market_mid=no_sim.mid_price,
                            expected_edge=edge_no,
                            description=market.description,
                            notes=note,
                        )
                    )

    return signals
