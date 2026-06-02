from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .config import WV_CRITERIA
from .score import normalize_llm_score


SYSTEM_PROMPT = """
You are a pragmatic venture capital sourcing analyst for Wittington Ventures.
Use only the compact external evidence supplied by the product. Do not invent
funding rounds, customers, or traction. Score whether this is specifically worth
Wittington's attention, not whether it is generally a good logistics company.
Return valid JSON only.
""".strip()


SCHEMA_HINT = {
    "company_type": "startup|incumbent|investor|service_provider|consulting|media_association|university_government_nonprofit|unknown",
    "is_startup_likely": True,
    "sector_tags": ["commerce", "logistics", "ai"],
    "venture_backability": "integer 0-25",
    "wv_sector_fit": "integer 0-25",
    "wittington_edge": "integer 0-20",
    "stage_signal": "integer 0-10",
    "traction_signal": "integer 0-10",
    "data_confidence": "integer 0-10",
    "total_score": "integer 0-100",
    "rationale": "one concise VC-style sentence",
    "evidence_summary": "one concise evidence sentence",
    "confidence": "high|medium|low",
}


def _safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def build_user_prompt(company: dict[str, Any]) -> str:
    evidence = {
        "company_name": company.get("canonical_name"),
        "deterministic_type": company.get("deterministic_type"),
        "deterministic_tags": _safe_json_loads(company.get("deterministic_tags"), []),
        "deterministic_exclusion_reason": company.get("deterministic_exclusion_reason"),
        "search_titles": _safe_json_loads(company.get("top_titles"), []),
        "search_urls": _safe_json_loads(company.get("top_urls"), []),
        "search_snippets": _safe_json_loads(company.get("top_snippets"), []),
        "website": company.get("website"),
        "evidence_source": company.get("evidence_source") or company.get("enrichment_provider") or "unknown",
        "homepage_route": company.get("homepage_route_decision"),
        "homepage_route_reason": company.get("homepage_route_reason"),
        "homepage_domain": company.get("homepage_resolved_url") or company.get("homepage_candidate_domain"),
        "homepage_domain_confidence": company.get("homepage_domain_confidence"),
        "homepage_evidence_quality": company.get("homepage_evidence_quality"),
        "homepage_positive_signals": _safe_json_loads(company.get("homepage_positive_signals"), []),
        "homepage_negative_signals": _safe_json_loads(company.get("homepage_negative_signals"), []),
        "homepage_fetch_error": company.get("homepage_fetch_error"),
    }
    return json.dumps(
        {
            "wittington_criteria": WV_CRITERIA,
            "score_breakdown": {
                "venture_backability": 25,
                "wv_sector_fit": 25,
                "wittington_edge": 20,
                "stage_signal": 10,
                "traction_signal": 10,
                "data_confidence": 10,
            },
            "score_caps": [
                "Non-startup incumbents should not exceed 35.",
                "Investors should not exceed 25.",
                "Media/associations/universities/government should not exceed 20.",
                "Service providers without software/platform evidence should not exceed 45.",
                "Confidence should be low when evidence is thin or ambiguous.",
            ],
            "external_evidence": evidence,
            "return_json_shape": SCHEMA_HINT,
        },
        ensure_ascii=True,
    )


def classify_with_openai(company: dict[str, Any], api_key: str, model: str) -> dict[str, Any]:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(company)},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    raw_content = response.choices[0].message.content or "{}"
    payload = json.loads(raw_content)
    normalized = normalize_llm_score(payload, company.get("deterministic_type"))
    usage = response.usage
    if usage:
        normalized["raw_json"]["usage"] = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
    normalized["raw_json"]["model"] = model
    return normalized
