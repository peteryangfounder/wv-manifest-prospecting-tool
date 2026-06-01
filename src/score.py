from __future__ import annotations

import json
from typing import Any


CAPS_BY_DETERMINISTIC_TYPE = {
    "incumbent_or_public_company": 35,
    "investor_or_financial_firm": 25,
    "media_event_association": 20,
    "university_government_nonprofit": 20,
    "duplicate_or_noisy_entry": 10,
}


def apply_score_caps(
    total_score: int,
    deterministic_type: str | None,
    *,
    has_external_evidence: bool,
    has_strong_startup_evidence: bool = False,
) -> int:
    score = max(0, min(100, int(total_score)))

    if deterministic_type in CAPS_BY_DETERMINISTIC_TYPE and not has_strong_startup_evidence:
        score = min(score, CAPS_BY_DETERMINISTIC_TYPE[deterministic_type])
    if deterministic_type == "logistics_service_provider" and not has_strong_startup_evidence:
        score = min(score, 45)
    if not has_external_evidence:
        score = min(score, 50)

    return score


def _baseline_components(deterministic_type: str | None, tags: list[str]) -> dict[str, int]:
    wv_tags = {"commerce", "healthcare", "consumer", "food", "climate"}
    infra_tags = {"logistics", "supply_chain", "retail_infrastructure", "warehouse_automation", "robotics", "ai"}

    sector_fit = 8
    if wv_tags.intersection(tags):
        sector_fit += 8
    if infra_tags.intersection(tags):
        sector_fit += 7
    sector_fit = min(25, sector_fit)

    if deterministic_type == "likely_startup_or_tech":
        venture = 18
        edge = 14 if infra_tags.intersection(tags) or wv_tags.intersection(tags) else 9
    elif deterministic_type == "unknown_needs_enrichment":
        venture = 11
        edge = 8
    elif deterministic_type == "logistics_service_provider":
        venture = 7
        edge = 9
    elif deterministic_type in {"retailer_or_brand_incumbent", "incumbent_or_public_company"}:
        venture = 4
        edge = 5
    else:
        venture = 3
        edge = 3

    return {
        "venture_backability": venture,
        "wv_sector_fit": sector_fit,
        "wittington_edge": edge,
        "stage_signal": 2,
        "traction_signal": 2,
        "data_confidence": 3,
    }


def baseline_score(company: dict[str, Any]) -> dict[str, Any]:
    deterministic_type = company.get("deterministic_type") or "unknown_needs_enrichment"
    tags = json.loads(company.get("deterministic_tags") or "[]")
    components = _baseline_components(deterministic_type, tags)
    raw_total = sum(components.values())
    capped_total = apply_score_caps(
        raw_total,
        deterministic_type,
        has_external_evidence=False,
        has_strong_startup_evidence=deterministic_type == "likely_startup_or_tech",
    )

    if deterministic_type == "likely_startup_or_tech":
        rationale = "Potential fit. Deterministic signals suggest a technology company. External evidence should confirm stage, product, and Wittington edge."
        company_type = "startup"
        startup_likely = 1
    elif deterministic_type == "unknown_needs_enrichment":
        rationale = "Needs enrichment: not an obvious non-prospect, but product, stage, and startup status are unclear from the attendee name alone."
        company_type = "unknown"
        startup_likely = 0
    else:
        reason = company.get("deterministic_exclusion_reason") or "Deterministic rules indicate this is probably outside Wittington's startup mandate."
        rationale = f"Low fit: {reason}"
        company_type = "service_provider" if deterministic_type == "logistics_service_provider" else "incumbent"
        startup_likely = 0

    return {
        "company_type": company_type,
        "is_startup_likely": startup_likely,
        "sector_tags": tags,
        **components,
        "total_score": capped_total,
        "rationale": rationale,
        "evidence_summary": "Baseline deterministic screen only. No paid external evidence used.",
        "confidence": "low",
        "raw_json": {"provider": "baseline", "deterministic_type": deterministic_type},
    }


def normalize_llm_score(payload: dict[str, Any], deterministic_type: str | None) -> dict[str, Any]:
    def bounded_int(name: str, maximum: int) -> int:
        value = payload.get(name, 0)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 0
        return max(0, min(maximum, value))

    sector_tags = payload.get("sector_tags") or []
    if isinstance(sector_tags, str):
        sector_tags = [tag.strip() for tag in sector_tags.split(",") if tag.strip()]
    if not isinstance(sector_tags, list):
        sector_tags = []

    normalized = {
        "company_type": str(payload.get("company_type") or "unknown"),
        "is_startup_likely": 1 if bool(payload.get("is_startup_likely")) else 0,
        "sector_tags": [str(tag) for tag in sector_tags],
        "venture_backability": bounded_int("venture_backability", 25),
        "wv_sector_fit": bounded_int("wv_sector_fit", 25),
        "wittington_edge": bounded_int("wittington_edge", 20),
        "stage_signal": bounded_int("stage_signal", 10),
        "traction_signal": bounded_int("traction_signal", 10),
        "data_confidence": bounded_int("data_confidence", 10),
        "rationale": str(payload.get("rationale") or "No rationale returned."),
        "evidence_summary": str(payload.get("evidence_summary") or "No evidence summary returned."),
        "confidence": str(payload.get("confidence") or "low").lower(),
    }
    summed = (
        normalized["venture_backability"]
        + normalized["wv_sector_fit"]
        + normalized["wittington_edge"]
        + normalized["stage_signal"]
        + normalized["traction_signal"]
        + normalized["data_confidence"]
    )
    has_strong_startup_evidence = bool(normalized["is_startup_likely"]) and normalized["venture_backability"] >= 15
    normalized["total_score"] = apply_score_caps(
        summed,
        deterministic_type,
        has_external_evidence=True,
        has_strong_startup_evidence=has_strong_startup_evidence,
    )
    normalized["raw_json"] = {"provider": "openai", "payload": payload}
    return normalized
