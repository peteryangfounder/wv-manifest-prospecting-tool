from __future__ import annotations

from dataclasses import dataclass

from .homepage_evidence import HomepageEvidence


@dataclass(frozen=True)
class RouteDecision:
    route: str
    confidence: float
    reason: str


def route_candidate_after_homepage_evidence(
    *,
    domain_status: str,
    evidence: HomepageEvidence | None,
    mode: str = "balanced",
) -> RouteDecision:
    if domain_status == "unresolved" or evidence is None:
        return RouteDecision("needs_tavily", 0.55, "Domain unresolved or homepage evidence missing.")

    positive_count = len(evidence.positive_signals) + len(evidence.wittington_signals)
    negative_count = len(evidence.negative_signals)

    if evidence.evidence_quality >= 0.60 and positive_count >= 2 and negative_count == 0:
        return RouteDecision("score_from_homepage", 0.80, "Homepage has sufficient positive technology and Wittington-fit evidence.")

    if evidence.evidence_quality >= 0.45 and negative_count >= 2 and positive_count == 0:
        return RouteDecision("soft_exclude", 0.75, "Homepage evidence points to a likely non-prospect; keep available for audit.")

    if evidence.evidence_quality < 0.35:
        return RouteDecision("needs_tavily", 0.60, "Homepage evidence is too thin for a reliable decision.")

    if mode == "precision-first":
        return RouteDecision("low_priority_data_gap", 0.55, "Evidence is not strong enough for precision-first scoring.")

    return RouteDecision("needs_tavily", 0.65, "Homepage evidence is ambiguous or incomplete.")
