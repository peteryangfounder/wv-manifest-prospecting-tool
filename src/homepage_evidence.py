from __future__ import annotations

from dataclasses import dataclass

from .web_metadata import PageMetadata


POSITIVE_SIGNALS = {
    "software",
    "platform",
    "saas",
    "api",
    "automation",
    "analytics",
    "machine learning",
    "artificial intelligence",
    " ai ",
    "robotics",
    "autonomous",
    "optimization",
    "optimisation",
    "visibility",
    "workflow",
    "operating system",
    "digital",
    "cloud",
    "wms",
    "tms",
}

WITTINGTON_SIGNALS = {
    "retail",
    "commerce",
    "grocery",
    "pharmacy",
    "healthcare",
    "consumer",
    "loyalty",
    "payments",
    "supply chain",
    "warehouse",
    "fulfillment",
    "fulfilment",
    "last mile",
    "cold chain",
    "climate",
    "carbon",
    "emissions",
    "real estate",
}

NEGATIVE_SIGNALS = {
    "freight forwarding",
    "truckload carrier",
    "customs brokerage",
    "consulting services",
    "staffing",
    "marketing agency",
    "law firm",
    "association",
    "nonprofit",
    "university",
    "bank",
    "media",
    "publication",
}


@dataclass(frozen=True)
class HomepageEvidence:
    evidence_text: str
    evidence_quality: float
    positive_signals: list[str]
    wittington_signals: list[str]
    negative_signals: list[str]


def _combined_text(metadata: PageMetadata) -> str:
    parts = [
        metadata.title,
        metadata.meta_description,
        metadata.og_title,
        metadata.og_description,
        metadata.og_site_name,
        metadata.jsonld_name,
        metadata.jsonld_description,
        " ".join(metadata.headings),
        metadata.text_snippet,
    ]
    return " ".join(part for part in parts if part).strip()


def _find_signals(text: str, signals: set[str]) -> list[str]:
    padded = f" {text.lower()} "
    return sorted(signal.strip() for signal in signals if signal in padded)


def extract_homepage_evidence(metadata: PageMetadata) -> HomepageEvidence:
    evidence_text = _combined_text(metadata)
    positive = _find_signals(evidence_text, POSITIVE_SIGNALS)
    wittington = _find_signals(evidence_text, WITTINGTON_SIGNALS)
    negative = _find_signals(evidence_text, NEGATIVE_SIGNALS)

    length_score = min(0.35, len(evidence_text) / 2000)
    metadata_score = 0.0
    if metadata.title:
        metadata_score += 0.10
    if metadata.meta_description or metadata.og_description:
        metadata_score += 0.15
    if metadata.headings:
        metadata_score += 0.10
    if metadata.jsonld_name or metadata.jsonld_description:
        metadata_score += 0.10
    signal_score = min(0.30, (len(positive) * 0.06) + (len(wittington) * 0.05))
    evidence_quality = max(0.0, min(1.0, length_score + metadata_score + signal_score))

    return HomepageEvidence(
        evidence_text=evidence_text[:1600],
        evidence_quality=evidence_quality,
        positive_signals=positive,
        wittington_signals=wittington,
        negative_signals=negative,
    )
