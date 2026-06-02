from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from urllib.parse import urlparse

from .clean import normalize_name
from .web_metadata import PageMetadata


DOMAIN_SUFFIXES = ("com", "ai", "io", "co")
DOMAIN_PREFIXES = ("", "get", "try")
DOMAIN_SUFFIX_PATTERNS = ("", "hq")


@dataclass(frozen=True)
class DomainResolution:
    company_id: int
    candidate_domain: str
    resolved_url: str | None
    confidence: float
    status: str
    metadata: PageMetadata | None = None
    fetch_error: str | None = None


def company_slug(company_name: str) -> str:
    normalized = normalize_name(company_name)
    normalized = re.sub(r"\b(inc|llc|ltd|limited|corp|corporation|company|co)\b", " ", normalized)
    return re.sub(r"[^a-z0-9]+", "", normalized)


def generate_domain_candidates(company_name: str) -> list[str]:
    slug = company_slug(company_name)
    if len(slug) < 3:
        return []

    domains: list[str] = []
    for suffix in DOMAIN_SUFFIXES:
        domains.append(f"{slug}.{suffix}")
    for prefix in DOMAIN_PREFIXES[1:]:
        domains.append(f"{prefix}{slug}.com")
    for pattern in DOMAIN_SUFFIX_PATTERNS[1:]:
        domains.append(f"{slug}{pattern}.com")

    deduped = []
    seen = set()
    for domain in domains:
        if domain not in seen:
            deduped.append(domain)
            seen.add(domain)
    return deduped


def _token_set(value: str) -> set[str]:
    normalized = normalize_name(value)
    return {token for token in normalized.split() if len(token) >= 3}


def _domain_tokens(domain: str) -> str:
    parsed = urlparse(domain if "://" in domain else f"https://{domain}")
    host = parsed.netloc or parsed.path
    host = host.lower().removeprefix("www.")
    return host.split(".")[0]


def _contains_company_token(metadata_text: str, company_tokens: set[str]) -> bool:
    text = normalize_name(metadata_text)
    return any(token in text for token in company_tokens)


def score_domain_confidence(company_name: str, domain: str, metadata: PageMetadata | None) -> float:
    slug = company_slug(company_name)
    if not slug:
        return 0.0

    domain_root = _domain_tokens(domain)
    company_tokens = _token_set(company_name)
    domain_similarity = SequenceMatcher(None, slug, domain_root).ratio()
    score = 0.25 * domain_similarity

    if domain_root == slug:
        score += 0.35
    elif slug in domain_root or domain_root in slug:
        score += 0.15

    if metadata is not None:
        title_blob = " ".join(
            [
                metadata.title or "",
                metadata.meta_description or "",
                metadata.og_title or "",
                metadata.og_description or "",
                metadata.og_site_name or "",
            ]
        )
        heading_blob = " ".join(metadata.headings)
        jsonld_blob = " ".join([metadata.jsonld_name or "", metadata.jsonld_description or ""])

        if _contains_company_token(title_blob, company_tokens):
            score += 0.20
        if _contains_company_token(heading_blob, company_tokens):
            score += 0.10
        if _contains_company_token(jsonld_blob, company_tokens):
            score += 0.15

        metadata_text = normalize_name(f"{title_blob} {heading_blob} {jsonld_blob}")
        wrong_entity_terms = ("university", "bank", "capital", "ventures", "law firm", "restaurant")
        if any(term in metadata_text for term in wrong_entity_terms) and not any(
            term in normalize_name(company_name) for term in wrong_entity_terms
        ):
            score -= 0.20

    return max(0.0, min(1.0, score))


def domain_status(confidence: float) -> str:
    if confidence >= 0.85:
        return "accepted"
    if confidence >= 0.60:
        return "provisional"
    return "unresolved"
