from __future__ import annotations

import re
from dataclasses import dataclass

from .clean import normalize_name


NON_TARGET_TYPES = {
    "incumbent_or_public_company",
    "investor_or_financial_firm",
    "consulting_or_agency",
    "media_event_association",
    "university_government_nonprofit",
    "retailer_or_brand_incumbent",
    "duplicate_or_noisy_entry",
}

KNOWN_INCUMBENTS = {
    "3m",
    "7 eleven",
    "abb robotics",
    "ab inbev",
    "acer",
    "airbus",
    "alaska airline",
    "alibaba com",
    "aliexpress",
    "amazon",
    "amazon freight",
    "amazon robotics",
    "amazon shipping",
    "amazon web services",
    "american airlines",
    "apple",
    "aritzia",
    "autozone",
    "best buy",
    "canadian tire",
    "caterpillar",
    "cisco",
    "coca cola",
    "costco",
    "dhl",
    "fedex",
    "google",
    "home depot",
    "ibm",
    "ikea",
    "intel",
    "j b hunt",
    "kroger",
    "loblaw",
    "lowes",
    "maersk",
    "microsoft",
    "nestle",
    "nike",
    "oracle",
    "pepsico",
    "procter and gamble",
    "samsung",
    "target",
    "tesla",
    "the home depot",
    "toyota",
    "ups",
    "usps",
    "walmart",
    "wayfair",
}

SELF_OR_RELATED = {
    "wittington ventures",
    "wittington",
}

INVESTOR_KEYWORDS = (
    "ventures",
    "venture",
    "capital",
    "private equity",
    "equity partners",
    "growth equity",
    "investment",
    "investments",
    "asset management",
    "wealth",
    "bank",
    "securities",
    "partners",
    "accelerator",
    "family office",
)

CONSULTING_KEYWORDS = (
    "consulting",
    "consultants",
    "advisory",
    "advisors",
    "agency",
    "communications",
    "public relations",
    "law",
    "legal",
    "llp",
    "mckinsey",
    "bain",
    "bcg",
    "deloitte",
    "accenture",
    "pwc",
    "kpmg",
    "alixpartners",
)

MEDIA_ASSOCIATION_KEYWORDS = (
    "association",
    "council",
    "institute",
    "magazine",
    "podcast",
    "journal",
    "media",
    "conference",
    "events",
    "expo",
    "news",
    "freightwaves",
)

UNIVERSITY_GOVERNMENT_KEYWORDS = (
    "university",
    "college",
    "government",
    "ministry",
    "department of",
    "city of",
    "state of",
    "port authority",
    "economic development",
    "nonprofit",
    "foundation",
)

RETAIL_BRAND_KEYWORDS = (
    "retail",
    "foods",
    "foodservice",
    "beverage",
    "apparel",
    "fashion",
    "beauty",
    "cosmetics",
    "brands",
    "cpg",
)

LOGISTICS_SERVICE_KEYWORDS = (
    "logistics",
    "freight",
    "trucking",
    "transport",
    "transportation",
    "shipping",
    "courier",
    "warehouse",
    "warehousing",
    "distribution",
    "3pl",
    "forwarding",
    "drayage",
    "carrier",
    "cargo",
    "fulfillment",
    "last mile",
    "supply chain service",
)

TECH_SIGNALS = (
    "ai",
    "robot",
    "robotics",
    "software",
    "systems",
    "platform",
    "automation",
    "data",
    "analytics",
    "saas",
    "autonomous",
    "visibility",
    "optimization",
    "optimisation",
    "tms",
    "wms",
    "digital",
    "cloud",
    "intelligence",
    "technology",
    "technologies",
    "tech",
    "marketplace",
    "api",
    "machine learning",
    "computer vision",
    "blockchain",
)

HIGH_PRIORITY_TECH_SIGNALS = (
    "ai",
    "robot",
    "robotics",
    "saas",
    "software",
    "platform",
    "automation",
    "analytics",
    "visibility",
    "autonomous",
    "optimization",
    "optimisation",
    "wms",
    "tms",
    "machine learning",
    "computer vision",
)

HIGH_PRIORITY_DOMAIN_SIGNALS = (
    "warehouse automation",
    "retail infrastructure",
    "healthcare operations",
    "supply chain tech",
    "supply chain technology",
    "climate",
    "sustainability",
    "carbon",
    "emissions",
)

SECTOR_KEYWORDS = {
    "commerce": ("commerce", "ecommerce", "e commerce", "retail", "marketplace", "checkout", "merchant"),
    "healthcare": ("health", "healthcare", "pharma", "pharmacy", "medical", "clinical", "care"),
    "consumer": ("consumer", "brand", "cpg", "beauty", "food", "apparel", "wellness"),
    "food": ("food", "grocery", "restaurant", "beverage", "agri", "agriculture"),
    "climate": ("climate", "carbon", "emission", "sustainability", "electric", "energy", "renewable"),
    "logistics": ("logistics", "freight", "transport", "shipping", "delivery", "fleet", "carrier"),
    "supply_chain": ("supply chain", "procurement", "warehouse", "inventory", "fulfillment", "planning"),
    "retail_infrastructure": ("retail", "store", "omnichannel", "checkout", "grocery", "merchandising"),
    "warehouse_automation": ("warehouse", "robot", "automation", "fulfillment", "picking"),
    "robotics": ("robot", "robotics", "autonomous"),
    "ai": (" ai", "artificial intelligence", "machine learning", "intelligence"),
    "fintech": ("fintech", "payments", "financial", "insurance"),
}


@dataclass(frozen=True)
class RuleResult:
    deterministic_type: str
    is_candidate: bool
    high_priority_enrichment: bool
    exclusion_reason: str | None
    tags: list[str]


def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    for keyword in keywords:
        clean_keyword = keyword.strip()
        if not clean_keyword:
            continue
        if len(clean_keyword) <= 3 and clean_keyword.replace(" ", "").isalnum():
            if f" {clean_keyword} " in text:
                return True
            continue
        if keyword in text:
            return True
    return False


def _has_ai_brand_signal(raw_name: str, normalized: str) -> bool:
    compact = (raw_name or "").strip()
    lower = compact.lower()
    if ".ai" in lower or lower.endswith(".ai"):
        return True
    if re.search(r"(^|[\s._-])ai($|[\s._-])", lower):
        return True
    if compact.endswith("AI") and len(normalized) > 3:
        return True
    return False


def _is_high_priority_candidate(
    raw_name: str,
    normalized: str,
    text: str,
    deterministic_type: str,
    tags: list[str],
) -> bool:
    if deterministic_type != "likely_startup_or_tech":
        return False

    strong_tech = _has_any(text, HIGH_PRIORITY_TECH_SIGNALS) or _has_ai_brand_signal(raw_name, normalized)
    domain_signal = _has_any(text, HIGH_PRIORITY_DOMAIN_SIGNALS)
    priority_tags = {
        "ai",
        "robotics",
        "warehouse_automation",
        "retail_infrastructure",
        "healthcare",
        "climate",
    }

    if strong_tech:
        return True
    if domain_signal and {"logistics", "supply_chain", "retail_infrastructure", "healthcare", "climate"}.intersection(tags):
        return True
    if priority_tags.intersection(tags) and _has_any(text, TECH_SIGNALS):
        return True

    return False


def detect_sector_tags(raw_name: str) -> list[str]:
    text = f" {normalize_name(raw_name)} "
    tags = []
    for tag, keywords in SECTOR_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            tags.append(tag)
    return tags


def classify_company_name(raw_name: str) -> RuleResult:
    normalized = normalize_name(raw_name)
    text = f" {normalized} "
    tags = detect_sector_tags(raw_name)
    has_tech = _has_any(text, TECH_SIGNALS)
    has_logistics = _has_any(text, LOGISTICS_SERVICE_KEYWORDS)
    has_tech = has_tech or _has_ai_brand_signal(raw_name, normalized)

    if len(normalized) < 2 or normalized in {"na", "none", "unknown", "test"}:
        return RuleResult("duplicate_or_noisy_entry", False, False, "Noisy or incomplete attendee entry.", tags)

    if normalized in SELF_OR_RELATED:
        return RuleResult("duplicate_or_noisy_entry", False, False, "Wittington-related entry, not a prospect.", tags)

    if normalized in KNOWN_INCUMBENTS:
        return RuleResult(
            "incumbent_or_public_company",
            False,
            False,
            "Known large incumbent or public company.",
            tags,
        )

    if _has_any(text, INVESTOR_KEYWORDS):
        return RuleResult(
            "investor_or_financial_firm",
            False,
            False,
            "Investor or financial services firm rather than an operating startup.",
            tags,
        )

    if _has_any(text, MEDIA_ASSOCIATION_KEYWORDS):
        return RuleResult(
            "media_event_association",
            False,
            False,
            "Media, event, or association attendee rather than a startup prospect.",
            tags,
        )

    if _has_any(text, UNIVERSITY_GOVERNMENT_KEYWORDS):
        return RuleResult(
            "university_government_nonprofit",
            False,
            False,
            "University, government, port authority, or nonprofit entry.",
            tags,
        )

    if _has_any(text, CONSULTING_KEYWORDS):
        return RuleResult(
            "consulting_or_agency",
            False,
            False,
            "Consulting, agency, advisory, or legal services firm.",
            tags,
        )

    if has_logistics and not has_tech:
        return RuleResult(
            "logistics_service_provider",
            False,
            False,
            "Likely logistics services provider; no obvious software or platform signal.",
            tags or ["logistics"],
        )

    if has_tech:
        priority_tags = tags or ["technology"]
        return RuleResult(
            "likely_startup_or_tech",
            True,
            _is_high_priority_candidate(raw_name, normalized, text, "likely_startup_or_tech", priority_tags),
            None,
            priority_tags,
        )

    if _has_any(text, RETAIL_BRAND_KEYWORDS) and not has_tech:
        return RuleResult(
            "retailer_or_brand_incumbent",
            False,
            False,
            "Likely retailer, brand, or CPG incumbent rather than venture-backed technology.",
            tags,
        )

    return RuleResult("unknown_needs_enrichment", True, False, None, tags)
