from pathlib import Path

from src import db, pipeline
from src.clean import CompanyRecord
from src.config import Settings
from src.domain_resolver import generate_domain_candidates, score_domain_confidence
from src.evidence_routing import route_candidate_after_homepage_evidence
from src.homepage_evidence import extract_homepage_evidence
from src.web_metadata import extract_page_metadata


POSITIVE_HTML = """
<html>
  <head>
    <title>Northstar Labs | Retail operations platform</title>
    <meta name="description" content="AI software platform for retail supply chain visibility and warehouse workflow automation.">
    <meta property="og:description" content="Modern analytics and automation for grocery fulfillment teams.">
    <script type="application/ld+json">
      {"@type":"Organization","name":"Northstar Labs","description":"Supply chain software platform"}
    </script>
  </head>
  <body><h1>Retail supply chain software</h1><h2>Warehouse automation and analytics</h2></body>
</html>
"""

NEGATIVE_HTML = """
<html>
  <head>
    <title>Northstar Freight Brokerage</title>
    <meta name="description" content="Freight forwarding, customs brokerage, truckload carrier, and staffing services.">
  </head>
  <body><h1>Logistics services</h1><h2>Customs brokerage</h2></body>
</html>
"""


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_path=tmp_path / "prospects.db",
        manifest_url="https://manife.st/who-attends/",
        openai_api_key=None,
        tavily_api_key=None,
        openai_model="test-model",
        max_enrich=10,
        max_score=10,
        tavily_max_results=1,
        openai_input_cost_per_1m_tokens=0.15,
        openai_output_cost_per_1m_tokens=0.60,
        tavily_cost_per_call_usd=0.001,
        homepage_evidence_max_per_run=10,
    )


def test_domain_candidate_generation_is_conservative() -> None:
    domains = generate_domain_candidates("Northstar Labs, Inc.")

    assert domains[:4] == ["northstarlabs.com", "northstarlabs.ai", "northstarlabs.io", "northstarlabs.co"]
    assert "getnorthstarlabs.com" in domains
    assert "northstar-labs.com" not in domains


def test_metadata_extraction_and_domain_confidence_accepts_correct_domain() -> None:
    metadata = extract_page_metadata(POSITIVE_HTML, "https://northstarlabs.com")
    confidence = score_domain_confidence("Northstar Labs", "northstarlabs.com", metadata)

    assert metadata.title == "Northstar Labs | Retail operations platform"
    assert metadata.meta_description.startswith("AI software platform")
    assert metadata.jsonld_name == "Northstar Labs"
    assert confidence >= 0.85


def test_domain_confidence_rejects_wrong_entity_domain() -> None:
    metadata = extract_page_metadata(NEGATIVE_HTML, "https://northstarbank.com")
    confidence = score_domain_confidence("Northstar Labs", "northstarbank.com", metadata)

    assert confidence < 0.60


def test_homepage_evidence_and_routing_detect_positive_and_negative_signals() -> None:
    positive = extract_homepage_evidence(extract_page_metadata(POSITIVE_HTML, "https://northstarlabs.com"))
    negative = extract_homepage_evidence(extract_page_metadata(NEGATIVE_HTML, "https://northstarfreight.com"))

    assert {"platform", "software", "automation", "analytics"}.intersection(positive.positive_signals)
    assert {"retail", "supply chain", "warehouse"}.intersection(positive.wittington_signals)
    assert route_candidate_after_homepage_evidence(domain_status="accepted", evidence=positive).route == "score_from_homepage"
    assert {"freight forwarding", "customs brokerage", "truckload carrier"}.intersection(negative.negative_signals)
    assert route_candidate_after_homepage_evidence(domain_status="accepted", evidence=negative).route == "soft_exclude"


def test_unresolved_domain_routes_to_tavily_not_exclusion() -> None:
    route = route_candidate_after_homepage_evidence(domain_status="unresolved", evidence=None)

    assert route.route == "needs_tavily"


def test_homepage_evidence_cache_prevents_repeated_fetch(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    conn = db.connect(settings.database_path)
    db.init_db(conn)
    db.upsert_companies(
        conn,
        [CompanyRecord(raw_name="Northstar Labs", canonical_name="Northstar Labs", normalized_name="northstar labs")],
    )
    company = db.list_companies(conn)[0]
    db.update_deterministic_result(conn, company["id"], "unknown_needs_enrichment", None, [], True, False)
    conn.commit()

    calls = {"count": 0}

    def fake_collect(company_row, session, *, timeout, max_bytes, mode):
        calls["count"] += 1
        metadata = extract_page_metadata(POSITIVE_HTML, "https://northstarlabs.com")
        evidence = extract_homepage_evidence(metadata)
        route = route_candidate_after_homepage_evidence(domain_status="accepted", evidence=evidence)
        return {
            "company_id": company_row["id"],
            "candidate_domain": "northstarlabs.com",
            "resolved_url": "https://northstarlabs.com",
            "domain_confidence": 0.95,
            "domain_status": "accepted",
            "metadata_json": metadata.to_dict(),
            "evidence_text": evidence.evidence_text,
            "evidence_quality": evidence.evidence_quality,
            "positive_signals": [*evidence.positive_signals, *evidence.wittington_signals],
            "negative_signals": evidence.negative_signals,
            "route_decision": route.route,
            "route_reason": route.reason,
            "fetch_error": None,
        }

    monkeypatch.setattr(pipeline, "_collect_homepage_for_company", fake_collect)

    first = pipeline.collect_homepage_evidence(conn, settings, limit=10)
    second = pipeline.collect_homepage_evidence(conn, settings, limit=10)

    assert first.counts["processed"] == 1
    assert second.counts["processed"] == 0
    assert calls["count"] == 1
    assert db.get_homepage_evidence(conn, company["id"])["route_decision"] == "score_from_homepage"
    assert db.enriched_for_openai_scoring(conn, limit=10)[0]["canonical_name"] == "Northstar Labs"


def test_homepage_evidence_summary_counts_cached_routes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    conn = db.connect(settings.database_path)
    db.init_db(conn)
    db.upsert_companies(
        conn,
        [
            CompanyRecord(raw_name="Northstar Labs", canonical_name="Northstar Labs", normalized_name="northstar labs"),
            CompanyRecord(raw_name="Northstar Freight", canonical_name="Northstar Freight", normalized_name="northstar freight"),
            CompanyRecord(raw_name="Opaque Robotics", canonical_name="Opaque Robotics", normalized_name="opaque robotics"),
        ],
    )
    for company in db.list_companies(conn):
        db.update_deterministic_result(conn, company["id"], "likely_startup_or_tech", None, [], True, False)

    for company, route in zip(db.list_companies(conn), ["score_from_homepage", "needs_tavily", "low_priority_data_gap"]):
        db.save_homepage_evidence(
            conn,
            {
                "company_id": company["id"],
                "candidate_domain": f"{company['normalized_name'].replace(' ', '')}.com",
                "resolved_url": f"https://{company['normalized_name'].replace(' ', '')}.com",
                "domain_confidence": 0.9,
                "domain_status": "accepted",
                "metadata_json": {},
                "evidence_text": "",
                "evidence_quality": 0.8,
                "positive_signals": [],
                "negative_signals": [],
                "route_decision": route,
                "route_reason": route,
                "fetch_error": None,
            },
        )

    summary = db.homepage_evidence_summary(conn)

    assert summary["resolved_domains"] == 3
    assert summary["score_from_homepage"] == 1
    assert summary["needs_tavily"] == 1
    assert summary["data_gaps"] == 1


def test_precision_mode_includes_homepage_positive_ambiguous_rows(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    conn = db.connect(settings.database_path)
    db.init_db(conn)
    db.upsert_companies(
        conn,
        [
            CompanyRecord(raw_name="Ambiguous Platform", canonical_name="Ambiguous Platform", normalized_name="ambiguous platform"),
            CompanyRecord(raw_name="Likely Tech", canonical_name="Likely Tech", normalized_name="likely tech"),
            CompanyRecord(raw_name="Unresolved Maybe", canonical_name="Unresolved Maybe", normalized_name="unresolved maybe"),
        ],
    )
    companies = {company["canonical_name"]: company for company in db.list_companies(conn)}
    db.update_deterministic_result(
        conn, companies["Ambiguous Platform"]["id"], "unknown_needs_enrichment", None, [], True, False
    )
    db.update_deterministic_result(
        conn, companies["Likely Tech"]["id"], "likely_startup_or_tech", None, [], True, False
    )
    db.update_deterministic_result(
        conn, companies["Unresolved Maybe"]["id"], "unknown_needs_enrichment", None, [], True, False
    )

    db.save_homepage_evidence(
        conn,
        {
            "company_id": companies["Ambiguous Platform"]["id"],
            "candidate_domain": "ambiguousplatform.com",
            "resolved_url": "https://ambiguousplatform.com",
            "domain_confidence": 0.92,
            "domain_status": "accepted",
            "metadata_json": {},
            "evidence_text": "AI platform for retail operations.",
            "evidence_quality": 0.9,
            "positive_signals": ["platform", "retail"],
            "negative_signals": [],
            "route_decision": "score_from_homepage",
            "route_reason": "strong positive homepage evidence",
            "fetch_error": None,
        },
    )
    db.save_enrichment(
        conn,
        {
            "company_id": companies["Ambiguous Platform"]["id"],
            "query": "Ambiguous Platform",
            "provider": "homepage",
            "raw_json": {},
            "top_titles": ["Ambiguous Platform"],
            "top_urls": ["https://ambiguousplatform.com"],
            "top_snippets": ["AI platform for retail operations."],
            "website": "https://ambiguousplatform.com",
            "status": "success",
            "error": None,
        },
    )
    db.save_homepage_evidence(
        conn,
        {
            "company_id": companies["Unresolved Maybe"]["id"],
            "candidate_domain": None,
            "resolved_url": None,
            "domain_confidence": 0.0,
            "domain_status": "unresolved",
            "metadata_json": {},
            "evidence_text": "",
            "evidence_quality": 0.0,
            "positive_signals": [],
            "negative_signals": [],
            "route_decision": "needs_tavily",
            "route_reason": "unresolved domain",
            "fetch_error": "not found",
        },
    )

    scoreable = db.enriched_for_openai_scoring(conn, limit=10, mode="precision-first")
    tavily_queue = db.candidates_for_enrichment(conn, limit=10, mode="precision-first")

    assert [company["canonical_name"] for company in scoreable] == ["Ambiguous Platform"]
    assert [company["canonical_name"] for company in tavily_queue] == ["Likely Tech"]
    assert db.count_candidate_universe(conn, mode="precision-first") == 2
