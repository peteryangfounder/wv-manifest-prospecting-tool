from src.clean import dedupe_names
from src import db
from src.clean import CompanyRecord
from src.config import Settings
from src.pipeline import run_deterministic_classification
from src.rules import classify_company_name
from src.scrape import load_seed_text, parse_company_lines


def test_high_priority_queue_is_much_smaller_than_full_manifest_list() -> None:
    raw_names = parse_company_lines(load_seed_text())
    companies = dedupe_names(raw_names)
    high_priority = [
        company
        for company in companies
        if classify_company_name(company.canonical_name).high_priority_enrichment
    ]

    assert len(companies) > 3000
    assert len(high_priority) > 25
    assert len(high_priority) < len(companies) * 0.25


def test_paid_queue_orders_high_signal_before_ambiguous_without_dropping_ambiguous(tmp_path) -> None:
    settings = Settings(
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
    )
    conn = db.connect(settings.database_path)
    db.init_db(conn)
    db.upsert_companies(
        conn,
        [
            CompanyRecord(raw_name="Alpha AI", canonical_name="Alpha AI", normalized_name="alpha ai"),
            CompanyRecord(raw_name="Northstar Labs", canonical_name="Northstar Labs", normalized_name="northstar labs"),
            CompanyRecord(raw_name="DHL", canonical_name="DHL", normalized_name="dhl"),
            CompanyRecord(
                raw_name="Wittington Ventures",
                canonical_name="Wittington Ventures",
                normalized_name="wittington ventures",
            ),
        ],
    )

    run_deterministic_classification(conn)
    candidates = db.candidates_for_enrichment(conn, limit=10)

    assert [candidate["canonical_name"] for candidate in candidates] == ["Alpha AI", "Northstar Labs"]
    assert candidates[0]["high_priority_enrichment"] == 1
    assert candidates[1]["deterministic_type"] == "unknown_needs_enrichment"
