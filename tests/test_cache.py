from pathlib import Path

from src import db
from src.clean import CompanyRecord
from src.config import Settings
from src.pipeline import verify_cache_reuse


def test_verify_cache_reuse_uses_sqlite_without_paid_calls(tmp_path: Path) -> None:
    settings = Settings(
        database_path=tmp_path / "prospects.db",
        manifest_url="https://manife.st/who-attends/",
        openai_api_key=None,
        tavily_api_key=None,
        openai_model="test-model",
        max_enrich=1,
        max_score=1,
        tavily_max_results=1,
    )
    conn = db.connect(settings.database_path)
    db.init_db(conn)
    db.upsert_companies(
        conn,
        [
            CompanyRecord(
                raw_name="1Logtech",
                canonical_name="1Logtech",
                normalized_name="1logtech",
            )
        ],
    )
    company = db.list_companies(conn)[0]
    db.update_deterministic_result(
        conn,
        company["id"],
        "likely_startup_or_tech",
        None,
        ["logistics", "ai"],
        True,
        True,
    )
    db.save_enrichment(
        conn,
        {
            "company_id": company["id"],
            "query": '"1Logtech" company startup logistics supply chain commerce healthcare climate funding',
            "provider": "tavily",
            "raw_json": {"results": [{"title": "1Logtech", "url": "https://example.com"}]},
            "top_titles": ["1Logtech"],
            "top_urls": ["https://example.com"],
            "top_snippets": ["Cached Tavily evidence."],
            "website": "https://example.com",
            "status": "success",
            "error": None,
        },
    )
    db.save_score(
        conn,
        company["id"],
        {
            "company_type": "startup",
            "is_startup_likely": 1,
            "sector_tags": ["logistics", "ai"],
            "wv_sector_fit": 20,
            "venture_backability": 20,
            "wittington_edge": 15,
            "stage_signal": 5,
            "traction_signal": 5,
            "data_confidence": 8,
            "total_score": 73,
            "rationale": "Cached OpenAI classification.",
            "evidence_summary": "Cached evidence summary.",
            "confidence": "medium",
            "raw_json": {"provider": "openai"},
        },
        provider="openai",
    )

    result = verify_cache_reuse(conn, settings)
    last_run = db.metrics(conn)["last_run"]

    assert result.counts["verified"] is True
    assert result.counts["company"] == "1Logtech"
    assert result.counts["cache_hits"] == 2
    assert result.counts["tavily_calls"] == 0
    assert result.counts["openai_calls"] == 0
    assert "No paid Tavily/OpenAI calls were made" in result.message
    assert last_run["run_type"] == "cache_verification"
    assert last_run["cache_hits"] == 2
    assert last_run["tavily_calls"] == 0
    assert last_run["openai_calls"] == 0
