from pathlib import Path

from src import db, pipeline
from src.clean import CompanyRecord
from src.config import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_path=tmp_path / "prospects.db",
        manifest_url="https://manife.st/who-attends/",
        openai_api_key="openai-key",
        tavily_api_key="tavily-key",
        openai_model="test-model",
        max_enrich=10,
        max_score=10,
        tavily_max_results=1,
        openai_input_cost_per_1m_tokens=0.15,
        openai_output_cost_per_1m_tokens=0.60,
        tavily_cost_per_call_usd=0.001,
        tavily_concurrency=2,
        openai_concurrency=3,
        db_commit_batch_size=2,
    )


def _seed_candidate_companies(conn, names: list[str]) -> list[dict]:
    db.upsert_companies(
        conn,
        [
            CompanyRecord(raw_name=name, canonical_name=name, normalized_name=name.lower().replace(" ", "-"))
            for name in names
        ],
    )
    companies = db.list_companies(conn)
    for company in companies:
        db.update_deterministic_result(
            conn,
            company["id"],
            "likely_startup_or_tech",
            None,
            ["technology"],
            True,
            True,
        )
    conn.commit()
    return companies


def test_enrich_candidates_persists_parallel_results(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    conn = db.connect(settings.database_path)
    db.init_db(conn)
    _seed_candidate_companies(conn, ["Alpha AI", "Beta Health", "Gamma Climate"])

    def fake_fetch(received_settings, company):
        assert received_settings is settings
        return {
            "api_call": 1,
            "cache_hit": 0,
            "enriched": 1,
            "error_count": 0,
            "status": "success",
            "company_name": company["canonical_name"],
            "enrichment": {
                "company_id": company["id"],
                "query": company["canonical_name"],
                "provider": "tavily",
                "raw_json": {"results": [{"title": company["canonical_name"], "url": "https://example.com"}]},
                "top_titles": [company["canonical_name"]],
                "top_urls": ["https://example.com"],
                "top_snippets": ["Evidence"],
                "website": "https://example.com",
                "status": "success",
                "error": None,
            },
        }

    monkeypatch.setattr(pipeline, "_fetch_enrichment_for_company", fake_fetch)

    result = pipeline.enrich_candidates(conn, settings, limit=3)
    metrics = db.metrics(conn)

    assert result.counts["tavily_calls"] == 3
    assert result.counts["enriched"] == 3
    assert result.counts["parallel_workers"] == 2
    assert metrics["enriched"] == 3


def test_score_enriched_candidates_persists_parallel_results(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    conn = db.connect(settings.database_path)
    db.init_db(conn)
    companies = _seed_candidate_companies(conn, ["Alpha AI", "Beta Health", "Gamma Climate"])
    for company in companies:
        db.save_enrichment(
            conn,
            {
                "company_id": company["id"],
                "query": company["canonical_name"],
                "provider": "tavily",
                "raw_json": {"results": [{"title": company["canonical_name"], "url": "https://example.com"}]},
                "top_titles": [company["canonical_name"]],
                "top_urls": ["https://example.com"],
                "top_snippets": ["Evidence"],
                "website": "https://example.com",
                "status": "success",
                "error": None,
            },
        )

    def fake_score(received_settings, company):
        assert received_settings is settings
        return {
            "api_call": 1,
            "cache_hit": 0,
            "scored": 1,
            "error_count": 0,
            "status": "success",
            "company_name": company["canonical_name"],
            "company_id": company["id"],
            "provider": "openai",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "estimated_cost_usd": 0.000027,
            "score": {
                "company_type": "startup",
                "is_startup_likely": 1,
                "sector_tags": ["technology"],
                "wv_sector_fit": 20,
                "venture_backability": 20,
                "wittington_edge": 15,
                "stage_signal": 5,
                "traction_signal": 5,
                "data_confidence": 8,
                "total_score": 73,
                "rationale": "Relevant technology startup.",
                "evidence_summary": "Evidence summary.",
                "confidence": "medium",
                "raw_json": {"provider": "openai"},
            },
        }

    monkeypatch.setattr(pipeline, "_score_company_with_openai", fake_score)

    result = pipeline.score_enriched_candidates(conn, settings, limit=3)
    metrics = db.metrics(conn)

    assert result.counts["openai_calls"] == 3
    assert result.counts["scored"] == 3
    assert result.counts["parallel_workers"] == 3
    assert result.counts["prompt_tokens"] == 300
    assert result.counts["completion_tokens"] == 60
    assert metrics["openai_scored"] == 3
