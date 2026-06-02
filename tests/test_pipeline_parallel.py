from pathlib import Path
from dataclasses import replace
import threading
import time

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
    companies = _seed_candidate_companies(conn, ["Alpha AI", "Beta Health", "Gamma Climate"])
    for company in companies:
        db.save_homepage_evidence(
            conn,
            {
                "company_id": company["id"],
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
                "route_reason": "company page did not provide enough data",
                "fetch_error": None,
            },
        )

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


def test_homepage_checks_run_in_parallel(monkeypatch, tmp_path: Path) -> None:
    settings = replace(_settings(tmp_path), homepage_concurrency=4, homepage_max_domain_attempts=1)
    conn = db.connect(settings.database_path)
    db.init_db(conn)
    _seed_candidate_companies(conn, ["Alpha AI", "Beta Health", "Gamma Climate", "Delta Robotics"])

    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_collect(company_row, session, *, timeout, max_bytes, mode, max_domain_attempts=None):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return {
            "company_id": company_row["id"],
            "candidate_domain": "example.com",
            "resolved_url": "https://example.com",
            "domain_confidence": 0.9,
            "domain_status": "accepted",
            "metadata_json": {"title": company_row["canonical_name"]},
            "evidence_text": "Software company page.",
            "evidence_quality": 0.9,
            "positive_signals": ["software"],
            "negative_signals": [],
            "route_decision": "score_from_homepage",
            "route_reason": "enough page data",
            "fetch_error": None,
        }

    monkeypatch.setattr(pipeline, "_collect_homepage_for_company", fake_collect)

    result = pipeline.collect_homepage_evidence(conn, settings, limit=4)

    assert result.counts["processed"] == 4
    assert result.counts["parallel_workers"] == 4
    assert max_active > 1
    assert db.homepage_evidence_summary(conn)["homepage_attempted"] == 4


def test_provider_retry_handles_429_then_success(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    calls = {"count": 0}
    sleeps: list[float] = []

    class RateLimitError(RuntimeError):
        status_code = 429

    def flaky_call():
        calls["count"] += 1
        if calls["count"] < 3:
            raise RateLimitError("rate limited")
        return "ok"

    monkeypatch.setattr(pipeline.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(pipeline.random, "uniform", lambda _start, _end: 0)

    value, attempts, retries = pipeline._call_provider_with_retries(settings, flaky_call)

    assert value == "ok"
    assert attempts == 3
    assert retries == 2
    assert sleeps == [1.0, 2.0]


def test_provider_retry_preserves_attempt_count_on_exhaustion(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    class ServerError(RuntimeError):
        status_code = 503

    monkeypatch.setattr(pipeline.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(pipeline.random, "uniform", lambda _start, _end: 0)
    monkeypatch.setattr(pipeline, "classify_with_openai", lambda *_args, **_kwargs: (_ for _ in ()).throw(ServerError("down")))

    result = pipeline._score_company_with_openai(
        settings,
        {"id": 1, "canonical_name": "Alpha AI", "deterministic_type": "likely_startup_or_tech"},
    )

    assert result["api_call"] == 1
    assert result["provider_attempts"] == settings.provider_max_retries + 1
    assert result["retry_attempts"] == settings.provider_max_retries
    assert result["error_count"] == 1
