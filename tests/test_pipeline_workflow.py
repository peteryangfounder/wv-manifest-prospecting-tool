from pathlib import Path

from src import pipeline
from src.config import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_path=tmp_path / "prospects.db",
        manifest_url="https://manife.st/who-attends/",
        openai_api_key=None,
        tavily_api_key=None,
        openai_model="test-model",
        max_enrich=7,
        max_score=5,
        tavily_max_results=1,
    )


def test_load_and_classify_runs_load_then_classify(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    settings = _settings(tmp_path)

    def fake_load(conn, received_settings):
        calls.append("load")
        assert received_settings is settings
        return pipeline.PipelineResult("load_attendees", "loaded", {})

    def fake_classify(conn):
        calls.append("classify")
        return pipeline.PipelineResult("deterministic_classification", "classified", {})

    monkeypatch.setattr(pipeline, "load_attendees", fake_load)
    monkeypatch.setattr(pipeline, "run_deterministic_classification", fake_classify)

    results = pipeline.load_and_classify_companies(object(), settings)

    assert calls == ["load", "classify"]
    assert [result.stage for result in results] == ["load_attendees", "deterministic_classification"]


def test_generate_verified_prospects_runs_enrichment_then_scoring(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, int, bool]] = []
    settings = _settings(tmp_path)

    def fake_enrich(conn, received_settings, limit=None, force=False):
        calls.append(("enrich", limit, force))
        assert received_settings is settings
        return pipeline.PipelineResult("tavily_enrichment", "enriched", {})

    def fake_score(conn, received_settings, limit=None, force=False):
        calls.append(("score", limit, force))
        assert received_settings is settings
        return pipeline.PipelineResult("openai_scoring", "scored", {})

    monkeypatch.setattr(pipeline, "enrich_candidates", fake_enrich)
    monkeypatch.setattr(pipeline, "score_enriched_candidates", fake_score)

    results = pipeline.generate_verified_prospects(object(), settings, enrich_limit=3, score_limit=2, force=True)

    assert calls == [("enrich", 3, True), ("score", 2, True)]
    assert [result.stage for result in results] == ["tavily_enrichment", "openai_scoring"]
