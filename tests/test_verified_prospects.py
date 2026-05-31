from pathlib import Path

import pandas as pd

from src import db
from src.clean import CompanyRecord
from src.config import Settings
from src.pipeline import run_deterministic_classification
from src.view_model import VERIFIED_EMPTY_STATE, add_review_metadata, verified_top_prospects


def test_generic_placeholders_are_excluded_from_paid_enrichment(tmp_path: Path) -> None:
    settings = Settings(
        database_path=tmp_path / "prospects.db",
        manifest_url="https://manife.st/who-attends/",
        openai_api_key=None,
        tavily_api_key=None,
        openai_model="test-model",
        max_enrich=10,
        max_score=10,
        tavily_max_results=1,
    )
    conn = db.connect(settings.database_path)
    db.init_db(conn)
    db.upsert_companies(
        conn,
        [
            CompanyRecord(raw_name="AI Startup", canonical_name="AI Startup", normalized_name="ai startup"),
            CompanyRecord(raw_name="Startup", canonical_name="Startup", normalized_name="startup"),
            CompanyRecord(raw_name="Stealth Company", canonical_name="Stealth Company", normalized_name="stealth"),
        ],
    )

    run_deterministic_classification(conn)
    candidates = db.candidates_for_enrichment(conn, limit=10)

    assert candidates == []


def test_verified_top_prospects_excludes_baseline_only_rows() -> None:
    frame = add_review_metadata(
        pd.DataFrame(
            [
                {
                    "canonical_name": "AI Startup",
                    "total_score": 50,
                    "score_provider": "baseline",
                    "enrichment_status": None,
                    "company_type": "startup",
                    "confidence": "low",
                },
                {
                    "canonical_name": "Supply Chain AI",
                    "total_score": 50,
                    "score_provider": "baseline",
                    "enrichment_status": None,
                    "company_type": "startup",
                    "confidence": "low",
                },
            ]
        )
    )

    top = verified_top_prospects(frame)

    assert top.empty
    assert VERIFIED_EMPTY_STATE.startswith("No verified prospects yet")
    assert frame["display_score"].max() == 40
    assert set(frame["evidence_status"]) == {"Baseline only"}


def test_verified_top_prospects_includes_enriched_openai_scored_rows() -> None:
    frame = add_review_metadata(
        pd.DataFrame(
            [
                {
                    "canonical_name": "Baseline AI",
                    "total_score": 50,
                    "score_provider": "baseline",
                    "enrichment_status": None,
                    "company_type": "startup",
                    "confidence": "low",
                },
                {
                    "canonical_name": "Verified Robotics",
                    "total_score": 84,
                    "score_provider": "openai",
                    "enrichment_status": "success",
                    "company_type": "startup",
                    "confidence": "high",
                },
            ]
        )
    )

    top = verified_top_prospects(frame)

    assert top["canonical_name"].tolist() == ["Verified Robotics"]
    assert top.iloc[0]["evidence_status"] == "OpenAI scored"
    assert top.iloc[0]["cache_status"] == "Cached"
