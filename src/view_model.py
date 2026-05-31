from __future__ import annotations

from typing import Any

import pandas as pd


VERIFIED_EMPTY_STATE = "No verified prospects yet. Run 'Generate verified prospects' to enrich and score high-priority companies."


def _is_successful_enrichment(row: dict[str, Any] | pd.Series) -> bool:
    return row.get("enrichment_status") == "success"


def _is_openai_scored(row: dict[str, Any] | pd.Series) -> bool:
    return row.get("score_provider") == "openai"


def evidence_status(row: dict[str, Any] | pd.Series) -> str:
    if _is_openai_scored(row):
        return "OpenAI scored"
    if _is_successful_enrichment(row):
        return "Enriched"
    return "Baseline only"


def cache_status(row: dict[str, Any] | pd.Series) -> str:
    if _is_openai_scored(row) or _is_successful_enrichment(row):
        return "Cached"
    return "Not cached"


def is_verified_prospect(row: dict[str, Any] | pd.Series) -> bool:
    return _is_openai_scored(row) and _is_successful_enrichment(row)


def display_score(row: dict[str, Any] | pd.Series) -> int:
    score = max(0, min(100, int(row.get("total_score") or 0)))
    if evidence_status(row) == "Baseline only":
        return min(score, 40)
    return score


def add_review_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    enriched = frame.copy()
    for column in ["score_provider", "enrichment_status"]:
        if column not in enriched.columns:
            enriched[column] = ""

    enriched["evidence_status"] = enriched.apply(evidence_status, axis=1)
    enriched["cache_status"] = enriched.apply(cache_status, axis=1)
    enriched["is_verified_prospect"] = enriched.apply(is_verified_prospect, axis=1)
    enriched["display_score"] = enriched.apply(display_score, axis=1).astype(int)
    enriched = enriched.sort_values(["display_score", "canonical_name"], ascending=[False, True])
    enriched["rank"] = range(1, len(enriched) + 1)
    return enriched


def verified_top_prospects(frame: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    working = add_review_metadata(frame) if "is_verified_prospect" not in frame.columns else frame
    verified = working[working["is_verified_prospect"].astype(bool)].copy()
    return verified.sort_values(["display_score", "canonical_name"], ascending=[False, True]).head(limit)
