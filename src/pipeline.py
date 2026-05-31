from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from . import db
from .clean import dedupe_names
from .classify import classify_with_openai
from .config import Settings
from .enrich import EnrichmentUnavailable, TavilyClient, compact_tavily_response
from .rules import classify_company_name
from .score import baseline_score
from .scrape import get_attendee_names


@dataclass
class PipelineResult:
    stage: str
    message: str
    counts: dict[str, Any]


def load_attendees(conn, settings: Settings) -> PipelineResult:
    run_id = db.start_run(conn, "load_attendees")
    raw_names, metadata = get_attendee_names(settings.manifest_url)
    records = dedupe_names(raw_names)
    unique_count = db.upsert_companies(conn, records)
    db.finish_run(
        conn,
        run_id,
        raw_count=len(raw_names),
        unique_count=unique_count,
        notes=f"Source: {metadata.get('source')}; live_error={metadata.get('live_error', '')}",
    )
    return PipelineResult(
        "load_attendees",
        f"Loaded {len(raw_names):,} raw rows and {unique_count:,} unique companies.",
        {"raw": len(raw_names), "unique": unique_count, **metadata},
    )


def run_deterministic_classification(conn) -> PipelineResult:
    run_id = db.start_run(conn, "deterministic_classification")
    companies = db.list_companies(conn)
    candidates = 0

    for company in companies:
        result = classify_company_name(company["canonical_name"])
        db.update_deterministic_result(
            conn,
            company["id"],
            result.deterministic_type,
            result.exclusion_reason,
            result.tags,
            result.is_candidate,
        )
        enriched_company = {
            **company,
            "deterministic_type": result.deterministic_type,
            "deterministic_exclusion_reason": result.exclusion_reason,
            "deterministic_tags": json.dumps(result.tags),
        }
        db.save_score(conn, company["id"], baseline_score(enriched_company), provider="baseline")
        candidates += 1 if result.is_candidate else 0

    db.finish_run(
        conn,
        run_id,
        unique_count=len(companies),
        candidates_count=candidates,
        scored_count=len(companies),
        notes="Baseline deterministic scores created for the full attendee list.",
    )
    return PipelineResult(
        "deterministic_classification",
        f"Classified {len(companies):,} companies; {candidates:,} are candidates for enrichment.",
        {"companies": len(companies), "candidates": candidates},
    )


def enrich_candidates(conn, settings: Settings, limit: int | None = None, force: bool = False) -> PipelineResult:
    limit = limit or settings.max_enrich
    run_id = db.start_run(conn, "tavily_enrichment")
    if not settings.tavily_api_key:
        db.finish_run(conn, run_id, notes="Skipped: TAVILY_API_KEY is not set.")
        return PipelineResult(
            "tavily_enrichment",
            "Skipped Tavily enrichment because TAVILY_API_KEY is not set.",
            {"tavily_calls": 0, "enriched": 0, "cache_hits": 0},
        )

    client = TavilyClient(settings.tavily_api_key, settings.tavily_max_results)
    companies = db.candidates_for_enrichment(conn, limit=limit, force=force)
    calls = 0
    enriched = 0
    errors = 0

    for company in companies:
        try:
            payload = client.search(company["canonical_name"])
            enrichment = compact_tavily_response(company["id"], payload)
            db.save_enrichment(conn, enrichment)
            calls += 1
            if enrichment["status"] == "success":
                enriched += 1
        except EnrichmentUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - persist provider failure per company.
            db.save_enrichment(
                conn,
                {
                    "company_id": company["id"],
                    "query": f'"{company["canonical_name"]}" company startup logistics supply chain commerce healthcare climate funding',
                    "provider": "tavily",
                    "raw_json": {},
                    "top_titles": [],
                    "top_urls": [],
                    "top_snippets": [],
                    "website": None,
                    "status": "error",
                    "error": str(exc),
                },
            )
            calls += 1
            errors += 1

    db.finish_run(
        conn,
        run_id,
        enriched_count=enriched,
        tavily_calls=calls,
        notes=f"Errors: {errors}. Force refresh: {force}.",
    )
    return PipelineResult(
        "tavily_enrichment",
        f"Made {calls:,} Tavily calls and stored {enriched:,} successful enrichments.",
        {"tavily_calls": calls, "enriched": enriched, "errors": errors},
    )


def score_enriched_candidates(conn, settings: Settings, limit: int | None = None, force: bool = False) -> PipelineResult:
    limit = limit or settings.max_score
    run_id = db.start_run(conn, "openai_scoring")
    if not settings.openai_api_key:
        db.finish_run(conn, run_id, notes="Skipped: OPENAI_API_KEY is not set.")
        return PipelineResult(
            "openai_scoring",
            "Skipped OpenAI scoring because OPENAI_API_KEY is not set. Baseline deterministic scores remain available.",
            {"openai_calls": 0, "scored": 0, "errors": 0},
        )

    companies = db.enriched_for_openai_scoring(conn, limit=limit, force=force)
    calls = 0
    scored = 0
    errors = 0

    for company in companies:
        try:
            score = classify_with_openai(company, settings.openai_api_key, settings.openai_model)
            db.save_score(conn, company["id"], score, provider="openai")
            calls += 1
            scored += 1
        except Exception as exc:  # noqa: BLE001 - keep partial scoring runs usable.
            fallback = baseline_score(company)
            fallback["rationale"] = f"OpenAI scoring failed; baseline retained. Error: {str(exc)[:120]}"
            db.save_score(conn, company["id"], fallback, provider="baseline")
            calls += 1
            errors += 1

    db.finish_run(
        conn,
        run_id,
        scored_count=scored,
        openai_calls=calls,
        notes=f"Errors: {errors}. Force refresh: {force}. Model: {settings.openai_model}.",
    )
    return PipelineResult(
        "openai_scoring",
        f"Made {calls:,} OpenAI calls and stored {scored:,} structured scores.",
        {"openai_calls": calls, "scored": scored, "errors": errors},
    )


def run_default_pipeline(conn, settings: Settings) -> list[PipelineResult]:
    results = [load_attendees(conn, settings), run_deterministic_classification(conn)]
    results.append(enrich_candidates(conn, settings, settings.max_enrich))
    results.append(score_enriched_candidates(conn, settings, settings.max_score))
    return results
