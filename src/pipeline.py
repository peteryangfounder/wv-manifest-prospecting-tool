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
    high_priority = 0

    for company in companies:
        result = classify_company_name(company["canonical_name"])
        db.update_deterministic_result(
            conn,
            company["id"],
            result.deterministic_type,
            result.exclusion_reason,
            result.tags,
            result.is_candidate,
            result.high_priority_enrichment,
        )
        enriched_company = {
            **company,
            "deterministic_type": result.deterministic_type,
            "deterministic_exclusion_reason": result.exclusion_reason,
            "deterministic_tags": json.dumps(result.tags),
        }
        db.save_baseline_score_if_missing_or_baseline(conn, company["id"], baseline_score(enriched_company))
        candidates += 1 if result.is_candidate else 0
        high_priority += 1 if result.high_priority_enrichment else 0

    db.finish_run(
        conn,
        run_id,
        unique_count=len(companies),
        candidates_count=candidates,
        scored_count=len(companies),
        notes=f"Baseline deterministic scores created for the full attendee list; high_priority_queue={high_priority}.",
    )
    return PipelineResult(
        "deterministic_classification",
        f"Classified {len(companies):,} companies; {candidates:,} broad candidates; {high_priority:,} high-priority for paid enrichment.",
        {"companies": len(companies), "candidates": candidates, "high_priority_queue": high_priority},
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
    companies = db.candidates_for_enrichment(conn, limit=limit, force=force, high_priority_only=True)
    calls = 0
    enriched = 0
    errors = 0
    cache_hits = 0

    for company in companies:
        result = _enrich_one_company(conn, settings, company, force=force, client=client)
        calls += result["api_call"]
        enriched += result["enriched"]
        errors += result["error_count"]
        cache_hits += result["cache_hit"]

    db.finish_run(
        conn,
        run_id,
        enriched_count=enriched,
        tavily_calls=calls,
        cache_hits=cache_hits,
        notes=f"Errors: {errors}. Force refresh: {force}.",
    )
    return PipelineResult(
        "tavily_enrichment",
        f"Made {calls:,} Tavily calls and stored {enriched:,} successful enrichments.",
        {"tavily_calls": calls, "enriched": enriched, "errors": errors, "cache_hits": cache_hits},
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

    companies = db.enriched_for_openai_scoring(conn, limit=limit, force=force, high_priority_only=True)
    calls = 0
    scored = 0
    errors = 0
    cache_hits = 0

    for company in companies:
        result = _score_one_company(conn, settings, company, force=force)
        calls += result["api_call"]
        scored += result["scored"]
        errors += result["error_count"]
        cache_hits += result["cache_hit"]

    db.finish_run(
        conn,
        run_id,
        scored_count=scored,
        openai_calls=calls,
        cache_hits=cache_hits,
        notes=f"Errors: {errors}. Force refresh: {force}. Model: {settings.openai_model}.",
    )
    return PipelineResult(
        "openai_scoring",
        f"Made {calls:,} OpenAI calls and stored {scored:,} structured scores.",
        {"openai_calls": calls, "scored": scored, "errors": errors, "cache_hits": cache_hits},
    )


def _enrich_one_company(
    conn,
    settings: Settings,
    company: dict[str, Any],
    *,
    force: bool = False,
    client: TavilyClient | None = None,
) -> dict[str, Any]:
    if not force:
        cached = db.get_successful_enrichment(conn, company["id"], provider="tavily")
        if cached:
            return {
                "api_call": 0,
                "cache_hit": 1,
                "enriched": 1,
                "error_count": 0,
                "status": "cache_hit",
                "company_name": company["canonical_name"],
            }

    if client is None:
        if not settings.tavily_api_key:
            return {
                "api_call": 0,
                "cache_hit": 0,
                "enriched": 0,
                "error_count": 1,
                "status": "missing_api_key",
                "company_name": company["canonical_name"],
            }
        client = TavilyClient(settings.tavily_api_key, settings.tavily_max_results)

    try:
        payload = client.search(company["canonical_name"])
        enrichment = compact_tavily_response(company["id"], payload)
        db.save_enrichment(conn, enrichment)
        return {
            "api_call": 1,
            "cache_hit": 0,
            "enriched": 1 if enrichment["status"] == "success" else 0,
            "error_count": 0 if enrichment["status"] == "success" else 1,
            "status": enrichment["status"],
            "company_name": company["canonical_name"],
        }
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
        return {
            "api_call": 1,
            "cache_hit": 0,
            "enriched": 0,
            "error_count": 1,
            "status": "error",
            "company_name": company["canonical_name"],
        }


def _score_one_company(
    conn,
    settings: Settings,
    company: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    if not force:
        cached = db.get_score(conn, company["id"], provider="openai")
        if cached:
            return {
                "api_call": 0,
                "cache_hit": 1,
                "scored": 1,
                "error_count": 0,
                "status": "cache_hit",
                "company_name": company["canonical_name"],
            }

    if not settings.openai_api_key:
        return {
            "api_call": 0,
            "cache_hit": 0,
            "scored": 0,
            "error_count": 1,
            "status": "missing_api_key",
            "company_name": company["canonical_name"],
        }

    try:
        score = classify_with_openai(company, settings.openai_api_key, settings.openai_model)
        db.save_score(conn, company["id"], score, provider="openai")
        return {
            "api_call": 1,
            "cache_hit": 0,
            "scored": 1,
            "error_count": 0,
            "status": "success",
            "company_name": company["canonical_name"],
        }
    except Exception as exc:  # noqa: BLE001 - keep partial scoring runs usable.
        fallback = baseline_score(company)
        fallback["rationale"] = f"OpenAI scoring failed; baseline retained. Error: {str(exc)[:120]}"
        db.save_score(conn, company["id"], fallback, provider="baseline")
        return {
            "api_call": 1,
            "cache_hit": 0,
            "scored": 0,
            "error_count": 1,
            "status": "error",
            "company_name": company["canonical_name"],
        }


def verify_cache_reuse(conn, settings: Settings) -> PipelineResult:
    run_id = db.start_run(conn, "cache_verification")
    company = db.cached_company_for_verification(conn)
    if company is None:
        message = "No cached company available yet. First run a small Tavily enrichment and OpenAI scoring pass."
        db.finish_run(conn, run_id, cache_hits=0, tavily_calls=0, openai_calls=0, notes=message)
        return PipelineResult(
            "cache_verification",
            message,
            {"verified": False, "cache_hits": 0, "tavily_calls": 0, "openai_calls": 0},
        )

    enrichment_result = _enrich_one_company(conn, settings, company, force=False)
    score_result = _score_one_company(conn, settings, company, force=False)
    cache_hits = enrichment_result["cache_hit"] + score_result["cache_hit"]
    tavily_calls = enrichment_result["api_call"]
    openai_calls = score_result["api_call"]
    verified = cache_hits == 2 and tavily_calls == 0 and openai_calls == 0

    db.finish_run(
        conn,
        run_id,
        enriched_count=1 if enrichment_result["cache_hit"] else 0,
        scored_count=1 if score_result["cache_hit"] else 0,
        tavily_calls=tavily_calls,
        openai_calls=openai_calls,
        cache_hits=cache_hits,
        notes=f"Verified={verified}; company={company['canonical_name']}; force_refresh=False.",
    )

    if verified:
        message = (
            f"Cache verified: reused stored enrichment and scoring for {company['canonical_name']}. "
            "No paid Tavily/OpenAI calls were made."
        )
    else:
        message = (
            f"Cache verification failed for {company['canonical_name']}: expected 2 cache hits and 0 paid calls, "
            f"got {cache_hits} cache hits, {tavily_calls} Tavily calls, and {openai_calls} OpenAI calls."
        )

    return PipelineResult(
        "cache_verification",
        message,
        {
            "verified": verified,
            "company": company["canonical_name"],
            "cache_hits": cache_hits,
            "tavily_calls": tavily_calls,
            "openai_calls": openai_calls,
        },
    )


def run_default_pipeline(conn, settings: Settings) -> list[PipelineResult]:
    results = [load_attendees(conn, settings), run_deterministic_classification(conn)]
    results.append(enrich_candidates(conn, settings, settings.max_enrich))
    results.append(score_enriched_candidates(conn, settings, settings.max_score))
    return results
