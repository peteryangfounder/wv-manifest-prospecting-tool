from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import time
from dataclasses import dataclass
from typing import Any
import requests

from . import db
from .billing import calculate_tavily_billing
from .clean import dedupe_names
from .classify import classify_with_openai
from .config import Settings
from .domain_resolver import domain_status, generate_domain_candidates, score_domain_confidence
from .enrich import EnrichmentUnavailable, TavilyClient, compact_tavily_response
from .evidence_routing import route_candidate_after_homepage_evidence
from .homepage_evidence import extract_homepage_evidence
from .rules import classify_company_name
from .score import baseline_score
from .scrape import get_attendee_names
from .web_metadata import fetch_homepage_metadata


@dataclass
class PipelineResult:
    stage: str
    message: str
    counts: dict[str, Any]


class ProviderCallFailed(RuntimeError):
    def __init__(self, original: Exception, attempts: int, retries: int):
        super().__init__(str(original))
        self.original = original
        self.attempts = attempts
        self.retries = retries


def _estimate_openai_cost(settings: Settings, prompt_tokens: int, completion_tokens: int) -> float:
    input_cost = (prompt_tokens / 1_000_000) * settings.openai_input_cost_per_1m_tokens
    output_cost = (completion_tokens / 1_000_000) * settings.openai_output_cost_per_1m_tokens
    return input_cost + output_cost


def _setting(settings: Settings, name: str, default):
    return getattr(settings, name, default)


def _setting_int(settings: Settings, name: str, default: int) -> int:
    try:
        return int(_setting(settings, name, default) or default)
    except (TypeError, ValueError):
        return default


def _bounded_worker_count(settings: Settings, name: str, default: int, item_count: int) -> int:
    if item_count <= 0:
        return 1
    configured = _setting_int(settings, name, default)
    return max(1, min(configured, item_count))


def _commit_interval(settings: Settings) -> int:
    return max(1, _setting_int(settings, "db_commit_batch_size", 25))


def _retryable_status_codes() -> set[int]:
    return {408, 409, 425, 429, 500, 502, 503, 504}


def _status_code_from_exception(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return int(status_code)
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) is not None:
        return int(response.status_code)
    return None


def _retry_after_from_exception(exc: Exception) -> float | None:
    headers = getattr(exc, "headers", None)
    response = getattr(exc, "response", None)
    if not headers and response is not None:
        headers = getattr(response, "headers", None)
    if not headers:
        return None
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None:
        return None
    try:
        return max(0.0, float(retry_after))
    except (TypeError, ValueError):
        return None


def _call_provider_with_retries(settings: Settings, func) -> tuple[Any, int, int]:
    max_retries = max(0, _setting_int(settings, "provider_max_retries", 4))
    initial = max(0.1, float(_setting(settings, "provider_backoff_initial_seconds", 1.0)))
    maximum = max(initial, float(_setting(settings, "provider_backoff_max_seconds", 20.0)))
    attempts = 0
    retries = 0
    while True:
        attempts += 1
        try:
            return func(), attempts, retries
        except Exception as exc:
            status_code = _status_code_from_exception(exc)
            retryable = status_code in _retryable_status_codes() or status_code is None
            if not retryable or retries >= max_retries:
                raise ProviderCallFailed(exc, attempts, retries) from exc
            retry_after = _retry_after_from_exception(exc)
            delay = retry_after if retry_after is not None else min(maximum, initial * (2**retries))
            delay += random.uniform(0, min(1.0, delay * 0.25))
            retries += 1
            time.sleep(delay)


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
        notes=f"Source: {metadata.get('source')}. live_error={metadata.get('live_error', '')}",
    )
    return PipelineResult(
        "load_attendees",
        f"Loaded {len(raw_names):,} raw rows and {unique_count:,} unique companies.",
        {"raw": len(raw_names), "unique": unique_count, **metadata},
    )


def load_manifest_rows(conn, settings: Settings) -> PipelineResult:
    run_id = db.start_run(conn, "load_manifest_rows")
    raw_names, metadata = get_attendee_names(settings.manifest_url)
    raw_count = db.replace_raw_manifest_rows(conn, raw_names)
    db.finish_run(
        conn,
        run_id,
        raw_count=raw_count,
        notes=f"Source: {metadata.get('source')}. live_error={metadata.get('live_error', '')}",
    )
    return PipelineResult(
        "load_manifest_rows",
        f"Loaded {raw_count:,} raw Manifest rows.",
        {"raw": raw_count, **metadata},
    )


def normalize_manifest_names(conn) -> PipelineResult:
    run_id = db.start_run(conn, "normalize_manifest_names")
    raw_names = db.list_raw_manifest_names(conn)
    records = dedupe_names(raw_names)
    unique_count = db.upsert_companies(conn, records)
    db.finish_run(
        conn,
        run_id,
        raw_count=len(raw_names),
        unique_count=unique_count,
    )
    duplicate_count = max(0, len(raw_names) - unique_count)
    return PipelineResult(
        "normalize_manifest_names",
        f"Normalized {len(raw_names):,} raw rows into {unique_count:,} company names. Merged {duplicate_count:,} duplicate rows.",
        {"raw": len(raw_names), "unique": unique_count, "duplicates": duplicate_count},
    )


def run_deterministic_classification(conn) -> PipelineResult:
    run_id = db.start_run(conn, "deterministic_classification")
    companies = db.list_companies(conn)
    candidates = 0
    high_priority = 0

    for index, company in enumerate(companies, start=1):
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
        db.save_baseline_score_if_missing_or_baseline(
            conn,
            company["id"],
            baseline_score(enriched_company),
            commit=False,
        )
        candidates += 1 if result.is_candidate else 0
        high_priority += 1 if result.high_priority_enrichment else 0
        if index % 250 == 0:
            conn.commit()

    conn.commit()
    db.finish_run(
        conn,
        run_id,
        unique_count=len(companies),
        candidates_count=candidates,
        scored_count=len(companies),
        notes=f"Baseline deterministic scores created for the full attendee list. high_priority_queue={high_priority}.",
    )
    return PipelineResult(
        "deterministic_classification",
        f"Classified {len(companies):,} companies. {candidates:,} broad candidates. {high_priority:,} first-priority for search and scoring.",
        {"companies": len(companies), "candidates": candidates, "high_priority_queue": high_priority},
    )


def enrich_candidates(
    conn,
    settings: Settings,
    limit: int | None = None,
    force: bool = False,
    progress_callback=None,
    mode: str = "balanced",
) -> PipelineResult:
    limit = limit or settings.max_enrich
    run_id = db.start_run(conn, "tavily_enrichment")
    if not settings.tavily_api_key:
        db.finish_run(conn, run_id, notes="Skipped: TAVILY_API_KEY is not set.")
        return PipelineResult(
            "tavily_enrichment",
            "Skipped Tavily enrichment because TAVILY_API_KEY is not set.",
            {"tavily_calls": 0, "enriched": 0, "cache_hits": 0},
        )

    companies = db.candidates_for_enrichment(conn, limit=limit, force=force, mode=mode)
    calls = 0
    enriched = 0
    errors = 0
    cache_hits = 0
    retry_attempts = 0
    provider_attempts = 0
    workers = _bounded_worker_count(settings, "tavily_concurrency", 48, len(companies))
    commit_interval = _commit_interval(settings)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_fetch_enrichment_for_company, settings, company) for company in companies]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            enrichment = result.pop("enrichment", None)
            if enrichment is not None:
                db.save_enrichment(conn, enrichment, commit=False)
            calls += result["api_call"]
            enriched += result["enriched"]
            errors += result["error_count"]
            cache_hits += result["cache_hit"]
            retry_attempts += result.get("retry_attempts", 0)
            provider_attempts += result.get("provider_attempts", result["api_call"])
            if index % commit_interval == 0:
                conn.commit()
            if progress_callback:
                progress_callback(
                    index,
                    len(companies),
                    result,
                    {
                        "tavily_calls": calls,
                        "enriched": enriched,
                        "errors": errors,
                        "cache_hits": cache_hits,
                        "retry_attempts": retry_attempts,
                    },
                )
    conn.commit()

    tavily_billing = calculate_tavily_billing(
        credits_used=calls,
        included_monthly_credits=_setting(settings, "tavily_included_monthly_credits", 1000),
        pay_as_you_go_enabled=_setting(settings, "tavily_pay_as_you_go_enabled", True),
        payg_price_per_credit_usd=_setting(settings, "tavily_payg_price_per_credit_usd", 0.008),
        plan_name=_setting(settings, "tavily_plan_name", "Researcher"),
        shadow_price_per_credit_usd=_setting(settings, "tavily_cost_per_call_usd", 0.001),
    )
    db.finish_run(
        conn,
        run_id,
        enriched_count=enriched,
        tavily_calls=calls,
        cache_hits=cache_hits,
        estimated_cost_usd=0.0,
        notes=f"Errors: {errors}. Retries: {retry_attempts}. Force refresh: {force}. Mode: {mode}.",
    )
    return PipelineResult(
        "tavily_enrichment",
        (
            f"Made {calls:,} Tavily calls with {workers:,} parallel workers and stored {enriched:,} successful enrichments. "
            f"Tavily billed spend: ${tavily_billing.actual_billed_usd:.4f}; "
            f"free credits remaining: {tavily_billing.free_credits_remaining:,}."
        ),
        {
            "tavily_calls": calls,
            "tavily_credits_used": tavily_billing.credits_used,
            "tavily_actual_billed_usd": tavily_billing.actual_billed_usd,
            "tavily_shadow_estimate_usd": tavily_billing.shadow_estimate_usd,
            "enriched": enriched,
            "errors": errors,
            "cache_hits": cache_hits,
            "provider_attempts": provider_attempts,
            "retry_attempts": retry_attempts,
            "parallel_workers": workers,
            "estimated_cost_usd": 0.0,
        },
    )


def collect_homepage_evidence(
    conn,
    settings: Settings,
    limit: int | None = None,
    force: bool = False,
    progress_callback=None,
    mode: str = "balanced",
    max_domain_attempts: int | None = None,
) -> PipelineResult:
    configured_limit = _setting_int(settings, "homepage_evidence_max_per_run", 100)
    limit = configured_limit if limit is None else max(1, int(limit))
    run_id = db.start_run(conn, "homepage_evidence")
    companies = db.candidates_for_homepage_evidence(conn, limit=limit, mode=mode, force=force)
    processed = 0
    accepted = 0
    provisional = 0
    unresolved = 0
    score_ready = 0
    needs_tavily = 0
    errors = 0
    commit_interval = _commit_interval(settings)
    timeout = float(_setting(settings, "homepage_fetch_timeout_seconds", 1.25))
    max_bytes = _setting_int(settings, "homepage_fetch_max_bytes", 100_000)
    workers = _bounded_worker_count(settings, "homepage_concurrency", 96, len(companies))
    configured_domain_attempts = _setting_int(settings, "homepage_max_domain_attempts", 3)
    if max_domain_attempts is None and configured_domain_attempts > 0:
        max_domain_attempts = configured_domain_attempts

    def collect_one(company: dict[str, Any]) -> dict[str, Any]:
        collect_kwargs = {
            "timeout": timeout,
            "max_bytes": max_bytes,
            "mode": mode,
        }
        if max_domain_attempts is not None:
            collect_kwargs["max_domain_attempts"] = max_domain_attempts
        with requests.Session() as session:
            return _collect_homepage_for_company(company, session, **collect_kwargs)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(collect_one, company) for company in companies]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            db.save_homepage_evidence(conn, result, commit=False)
            if result.get("route_decision") == "score_from_homepage":
                db.save_enrichment(conn, _homepage_enrichment_from_evidence(result), commit=False)
            processed += 1
            accepted += 1 if result.get("domain_status") == "accepted" else 0
            provisional += 1 if result.get("domain_status") == "provisional" else 0
            unresolved += 1 if result.get("domain_status") == "unresolved" else 0
            score_ready += 1 if result.get("route_decision") == "score_from_homepage" else 0
            needs_tavily += 1 if result.get("route_decision") == "needs_tavily" else 0
            errors += 1 if result.get("fetch_error") else 0
            if index % commit_interval == 0:
                conn.commit()
            if progress_callback:
                progress_callback(index, len(companies), result, {
                    "processed": processed,
                    "accepted": accepted,
                    "provisional": provisional,
                    "unresolved": unresolved,
                    "score_ready": score_ready,
                    "needs_tavily": needs_tavily,
                    "errors": errors,
                    "parallel_workers": workers,
                })
    conn.commit()
    db.finish_run(
        conn,
        run_id,
        enriched_count=score_ready,
        cache_hits=0,
        estimated_cost_usd=0.0,
        notes=f"Processed: {processed}. Accepted: {accepted}. Provisional: {provisional}. Unresolved: {unresolved}. Needs Tavily: {needs_tavily}. Mode: {mode}.",
    )
    return PipelineResult(
        "homepage_evidence",
        f"Checked company pages for {processed:,} companies with {workers:,} parallel workers. {score_ready:,} can be scored from company-page data; {needs_tavily:,} need web search.",
        {
            "processed": processed,
            "accepted_domains": accepted,
            "provisional_domains": provisional,
            "unresolved_domains": unresolved,
            "homepage_score_ready": score_ready,
            "homepage_needs_tavily": needs_tavily,
            "errors": errors,
            "tavily_call_avoided_by_homepage_evidence": score_ready,
            "estimated_tavily_credits_saved": score_ready,
            "estimated_tavily_cost_saved": score_ready * float(_setting(settings, "tavily_cost_per_call_usd", 0.001) or 0.0),
            "parallel_workers": workers,
            "domain_attempts_per_company": max_domain_attempts or len(generate_domain_candidates("example")),
            "estimated_cost_usd": 0.0,
        },
    )


def _collect_homepage_for_company(
    company: dict[str, Any],
    session: requests.Session,
    *,
    timeout: float,
    max_bytes: int,
    mode: str,
    max_domain_attempts: int | None = None,
) -> dict[str, Any]:
    company_name = company["canonical_name"]
    best = None
    best_domain = None
    best_confidence = 0.0
    fetch_error = None

    domain_candidates = generate_domain_candidates(company_name)
    if max_domain_attempts is not None:
        domain_candidates = domain_candidates[: max(1, int(max_domain_attempts))]

    for domain in domain_candidates:
        metadata = fetch_homepage_metadata(domain, session=session, timeout=timeout, max_bytes=max_bytes)
        confidence = score_domain_confidence(company_name, domain, metadata)
        fetch_error = metadata.fetch_error or fetch_error
        if confidence > best_confidence:
            best = metadata
            best_domain = domain
            best_confidence = confidence
        if confidence >= 0.85:
            break

    status = domain_status(best_confidence)
    homepage = extract_homepage_evidence(best) if best is not None else None
    route = route_candidate_after_homepage_evidence(domain_status=status, evidence=homepage, mode=mode)

    return {
        "company_id": company["id"],
        "candidate_domain": best_domain,
        "resolved_url": best.final_url if best is not None else None,
        "domain_confidence": best_confidence,
        "domain_status": status,
        "metadata_json": best.to_dict() if best is not None else {},
        "evidence_text": homepage.evidence_text if homepage is not None else "",
        "evidence_quality": homepage.evidence_quality if homepage is not None else 0.0,
        "positive_signals": ([*homepage.positive_signals, *homepage.wittington_signals] if homepage is not None else []),
        "negative_signals": homepage.negative_signals if homepage is not None else [],
        "route_decision": route.route,
        "route_reason": route.reason,
        "fetch_error": fetch_error,
    }


def _homepage_enrichment_from_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    metadata = evidence.get("metadata_json") or {}
    return {
        "company_id": evidence["company_id"],
        "query": f"homepage:{evidence.get('candidate_domain') or ''}",
        "provider": "homepage",
        "raw_json": {
            "metadata": metadata,
            "domain_confidence": evidence.get("domain_confidence"),
            "evidence_quality": evidence.get("evidence_quality"),
            "positive_signals": evidence.get("positive_signals") or [],
            "negative_signals": evidence.get("negative_signals") or [],
            "route_decision": evidence.get("route_decision"),
        },
        "top_titles": [metadata.get("title") or metadata.get("og_title") or "Company page"],
        "top_urls": [evidence.get("resolved_url")] if evidence.get("resolved_url") else [],
        "top_snippets": [evidence.get("evidence_text") or ""],
        "website": evidence.get("resolved_url"),
        "status": "success",
        "error": None,
    }


def score_enriched_candidates(
    conn,
    settings: Settings,
    limit: int | None = None,
    force: bool = False,
    progress_callback=None,
    mode: str = "balanced",
) -> PipelineResult:
    limit = limit or settings.max_score
    run_id = db.start_run(conn, "openai_scoring")
    if not settings.openai_api_key:
        db.finish_run(conn, run_id, notes="Skipped: OPENAI_API_KEY is not set.")
        return PipelineResult(
            "openai_scoring",
            "Skipped OpenAI scoring because OPENAI_API_KEY is not set. Baseline deterministic scores remain available.",
            {"openai_calls": 0, "scored": 0, "errors": 0},
        )

    companies = db.enriched_for_openai_scoring(conn, limit=limit, force=force, mode=mode)
    calls = 0
    scored = 0
    errors = 0
    cache_hits = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    estimated_cost_usd = 0.0
    retry_attempts = 0
    provider_attempts = 0
    workers = _bounded_worker_count(settings, "openai_concurrency", 48, len(companies))
    commit_interval = _commit_interval(settings)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_score_company_with_openai, settings, company) for company in companies]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            score = result.pop("score", None)
            provider = result.pop("provider", "openai")
            company_id = result.pop("company_id", None)
            if score is not None and company_id is not None:
                db.save_score(conn, company_id, score, provider=provider, commit=False)
            calls += result["api_call"]
            scored += result["scored"]
            errors += result["error_count"]
            cache_hits += result["cache_hit"]
            prompt_tokens += result.get("prompt_tokens", 0)
            completion_tokens += result.get("completion_tokens", 0)
            total_tokens += result.get("total_tokens", 0)
            estimated_cost_usd += result.get("estimated_cost_usd", 0.0)
            retry_attempts += result.get("retry_attempts", 0)
            provider_attempts += result.get("provider_attempts", result["api_call"])
            if index % commit_interval == 0:
                conn.commit()
            if progress_callback:
                progress_callback(
                    index,
                    len(companies),
                    result,
                    {
                        "openai_calls": calls,
                        "scored": scored,
                        "errors": errors,
                        "cache_hits": cache_hits,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "estimated_cost_usd": estimated_cost_usd,
                        "retry_attempts": retry_attempts,
                    },
                )
    conn.commit()

    db.finish_run(
        conn,
        run_id,
        scored_count=scored,
        openai_calls=calls,
        cache_hits=cache_hits,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
        notes=f"Errors: {errors}. Retries: {retry_attempts}. Force refresh: {force}. Model: {settings.openai_model}. Mode: {mode}.",
    )
    return PipelineResult(
        "openai_scoring",
        f"Made {calls:,} OpenAI calls with {workers:,} parallel workers and stored {scored:,} structured scores. Tokens: {total_tokens:,}. Estimated model spend: ${estimated_cost_usd:.4f}.",
        {
            "openai_calls": calls,
            "scored": scored,
            "errors": errors,
            "cache_hits": cache_hits,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "provider_attempts": provider_attempts,
            "retry_attempts": retry_attempts,
            "parallel_workers": workers,
            "estimated_cost_usd": estimated_cost_usd,
        },
    )


def _fetch_enrichment_for_company(settings: Settings, company: dict[str, Any]) -> dict[str, Any]:
    try:
        client = TavilyClient(settings.tavily_api_key, settings.tavily_max_results)
        payload, attempts, retries = _call_provider_with_retries(
            settings,
            lambda: client.search(company["canonical_name"]),
        )
        enrichment = compact_tavily_response(company["id"], payload)
        return {
            "api_call": 1,
            "provider_attempts": attempts,
            "retry_attempts": retries,
            "cache_hit": 0,
            "enriched": 1 if enrichment["status"] == "success" else 0,
            "error_count": 0 if enrichment["status"] == "success" else 1,
            "status": enrichment["status"],
            "company_name": company["canonical_name"],
            "enrichment": enrichment,
        }
    except EnrichmentUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - persist provider failure per company.
        original = exc.original if isinstance(exc, ProviderCallFailed) else exc
        status_code = _status_code_from_exception(original)
        attempts = exc.attempts if isinstance(exc, ProviderCallFailed) else 1
        retries = exc.retries if isinstance(exc, ProviderCallFailed) else 0
        return {
            "api_call": 1,
            "provider_attempts": attempts,
            "retry_attempts": retries,
            "cache_hit": 0,
            "enriched": 0,
            "error_count": 1,
            "status": "error",
            "company_name": company["canonical_name"],
            "enrichment": {
                "company_id": company["id"],
                "query": f'"{company["canonical_name"]}" company startup logistics supply chain commerce healthcare climate funding',
                "provider": "tavily",
                "raw_json": {},
                "top_titles": [],
                "top_urls": [],
                "top_snippets": [],
                "website": None,
                "status": "error",
                "error": f"status={status_code or 'unknown'} {str(original)[:180]}",
            },
        }


def _score_company_with_openai(settings: Settings, company: dict[str, Any]) -> dict[str, Any]:
    try:
        score, attempts, retries = _call_provider_with_retries(
            settings,
            lambda: classify_with_openai(
                company,
                settings.openai_api_key or "",
                settings.openai_model,
                max(1, _setting_int(settings, "openai_max_completion_tokens", 500)),
            ),
        )
        usage = (score.get("raw_json") or {}).get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        estimated_cost_usd = _estimate_openai_cost(settings, prompt_tokens, completion_tokens)
        score["raw_json"]["estimated_cost_usd"] = estimated_cost_usd
        return {
            "api_call": 1,
            "provider_attempts": attempts,
            "retry_attempts": retries,
            "cache_hit": 0,
            "scored": 1,
            "error_count": 0,
            "status": "success",
            "company_name": company["canonical_name"],
            "company_id": company["id"],
            "provider": "openai",
            "score": score,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost_usd,
        }
    except Exception as exc:  # noqa: BLE001 - keep partial scoring runs usable.
        original = exc.original if isinstance(exc, ProviderCallFailed) else exc
        status_code = _status_code_from_exception(original)
        attempts = exc.attempts if isinstance(exc, ProviderCallFailed) else 1
        retries = exc.retries if isinstance(exc, ProviderCallFailed) else 0
        fallback = baseline_score(company)
        fallback["rationale"] = f"OpenAI scoring failed. Baseline retained. status={status_code or 'unknown'}."
        return {
            "api_call": 1,
            "provider_attempts": attempts,
            "retry_attempts": retries,
            "cache_hit": 0,
            "scored": 0,
            "error_count": 1,
            "status": "error",
            "company_name": company["canonical_name"],
            "company_id": company["id"],
            "provider": "baseline",
            "score": fallback,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }


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
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }

    try:
        score = classify_with_openai(
            company,
            settings.openai_api_key,
            settings.openai_model,
            max(1, _setting_int(settings, "openai_max_completion_tokens", 500)),
        )
        usage = (score.get("raw_json") or {}).get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        estimated_cost_usd = _estimate_openai_cost(settings, prompt_tokens, completion_tokens)
        score["raw_json"]["estimated_cost_usd"] = estimated_cost_usd
        db.save_score(conn, company["id"], score, provider="openai")
        return {
            "api_call": 1,
            "cache_hit": 0,
            "scored": 1,
            "error_count": 0,
            "status": "success",
            "company_name": company["canonical_name"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost_usd,
        }
    except Exception as exc:  # noqa: BLE001 - keep partial scoring runs usable.
        fallback = baseline_score(company)
        fallback["rationale"] = f"OpenAI scoring failed. Baseline retained. Error: {str(exc)[:120]}"
        db.save_score(conn, company["id"], fallback, provider="baseline")
        return {
            "api_call": 1,
            "cache_hit": 0,
            "scored": 0,
            "error_count": 1,
            "status": "error",
            "company_name": company["canonical_name"],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
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
        notes=f"Verified={verified}. company={company['canonical_name']}. force_refresh=False.",
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


def load_and_classify_companies(conn, settings: Settings) -> list[PipelineResult]:
    return [load_attendees(conn, settings), run_deterministic_classification(conn)]


def generate_verified_prospects(
    conn,
    settings: Settings,
    enrich_limit: int | None = None,
    score_limit: int | None = None,
    force: bool = False,
    mode: str = "balanced",
) -> list[PipelineResult]:
    return [
        enrich_candidates(conn, settings, enrich_limit or settings.max_enrich, force=force, mode=mode),
        score_enriched_candidates(conn, settings, score_limit or settings.max_score, force=force, mode=mode),
    ]
