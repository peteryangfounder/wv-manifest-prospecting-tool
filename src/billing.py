from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import requests


OPENAI_COSTS_URL = "https://api.openai.com/v1/organization/costs"
OPENAI_COMPLETIONS_USAGE_URL = "https://api.openai.com/v1/organization/usage/completions"


@dataclass(frozen=True)
class TavilyBillingSummary:
    plan_name: str
    credits_used: int
    included_monthly_credits: int
    free_credits_remaining: int
    pay_as_you_go_enabled: bool
    overage_credits: int
    actual_billed_usd: float
    shadow_estimate_usd: float
    status: str


@dataclass(frozen=True)
class OpenAICostSummary:
    total: float = 0.0
    currency: str = "usd"
    amount_by_currency: dict[str, float] = field(default_factory=dict)
    bucket_count: int = 0
    result_count: int = 0
    project_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class OpenAIUsageSummary:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    request_count: int = 0
    results: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class OpenAIBillingSnapshot:
    available: bool
    status: str
    source_label: str
    scope_label: str
    cost: OpenAICostSummary = field(default_factory=OpenAICostSummary)
    usage: OpenAIUsageSummary = field(default_factory=OpenAIUsageSummary)
    error: str | None = None


@dataclass(frozen=True)
class ProviderBilledSpend:
    total_billed_usd: float
    openai_billed_usd: float | None
    tavily_billed_usd: float
    hosting_billed_usd: float
    is_complete: bool
    status: str


_OPENAI_BILLING_CACHE: dict[tuple[str | None, int], tuple[float, OpenAIBillingSnapshot]] = {}


def calculate_tavily_billing(
    *,
    credits_used: int,
    included_monthly_credits: int = 1000,
    pay_as_you_go_enabled: bool = False,
    payg_price_per_credit_usd: float = 0.008,
    plan_name: str = "Researcher",
    shadow_price_per_credit_usd: float = 0.001,
) -> TavilyBillingSummary:
    credits = max(0, int(credits_used or 0))
    included = max(0, int(included_monthly_credits or 0))
    overage = max(0, credits - included)
    remaining = max(0, included - credits)
    actual_billed_usd = overage * max(0.0, float(payg_price_per_credit_usd or 0.0)) if pay_as_you_go_enabled else 0.0
    shadow_estimate_usd = credits * max(0.0, float(shadow_price_per_credit_usd or 0.0))

    if overage > 0 and not pay_as_you_go_enabled:
        status = "over_limit_no_payg"
    elif overage > 0:
        status = "payg_overage_billed"
    else:
        status = "within_included_credits"

    return TavilyBillingSummary(
        plan_name=plan_name or "Researcher",
        credits_used=credits,
        included_monthly_credits=included,
        free_credits_remaining=remaining,
        pay_as_you_go_enabled=bool(pay_as_you_go_enabled),
        overage_credits=overage,
        actual_billed_usd=actual_billed_usd,
        shadow_estimate_usd=shadow_estimate_usd,
        status=status,
    )


def parse_openai_costs_response(payload: dict[str, Any]) -> OpenAICostSummary:
    amount_by_currency: dict[str, float] = {}
    project_ids: set[str] = set()
    bucket_count = 0
    result_count = 0

    for bucket in payload.get("data") or []:
        bucket_count += 1
        for result in bucket.get("results") or []:
            amount = result.get("amount") or {}
            currency = str(amount.get("currency") or "usd").lower()
            value = _safe_float(amount.get("value"))
            amount_by_currency[currency] = amount_by_currency.get(currency, 0.0) + value
            if result.get("project_id"):
                project_ids.add(str(result["project_id"]))
            result_count += 1

    if len(amount_by_currency) == 1:
        currency = next(iter(amount_by_currency))
        total = amount_by_currency[currency]
    else:
        currency = "mixed" if amount_by_currency else "usd"
        total = amount_by_currency.get("usd", 0.0)

    return OpenAICostSummary(
        total=total,
        currency=currency,
        amount_by_currency=amount_by_currency,
        bucket_count=bucket_count,
        result_count=result_count,
        project_ids=tuple(sorted(project_ids)),
    )


def parse_openai_completions_usage_response(payload: dict[str, Any]) -> OpenAIUsageSummary:
    input_tokens = 0
    output_tokens = 0
    cached_input_tokens = 0
    request_count = 0
    results: list[dict[str, Any]] = []

    for bucket in payload.get("data") or []:
        for result in bucket.get("results") or []:
            row = {
                "model": result.get("model"),
                "project_id": result.get("project_id"),
                "api_key_id": result.get("api_key_id"),
                "input_tokens": int(result.get("input_tokens") or 0),
                "output_tokens": int(result.get("output_tokens") or 0),
                "cached_input_tokens": int(result.get("input_cached_tokens") or 0),
                "request_count": int(result.get("num_model_requests") or 0),
            }
            input_tokens += row["input_tokens"]
            output_tokens += row["output_tokens"]
            cached_input_tokens += row["cached_input_tokens"]
            request_count += row["request_count"]
            results.append(row)

    return OpenAIUsageSummary(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        request_count=request_count,
        results=tuple(results),
    )


def fetch_openai_billing_snapshot(
    *,
    admin_key: str | None,
    project_id: str | None = None,
    lookback_days: int = 31,
    cache_ttl_seconds: int = 300,
    session: requests.Session | None = None,
    now: float | None = None,
) -> OpenAIBillingSnapshot:
    if not admin_key:
        return OpenAIBillingSnapshot(
            available=False,
            status="missing_admin_key",
            source_label="Live OpenAI billing unavailable",
            scope_label="No OpenAI admin key configured",
            error="OPENAI_ADMIN_KEY is not configured.",
        )

    current_time = float(now if now is not None else time.time())
    ttl = max(0, int(cache_ttl_seconds or 0))
    cache_key = (project_id or None, max(1, int(lookback_days or 1)))
    if ttl > 0 and session is None:
        cached = _OPENAI_BILLING_CACHE.get(cache_key)
        if cached and current_time - cached[0] < ttl:
            return cached[1]

    client = session or requests.Session()
    headers = {"Authorization": f"Bearer {admin_key}", "Content-Type": "application/json"}
    start_time = int(current_time) - (max(1, int(lookback_days or 1)) * 86_400)
    end_time = int(current_time)
    limit = min(31, max(1, int(lookback_days or 1)))
    common_params: list[tuple[str, Any]] = [
        ("start_time", start_time),
        ("end_time", end_time),
        ("bucket_width", "1d"),
        ("limit", limit),
    ]
    if project_id:
        common_params.extend([("project_ids", project_id), ("group_by", "project_id")])

    try:
        costs_payload = _get_paginated(client, OPENAI_COSTS_URL, headers, common_params)
        usage_params = [*common_params, ("group_by", "model"), ("group_by", "api_key_id")]
        usage_payload = _get_paginated(client, OPENAI_COMPLETIONS_USAGE_URL, headers, usage_params)
    except Exception as exc:  # noqa: BLE001 - show availability status without leaking secrets.
        if project_id:
            try:
                org_params = [
                    ("start_time", start_time),
                    ("end_time", end_time),
                    ("bucket_width", "1d"),
                    ("limit", limit),
                ]
                costs_payload = _get_paginated(client, OPENAI_COSTS_URL, headers, org_params)
                usage_payload = _get_paginated(
                    client,
                    OPENAI_COMPLETIONS_USAGE_URL,
                    headers,
                    [*org_params, ("group_by", "model"), ("group_by", "api_key_id")],
                )
                cost = parse_openai_costs_response(costs_payload)
                usage = parse_openai_completions_usage_response(usage_payload)
                snapshot = OpenAIBillingSnapshot(
                    available=True,
                    status="live_org_fallback",
                    source_label="Live from OpenAI organization costs API",
                    scope_label=f"Organization-level billing; project filter unavailable: {project_id}",
                    cost=cost,
                    usage=usage,
                    error=_sanitize_error(str(exc)),
                )
                if ttl > 0 and session is None:
                    _OPENAI_BILLING_CACHE[cache_key] = (current_time, snapshot)
                return snapshot
            except Exception as fallback_exc:  # noqa: BLE001 - retain graceful UI fallback.
                exc = fallback_exc
        return OpenAIBillingSnapshot(
            available=False,
            status="api_error",
            source_label="Live OpenAI billing unavailable",
            scope_label=_scope_label(project_id, False),
            error=_sanitize_error(str(exc)),
        )

    cost = parse_openai_costs_response(costs_payload)
    usage = parse_openai_completions_usage_response(usage_payload)
    snapshot = OpenAIBillingSnapshot(
        available=True,
        status="live",
        source_label="Live from OpenAI organization costs API",
        scope_label=_scope_label(project_id, True),
        cost=cost,
        usage=usage,
    )
    if ttl > 0 and session is None:
        _OPENAI_BILLING_CACHE[cache_key] = (current_time, snapshot)
    return snapshot


def calculate_provider_billed_spend(
    *,
    tavily_billing: TavilyBillingSummary,
    openai_billing: OpenAIBillingSnapshot,
    hosting_billed_usd: float = 0.0,
) -> ProviderBilledSpend:
    hosting = max(0.0, float(hosting_billed_usd or 0.0))
    tavily = max(0.0, float(tavily_billing.actual_billed_usd or 0.0))
    openai_billed_usd = None
    if openai_billing.available and openai_billing.cost.currency == "usd":
        openai_billed_usd = max(0.0, float(openai_billing.cost.total or 0.0))

    is_complete = openai_billed_usd is not None
    total = tavily + hosting + (openai_billed_usd or 0.0)
    status = "complete" if is_complete else "openai_unavailable"
    return ProviderBilledSpend(
        total_billed_usd=total,
        openai_billed_usd=openai_billed_usd,
        tavily_billed_usd=tavily,
        hosting_billed_usd=hosting,
        is_complete=is_complete,
        status=status,
    )


def _get_paginated(
    session: requests.Session,
    url: str,
    headers: dict[str, str],
    params: list[tuple[str, Any]],
) -> dict[str, Any]:
    combined_data: list[dict[str, Any]] = []
    next_page: str | None = None
    has_more = True
    response_object = "page"

    while has_more:
        request_params = list(params)
        if next_page:
            request_params.append(("page", next_page))
        response = session.get(url, headers=headers, params=request_params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        response_object = payload.get("object") or response_object
        combined_data.extend(payload.get("data") or [])
        has_more = bool(payload.get("has_more"))
        next_page = payload.get("next_page")
        if has_more and not next_page:
            break

    return {"object": response_object, "data": combined_data, "has_more": False, "next_page": None}


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _scope_label(project_id: str | None, filtered: bool) -> str:
    if project_id and filtered:
        return f"Project filtered: {project_id}"
    if project_id:
        return f"Project requested but unavailable: {project_id}"
    return "Organization-level billing"


def _sanitize_error(message: str) -> str:
    clean = " ".join(str(message or "").split())
    return clean[:180] if clean else "OpenAI billing request failed."
