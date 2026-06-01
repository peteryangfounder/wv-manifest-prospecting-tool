from __future__ import annotations

import pytest

from src.billing import (
    OpenAIBillingSnapshot,
    calculate_provider_billed_spend,
    calculate_tavily_billing,
    fetch_openai_billing_snapshot,
    parse_openai_completions_usage_response,
    parse_openai_costs_response,
)


class MockResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self.payload


class MockSession:
    def __init__(self, responses: list[MockResponse]):
        self.responses = responses
        self.calls: list[dict] = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "params": params, "timeout": timeout})
        if not self.responses:
            raise RuntimeError("No mocked response left")
        return self.responses.pop(0)


@pytest.mark.parametrize(
    ("credits_used", "included", "payg_enabled", "expected_bill", "expected_status"),
    [
        (20, 1000, False, 0.00, "within_included_credits"),
        (1000, 1000, False, 0.00, "within_included_credits"),
        (1100, 1000, False, 0.00, "over_limit_no_payg"),
        (1100, 1000, True, 0.80, "payg_overage_billed"),
    ],
)
def test_tavily_billing_actual_spend(credits_used, included, payg_enabled, expected_bill, expected_status) -> None:
    summary = calculate_tavily_billing(
        credits_used=credits_used,
        included_monthly_credits=included,
        pay_as_you_go_enabled=payg_enabled,
        payg_price_per_credit_usd=0.008,
    )

    assert summary.actual_billed_usd == pytest.approx(expected_bill)
    assert summary.status == expected_status
    assert summary.overage_credits == max(0, credits_used - included)


def test_openai_cost_parser_sums_amount_values_and_currency() -> None:
    summary = parse_openai_costs_response(
        {
            "data": [
                {
                    "results": [
                        {"amount": {"value": 0.06, "currency": "usd"}, "project_id": "proj_a"},
                        {"amount": {"value": 0.04, "currency": "usd"}, "project_id": "proj_a"},
                    ]
                },
                {"results": [{"amount": {"value": 0.15, "currency": "usd"}, "project_id": "proj_b"}]},
            ]
        }
    )

    assert summary.total == pytest.approx(0.25)
    assert summary.currency == "usd"
    assert summary.amount_by_currency == {"usd": pytest.approx(0.25)}
    assert summary.result_count == 3
    assert summary.project_ids == ("proj_a", "proj_b")


def test_openai_usage_parser_extracts_token_model_project_key_and_requests() -> None:
    summary = parse_openai_completions_usage_response(
        {
            "data": [
                {
                    "results": [
                        {
                            "input_tokens": 100,
                            "output_tokens": 40,
                            "input_cached_tokens": 25,
                            "num_model_requests": 2,
                            "model": "gpt-test",
                            "project_id": "proj_a",
                            "api_key_id": "key_a",
                        }
                    ]
                }
            ]
        }
    )

    assert summary.input_tokens == 100
    assert summary.output_tokens == 40
    assert summary.cached_input_tokens == 25
    assert summary.request_count == 2
    assert summary.results[0]["model"] == "gpt-test"
    assert summary.results[0]["project_id"] == "proj_a"
    assert summary.results[0]["api_key_id"] == "key_a"


def test_openai_billing_missing_admin_key_does_not_crash() -> None:
    snapshot = fetch_openai_billing_snapshot(admin_key=None)

    assert snapshot.available is False
    assert snapshot.status == "missing_admin_key"
    assert "unavailable" in snapshot.source_label.lower()


def test_openai_billing_api_failure_does_not_crash() -> None:
    session = MockSession([MockResponse({}, status_code=500)])

    snapshot = fetch_openai_billing_snapshot(admin_key="admin-key", session=session, cache_ttl_seconds=0)

    assert snapshot.available is False
    assert snapshot.status == "api_error"
    assert "admin-key" not in str(snapshot.error)


def test_openai_billing_client_parses_cost_and_usage_responses() -> None:
    session = MockSession(
        [
            MockResponse(
                {
                    "object": "page",
                    "data": [{"results": [{"amount": {"value": 0.06, "currency": "usd"}}]}],
                    "has_more": False,
                }
            ),
            MockResponse(
                {
                    "object": "page",
                    "data": [
                        {
                            "results": [
                                {
                                    "input_tokens": 10,
                                    "output_tokens": 5,
                                    "input_cached_tokens": 2,
                                    "num_model_requests": 1,
                                    "model": "gpt-test",
                                    "project_id": "proj_a",
                                    "api_key_id": "key_a",
                                }
                            ]
                        }
                    ],
                    "has_more": False,
                }
            ),
        ]
    )

    snapshot = fetch_openai_billing_snapshot(
        admin_key="admin-key",
        project_id="proj_a",
        lookback_days=7,
        cache_ttl_seconds=0,
        session=session,
        now=1_800_000_000,
    )

    assert snapshot.available is True
    assert snapshot.cost.total == pytest.approx(0.06)
    assert snapshot.cost.currency == "usd"
    assert snapshot.usage.input_tokens == 10
    assert snapshot.usage.output_tokens == 5
    assert snapshot.usage.cached_input_tokens == 2
    assert snapshot.usage.request_count == 1
    assert all(call["headers"]["Authorization"] == "Bearer admin-key" for call in session.calls)


def test_actual_provider_billed_total_excludes_tavily_free_credits() -> None:
    tavily = calculate_tavily_billing(
        credits_used=20,
        included_monthly_credits=1000,
        pay_as_you_go_enabled=False,
        shadow_price_per_credit_usd=0.001,
    )
    openai = OpenAIBillingSnapshot(available=False, status="missing", source_label="unavailable", scope_label="org")

    total = calculate_provider_billed_spend(tavily_billing=tavily, openai_billing=openai, hosting_billed_usd=0.0)

    assert tavily.shadow_estimate_usd == pytest.approx(0.02)
    assert total.tavily_billed_usd == 0.0
    assert total.total_billed_usd == 0.0
    assert total.is_complete is False
