from src.score import apply_score_caps


def test_investor_score_cap() -> None:
    assert apply_score_caps(90, "investor_or_financial_firm", has_external_evidence=True) == 25


def test_no_external_evidence_cap() -> None:
    assert apply_score_caps(90, "likely_startup_or_tech", has_external_evidence=False, has_strong_startup_evidence=True) == 50


def test_service_provider_without_tech_cap() -> None:
    assert apply_score_caps(90, "logistics_service_provider", has_external_evidence=True) == 45
