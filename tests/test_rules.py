from src.rules import classify_company_name


def test_classifies_known_incumbent() -> None:
    result = classify_company_name("Amazon")
    assert result.deterministic_type == "incumbent_or_public_company"
    assert result.is_candidate is False


def test_classifies_investor_as_non_candidate() -> None:
    result = classify_company_name("Alpha Ventures")
    assert result.deterministic_type == "investor_or_financial_firm"
    assert result.is_candidate is False


def test_classifies_robotics_company_as_candidate() -> None:
    result = classify_company_name("Pickle Robot")
    assert result.deterministic_type == "likely_startup_or_tech"
    assert result.is_candidate is True


def test_classifies_plain_logistics_provider_as_service_provider() -> None:
    result = classify_company_name("Apex Logistics")
    assert result.deterministic_type == "logistics_service_provider"
    assert result.is_candidate is False


def test_short_ai_signal_does_not_match_inside_supply_chain() -> None:
    result = classify_company_name("Access Supply Chain Service")
    assert result.deterministic_type == "logistics_service_provider"
    assert result.is_candidate is False
