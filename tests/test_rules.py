from src.rules import classify_company_name


def test_classifies_known_incumbent() -> None:
    result = classify_company_name("Amazon")
    assert result.deterministic_type == "incumbent_or_public_company"
    assert result.is_candidate is False
    assert result.high_priority_enrichment is False


def test_classifies_investor_as_non_candidate() -> None:
    result = classify_company_name("Alpha Ventures")
    assert result.deterministic_type == "investor_or_financial_firm"
    assert result.is_candidate is False
    assert result.high_priority_enrichment is False


def test_classifies_robotics_company_as_candidate() -> None:
    result = classify_company_name("Pickle Robot")
    assert result.deterministic_type == "likely_startup_or_tech"
    assert result.is_candidate is True
    assert result.high_priority_enrichment is True


def test_classifies_plain_logistics_provider_as_service_provider() -> None:
    result = classify_company_name("Apex Logistics")
    assert result.deterministic_type == "logistics_service_provider"
    assert result.is_candidate is False
    assert result.high_priority_enrichment is False


def test_short_ai_signal_does_not_match_inside_supply_chain() -> None:
    result = classify_company_name("Access Supply Chain Service")
    assert result.deterministic_type == "logistics_service_provider"
    assert result.is_candidate is False
    assert result.high_priority_enrichment is False


def test_obvious_non_targets_are_excluded_from_paid_enrichment() -> None:
    names = [
        "Walmart",
        "DHL",
        "FedEx",
        "Amazon",
        "Google",
        "Wittington Ventures",
        "Royal Bank of Canada",
        "Alpha Ventures",
        "University of Arkansas",
        "American Association of Port Authorities",
    ]

    for name in names:
        result = classify_company_name(name)
        assert result.high_priority_enrichment is False, name


def test_likely_tech_startups_enter_high_priority_queue() -> None:
    names = [
        "AutoScheduler.AI",
        "Pickle Robot",
        "Warehouse Automation Platform",
        "RetailReadyAI",
        "Supply Chain Visibility Software",
    ]

    for name in names:
        result = classify_company_name(name)
        assert result.is_candidate is True, name
        assert result.high_priority_enrichment is True, name


def test_generic_placeholders_are_not_high_priority() -> None:
    names = [
        "AI Startup",
        "Startup",
        "Stealth Company",
        "TBD",
        "Student",
        "COO",
        "CEO",
        "Founder",
        "N/A",
        "Unknown",
    ]

    for name in names:
        result = classify_company_name(name)
        assert result.deterministic_type == "duplicate_or_noisy_entry", name
        assert result.is_candidate is False, name
        assert result.high_priority_enrichment is False, name
