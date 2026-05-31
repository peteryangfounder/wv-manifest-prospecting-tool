from src.clean import dedupe_names
from src.rules import classify_company_name
from src.scrape import load_seed_text, parse_company_lines


def test_high_priority_queue_is_much_smaller_than_full_manifest_list() -> None:
    raw_names = parse_company_lines(load_seed_text())
    companies = dedupe_names(raw_names)
    high_priority = [
        company
        for company in companies
        if classify_company_name(company.canonical_name).high_priority_enrichment
    ]

    assert len(companies) > 3000
    assert len(high_priority) > 25
    assert len(high_priority) < len(companies) * 0.25
