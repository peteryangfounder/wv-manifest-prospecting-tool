from src.clean import dedupe_names, normalize_name


def test_normalize_strips_legal_suffix_for_dedupe() -> None:
    assert normalize_name("PAXAFE, Inc.") == "paxafe"
    assert normalize_name("GoodShip LLC") == "goodship"


def test_dedupe_merges_obvious_legal_suffix_duplicates() -> None:
    records = dedupe_names(["Acme Logistics LLC", "Acme Logistics, Inc.", "Other Co"])
    assert len(records) == 2
    acme = next(record for record in records if record.normalized_name == "acme logistics")
    assert acme.duplicate_count == 2
