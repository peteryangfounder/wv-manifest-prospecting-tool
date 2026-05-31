from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass


LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "co",
    "company",
    "companies",
    "group",
    "holdings",
    "holding",
    "plc",
    "gmbh",
    "ag",
    "sa",
    "pte",
    "pty",
    "lp",
    "llp",
}


@dataclass(frozen=True)
class CompanyRecord:
    raw_name: str
    canonical_name: str
    normalized_name: str
    duplicate_count: int = 1


def canonicalize_name(raw_name: str) -> str:
    name = unicodedata.normalize("NFKC", raw_name or "")
    name = name.replace("\u00a0", " ")
    name = re.sub(r"\s+", " ", name).strip()
    name = name.strip("`'\"“”‘’")
    return name


def normalize_name(raw_name: str) -> str:
    name = canonicalize_name(raw_name).lower()
    name = name.replace("&", " and ")
    name = re.sub(r"[/|]", " ", name)
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    tokens = [token for token in name.split() if token]

    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()

    return " ".join(tokens)


def dedupe_names(raw_names: list[str]) -> list[CompanyRecord]:
    records: OrderedDict[str, CompanyRecord] = OrderedDict()

    for raw in raw_names:
        canonical = canonicalize_name(raw)
        normalized = normalize_name(canonical)
        if len(normalized) < 2:
            continue

        existing = records.get(normalized)
        if existing is None:
            records[normalized] = CompanyRecord(
                raw_name=raw,
                canonical_name=canonical,
                normalized_name=normalized,
                duplicate_count=1,
            )
        else:
            records[normalized] = CompanyRecord(
                raw_name=existing.raw_name,
                canonical_name=existing.canonical_name,
                normalized_name=existing.normalized_name,
                duplicate_count=existing.duplicate_count + 1,
            )

    return list(records.values())
