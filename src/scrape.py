from __future__ import annotations

import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .config import DATA_DIR, MANIFEST_URL


SEED_PATH = DATA_DIR / "manifest_attendees_seed.txt"

NOISE_EXACT = {
    "sponsored by",
    "companies attending",
    "companies who attend include:",
    "companies who attend include",
    "make connections to accelerate your business forward at manifest vegas.",
    "skip to content",
    "privacy",
    "terms",
    "contact",
    "follow",
    "manifest",
    "register",
    "agenda",
    "speakers",
    "sponsors",
    "venue",
    "faq",
}

NOISE_CONTAINS = (
    "cookie",
    "copyright",
    "all rights reserved",
    "subscribe",
    "newsletter",
    "book a meeting",
    "download",
    "menu",
    "search",
    "privacy policy",
    "terms of use",
)


def fetch_manifest_text(url: str = MANIFEST_URL, timeout: int = 25) -> str:
    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
            )
        },
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return soup.get_text("\n")


def load_seed_text(path: Path = SEED_PATH) -> str:
    return path.read_text(encoding="utf-8")


def _looks_like_company_line(line: str) -> bool:
    compact = line.strip()
    lower = compact.lower()

    if not compact or lower in NOISE_EXACT:
        return False
    if any(part in lower for part in NOISE_CONTAINS):
        return False
    if "http://" in lower or "https://" in lower or "@" in compact:
        return False
    if len(compact) > 120:
        return False
    if re.fullmatch(r"[-_'\".,:;|/\\]+", compact):
        return False
    if len(re.findall(r"[A-Za-z0-9]", compact)) < 2:
        return False
    if compact.lower().startswith(("page ", "previous", "next")):
        return False

    return True


def parse_company_lines(text: str) -> list[str]:
    companies: list[str] = []
    seen = set()

    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not _looks_like_company_line(line):
            continue

        key = line.casefold()
        if key not in seen:
            companies.append(line)
            seen.add(key)

    return companies


def get_attendee_names(
    url: str = MANIFEST_URL,
    fallback_path: Path = SEED_PATH,
    min_live_rows: int = 200,
) -> tuple[list[str], dict[str, str | int | bool]]:
    metadata: dict[str, str | int | bool] = {"used_live_scrape": False}
    live_error = None

    try:
        live_text = fetch_manifest_text(url)
        live_names = parse_company_lines(live_text)
        if len(live_names) >= min_live_rows:
            metadata.update(
                {
                    "used_live_scrape": True,
                    "source": url,
                    "raw_count": len(live_names),
                }
            )
            return live_names, metadata
        live_error = f"Live scrape returned only {len(live_names)} company-like rows."
    except Exception as exc:  # noqa: BLE001 - surface the scrape failure in metadata.
        live_error = str(exc)

    seed_text = load_seed_text(fallback_path)
    seed_names = parse_company_lines(seed_text)
    metadata.update(
        {
            "source": str(fallback_path),
            "raw_count": len(seed_names),
            "live_error": live_error or "",
        }
    )
    return seed_names, metadata
