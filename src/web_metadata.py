from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import requests


@dataclass(frozen=True)
class PageMetadata:
    url: str
    final_url: str
    status_code: int | None
    title: str | None
    meta_description: str | None
    og_title: str | None
    og_description: str | None
    og_site_name: str | None
    canonical_url: str | None
    headings: list[str]
    jsonld_name: str | None
    jsonld_description: str | None
    text_snippet: str
    fetch_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "final_url": self.final_url,
            "status_code": self.status_code,
            "title": self.title,
            "meta_description": self.meta_description,
            "og_title": self.og_title,
            "og_description": self.og_description,
            "og_site_name": self.og_site_name,
            "canonical_url": self.canonical_url,
            "headings": self.headings,
            "jsonld_name": self.jsonld_name,
            "jsonld_description": self.jsonld_description,
            "text_snippet": self.text_snippet,
            "fetch_error": self.fetch_error,
        }


def _clean_text(value: str | None, limit: int = 600) -> str | None:
    if not value:
        return None
    compact = " ".join(value.split())
    return compact[:limit] if compact else None


def _meta_content(soup: BeautifulSoup, *, name: str | None = None, prop: str | None = None) -> str | None:
    if name:
        tag = soup.find("meta", attrs={"name": name})
    else:
        tag = soup.find("meta", attrs={"property": prop})
    if not tag:
        return None
    return _clean_text(tag.get("content"))


def _extract_jsonld_org(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(script.string or "")
        except json.JSONDecodeError:
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            graph = item.get("@graph")
            candidates = graph if isinstance(graph, list) else [item]
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                type_value = candidate.get("@type")
                types = type_value if isinstance(type_value, list) else [type_value]
                if any(str(value).lower() in {"organization", "corporation", "localbusiness"} for value in types):
                    return _clean_text(candidate.get("name")), _clean_text(candidate.get("description"))
    return None, None


def extract_page_metadata(html: str, url: str, *, final_url: str | None = None, status_code: int | None = None) -> PageMetadata:
    soup = BeautifulSoup(html or "", "html.parser")
    jsonld_name, jsonld_description = _extract_jsonld_org(soup)
    for tag_name in ("script", "style", "noscript"):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    title = _clean_text(soup.title.string if soup.title else None)
    canonical_tag = soup.find("link", attrs={"rel": "canonical"})
    canonical_url = urljoin(final_url or url, canonical_tag.get("href")) if canonical_tag and canonical_tag.get("href") else None
    headings = []
    for tag in soup.find_all(["h1", "h2"])[:8]:
        text = _clean_text(tag.get_text(" "))
        if text:
            headings.append(text)

    body_text = _clean_text(soup.get_text(" "), limit=1200) or ""

    return PageMetadata(
        url=url,
        final_url=final_url or url,
        status_code=status_code,
        title=title,
        meta_description=_meta_content(soup, name="description"),
        og_title=_meta_content(soup, prop="og:title"),
        og_description=_meta_content(soup, prop="og:description"),
        og_site_name=_meta_content(soup, prop="og:site_name"),
        canonical_url=canonical_url,
        headings=headings,
        jsonld_name=jsonld_name,
        jsonld_description=jsonld_description,
        text_snippet=body_text,
    )


def fetch_homepage_metadata(
    url: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 4.0,
    max_bytes: int = 200_000,
) -> PageMetadata:
    client = session or requests.Session()
    fetch_url = url if url.startswith(("http://", "https://")) else f"https://{url}"
    try:
        response = client.get(fetch_url, timeout=timeout, allow_redirects=True, headers={"User-Agent": "WVProspectingBot/1.0"})
        content = response.content[:max_bytes]
        text = content.decode(response.encoding or "utf-8", errors="replace")
        return extract_page_metadata(text, fetch_url, final_url=response.url, status_code=response.status_code)
    except Exception as exc:  # noqa: BLE001 - network failures become data gaps.
        return PageMetadata(
            url=fetch_url,
            final_url=fetch_url,
            status_code=None,
            title=None,
            meta_description=None,
            og_title=None,
            og_description=None,
            og_site_name=None,
            canonical_url=None,
            headings=[],
            jsonld_name=None,
            jsonld_description=None,
            text_snippet="",
            fetch_error=str(exc)[:180],
        )
