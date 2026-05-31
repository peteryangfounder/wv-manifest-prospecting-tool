from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import requests


class EnrichmentUnavailable(RuntimeError):
    """Raised when an external enrichment provider is not configured."""


@dataclass
class TavilyClient:
    api_key: str | None
    max_results: int = 3

    def search(self, company_name: str) -> dict:
        if not self.api_key:
            raise EnrichmentUnavailable("TAVILY_API_KEY is not set.")

        query = f'"{company_name}" company startup logistics supply chain commerce healthcare climate funding'
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": self.max_results,
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=30,
        )
        response.raise_for_status()
        return {"query": query, "response": response.json()}


def compact_tavily_response(company_id: int, search_payload: dict) -> dict:
    response = search_payload.get("response") or {}
    results = response.get("results") or []
    top_titles = []
    top_urls = []
    top_snippets = []

    for result in results:
        title = result.get("title")
        url = result.get("url")
        snippet = result.get("content") or result.get("snippet")
        if title:
            top_titles.append(title)
        if url:
            top_urls.append(url)
        if snippet:
            top_snippets.append(snippet[:600])

    website = None
    if top_urls:
        parsed = urlparse(top_urls[0])
        if parsed.netloc:
            website = f"{parsed.scheme}://{parsed.netloc}"

    return {
        "company_id": company_id,
        "query": search_payload.get("query") or "",
        "provider": "tavily",
        "raw_json": response,
        "top_titles": top_titles,
        "top_urls": top_urls,
        "top_snippets": top_snippets,
        "website": website,
        "status": "success" if results else "no_results",
        "error": None if results else "Tavily returned no results.",
    }
