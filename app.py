from __future__ import annotations

import html
import json
from collections import Counter
from dataclasses import replace

import altair as alt
import pandas as pd
import streamlit as st

from src import db
from src.billing import (
    OpenAIBillingSnapshot,
    ProviderBilledSpend,
    TavilyBillingSummary,
    calculate_provider_billed_spend,
    calculate_tavily_billing,
    fetch_openai_billing_snapshot,
)
from src.config import PROJECT_ROOT, get_settings
from src.pipeline import (
    enrich_candidates,
    load_attendees,
    run_deterministic_classification,
    score_enriched_candidates,
)


st.set_page_config(
    page_title="Manifest Prospecting Tool",
    page_icon="WV",
    layout="wide",
)


DEFAULT_WEIGHTS = {
    "venture_backability": 25,
    "wv_sector_fit": 25,
    "wittington_edge": 20,
    "stage_signal": 10,
    "traction_signal": 10,
    "data_confidence": 10,
}

COMPONENT_MAX = {
    "venture_backability": 25,
    "wv_sector_fit": 25,
    "wittington_edge": 20,
    "stage_signal": 10,
    "traction_signal": 10,
    "data_confidence": 10,
}

DISPLAY_LABELS = {
    "ai": "AI",
    "baseline": "Rule screen",
    "cache_verification": "Cache check",
    "climate": "Climate",
    "commerce": "Commerce",
    "completion_tokens": "Output tokens",
    "consulting": "Consulting",
    "consulting_or_agency": "Consulting or agency",
    "consumer": "Consumer",
    "data_confidence": "Evidence confidence",
    "duplicate_or_noisy_entry": "Placeholder or noisy entry",
    "fintech": "Fintech",
    "food": "Food",
    "healthcare": "Healthcare",
    "incumbent": "Incumbent",
    "incumbent_or_public_company": "Incumbent or public company",
    "investor": "Investor",
    "investor_or_financial_firm": "Investor or financial firm",
    "likely_startup_or_tech": "Technology company",
    "load_attendees": "Source load",
    "logistics": "Logistics",
    "logistics_service_provider": "Logistics service provider",
    "media_association": "Media or association",
    "media_event_association": "Media or event group",
    "openai": "API verified",
    "openai_scoring": "OpenAI scoring",
    "prompt_tokens": "Input tokens",
    "retail_infrastructure": "Retail infrastructure",
    "retailer_or_brand_incumbent": "Retail or brand incumbent",
    "robotics": "Robotics",
    "service_provider": "Service provider",
    "stage_signal": "Stage signal",
    "startup": "Startup",
    "supply_chain": "Supply chain",
    "tavily_enrichment": "Search enrichment",
    "technology": "Technology",
    "traction_signal": "Traction signal",
    "university_government_nonprofit": "University, government, or nonprofit",
    "unknown": "Needs evidence",
    "unknown_needs_enrichment": "Needs evidence",
    "venture_backability": "Venture backability",
    "warehouse_automation": "Warehouse automation",
    "wittington_edge": "Wittington edge",
    "wv_sector_fit": "WV sector fit",
}

GENERIC_PLACEHOLDERS = {
    "ai company",
    "ai startup",
    "company",
    "early stage startup",
    "logistics startup",
    "my company",
    "new startup",
    "startup",
    "startup company",
    "stealth",
    "stealth company",
    "stealth mode",
    "stealth startup",
    "tech startup",
    "technology startup",
    "tbd",
}

OPENAI_MODEL_PRESETS = {
    "gpt-5.5": {"input": 1.25, "output": 10.00},
    "gpt-5.4": {"input": 1.25, "output": 10.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


CUSTOM_CSS = """
<style>
  .block-container {
    max-width: 1240px;
    padding-top: 1rem;
    padding-bottom: 2.5rem;
  }
  h1, h2, h3 {
    letter-spacing: 0 !important;
  }
  .wv-header {
    border-bottom: 1px solid #eceff3;
    margin-bottom: 0.9rem;
    padding: 0.35rem 0 1rem 0;
  }
  .wv-eyebrow {
    color: #5d6675;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    line-height: 1.35;
    margin: 0 0 0.45rem 0;
    text-transform: uppercase;
    white-space: normal;
  }
  .wv-title {
    color: #202332;
    font-size: clamp(2rem, 4.4vw, 3.05rem);
    font-weight: 780;
    line-height: 1.08;
    margin: 0;
    overflow-wrap: normal;
    white-space: normal;
  }
  .wv-subtitle {
    color: #697386;
    font-size: 1rem;
    margin: 0.6rem 0 0 0;
    max-width: 800px;
  }
  .guided-panel {
    background: #ffffff;
    border: 1px solid #dfe5ee;
    border-radius: 10px;
    margin: 1rem 0 1.15rem 0;
    padding: 1rem;
  }
  .guided-kicker {
    color: #697386;
    font-size: 0.78rem;
    font-weight: 760;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .guided-title {
    color: #202332;
    font-size: 1.35rem;
    font-weight: 780;
    margin-top: 0.15rem;
  }
  .guided-copy {
    color: #5d6675;
    font-size: 0.96rem;
    line-height: 1.45;
    margin: 0.35rem 0 0.85rem 0;
    max-width: 760px;
  }
  .step-list {
    display: grid;
    gap: 0.55rem;
    grid-template-columns: 1fr;
    margin: 0.8rem 0 1rem 0;
  }
  .step-row {
    align-items: flex-start;
    background: #f8fafc;
    border: 1px solid #e5e9ef;
    border-radius: 8px;
    display: grid;
    gap: 0.75rem;
    grid-template-columns: 2rem 1fr auto;
    padding: 0.72rem 0.8rem;
  }
  .step-row.active {
    background: #f0f6ff;
    border-color: #2f6fed;
  }
  .step-row.complete {
    background: #f1fbf5;
    border-color: #44a463;
  }
  .step-number {
    align-items: center;
    background: #202332;
    border-radius: 999px;
    color: #ffffff;
    display: flex;
    font-size: 0.85rem;
    font-weight: 780;
    height: 1.7rem;
    justify-content: center;
    line-height: 1;
    width: 1.7rem;
  }
  .step-row.complete .step-number {
    background: #16823d;
  }
  .step-row.active .step-number {
    background: #2f6fed;
  }
  .step-label {
    color: #202332;
    font-size: 0.98rem;
    font-weight: 760;
  }
  .step-detail {
    color: #8b94a5;
    font-size: 0.82rem;
    line-height: 1.35;
    margin-top: 0.12rem;
  }
  .step-state {
    color: #697386;
    font-size: 0.78rem;
    font-weight: 720;
    white-space: nowrap;
  }
  .primary-cta button {
    background: #202332 !important;
    border-color: #202332 !important;
    color: #ffffff !important;
    font-weight: 760 !important;
  }
  .stButton > button[kind="primary"],
  .stButton > button[data-testid="baseButton-primary"] {
    background: #202332 !important;
    border-color: #202332 !important;
    color: #ffffff !important;
    font-weight: 760 !important;
  }
  .stButton > button[kind="primary"]:hover,
  .stButton > button[data-testid="baseButton-primary"]:hover {
    background: #111827 !important;
    border-color: #111827 !important;
    color: #ffffff !important;
  }
  .run-settings {
    background: #f8fafc;
    border: 1px solid #e6eaf1;
    border-radius: 8px;
    margin-top: 0.8rem;
    padding: 0.85rem;
  }
  .summary-grid {
    display: grid;
    gap: 0.75rem;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    margin: 1rem 0 1.2rem 0;
  }
  .cost-hero {
    background: #ffffff;
    border: 1px solid #dfe5ee;
    border-radius: 8px;
    margin: 1rem 0;
    padding: 1.05rem;
  }
  .cost-hero-title {
    color: #202332;
    font-size: 0.9rem;
    font-weight: 760;
    margin-bottom: 0.65rem;
    text-transform: uppercase;
  }
  .cost-hero-grid {
    display: grid;
    gap: 0.8rem;
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
  .cost-hero-label {
    color: #697386;
    font-size: 0.82rem;
    line-height: 1.35;
  }
  .cost-hero-value {
    color: #202332;
    font-size: 1.35rem;
    font-weight: 780;
    line-height: 1.2;
    margin-top: 0.18rem;
    overflow-wrap: anywhere;
  }
  .cost-hero-value.compact {
    font-size: 1.16rem;
    white-space: nowrap;
  }
  .summary-card {
    background: #ffffff;
    border: 1px solid #e5e9ef;
    border-radius: 8px;
    padding: 0.95rem;
  }
  .summary-title {
    color: #202332;
    font-size: 0.92rem;
    font-weight: 760;
    margin-bottom: 0.5rem;
  }
  .summary-line {
    align-items: baseline;
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.24rem 0;
  }
  .summary-label {
    color: #697386;
    font-size: 0.86rem;
  }
  .summary-value {
    color: #202332;
    font-size: 0.96rem;
    font-weight: 760;
    text-align: right;
  }
  .status-strip {
    background: #f6f8fb;
    border: 1px solid #e5e9ef;
    border-left: 4px solid #2f6fed;
    border-radius: 8px;
    color: #313647;
    font-size: 0.92rem;
    margin: 0.65rem 0 1.1rem 0;
    padding: 0.72rem 0.9rem;
  }
  .status-strip.success {
    border-left-color: #12823b;
  }
  .status-strip.warning {
    border-left-color: #bf7a00;
  }
  .status-strip.error {
    border-left-color: #d63d3d;
  }
  .section-label {
    color: #202332;
    font-size: 1.05rem;
    font-weight: 740;
    margin: 0.3rem 0 0.6rem 0;
  }
  .quiet-note {
    color: #697386;
    font-size: 0.88rem;
    margin: -0.15rem 0 0.75rem 0;
  }
  .detail-box {
    background: #ffffff;
    border: 1px solid #e5e9ef;
    border-radius: 8px;
    padding: 1rem;
  }
  .detail-title {
    color: #202332;
    font-size: 1.15rem;
    font-weight: 760;
    margin: 0 0 0.25rem 0;
  }
  .detail-text {
    color: #313647;
    font-size: 0.95rem;
    line-height: 1.5;
    margin-top: 0.45rem;
  }
  .sidebar-badge {
    background: #f6f8fb;
    border: 1px solid #e5e9ef;
    border-radius: 8px;
    color: #313647;
    font-size: 0.83rem;
    margin-bottom: 0.45rem;
    padding: 0.48rem 0.6rem;
  }
  .sidebar-badge strong {
    color: #202332;
  }
  .wv-table {
    border: 1px solid #e5e9ef;
    border-collapse: separate;
    border-radius: 8px;
    border-spacing: 0;
    overflow: hidden;
    table-layout: fixed;
    width: 100%;
  }
  .wv-table th {
    background: #f8fafc;
    border-bottom: 1px solid #e5e9ef;
    color: #687084;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 0.72rem 0.65rem;
    text-align: left;
    text-transform: uppercase;
    word-break: normal;
  }
  .wv-table td {
    border-bottom: 1px solid #edf0f4;
    color: #272b3a;
    font-size: 0.88rem;
    line-height: 1.35;
    padding: 0.68rem 0.65rem;
    vertical-align: top;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: normal;
  }
  .wv-table tr:last-child td {
    border-bottom: 0;
  }
  .wv-table a {
    color: #1559b7;
    font-weight: 700;
    text-decoration: none;
  }
  .wv-pill {
    background: #f0f3f7;
    border-radius: 999px;
    color: #414a5c;
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 650;
    margin: 0 0.25rem 0.25rem 0;
    padding: 0.18rem 0.48rem;
  }
  .score-cell {
    align-items: center;
    display: flex;
    gap: 0.55rem;
  }
  .score-track {
    background: #edf0f4;
    border-radius: 999px;
    flex: 1;
    height: 0.55rem;
    min-width: 64px;
    overflow: hidden;
  }
  .score-fill {
    background: #1474c9;
    border-radius: inherit;
    display: block;
    height: 100%;
  }
  .score-number {
    color: #202332;
    font-weight: 740;
    min-width: 2.2rem;
    text-align: right;
  }
  .empty-state {
    background: #f8fafc;
    border: 1px dashed #d7dde6;
    border-radius: 8px;
    color: #697386;
    padding: 1rem;
  }
  .prospect-list {
    display: grid;
    gap: 0.75rem;
    margin: 0.4rem 0 1rem 0;
  }
  .prospect-card {
    background: #ffffff;
    border: 1px solid #e5e9ef;
    border-radius: 8px;
    display: grid;
    gap: 0.75rem;
    grid-template-columns: minmax(12rem, 1.15fr) minmax(8rem, 0.75fr) minmax(14rem, 1.4fr);
    padding: 0.9rem;
  }
  .prospect-main {
    min-width: 0;
  }
  .prospect-rank {
    color: #697386;
    font-size: 0.78rem;
    font-weight: 760;
    margin-bottom: 0.24rem;
    text-transform: uppercase;
  }
  .prospect-name {
    color: #202332;
    font-size: 1rem;
    font-weight: 760;
    line-height: 1.32;
    overflow-wrap: anywhere;
  }
  .prospect-score {
    min-width: 0;
  }
  .prospect-score-label {
    color: #697386;
    font-size: 0.78rem;
    font-weight: 760;
    margin-bottom: 0.32rem;
    text-transform: uppercase;
  }
  .prospect-tags {
    min-width: 0;
  }
  .prospect-evidence {
    color: #313647;
    font-size: 0.92rem;
    line-height: 1.45;
    min-width: 0;
    overflow-wrap: anywhere;
  }
  .source-list {
    display: grid;
    gap: 0.75rem;
    margin: 0.4rem 0 1rem 0;
  }
  .source-card {
    background: #ffffff;
    border: 1px solid #e5e9ef;
    border-radius: 8px;
    display: grid;
    gap: 0.75rem;
    grid-template-columns: minmax(14rem, 1.2fr) minmax(12rem, 0.8fr);
    padding: 0.9rem;
  }
  .source-name {
    color: #202332;
    font-size: 1rem;
    font-weight: 760;
    line-height: 1.32;
    overflow-wrap: anywhere;
  }
  .source-raw {
    color: #697386;
    font-size: 0.84rem;
    line-height: 1.35;
    margin-top: 0.26rem;
    overflow-wrap: anywhere;
  }
  .source-meta {
    align-items: baseline;
    display: flex;
    gap: 0.45rem;
    color: #202332;
    font-size: 0.84rem;
    font-weight: 650;
    margin-top: 0.45rem;
  }
  .source-score {
    min-width: 0;
  }
  .evidence-list {
    display: grid;
    gap: 0.75rem;
    margin-top: 0.35rem;
  }
  .evidence-card {
    background: #ffffff;
    border: 1px solid #e5e9ef;
    border-radius: 8px;
    padding: 0.85rem;
  }
  .evidence-link {
    color: #1559b7;
    font-size: 0.98rem;
    font-weight: 720;
    line-height: 1.35;
    text-decoration: none;
  }
  @media (max-width: 1100px) {
    .summary-grid {
      grid-template-columns: 1fr;
    }
    .cost-hero-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .prospect-card {
      grid-template-columns: minmax(12rem, 1fr) minmax(8rem, 0.7fr);
    }
    .prospect-evidence {
      grid-column: 1 / -1;
    }
    .source-card {
      grid-template-columns: 1fr;
    }
  }
  @media (max-width: 760px) {
    .step-row {
      grid-template-columns: 2rem 1fr;
    }
    .step-state {
      grid-column: 2;
    }
    .wv-table th, .wv-table td {
      font-size: 0.78rem;
      padding: 0.52rem 0.45rem;
    }
    .source-table th:nth-child(1),
    .source-table td:nth-child(1),
    .source-table th:nth-child(3),
    .source-table td:nth-child(3),
    .source-table th:nth-child(7),
    .source-table td:nth-child(7) {
      display: none;
    }
    .cost-hero-grid {
      grid-template-columns: 1fr;
    }
    .prospect-card {
      grid-template-columns: 1fr;
    }
    .prospect-evidence {
      grid-column: auto;
    }
    .source-card {
      grid-template-columns: 1fr;
    }
    .source-score {
      grid-column: auto;
    }
  }
</style>
"""


def _connect():
    settings = get_settings()
    conn = db.connect(settings.database_path)
    db.init_db(conn)
    return settings, conn


def _json_list(value) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def _format_int(value: int | float | None) -> str:
    return f"{int(value or 0):,}"


def _format_currency(value: int | float | None) -> str:
    amount = float(value or 0)
    if amount < 0.01:
        return f"${amount:.4f}"
    return f"${amount:,.2f}"


def _format_money(value: int | float | None, currency: str = "usd") -> str:
    currency_clean = (currency or "usd").upper()
    if currency_clean == "USD":
        return _format_currency(value)
    return f"{float(value or 0):,.4f} {currency_clean}"


def _format_billed_total(spend: ProviderBilledSpend) -> str:
    if spend.is_complete:
        return _format_currency(spend.total_billed_usd)
    return "Unavailable"


def _pct(part: int | float | None, whole: int | float | None) -> str:
    if not whole:
        return "0%"
    return f"{(float(part or 0) / float(whole)):.0%}"


def _truncate(value: str | None, max_chars: int = 110) -> str:
    text = (value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1].rstrip()}..."


def _humanize(value: str | None) -> str:
    if value is None or value == "":
        return "None"
    clean = str(value).strip()
    return DISPLAY_LABELS.get(clean, clean.replace("_", " ").title())


def _clean_ui_text(value: object) -> str:
    clean = str(value or "").replace(chr(59), ",").replace(chr(8212), "-")
    clean = clean.replace("##", "").replace("#", "")
    return " ".join(clean.split())


def _normalized_display_name(value: str | None) -> str:
    return "".join(char.lower() if char.isalnum() else " " for char in str(value or "")).strip()


def _is_generic_placeholder(value: str | None) -> bool:
    return " ".join(_normalized_display_name(value).split()) in GENERIC_PLACEHOLDERS


def _tag_pills(tags: list[str] | str | None) -> str:
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
    tags = tags or []
    if not tags:
        return "<span class='wv-pill'>None</span>"
    return "".join(f"<span class='wv-pill'>{html.escape(_clean_ui_text(_humanize(tag)))}</span>" for tag in tags[:4])


def _safe_link(url: str | None, label: str) -> str:
    clean_url = (url or "").strip()
    safe_label = html.escape(label)
    if not clean_url:
        return safe_label
    return f"<a href='{html.escape(clean_url, quote=True)}' target='_blank' rel='noopener noreferrer'>{safe_label}</a>"


def _reset_database(conn) -> None:
    conn.executescript(
        """
        DELETE FROM scores;
        DELETE FROM enrichments;
        DELETE FROM companies;
        DELETE FROM runs;
        """
    )
    conn.commit()


def _rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    for column in ["sector_tags", "deterministic_tags", "top_titles", "top_urls", "top_snippets"]:
        if column in frame.columns:
            frame[column] = frame[column].apply(_json_list)

    frame["sector_tags_text"] = frame.get("sector_tags", pd.Series(dtype=object)).apply(lambda tags: ", ".join(tags or []))
    frame["source_urls"] = frame.get("top_urls", pd.Series(dtype=object)).apply(lambda urls: "\n".join(urls or []))
    frame["primary_source_url"] = frame.get("top_urls", pd.Series(dtype=object)).apply(lambda urls: urls[0] if urls else "")
    frame["total_score"] = frame["total_score"].fillna(0).astype(int)
    frame["is_candidate"] = frame["is_candidate"].fillna(0).astype(int)
    frame["is_startup_likely"] = frame["is_startup_likely"].fillna(0).astype(int)
    frame["wittington_edge"] = frame["wittington_edge"].fillna(0).astype(int)
    frame["company_type"] = frame["company_type"].fillna(frame["deterministic_type"]).fillna("unscored")
    frame["confidence"] = frame["confidence"].fillna("low")
    frame["rationale"] = frame["rationale"].fillna("")
    frame["evidence_summary"] = frame["evidence_summary"].fillna("")
    frame["score_provider"] = frame["score_provider"].fillna("none")
    frame["is_api_verified"] = frame["score_provider"].eq("openai")
    frame["is_refined_prospect"] = (
        frame["is_api_verified"]
        & frame["is_candidate"].eq(1)
        & frame["is_startup_likely"].eq(1)
        & frame["company_type"].fillna("").isin(["startup"])
    )
    frame["company_type_display"] = frame["company_type"].apply(_humanize)
    frame["deterministic_type_display"] = frame["deterministic_type"].apply(_humanize)
    frame["confidence_display"] = frame["confidence"].apply(_humanize)
    frame["score_source_display"] = frame["score_provider"].apply(_humanize)
    frame["source_status"] = frame["is_candidate"].apply(lambda value: "Candidate" if int(value or 0) else "Filtered out")
    placeholder_mask = frame["canonical_name"].apply(_is_generic_placeholder)
    if placeholder_mask.any():
        frame.loc[placeholder_mask, "is_candidate"] = 0
        frame.loc[placeholder_mask, "is_startup_likely"] = 0
        frame.loc[placeholder_mask, "total_score"] = 0
        for component in DEFAULT_WEIGHTS:
            frame.loc[placeholder_mask, component] = 0
        frame.loc[placeholder_mask, "company_type"] = "duplicate_or_noisy_entry"
        frame.loc[placeholder_mask, "company_type_display"] = _humanize("duplicate_or_noisy_entry")
        frame.loc[placeholder_mask, "deterministic_type_display"] = _humanize("duplicate_or_noisy_entry")
        frame.loc[placeholder_mask, "is_refined_prospect"] = False
        frame.loc[placeholder_mask, "source_status"] = "Filtered out"
        frame.loc[placeholder_mask, "rationale"] = "Filtered out, generic placeholder entry rather than a named company."
    frame["rationale_preview"] = frame["rationale"].apply(lambda value: _clean_ui_text(_truncate(value, 140)))
    frame["evidence_preview"] = frame["evidence_summary"].apply(lambda value: _clean_ui_text(_truncate(value, 180)))
    return frame.sort_values(["total_score", "canonical_name"], ascending=[False, True]).reset_index(drop=True)


def _load_frame_and_metrics(conn) -> tuple[pd.DataFrame, dict]:
    return _rows_to_frame(db.dashboard_rows(conn)), db.metrics(conn)


def _apply_weighted_scores(frame: pd.DataFrame, weights: dict[str, int]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    scored = frame.copy()
    total_weight = sum(max(0, int(value)) for value in weights.values())
    if total_weight <= 0:
        scored["weighted_score"] = scored["total_score"].fillna(0).astype(int)
    else:
        weighted = 0
        for column, weight in weights.items():
            component = scored[column].fillna(0).astype(float).clip(lower=0, upper=COMPONENT_MAX[column])
            weighted += (component / COMPONENT_MAX[column]) * int(weight)
        scored["weighted_score"] = ((weighted / total_weight) * 100).round().astype(int)
    scored = scored.sort_values(["weighted_score", "total_score", "canonical_name"], ascending=[False, False, True])
    scored["rank"] = range(1, len(scored) + 1)
    return scored


def _render_metric_cards(cards: list[tuple[str, str, str]]) -> None:
    body = ["<div class='metric-grid'>"]
    for label, value, caption in cards:
        body.append(
            "<div class='metric-card'>"
            f"<div class='metric-label'>{html.escape(label)}</div>"
            f"<div class='metric-value'>{html.escape(value)}</div>"
            f"<div class='metric-caption'>{html.escape(caption)}</div>"
            "</div>"
        )
    body.append("</div>")
    st.markdown("".join(body), unsafe_allow_html=True)


def _render_summary_card(title: str, rows: list[tuple[str, str]]) -> None:
    body = [f"<div class='summary-card'><div class='summary-title'>{html.escape(_clean_ui_text(title))}</div>"]
    for label, value in rows:
        body.append(
            "<div class='summary-line'>"
            f"<span class='summary-label'>{html.escape(_clean_ui_text(label))}</span>"
            f"<span class='summary-value'>{html.escape(_clean_ui_text(value))}</span>"
            "</div>"
        )
    body.append("</div>")
    st.markdown("".join(body), unsafe_allow_html=True)


def _render_cost_hero(
    *,
    provider_spend: ProviderBilledSpend,
    openai_billing: OpenAIBillingSnapshot,
    tavily_billing: TavilyBillingSummary,
    local_openai_estimate: float,
    total_tokens: int,
    openai_calls: int,
    last_api_calls: int,
    last_run_local_openai_estimate: float,
    model_name: str,
) -> None:
    openai_billed = (
        _format_money(openai_billing.cost.total, openai_billing.cost.currency)
        if openai_billing.available
        else "Unavailable"
    )
    items = [
        ("Actual provider-billed spend", _format_billed_total(provider_spend)),
        ("Live OpenAI billed cost", openai_billed),
        ("Local OpenAI token estimate", _format_currency(local_openai_estimate)),
        ("OpenAI tokens tracked", _format_int(total_tokens)),
        (
            "Tavily free credits consumed",
            f"{_format_int(tavily_billing.credits_used)} / {_format_int(tavily_billing.included_monthly_credits)}",
        ),
        ("OpenAI scoring calls", _format_int(openai_calls)),
        ("Last run estimate", f"{_format_int(last_api_calls)} calls / {_format_currency(last_run_local_openai_estimate)}"),
    ]
    body = ["<div class='cost-hero'><div class='cost-hero-title'>Provider billing and local estimates</div><div class='cost-hero-grid'>"]
    for label, value in items:
        value_class = "cost-hero-value compact" if label == "Last run estimate" else "cost-hero-value"
        body.append(
            "<div>"
            f"<div class='cost-hero-label'>{html.escape(_clean_ui_text(label))}</div>"
            f"<div class='{value_class}'>{html.escape(_clean_ui_text(value))}</div>"
            "</div>"
        )
    body.append("</div>")
    body.append(
        "<div class='quiet-note'>Actual provider-billed spend is the reimbursement-relevant figure. "
        f"OpenAI source: {html.escape(_clean_ui_text(openai_billing.source_label))}. "
        f"Scope: {html.escape(_clean_ui_text(openai_billing.scope_label))}. "
        f"Tavily plan: {html.escape(_clean_ui_text(tavily_billing.plan_name))}; "
        f"Tavily billed spend: {html.escape(_format_currency(tavily_billing.actual_billed_usd))}. "
        f"Streamlit Community Cloud hosting: {html.escape(_format_currency(provider_spend.hosting_billed_usd))}. "
        f"Model for local estimate: {html.escape(_clean_ui_text(model_name))}.</div>"
    )
    if (
        openai_billing.available
        and openai_billing.cost.currency == "usd"
        and openai_billing.cost.total == 0
        and local_openai_estimate > 0
    ):
        body.append(
            "<div class='quiet-note'>Provider-billed cost is the source of truth. "
            "The local estimate is a token-rate estimate and may differ because it is not the billing ledger.</div>"
        )
    if not openai_billing.available:
        body.append(
            "<div class='quiet-note'>Live OpenAI billing unavailable. The local OpenAI token estimate is shown for planning only and is not an invoice.</div>"
        )
    body.append("</div>")
    st.markdown("".join(body), unsafe_allow_html=True)


def _workflow_stage(metrics: dict) -> int:
    if int(metrics.get("unique_companies") or 0) <= 0:
        return 1
    if int(metrics.get("openai_scored") or 0) <= 0:
        return 2
    return 3


def _step_row(number: int, label: str, detail: str, state: str) -> str:
    state_class = "active" if state == "Active" else "complete" if state == "Done" else ""
    return (
        f"<div class='step-row {state_class}'>"
        f"<div class='step-number'>{number}</div>"
        "<div>"
        f"<div class='step-label'>{html.escape(label)}</div>"
        f"<div class='step-detail'>{html.escape(detail)}</div>"
        "</div>"
        f"<div class='step-state'>{html.escape(state)}</div>"
        "</div>"
    )


def _render_guided_steps(metrics: dict) -> None:
    stage = _workflow_stage(metrics)
    rows = [
        _step_row(
            1,
            "Prepare source list",
            "Load the Manifest attendee file, remove duplicates, and screen out obvious non-prospects.",
            "Done" if stage > 1 else "Active",
        ),
        _step_row(
            2,
            "Verify prospects",
            "Run search enrichment and OpenAI scoring for the next capped batch.",
            "Done" if stage > 2 else "Active" if stage == 2 else "Locked",
        ),
        _step_row(
            3,
            "Review results",
            "Use the verified prospect list, charts, company detail, and CSV exports.",
            "Active" if stage == 3 else "Locked",
        ),
    ]
    st.markdown(
        "<div class='step-list'>" + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )


def _selected_model_pricing(model_name: str) -> dict[str, float]:
    return OPENAI_MODEL_PRESETS.get(
        model_name,
        {
            "input": st.session_state.get("custom_input_cost", 1.25),
            "output": st.session_state.get("custom_output_cost", 10.00),
        },
    )


def _settings_for_run(settings, model_name: str, input_cost: float, output_cost: float):
    return replace(
        settings,
        openai_model=model_name,
        openai_input_cost_per_1m_tokens=float(input_cost),
        openai_output_cost_per_1m_tokens=float(output_cost),
    )


def _setting(settings, name: str, default):
    return getattr(settings, name, default)


def _local_openai_estimate_usd(metrics: dict, settings) -> float:
    openai_stage_total = 0.0
    for row in metrics.get("cost_by_stage", []):
        if row.get("run_type") == "openai_scoring":
            openai_stage_total += float(row.get("estimated_cost_usd") or 0.0)
    if openai_stage_total > 0:
        return openai_stage_total

    totals = metrics.get("run_totals") or {}
    prompt_tokens = int(totals.get("prompt_tokens") or 0)
    completion_tokens = int(totals.get("completion_tokens") or 0)
    input_cost = (prompt_tokens / 1_000_000) * float(settings.openai_input_cost_per_1m_tokens or 0)
    output_cost = (completion_tokens / 1_000_000) * float(settings.openai_output_cost_per_1m_tokens or 0)
    return input_cost + output_cost


def _last_run_local_openai_estimate_usd(last_run: dict) -> float:
    if last_run.get("run_type") != "openai_scoring":
        return 0.0
    return float(last_run.get("estimated_cost_usd") or 0.0)


def _tavily_billing_from_metrics(metrics: dict, settings) -> TavilyBillingSummary:
    totals = metrics.get("run_totals") or {}
    return calculate_tavily_billing(
        credits_used=int(totals.get("tavily_calls") or 0),
        included_monthly_credits=_setting(settings, "tavily_included_monthly_credits", 1000),
        pay_as_you_go_enabled=_setting(settings, "tavily_pay_as_you_go_enabled", False),
        payg_price_per_credit_usd=_setting(settings, "tavily_payg_price_per_credit_usd", 0.008),
        plan_name=_setting(settings, "tavily_plan_name", "Researcher"),
        shadow_price_per_credit_usd=_setting(settings, "tavily_cost_per_call_usd", 0.001),
    )


def _score_band_frame(frame: pd.DataFrame) -> pd.DataFrame:
    labels = ["0-19", "20-39", "40-59", "60-79", "80-100"]
    if frame.empty:
        return pd.DataFrame({"Score band": labels, "Companies": [0] * len(labels)})

    bands = pd.cut(
        frame["weighted_score"],
        bins=[0, 20, 40, 60, 80, 101],
        labels=labels,
        right=False,
        include_lowest=True,
    )
    counts = bands.value_counts().reindex(labels, fill_value=0)
    return pd.DataFrame({"Score band": labels, "Companies": counts.astype(int).tolist()})


def _sector_frame(frame: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    if not frame.empty:
        for tags in frame["sector_tags"]:
            counter.update(_humanize(tag) for tag in (tags or []))

    rows = counter.most_common(limit)
    if not rows:
        rows = [("None", 0)]
    return pd.DataFrame(rows, columns=["Sector", "Companies"])


def _type_frame(frame: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame({"Type": ["None"], "Companies": [0]})
    counts = frame["company_type_display"].value_counts().head(limit)
    return counts.rename_axis("Type").reset_index(name="Companies")


def _stage_cost_frame(metrics: dict, settings) -> pd.DataFrame:
    rows = []
    for row in metrics.get("cost_by_stage", []):
        if row.get("run_type") != "openai_scoring":
            continue
        cost = float(row.get("estimated_cost_usd") or 0)
        if cost > 0:
            rows.append({"Stage": _humanize(row.get("run_type")), "Estimated USD": cost})
    if not rows:
        rows = [{"Stage": "No local OpenAI estimate", "Estimated USD": 0.0}]
    return pd.DataFrame(rows)


def _resource_cost_frame(metrics: dict, settings) -> pd.DataFrame:
    tavily_billing = _tavily_billing_from_metrics(metrics, settings)
    openai_cost = _local_openai_estimate_usd(metrics, settings)
    rows = [
        {"Resource": "Local OpenAI token estimate", "Estimated USD": openai_cost},
        {"Resource": "Tavily shadow value, not billed", "Estimated USD": tavily_billing.shadow_estimate_usd},
    ]
    return pd.DataFrame(rows)


def _pie_chart(frame: pd.DataFrame, category: str, value: str, height: int = 230) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_arc(innerRadius=48)
        .encode(
            theta=alt.Theta(f"{value}:Q"),
            color=alt.Color(f"{category}:N", legend=alt.Legend(title=None)),
            tooltip=[category, alt.Tooltip(f"{value}:Q", format=",.4f")],
        )
        .properties(height=height)
    )


def _horizontal_bar_chart(
    frame: pd.DataFrame,
    category: str,
    value: str,
    *,
    height: int = 230,
    sort: str | list[str] = "-x",
) -> alt.Chart:
    frame = frame.copy()
    row_count = max(1, len(frame))
    chart_height = max(height, row_count * 46)
    max_value = float(frame[value].max() or 0) if value in frame else 0
    domain_max = max(1.0, max_value * 1.18)
    base = alt.Chart(frame).encode(
        y=alt.Y(
            f"{category}:N",
            sort=sort,
            title=None,
            axis=alt.Axis(labelAngle=0, labelLimit=320, labelPadding=12),
        ),
        x=alt.X(
            f"{value}:Q",
            title=None,
            scale=alt.Scale(domain=[0, domain_max]),
            axis=alt.Axis(labelAngle=0, grid=True, format="~s", tickCount=5),
        ),
        tooltip=[
            alt.Tooltip(f"{category}:N", title=category),
            alt.Tooltip(f"{value}:Q", title=value, format=",.4f"),
        ],
    )
    bars = base.mark_bar(color="#1474c9", cornerRadiusEnd=4, size=26)
    labels = base.mark_text(
        align="left",
        baseline="middle",
        dx=8,
        color="#313647",
        fontSize=13,
    ).encode(text=alt.Text(f"{value}:Q", format=",.0f"))
    return (
        (bars + labels)
        .properties(height=chart_height)
        .configure_axis(labelFontSize=13, titleFontSize=13)
        .configure_view(stroke=None)
    )


def _table_html(frame: pd.DataFrame, columns: list[tuple[str, str, str]], empty_message: str) -> str:
    if frame.empty:
        return f"<div class='empty-state'>{html.escape(empty_message)}</div>"

    widths = {
        "rank": "6%",
        "score": "14%",
        "company": "22%",
        "evidence": "38%",
        "tags": "20%",
    }
    table_kind = "source-table" if columns and columns[0][0] == "raw_name" else "prospect-table"
    body = [f"<table class='wv-table {table_kind}'><thead><tr>"]
    for key, label, kind in columns:
        width = widths.get(kind) or widths.get(key) or "auto"
        body.append(f"<th style='width:{width}'>{html.escape(label)}</th>")
    body.append("</tr></thead><tbody>")

    for _, row in frame.iterrows():
        body.append("<tr>")
        for key, _label, kind in columns:
            value = row.get(key)
            if kind == "company":
                cell = html.escape(_clean_ui_text(value))
            elif kind == "score":
                score = int(value or 0)
                cell = (
                    "<div class='score-cell'>"
                    "<div class='score-track'>"
                    f"<span class='score-fill' style='width:{max(0, min(100, score))}%'></span>"
                    "</div>"
                    f"<span class='score-number'>{score}</span>"
                    "</div>"
                )
            elif kind == "tags":
                cell = _tag_pills(value)
            elif kind == "label":
                cell = html.escape(_clean_ui_text(_humanize(str(value or ""))))
            elif kind == "link":
                cell = _safe_link(str(value or ""), "Open")
            elif kind == "number":
                cell = _format_int(value)
            else:
                cell = html.escape(_clean_ui_text(value))
            body.append(f"<td>{cell}</td>")
        body.append("</tr>")
    body.append("</tbody></table>")
    return "".join(body)


def _render_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]], empty_message: str) -> None:
    st.markdown(_table_html(frame, columns, empty_message), unsafe_allow_html=True)


def _prospect_cards_html(frame: pd.DataFrame, empty_message: str) -> str:
    if frame.empty:
        return f"<div class='empty-state'>{html.escape(empty_message)}</div>"
    body = ["<div class='prospect-list'>"]
    for _, row in frame.iterrows():
        score = int(row.get("weighted_score") or 0)
        rank = int(row.get("rank") or 0)
        body.append(
            "<div class='prospect-card'>"
            "<div class='prospect-main'>"
            f"<div class='prospect-rank'>Rank {rank}</div>"
            f"<div class='prospect-name'>{html.escape(_clean_ui_text(row.get('canonical_name')))}</div>"
            "</div>"
            "<div class='prospect-score'>"
            "<div class='prospect-score-label'>Fit score</div>"
            "<div class='score-cell'>"
            "<div class='score-track'>"
            f"<span class='score-fill' style='width:{max(0, min(100, score))}%'></span>"
            "</div>"
            f"<span class='score-number'>{score}</span>"
            "</div>"
            f"<div class='prospect-tags'>{_tag_pills(row.get('sector_tags'))}</div>"
            "</div>"
            f"<div class='prospect-evidence'>{html.escape(_clean_ui_text(row.get('evidence_summary') or 'No external evidence yet.'))}</div>"
            "</div>"
        )
    body.append("</div>")
    return "".join(body)


def _render_prospect_cards(frame: pd.DataFrame, empty_message: str) -> None:
    st.markdown(_prospect_cards_html(frame, empty_message), unsafe_allow_html=True)


def _source_cards_html(frame: pd.DataFrame, empty_message: str) -> str:
    if frame.empty:
        return f"<div class='empty-state'>{html.escape(empty_message)}</div>"
    body = ["<div class='source-list'>"]
    for _, row in frame.iterrows():
        score = int(row.get("weighted_score") or 0)
        body.append(
            "<div class='source-card'>"
            "<div>"
            f"<div class='source-name'>{html.escape(_clean_ui_text(row.get('canonical_name')))}</div>"
            f"<div class='source-raw'>Raw entry: {html.escape(_clean_ui_text(row.get('raw_name')))}</div>"
            f"<div class='source-meta'>{_format_int(row.get('duplicate_count'))} source row{'s' if int(row.get('duplicate_count') or 0) != 1 else ''}</div>"
            "</div>"
            "<div class='source-score'>"
            "<div class='prospect-score-label'>Screen score</div>"
            "<div class='score-cell'>"
            "<div class='score-track'>"
            f"<span class='score-fill' style='width:{max(0, min(100, score))}%'></span>"
            "</div>"
            f"<span class='score-number'>{score}</span>"
            "</div>"
            f"<div class='prospect-tags'>{_tag_pills(row.get('sector_tags'))}</div>"
            "</div>"
            "</div>"
        )
    body.append("</div>")
    return "".join(body)


def _render_source_cards(frame: pd.DataFrame, empty_message: str) -> None:
    st.markdown(_source_cards_html(frame, empty_message), unsafe_allow_html=True)


def _evidence_cards_html(urls: list[str], titles: list[str], snippets: list[str]) -> str:
    body = ["<div class='evidence-list'>"]
    for index, url in enumerate(urls):
        title = _clean_ui_text(titles[index] if index < len(titles) else url)
        safe_url = html.escape(str(url or ""), quote=True)
        body.append(
            "<div class='evidence-card'>"
            f"<a class='evidence-link' href='{safe_url}' target='_blank' rel='noopener noreferrer'>{html.escape(title)}</a>"
            "</div>"
        )
    body.append("</div>")
    return "".join(body)


def _render_workflow(metrics: dict) -> None:
    has_source = int(metrics.get("unique_companies") or 0) > 0
    has_verified = int(metrics.get("openai_scored") or 0) > 0
    source_class = "complete" if has_source else "active"
    prospect_class = "complete" if has_verified else ("active" if has_source else "")
    st.markdown(
        "<div class='workflow-step {source_class}'>"
        "<div class='workflow-title'>1. Source data</div>"
        "<div class='workflow-caption'>{source_caption}</div>"
        "</div>"
        "<div class='workflow-step {prospect_class}'>"
        "<div class='workflow-title'>2. Refined prospects</div>"
        "<div class='workflow-caption'>{prospect_caption}</div>"
        "</div>".format(
            source_class=source_class,
            prospect_class=prospect_class,
            source_caption=(
                f"{_format_int(metrics.get('unique_companies'))} unique names loaded"
                if has_source
                else "Load and classify the attendee file"
            ),
            prospect_caption=(
                f"{_format_int(metrics.get('openai_scored'))} companies verified with external evidence"
                if has_verified
                else "Enrich and score a capped batch"
            ),
        ),
        unsafe_allow_html=True,
    )


def _run_and_store(label: str, func, *, level: str = "success"):
    with st.spinner(label):
        result = func()
    st.session_state["last_action"] = {
        "message": result.message,
        "level": level,
    }
    return result


settings, conn = _connect()
try:
    display_database_path = settings.database_path.relative_to(PROJECT_ROOT)
except ValueError:
    display_database_path = settings.database_path

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
frame, metrics = _load_frame_and_metrics(conn)

weights: dict[str, int] = {}
min_score = 0
show_verified_only = True
selected_types: list[str] = []
selected_sectors: list[str] = []
selected_confidence: list[str] = []

with st.sidebar:
    st.markdown("### Display")
    st.markdown(
        "<div class='sidebar-badge'><strong>Tavily</strong>: "
        f"{'configured' if settings.tavily_api_key else 'missing key'}</div>"
        "<div class='sidebar-badge'><strong>OpenAI</strong>: "
        f"{'configured' if settings.openai_api_key else 'missing key'}</div>"
        "<div class='sidebar-badge'><strong>OpenAI billing</strong>: "
        f"{'admin key configured' if _setting(settings, 'openai_admin_key', None) else 'admin key missing'}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"SQLite: `{display_database_path}`")

    source_rows_shown = st.number_input(
        "Source cards shown",
        min_value=10,
        max_value=500,
        value=int(st.session_state.get("source_rows_shown", 75)),
        step=10,
        key="source_rows_shown",
        help="How many source cards to show in the Source list tab. This does not change API calls.",
    )

    with st.expander("Review scoring weights", expanded=False):
        for key, default in DEFAULT_WEIGHTS.items():
            weights[key] = st.slider(
                _humanize(key),
                min_value=0,
                max_value=50,
                value=int(st.session_state.get(f"weight_{key}", default)),
                step=5,
                key=f"weight_{key}",
            )

    with st.expander("Filters", expanded=False):
        if frame.empty:
            st.caption("Load source data to enable filters.")
        else:
            min_score = st.slider("Minimum fit score", min_value=0, max_value=100, value=0)
            show_verified_only = st.checkbox("Verified prospects only", value=True)
            type_options = sorted([value for value in frame["company_type"].dropna().unique().tolist() if value])
            confidence_options = sorted([value for value in frame["confidence"].dropna().unique().tolist() if value])
            all_sectors = sorted({tag for tags in frame["sector_tags"] for tag in (tags or [])})
            selected_types = st.multiselect("Company type", type_options, default=[], format_func=_humanize)
            selected_sectors = st.multiselect("Sector", all_sectors, default=[], format_func=_humanize)
            selected_confidence = st.multiselect("Confidence", confidence_options, default=[], format_func=_humanize)

    if st.button(
        "Reset app data",
        use_container_width=True,
        help="Clear source rows, cached enrichments, scores, and run history so the workflow starts fresh.",
    ):
        _reset_database(conn)
        st.session_state["last_action"] = {
            "message": "App data reset. Start again with step 1.",
            "level": "success",
        }
        st.rerun()

st.markdown(
    """
    <div class="wv-header">
      <h1 class="wv-title">Manifest Prospecting Tool</h1>
      <p class="wv-subtitle">Wittington Ventures workflow for turning the messy Manifest attendee file into a smaller set of externally verified venture prospects.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

workflow_stage = _workflow_stage(metrics)

st.markdown("<div class='guided-kicker'>Guided workflow</div>", unsafe_allow_html=True)
if workflow_stage == 1:
    st.markdown("<div class='guided-title'>Step 1: Prepare the source list</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='guided-copy'>Start here. This loads the Manifest attendee file, deduplicates names, and creates the first screen without using paid APIs.</div>",
        unsafe_allow_html=True,
    )
elif workflow_stage == 2:
    st.markdown("<div class='guided-title'>Step 2: Verify prospects with APIs</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='guided-copy'>Next, choose the batch size and model, then run search enrichment and OpenAI scoring for the next high-priority companies.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown("<div class='guided-title'>Step 3: Review verified prospects</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='guided-copy'>The verified prospect list is ready. Review the results below, or run another API batch if you want more companies scored.</div>",
        unsafe_allow_html=True,
    )
_render_guided_steps(metrics)

selected_model = st.session_state.get("selected_openai_model", settings.openai_model)
if selected_model not in OPENAI_MODEL_PRESETS:
    selected_model = settings.openai_model if settings.openai_model in OPENAI_MODEL_PRESETS else "gpt-4o-mini"
model_pricing = _selected_model_pricing(selected_model)
prospect_cap = int(st.session_state.get("prospect_cap", min(settings.max_score, 100)))
runtime_settings = _settings_for_run(settings, selected_model, model_pricing["input"], model_pricing["output"])

if workflow_stage >= 2:
    settings_cols = st.columns((1, 1, 1))
    prospect_cap = settings_cols[0].number_input(
        "Companies to verify now",
        min_value=1,
        max_value=500,
        value=int(st.session_state.get("prospect_cap", min(settings.max_score, 100))),
        step=5,
        key="prospect_cap",
        help="Maximum number of high-priority companies to enrich and score in this API run.",
    )
    selected_model = settings_cols[1].selectbox(
        "OpenAI scoring model",
        list(OPENAI_MODEL_PRESETS.keys()),
        index=list(OPENAI_MODEL_PRESETS.keys()).index(selected_model),
        key="selected_openai_model",
        help="Model name sent to OpenAI for the scoring step.",
    )
    model_pricing = _selected_model_pricing(selected_model)
    runtime_settings = _settings_for_run(settings, selected_model, model_pricing["input"], model_pricing["output"])
    settings_cols[2].markdown(
        f"**Local OpenAI estimate**  \n"
        f"Input: `${model_pricing['input']:g}` / 1M tokens  \n"
        f"Output: `${model_pricing['output']:g}` / 1M tokens  \n"
        f"Tavily: `{_setting(settings, 'tavily_plan_name', 'Researcher')}` plan credits"
    )

if workflow_stage == 1:
    primary_label = "1. Load and screen source data"
elif workflow_stage == 2:
    primary_label = "2. Verify prospects with APIs"
else:
    primary_label = "Verify another batch"

if st.button(primary_label, type="primary", use_container_width=True):
    if workflow_stage == 1:
        run_step = "source"
    else:
        run_step = "verify"
else:
    run_step = None

if run_step == "source":
    load_result = _run_and_store("Loading attendee names...", lambda: load_attendees(conn, settings))
    classify_result = _run_and_store("Classifying source data...", lambda: run_deterministic_classification(conn))
    st.session_state["last_action"] = {
        "message": f"{load_result.message} {classify_result.message}",
        "level": "success",
    }
    frame, metrics = _load_frame_and_metrics(conn)
    st.rerun()

if run_step == "verify":
    cap = int(prospect_cap)
    progress = st.progress(0, text=f"Refreshing source screening before verifying up to {cap:,} companies...")
    preview = st.empty()
    classify_result = run_deterministic_classification(conn)

    def enrichment_progress(index, total, result, counts):
        if total:
            progress.progress(
                min(0.5, (index / total) * 0.5),
                text=f"Search enrichment {index:,}/{total:,}: {result.get('company_name', '')}",
            )

    def scoring_progress(index, total, result, counts):
        if total:
            progress.progress(
                min(1.0, 0.5 + (index / total) * 0.5),
                text=f"Prospect scoring {index:,}/{total:,}: {result.get('company_name', '')}",
            )
        latest_frame, _latest_metrics = _load_frame_and_metrics(conn)
        latest_weighted = _apply_weighted_scores(latest_frame, weights)
        latest_prospects = latest_weighted[latest_weighted["is_refined_prospect"]].head(8)
        preview.markdown(
            _prospect_cards_html(
                latest_prospects,
                "Verified prospects will appear here as scoring completes.",
            ),
            unsafe_allow_html=True,
        )

    enrich_result = enrich_candidates(
        conn,
        runtime_settings,
        cap,
        False,
        progress_callback=enrichment_progress,
    )
    score_result = score_enriched_candidates(
        conn,
        runtime_settings,
        cap,
        False,
        progress_callback=scoring_progress,
    )
    progress.progress(1.0, text="Verification run finished.")
    level = "warning" if enrich_result.counts.get("errors") or score_result.counts.get("errors") else "success"
    run_openai_estimate = float(score_result.counts.get("estimated_cost_usd") or 0)
    run_tavily_credits = int(enrich_result.counts.get("tavily_credits_used") or enrich_result.counts.get("tavily_calls") or 0)
    run_tavily_billed = float(enrich_result.counts.get("tavily_actual_billed_usd") or 0)
    st.session_state["last_action"] = {
        "message": (
            f"Verified {score_result.counts.get('scored', 0):,} companies from a {cap:,}-company batch. "
            f"Local OpenAI token estimate: {_format_currency(run_openai_estimate)}. "
            f"Tavily credits consumed: {run_tavily_credits:,}; Tavily billed spend: {_format_currency(run_tavily_billed)}."
        ),
        "level": level,
    }
    frame, metrics = _load_frame_and_metrics(conn)
    st.rerun()

last_action = st.session_state.get("last_action")
if last_action:
    level = html.escape(last_action.get("level", "info"))
    message = html.escape(_clean_ui_text(last_action.get("message", "")))
    st.markdown(f"<div class='status-strip {level}'>{message}</div>", unsafe_allow_html=True)

weighted_frame = _apply_weighted_scores(frame, weights)
filtered = weighted_frame[weighted_frame["weighted_score"] >= min_score].copy() if not weighted_frame.empty else weighted_frame.copy()
if show_verified_only and not filtered.empty:
    filtered = filtered[filtered["is_refined_prospect"]]
if selected_types:
    filtered = filtered[filtered["company_type"].isin(selected_types)]
if selected_confidence:
    filtered = filtered[filtered["confidence"].isin(selected_confidence)]
if selected_sectors:
    selected = set(selected_sectors)
    filtered = filtered[filtered["sector_tags"].apply(lambda tags: bool(selected.intersection(tags or [])))]

prospects = weighted_frame[weighted_frame["is_refined_prospect"]].copy() if not weighted_frame.empty else weighted_frame.copy()
visible_prospects = filtered if show_verified_only else prospects
source_rows = weighted_frame.head(int(source_rows_shown)).copy()
candidate_count = int(weighted_frame["is_candidate"].sum()) if not weighted_frame.empty else int(metrics.get("candidates") or 0)
run_totals = metrics.get("run_totals") or {}
last_run = metrics.get("last_run") or {}
last_api_calls = int(last_run.get("tavily_calls") or 0) + int(last_run.get("openai_calls") or 0)
tavily_billing = _tavily_billing_from_metrics(metrics, settings)
openai_billing = fetch_openai_billing_snapshot(
    admin_key=_setting(settings, "openai_admin_key", None),
    project_id=_setting(settings, "openai_billing_project_id", None),
    lookback_days=_setting(settings, "openai_billing_lookback_days", 31),
    cache_ttl_seconds=_setting(settings, "openai_billing_cache_ttl_seconds", 300),
)
provider_spend = calculate_provider_billed_spend(
    tavily_billing=tavily_billing,
    openai_billing=openai_billing,
    hosting_billed_usd=0.0,
)
local_openai_estimate = _local_openai_estimate_usd(metrics, runtime_settings)
last_run_local_openai_estimate = _last_run_local_openai_estimate_usd(last_run)

_render_cost_hero(
    provider_spend=provider_spend,
    openai_billing=openai_billing,
    tavily_billing=tavily_billing,
    local_openai_estimate=local_openai_estimate,
    total_tokens=int(run_totals.get("total_tokens") or 0),
    openai_calls=int(run_totals.get("openai_calls") or 0),
    last_api_calls=last_api_calls,
    last_run_local_openai_estimate=last_run_local_openai_estimate,
    model_name=str(runtime_settings.openai_model),
)

summary_cols = st.columns((1, 1))
with summary_cols[0]:
    _render_summary_card(
        "Source and prospect status",
        [
            ("Source rows", f"{_format_int(metrics['raw_companies'])} raw / {_format_int(metrics['unique_companies'])} unique"),
            ("Candidates after rule screen", f"{_format_int(candidate_count)} ({_pct(candidate_count, metrics['unique_companies'])})"),
            ("API-scored companies", _format_int(metrics["openai_scored"])),
            ("Verified prospects shown", _format_int(len(prospects))),
        ],
    )
with summary_cols[1]:
    _render_summary_card(
        "Billing configuration",
        [
            ("OpenAI billing source", openai_billing.source_label),
            ("OpenAI scope", openai_billing.scope_label),
            ("Tavily plan", str(tavily_billing.plan_name)),
            ("Tavily pay-as-you-go", "enabled" if tavily_billing.pay_as_you_go_enabled else "disabled"),
            ("Tavily free credits remaining", _format_int(tavily_billing.free_credits_remaining)),
            ("Streamlit Community Cloud hosting", _format_currency(provider_spend.hosting_billed_usd)),
        ],
    )

if frame.empty:
    st.warning("No companies loaded yet. Use Prepare source list to load and classify the Manifest attendee file.")
else:
    overview_tab, source_tab, prospects_tab, detail_tab = st.tabs(
        ["Overview", "Source list", "Verified prospects", "Company detail"]
    )

    with overview_tab:
        funnel_df = pd.DataFrame(
            {
                "Stage": ["Unique names", "Screened candidates", "Evidence enriched", "API-scored"],
                "Companies": [
                    metrics["unique_companies"],
                    candidate_count,
                    metrics["enriched"],
                    metrics["openai_scored"],
                ],
            }
        )
        st.markdown("<div class='section-label'>Source to prospects</div>", unsafe_allow_html=True)
        st.altair_chart(
            _horizontal_bar_chart(funnel_df, "Stage", "Companies", height=320, sort=None),
            use_container_width=True,
        )
        st.divider()

        st.markdown("<div class='section-label'>Fit score distribution</div>", unsafe_allow_html=True)
        st.altair_chart(
            _horizontal_bar_chart(_score_band_frame(weighted_frame), "Score band", "Companies", height=340, sort=None),
            use_container_width=True,
        )
        st.divider()

        st.markdown("<div class='section-label'>Sector mix</div>", unsafe_allow_html=True)
        st.altair_chart(
            _horizontal_bar_chart(_sector_frame(weighted_frame), "Sector", "Companies", height=380),
            use_container_width=True,
        )
        st.divider()

        st.markdown("<div class='section-label'>Company types</div>", unsafe_allow_html=True)
        st.altair_chart(
            _horizontal_bar_chart(_type_frame(weighted_frame), "Type", "Companies", height=320),
            use_container_width=True,
        )
        st.divider()

        st.markdown("<div class='section-label'>Internal estimates and shadow values</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='quiet-note'>These values support planning and projections. They are not provider-billed reimbursement totals.</div>",
            unsafe_allow_html=True,
        )
        st.altair_chart(
            _horizontal_bar_chart(_resource_cost_frame(metrics, settings), "Resource", "Estimated USD", height=220),
            use_container_width=True,
        )
        st.divider()

        cost_stage_df = _stage_cost_frame(metrics, settings)
        st.markdown("<div class='section-label'>Local OpenAI estimate by stage</div>", unsafe_allow_html=True)
        st.altair_chart(
            _horizontal_bar_chart(cost_stage_df, "Stage", "Estimated USD", height=220, sort=None),
            use_container_width=True,
        )
        st.divider()

        st.markdown("<div class='section-label'>Top verified prospects</div>", unsafe_allow_html=True)
        _render_prospect_cards(
            prospects.head(10),
            "No externally verified prospects yet. Generate a capped batch to populate this list.",
        )

    with source_tab:
        st.markdown("<div class='section-label'>Source list</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='quiet-note'>Showing {_format_int(len(source_rows))} de-duplicated source rows from {_format_int(len(weighted_frame))} unique company names.</div>",
            unsafe_allow_html=True,
        )
        _render_source_cards(
            source_rows,
            "No source rows loaded.",
        )
        source_export = weighted_frame[
            [
                "raw_name",
                "canonical_name",
                "source_status",
                "deterministic_type_display",
                "sector_tags_text",
                "weighted_score",
                "duplicate_count",
                "deterministic_exclusion_reason",
            ]
        ].rename(
            columns={
                "raw_name": "Raw attendee entry",
                "canonical_name": "Cleaned company",
                "source_status": "Status",
                "deterministic_type_display": "Rule screen",
                "sector_tags_text": "Sectors",
                "weighted_score": "Screen score",
                "duplicate_count": "Rows",
                "deterministic_exclusion_reason": "Filter reason",
            }
        )
        st.download_button(
            "Download source CSV",
            data=source_export.to_csv(index=False),
            file_name="manifest_source_list.csv",
            mime="text/csv",
        )

    with prospects_tab:
        st.markdown("<div class='section-label'>Verified prospects</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='quiet-note'>Showing {_format_int(len(visible_prospects))} refined prospects that have external evidence and API scoring.</div>",
            unsafe_allow_html=True,
        )
        _render_prospect_cards(
            visible_prospects,
            "No verified prospects match the current filters.",
        )
        prospect_export = prospects[
            [
                "rank",
                "canonical_name",
                "weighted_score",
                "total_score",
                "company_type_display",
                "confidence_display",
                "sector_tags_text",
                "website",
                "primary_source_url",
                "rationale",
                "evidence_summary",
                "source_urls",
            ]
        ].rename(
            columns={
                "rank": "Rank",
                "canonical_name": "Company",
                "weighted_score": "Fit score",
                "total_score": "Stored score",
                "company_type_display": "Type",
                "confidence_display": "Confidence",
                "sector_tags_text": "Sectors",
                "website": "Website",
                "primary_source_url": "Primary source",
                "rationale": "Rationale",
                "evidence_summary": "Evidence summary",
                "source_urls": "Source URLs",
            }
        )
        st.download_button(
            "Download prospects CSV",
            data=prospect_export.to_csv(index=False),
            file_name="manifest_verified_prospects.csv",
            mime="text/csv",
        )

    with detail_tab:
        st.markdown("<div class='section-label'>Company detail</div>", unsafe_allow_html=True)
        detail_source = filtered if not filtered.empty else weighted_frame
        if detail_source.empty:
            st.warning("No companies match the current filters.")
        else:
            selected_company = st.selectbox("Company", detail_source["canonical_name"].tolist())
            selected_row = detail_source[detail_source["canonical_name"] == selected_company].iloc[0]

            detail_cols = st.columns(5)
            detail_cols[0].metric("Fit score", f"{int(selected_row['weighted_score'])}/100")
            detail_cols[1].metric("Stored score", f"{int(selected_row['total_score'])}/100")
            detail_cols[2].metric("WV edge", f"{int(selected_row['wittington_edge'])}/20")
            detail_cols[3].metric("Type", str(selected_row.get("company_type_display") or "Needs evidence"))
            detail_cols[4].metric("Confidence", str(selected_row.get("confidence_display") or "Low"))

            st.markdown(
                "<div class='detail-box'>"
                f"<div class='detail-title'>{html.escape(_clean_ui_text(selected_row['canonical_name']))}</div>"
                f"<div class='detail-text'><strong>Rationale:</strong> {html.escape(_clean_ui_text(selected_row.get('rationale') or 'No rationale yet.'))}</div>"
                f"<div class='detail-text'><strong>Evidence:</strong> {html.escape(_clean_ui_text(selected_row.get('evidence_summary') or 'No external evidence yet.'))}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

            component_df = pd.DataFrame(
                [
                    {"Criterion": _humanize(key), "Score": int(selected_row.get(key) or 0), "Max": COMPONENT_MAX[key], "Weight": weights[key]}
                    for key in DEFAULT_WEIGHTS
                ]
            )
            st.markdown("<div class='section-label'>Score components</div>", unsafe_allow_html=True)
            st.altair_chart(
                _horizontal_bar_chart(component_df, "Criterion", "Score", height=260, sort=None),
                use_container_width=True,
            )

            urls = selected_row.get("top_urls") or []
            titles = selected_row.get("top_titles") or []
            snippets = selected_row.get("top_snippets") or []
            if urls:
                st.markdown("<div class='section-label'>Retrieved evidence</div>", unsafe_allow_html=True)
                st.markdown(_evidence_cards_html(urls, titles, snippets), unsafe_allow_html=True)
