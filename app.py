from __future__ import annotations

import html
import json
from collections import Counter
from urllib.parse import quote_plus

import altair as alt
import pandas as pd
import streamlit as st

from src import db
from src.config import PROJECT_ROOT, get_settings
from src.pipeline import (
    enrich_candidates,
    load_attendees,
    run_deterministic_classification,
    score_enriched_candidates,
    verify_cache_reuse,
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
    "likely_startup_or_tech": "Tech candidate",
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


CUSTOM_CSS = """
<style>
  .block-container {
    max-width: 1240px;
    padding-top: 2rem;
    padding-bottom: 2.5rem;
  }
  h1, h2, h3 {
    letter-spacing: 0 !important;
  }
  .wv-header {
    border-bottom: 1px solid #eceff3;
    margin-bottom: 1rem;
    padding-bottom: 1rem;
  }
  .wv-eyebrow {
    color: #5d6675;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin: 0 0 0.25rem 0;
    text-transform: uppercase;
  }
  .wv-title {
    color: #202332;
    font-size: clamp(2.1rem, 5vw, 3.25rem);
    font-weight: 780;
    line-height: 1.02;
    margin: 0;
  }
  .wv-subtitle {
    color: #697386;
    font-size: 1rem;
    margin: 0.6rem 0 0 0;
    max-width: 800px;
  }
  .metric-grid {
    display: grid;
    gap: 0.75rem;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    margin: 1rem 0 1.2rem 0;
  }
  .metric-card {
    background: #ffffff;
    border: 1px solid #e5e9ef;
    border-radius: 8px;
    padding: 0.85rem 0.9rem;
  }
  .metric-label {
    color: #697386;
    font-size: 0.76rem;
    font-weight: 650;
    line-height: 1.2;
    margin-bottom: 0.35rem;
    text-transform: uppercase;
  }
  .metric-value {
    color: #202332;
    font-size: 1.45rem;
    font-weight: 760;
    line-height: 1.1;
  }
  .metric-caption {
    color: #8b94a5;
    font-size: 0.78rem;
    line-height: 1.25;
    margin-top: 0.3rem;
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
  .workflow-step {
    background: #ffffff;
    border: 1px solid #dfe4ec;
    border-radius: 8px;
    color: #313647;
    margin-bottom: 0.55rem;
    padding: 0.7rem 0.75rem;
  }
  .workflow-step.active {
    background: #fff2f2;
    border-color: #ff4b4b;
  }
  .workflow-step.complete {
    background: #f0fbf4;
    border-color: #43a15f;
  }
  .workflow-title {
    font-size: 0.92rem;
    font-weight: 740;
    margin-bottom: 0.18rem;
  }
  .workflow-caption {
    color: #697386;
    font-size: 0.8rem;
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
    background: #ff4b4b;
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
  @media (max-width: 1100px) {
    .metric-grid {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
  }
  @media (max-width: 760px) {
    .metric-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
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
    .prospect-table th:nth-child(4),
    .prospect-table td:nth-child(4),
    .prospect-table th:nth-child(5),
    .prospect-table td:nth-child(5),
    .prospect-table th:nth-child(7),
    .prospect-table td:nth-child(7) {
      display: none;
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
    return "".join(f"<span class='wv-pill'>{html.escape(_humanize(tag))}</span>" for tag in tags[:4])


def _safe_link(url: str | None, label: str) -> str:
    clean_url = (url or "").strip()
    safe_label = html.escape(label)
    if not clean_url:
        return safe_label
    return f"<a href='{html.escape(clean_url, quote=True)}' target='_blank' rel='noopener noreferrer'>{safe_label}</a>"


def _company_url(row: pd.Series) -> str:
    for column in ("website", "primary_source_url"):
        value = str(row.get(column) or "").strip()
        if value.startswith(("http://", "https://")):
            return value
    return f"https://www.google.com/search?q={quote_plus(str(row.get('canonical_name') or ''))}"


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
        frame.loc[placeholder_mask, "rationale"] = "Filtered out: generic placeholder entry rather than a named company."
    frame["rationale_preview"] = frame["rationale"].apply(lambda value: _truncate(value, 95))
    frame["evidence_preview"] = frame["evidence_summary"].apply(lambda value: _truncate(value, 95))
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
        cost = float(row.get("estimated_cost_usd") or 0)
        if cost > 0:
            rows.append({"Stage": _humanize(row.get("run_type")), "Estimated USD": cost})
    if not rows:
        rows = [{"Stage": "No tracked API spend", "Estimated USD": 0.0}]
    return pd.DataFrame(rows)


def _resource_cost_frame(metrics: dict, settings) -> pd.DataFrame:
    totals = metrics.get("run_totals") or {}
    tavily_cost = int(totals.get("tavily_calls") or 0) * settings.tavily_cost_per_call_usd
    tracked_cost = float(totals.get("estimated_cost_usd") or 0)
    openai_cost = max(0.0, tracked_cost - tavily_cost)
    rows = [
        {"Resource": "Search API", "Estimated USD": tavily_cost},
        {"Resource": "OpenAI model", "Estimated USD": openai_cost},
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
    chart_height = max(height, row_count * 36)
    max_value = float(frame[value].max() or 0) if value in frame else 0
    domain_max = max(1.0, max_value * 1.18)
    base = alt.Chart(frame).encode(
        y=alt.Y(
            f"{category}:N",
            sort=sort,
            title=None,
            axis=alt.Axis(labelAngle=0, labelLimit=220, labelPadding=8),
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
    bars = base.mark_bar(color="#1474c9", cornerRadiusEnd=3)
    labels = base.mark_text(
        align="left",
        baseline="middle",
        dx=5,
        color="#313647",
        fontSize=12,
    ).encode(text=alt.Text(f"{value}:Q", format=",.0f"))
    return (bars + labels).properties(height=chart_height).configure_axis(labelFontSize=12, titleFontSize=12)


def _table_html(frame: pd.DataFrame, columns: list[tuple[str, str, str]], empty_message: str) -> str:
    if frame.empty:
        return f"<div class='empty-state'>{html.escape(empty_message)}</div>"

    widths = {
        "rank": "7%",
        "score": "18%",
        "company": "22%",
        "evidence": "28%",
        "tags": "19%",
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
                cell = _safe_link(_company_url(row), str(value or ""))
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
                cell = html.escape(_humanize(str(value or "")))
            elif kind == "link":
                cell = _safe_link(str(value or ""), "Open")
            elif kind == "number":
                cell = _format_int(value)
            else:
                cell = html.escape(str(value or ""))
            body.append(f"<td>{cell}</td>")
        body.append("</tr>")
    body.append("</tbody></table>")
    return "".join(body)


def _render_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]], empty_message: str) -> None:
    st.markdown(_table_html(frame, columns, empty_message), unsafe_allow_html=True)


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
    st.markdown("### Workflow")
    _render_workflow(metrics)
    st.markdown(
        "<div class='sidebar-badge'><strong>Tavily</strong>: "
        f"{'configured' if settings.tavily_api_key else 'missing key'}</div>"
        "<div class='sidebar-badge'><strong>OpenAI</strong>: "
        f"{'configured' if settings.openai_api_key else 'missing key'}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"SQLite: `{display_database_path}`")

    prospect_cap = st.number_input(
        "Prospects to verify",
        min_value=1,
        max_value=500,
        value=int(st.session_state.get("prospect_cap", min(settings.max_score, 100))),
        step=5,
        key="prospect_cap",
    )
    source_rows_shown = st.number_input(
        "Source rows shown",
        min_value=10,
        max_value=500,
        value=int(st.session_state.get("source_rows_shown", 75)),
        step=10,
        key="source_rows_shown",
    )

    with st.expander("Scoring weights", expanded=True):
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

    with st.expander("Cache and pricing", expanded=False):
        st.caption(
            f"OpenAI {settings.openai_model}: "
            f"${settings.openai_input_cost_per_1m_tokens:g}/1M input tokens, "
            f"${settings.openai_output_cost_per_1m_tokens:g}/1M output tokens."
        )
        st.caption(f"Tavily estimate: ${settings.tavily_cost_per_call_usd:g} per search call.")
        force_refresh = st.checkbox("Force refresh cached API records", value=False, key="force_refresh")
        if st.button("Check cache reuse", use_container_width=True):
            with st.spinner("Checking SQLite cache reuse..."):
                result = verify_cache_reuse(conn, settings)
            level = "success" if result.counts.get("verified") else "warning"
            st.session_state["last_action"] = {"message": result.message, "level": level}
        if st.button("Initialize database", use_container_width=True):
            db.init_db(conn)
            st.session_state["last_action"] = {"message": "Database initialized.", "level": "success"}

st.markdown(
    """
    <div class="wv-header">
      <p class="wv-eyebrow">Wittington Ventures</p>
      <h1 class="wv-title">Manifest Prospecting Tool</h1>
      <p class="wv-subtitle">Turn the messy Manifest attendee file into a smaller set of externally verified venture prospects.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

action_cols = st.columns((1, 1, 2))
if action_cols[0].button("Prepare source list", use_container_width=True):
    load_result = _run_and_store("Loading attendee names...", lambda: load_attendees(conn, settings))
    classify_result = _run_and_store("Classifying source data...", lambda: run_deterministic_classification(conn))
    st.session_state["last_action"] = {
        "message": f"{load_result.message} {classify_result.message}",
        "level": "success",
    }
    frame, metrics = _load_frame_and_metrics(conn)

if action_cols[1].button("Generate refined prospects", use_container_width=True):
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
            _table_html(
                latest_prospects,
                [
                    ("rank", "Rank", "rank"),
                    ("canonical_name", "Company", "company"),
                    ("weighted_score", "Fit score", "score"),
                    ("company_type_display", "Type", "text"),
                    ("confidence_display", "Confidence", "text"),
                ],
                "Verified prospects will appear here as scoring completes.",
            ),
            unsafe_allow_html=True,
        )

    enrich_result = enrich_candidates(
        conn,
        settings,
        cap,
        bool(st.session_state.get("force_refresh", False)),
        progress_callback=enrichment_progress,
    )
    score_result = score_enriched_candidates(
        conn,
        settings,
        cap,
        bool(st.session_state.get("force_refresh", False)),
        progress_callback=scoring_progress,
    )
    progress.progress(1.0, text="Verification run finished.")
    level = "warning" if enrich_result.counts.get("errors") or score_result.counts.get("errors") else "success"
    st.session_state["last_action"] = {
        "message": f"{classify_result.message} {enrich_result.message} {score_result.message}",
        "level": level,
    }
    frame, metrics = _load_frame_and_metrics(conn)

last_action = st.session_state.get("last_action")
if last_action:
    level = html.escape(last_action.get("level", "info"))
    message = html.escape(last_action.get("message", ""))
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
lifetime_api_calls = int(run_totals.get("tavily_calls") or 0) + int(run_totals.get("openai_calls") or 0)
lifetime_cost = float(run_totals.get("estimated_cost_usd") or 0)
cost_per_verified = lifetime_cost / int(metrics["openai_scored"] or 1) if metrics.get("openai_scored") else 0

_render_metric_cards(
    [
        ("Raw attendee rows", _format_int(metrics["raw_companies"]), f"{_format_int(metrics['unique_companies'])} unique names"),
        ("Screened candidates", _format_int(candidate_count), f"{_pct(candidate_count, metrics['unique_companies'])} of unique"),
        ("Evidence enriched", _format_int(metrics["enriched"]), f"{_pct(metrics['enriched'], candidate_count)} of candidates"),
        ("Verified prospects", _format_int(len(prospects)), f"{_format_int(metrics['openai_scored'])} API-scored"),
        ("Last API calls", _format_int(last_api_calls), f"{_format_currency(last_run.get('estimated_cost_usd'))} latest run"),
        ("Tokens tracked", _format_int(run_totals.get("total_tokens")), f"{_format_currency(cost_per_verified)} per API-scored company"),
    ]
)

if frame.empty:
    st.warning("No companies loaded yet. Use Prepare source list to load and classify the Manifest attendee file.")
else:
    overview_tab, source_tab, prospects_tab, detail_tab = st.tabs(
        ["Overview", "Source list", "Verified prospects", "Company detail"]
    )

    with overview_tab:
        chart_cols = st.columns((1.15, 1, 1))

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
        chart_cols[0].markdown("<div class='section-label'>Source to prospects</div>", unsafe_allow_html=True)
        chart_cols[0].altair_chart(
            _horizontal_bar_chart(funnel_df, "Stage", "Companies", height=230, sort=None),
            use_container_width=True,
        )

        chart_cols[1].markdown("<div class='section-label'>Fit score distribution</div>", unsafe_allow_html=True)
        chart_cols[1].altair_chart(
            _horizontal_bar_chart(_score_band_frame(weighted_frame), "Score band", "Companies", height=230, sort=None),
            use_container_width=True,
        )

        chart_cols[2].markdown("<div class='section-label'>Sector mix</div>", unsafe_allow_html=True)
        chart_cols[2].altair_chart(_pie_chart(_sector_frame(weighted_frame), "Sector", "Companies"), use_container_width=True)

        lower_cols = st.columns((1, 1, 1))
        lower_cols[0].markdown("<div class='section-label'>Company types</div>", unsafe_allow_html=True)
        lower_cols[0].altair_chart(_pie_chart(_type_frame(weighted_frame), "Type", "Companies"), use_container_width=True)

        lower_cols[1].markdown("<div class='section-label'>Estimated API spend</div>", unsafe_allow_html=True)
        lower_cols[1].altair_chart(
            _pie_chart(_resource_cost_frame(metrics, settings), "Resource", "Estimated USD"),
            use_container_width=True,
        )

        cost_stage_df = _stage_cost_frame(metrics, settings)
        lower_cols[2].markdown("<div class='section-label'>Cost by stage</div>", unsafe_allow_html=True)
        lower_cols[2].altair_chart(
            _horizontal_bar_chart(cost_stage_df, "Stage", "Estimated USD", height=230, sort=None),
            use_container_width=True,
        )

        st.markdown("<div class='section-label'>Top verified prospects</div>", unsafe_allow_html=True)
        _render_table(
            prospects.head(10),
            [
                ("rank", "Rank", "rank"),
                ("canonical_name", "Company", "company"),
                ("weighted_score", "Fit score", "score"),
                ("company_type_display", "Type", "text"),
                ("confidence_display", "Confidence", "text"),
                ("sector_tags", "Sectors", "tags"),
                ("evidence_preview", "Evidence", "text"),
            ],
            "No externally verified prospects yet. Generate a capped batch to populate this list.",
        )

    with source_tab:
        st.markdown("<div class='section-label'>Source list</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='quiet-note'>Showing {_format_int(len(source_rows))} de-duplicated source rows from {_format_int(len(weighted_frame))} unique company names.</div>",
            unsafe_allow_html=True,
        )
        _render_table(
            source_rows,
            [
                ("raw_name", "Raw attendee entry", "text"),
                ("canonical_name", "Cleaned company", "company"),
                ("source_status", "Status", "text"),
                ("deterministic_type_display", "Rule bucket", "text"),
                ("sector_tags", "Sectors", "tags"),
                ("weighted_score", "Screen score", "score"),
                ("duplicate_count", "Rows", "number"),
            ],
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
                "deterministic_type_display": "Rule bucket",
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
        _render_table(
            visible_prospects,
            [
                ("rank", "Rank", "rank"),
                ("canonical_name", "Company", "company"),
                ("weighted_score", "Fit score", "score"),
                ("company_type_display", "Type", "text"),
                ("confidence_display", "Confidence", "text"),
                ("sector_tags", "Sectors", "tags"),
                ("wittington_edge", "WV edge", "number"),
                ("evidence_preview", "Evidence", "text"),
            ],
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

            website = _company_url(selected_row)
            st.markdown(
                "<div class='detail-box'>"
                f"<div class='detail-title'>{_safe_link(website, selected_row['canonical_name'])}</div>"
                f"<div class='detail-text'><strong>Rule bucket:</strong> {html.escape(str(selected_row.get('deterministic_type_display') or ''))}</div>"
                f"<div class='detail-text'><strong>Rationale:</strong> {html.escape(selected_row.get('rationale') or 'No rationale yet.')}</div>"
                f"<div class='detail-text'><strong>Evidence:</strong> {html.escape(selected_row.get('evidence_summary') or 'No external evidence yet.')}</div>"
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
                for index, url in enumerate(urls):
                    title = titles[index] if index < len(titles) else url
                    snippet = snippets[index] if index < len(snippets) else ""
                    st.markdown(f"- [{html.escape(title)}]({url})")
                    if snippet:
                        st.caption(snippet)
