from __future__ import annotations

import html
import json
from collections import Counter

import altair as alt
import pandas as pd
import streamlit as st

from src import db
from src.config import get_settings
from src.pipeline import (
    enrich_candidates,
    load_attendees,
    run_deterministic_classification,
    score_enriched_candidates,
    verify_cache_reuse,
)
from src.view_model import VERIFIED_EMPTY_STATE, add_review_metadata, verified_top_prospects


st.set_page_config(
    page_title="Manifest Prospecting Tool",
    page_icon="WV",
    layout="wide",
)


CUSTOM_CSS = """
<style>
  .block-container {
    max-width: 1180px;
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
  }
  h1, h2, h3 {
    letter-spacing: 0 !important;
  }
  .wv-header {
    border-bottom: 1px solid #eceff3;
    margin-bottom: 0.9rem;
    padding: 0.25rem 0 1rem 0;
  }
  .wv-title {
    color: #202332;
    font-size: clamp(2rem, 4.6vw, 3rem);
    font-weight: 780;
    line-height: 1.14;
    margin: 0;
  }
  .wv-subtitle {
    color: #697386;
    font-size: 1.02rem;
    margin: 0.45rem 0 0 0;
    max-width: 760px;
  }
  .metric-grid {
    display: grid;
    gap: 0.85rem;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    margin: 1.05rem 0 1.4rem 0;
  }
  .metric-card {
    background: #ffffff;
    border: 1px solid #e5e9ef;
    border-radius: 8px;
    padding: 1rem;
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
    font-size: 1.55rem;
    font-weight: 760;
    line-height: 1.1;
  }
  .metric-caption {
    color: #8b94a5;
    font-size: 0.78rem;
    line-height: 1.25;
    margin-top: 0.3rem;
  }
  .empty-state {
    background: #ffffff;
    border: 1px solid #e5e9ef;
    border-radius: 8px;
    margin-top: 1.2rem;
    padding: 1.1rem;
  }
  .empty-state-title {
    color: #202332;
    font-size: 1.15rem;
    font-weight: 760;
    margin-bottom: 0.25rem;
  }
  .empty-state-copy {
    color: #697386;
    font-size: 0.95rem;
    margin-bottom: 0.8rem;
  }
  .small-note {
    color: #697386;
    font-size: 0.84rem;
    margin-top: 0.45rem;
  }
  .status-strip {
    background: #f6f8fb;
    border: 1px solid #e5e9ef;
    border-left: 4px solid #2f6fed;
    border-radius: 8px;
    color: #313647;
    font-size: 0.92rem;
    margin: 0.2rem 0 1.1rem 0;
    padding: 0.72rem 0.9rem;
  }
  .status-strip.success {
    border-left-color: #12823b;
  }
  .status-strip.warning {
    border-left-color: #bf7a00;
  }
  .section-label {
    color: #202332;
    font-size: 1.12rem;
    font-weight: 740;
    margin: 1.1rem 0 0.55rem 0;
  }
  .quiet-note {
    color: #697386;
    font-size: 0.88rem;
    margin-top: -0.15rem;
  }
  section[data-testid="stSidebar"] .stButton > button {
    min-height: 2.7rem;
  }
  section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
    line-height: 1.35;
  }
  div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e9ef;
    border-radius: 8px;
    padding: 0.65rem 0.75rem;
  }
  div[data-testid="stMetricLabel"] p {
    font-size: 0.78rem;
  }
  div[data-testid="stMetricValue"] {
    font-size: 1.35rem;
  }
  @media (max-width: 1100px) {
    .metric-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
  @media (max-width: 700px) {
    .metric-grid {
      grid-template-columns: 1fr;
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


def _pct(part: int | float | None, whole: int | float | None) -> str:
    if not whole:
        return "0%"
    return f"{(float(part or 0) / float(whole)):.0%}"


def _truncate(value: str | None, max_chars: int = 110) -> str:
    text = (value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1].rstrip()}..."


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
    frame["is_startup_likely"] = frame["is_startup_likely"].fillna(0).astype(int)
    if "high_priority_enrichment" not in frame.columns:
        frame["high_priority_enrichment"] = 0
    frame["high_priority_enrichment"] = frame["high_priority_enrichment"].fillna(0).astype(int)
    frame["wittington_edge"] = frame["wittington_edge"].fillna(0).astype(int)
    frame["company_type"] = frame["company_type"].fillna(frame["deterministic_type"]).fillna("unscored")
    frame["confidence"] = frame["confidence"].fillna("low")
    frame["rationale"] = frame["rationale"].fillna("")
    frame["rationale_preview"] = frame["rationale"].apply(lambda value: _truncate(value, 95))
    return add_review_metadata(frame)


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
        frame["display_score"],
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
            counter.update(tags or [])

    rows = counter.most_common(limit)
    if not rows:
        rows = [("untagged", 0)]
    return pd.DataFrame(rows, columns=["Sector", "Companies"])


def _type_frame(frame: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame({"Type": ["none"], "Companies": [0]})
    counts = frame["company_type"].value_counts().head(limit)
    return counts.rename_axis("Type").reset_index(name="Companies")


def _horizontal_bar_chart(
    data: pd.DataFrame,
    label_column: str,
    value_column: str,
    *,
    height: int = 260,
    sort: str | list[str] = "-x",
) -> None:
    chart = (
        alt.Chart(data)
        .mark_bar(color="#0b72d0", cornerRadiusEnd=3)
        .encode(
            x=alt.X(f"{value_column}:Q", title=None, axis=alt.Axis(grid=True, labelColor="#697386")),
            y=alt.Y(
                f"{label_column}:N",
                sort=sort,
                title=None,
                axis=alt.Axis(labelLimit=260, labelColor="#697386"),
            ),
            tooltip=[alt.Tooltip(f"{label_column}:N", title=label_column), alt.Tooltip(f"{value_column}:Q", title=value_column)],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def _run_sequence_and_store(label: str, func, *, success_message: str | None = None, level: str = "success"):
    with st.spinner(label):
        results = func()
    message = success_message or " ".join(result.message for result in results)
    st.session_state["last_action"] = {
        "message": message,
        "level": level,
    }
    return results


def _load_and_classify_companies():
    return [load_attendees(conn, settings), run_deterministic_classification(conn)]


def _generate_verified_prospects(enrich_limit: int, score_limit: int, force: bool):
    return [
        enrich_candidates(conn, settings, enrich_limit, force=force),
        score_enriched_candidates(conn, settings, score_limit, force=force),
    ]


settings, conn = _connect()

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Workflow")
    st.caption("Three steps. The first is free; the second uses the cached high-priority queue.")

    if st.button("1. Load & classify", type="primary", width="stretch"):
        _run_sequence_and_store(
            "Loading and classifying Manifest companies...",
            _load_and_classify_companies,
            success_message="Manifest companies loaded and prioritized. No paid API calls were made.",
        )

    if st.button("2. Generate verified prospects", width="stretch"):
        _run_sequence_and_store(
            "Generating verified prospects...",
            lambda: _generate_verified_prospects(settings.max_enrich, settings.max_score, False),
            success_message="Verified prospect generation finished.",
        )

    if st.button("3. Verify cache reuse", width="stretch"):
        with st.spinner("Checking SQLite cache reuse..."):
            result = verify_cache_reuse(conn, settings)
        level = "success" if result.counts.get("verified") else "warning"
        st.session_state["last_action"] = {"message": result.message, "level": level}
        if result.counts.get("verified"):
            st.success(result.message)
        else:
            st.warning(result.message)

    st.caption(
        f"Tavily {'configured' if settings.tavily_api_key else 'missing'} · "
        f"OpenAI {'configured' if settings.openai_api_key else 'missing'} · "
        f"Limits {settings.max_enrich}/{settings.max_score}"
    )

rows = db.dashboard_rows(conn)
frame = _rows_to_frame(rows)
metrics = db.metrics(conn)
verified_count = int(frame["is_verified_prospect"].sum()) if not frame.empty else 0

st.markdown(
    """
    <div class="wv-header">
      <h1 class="wv-title">Manifest Prospecting Tool</h1>
      <p class="wv-subtitle">A Wittington Ventures sourcing workflow for finding evidence-backed prospects from the Manifest attendee list.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

last_run = metrics.get("last_run") or {}
last_api_calls = int(last_run.get("tavily_calls") or 0) + int(last_run.get("openai_calls") or 0)
last_cache_hits = int(last_run.get("cache_hits") or 0)

last_action = st.session_state.get("last_action")
if last_action:
    level = html.escape(last_action.get("level", "info"))
    message = html.escape(last_action.get("message", ""))
    st.markdown(f"<div class='status-strip {level}'>{message}</div>", unsafe_allow_html=True)

if frame.empty:
    st.markdown(
        """
        <div class="empty-state">
          <div class="empty-state-title">Start with the Manifest attendee universe.</div>
          <div class="empty-state-copy">Load, clean, dedupe, and classify the attendee list before spending on enrichment.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Load & classify Manifest companies", type="primary", key="main_load_classify"):
        _run_sequence_and_store(
            "Loading and classifying Manifest companies...",
            _load_and_classify_companies,
            success_message="Manifest companies loaded and prioritized. No paid API calls were made.",
        )
        st.rerun()
    st.markdown("<div class='small-note'>No paid API calls are made in this step.</div>", unsafe_allow_html=True)
    st.stop()
else:
    _render_metric_cards(
        [
            ("Universe", _format_int(metrics["unique_companies"]), f"{_format_int(metrics['raw_companies'])} raw rows"),
            ("High-priority queue", _format_int(metrics["high_priority_queue"]), "paid-call default"),
            ("Verified prospects", _format_int(verified_count), "evidence-backed"),
            ("Last API calls", _format_int(last_api_calls), "latest run"),
            ("Cache hits", _format_int(last_cache_hits), "latest run"),
        ]
    )

    top_prospects = verified_top_prospects(frame, limit=12)
    st.markdown("<div class='section-label'>Verified prospects</div>", unsafe_allow_html=True)
    if top_prospects.empty:
        st.info(VERIFIED_EMPTY_STATE)
    else:
        st.dataframe(
            top_prospects[
                [
                    "rank",
                    "canonical_name",
                    "display_score",
                    "company_type",
                    "confidence",
                    "evidence_summary",
                    "primary_source_url",
                ]
            ],
            width="stretch",
            height=430,
            hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "canonical_name": st.column_config.TextColumn("Company", width="medium"),
                "display_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
                "company_type": st.column_config.TextColumn("Type", width="small"),
                "confidence": st.column_config.TextColumn("Confidence", width="small"),
                "evidence_summary": st.column_config.TextColumn("Evidence", width="large"),
                "primary_source_url": st.column_config.LinkColumn("Source", width="small"),
            },
        )
        st.caption("Select a cell and press Cmd/Ctrl+C to copy.")

    st.markdown("<div class='section-label'>Sourcing analytics</div>", unsafe_allow_html=True)
    chart_left, chart_right = st.columns(2)
    funnel_df = pd.DataFrame(
        {
            "Stage": ["Universe", "High-priority", "Enriched", "OpenAI scored"],
            "Companies": [
                metrics["unique_companies"],
                metrics["high_priority_queue"],
                metrics["enriched"],
                metrics["openai_scored"],
            ],
        }
    )
    with chart_left:
        st.markdown("<div class='quiet-note'>Pipeline funnel</div>", unsafe_allow_html=True)
        _horizontal_bar_chart(
            funnel_df,
            "Stage",
            "Companies",
            height=250,
            sort=["OpenAI scored", "Enriched", "High-priority", "Universe"],
        )
    with chart_right:
        st.markdown("<div class='quiet-note'>Score distribution</div>", unsafe_allow_html=True)
        _horizontal_bar_chart(_score_band_frame(frame), "Score band", "Companies", height=250, sort="-x")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("<div class='quiet-note'>Sector mix</div>", unsafe_allow_html=True)
        _horizontal_bar_chart(_sector_frame(frame), "Sector", "Companies", height=300, sort="-x")
    with chart_right:
        st.markdown("<div class='quiet-note'>Company types</div>", unsafe_allow_html=True)
        _horizontal_bar_chart(_type_frame(frame), "Type", "Companies", height=300, sort="-x")

    st.markdown("<div class='section-label'>Full ranked pipeline</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='quiet-note'>Full list retained for review. Baseline-only rows are screening signals, not verified prospects.</div>",
        unsafe_allow_html=True,
    )

    display_columns = [
        "rank",
        "canonical_name",
        "display_score",
        "evidence_status",
        "cache_status",
        "company_type",
        "sector_tags_text",
        "confidence",
        "primary_source_url",
    ]
    st.dataframe(
        frame[display_columns],
        width="stretch",
        height=640,
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "canonical_name": st.column_config.TextColumn("Company", width="medium"),
            "display_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
            "evidence_status": st.column_config.TextColumn("Evidence", width="small"),
            "cache_status": st.column_config.TextColumn("Cache", width="small"),
            "company_type": st.column_config.TextColumn("Type", width="small"),
            "sector_tags_text": st.column_config.TextColumn("Sectors", width="medium"),
            "confidence": st.column_config.TextColumn("Confidence", width="small"),
            "primary_source_url": st.column_config.LinkColumn("Source", width="small"),
        },
    )
    st.caption("Select cells to copy, or download the CSV.")

    csv_columns = display_columns + [
        "high_priority_enrichment",
        "total_score",
        "rationale",
        "source_urls",
        "evidence_summary",
    ]
    csv = frame[csv_columns].to_csv(index=False)
    st.download_button(
        "Download CSV",
        data=csv,
        file_name="manifest_ranked_pipeline.csv",
        mime="text/csv",
    )
