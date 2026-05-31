from __future__ import annotations

import html
import json
from collections import Counter

import pandas as pd
import streamlit as st

from src import db
from src.config import PROJECT_ROOT, get_settings
from src.pipeline import (
    enrich_candidates,
    generate_verified_prospects,
    load_attendees,
    load_and_classify_companies,
    run_default_pipeline,
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
    font-size: 1rem;
    margin: 0.45rem 0 0 0;
    max-width: 760px;
  }
  .workflow-strip {
    align-items: center;
    display: grid;
    gap: 0.5rem;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    margin: 1rem 0 1.1rem 0;
  }
  .workflow-step {
    background: #f6f8fb;
    border: 1px solid #e5e9ef;
    border-radius: 8px;
    color: #697386;
    font-size: 0.86rem;
    font-weight: 680;
    padding: 0.62rem 0.72rem;
  }
  .workflow-step.done {
    background: #edf8f1;
    border-color: #cfe8d7;
    color: #176c35;
  }
  .workflow-step.current {
    background: #eef4ff;
    border-color: #cfe0ff;
    color: #235ab7;
  }
  .metric-grid {
    display: grid;
    gap: 0.75rem;
    grid-template-columns: repeat(5, minmax(0, 1fr));
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
    font-size: 1.05rem;
    font-weight: 740;
    margin: 0.3rem 0 0.6rem 0;
  }
  .quiet-note {
    color: #697386;
    font-size: 0.88rem;
    margin-top: -0.15rem;
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
    .workflow-strip {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
  @media (max-width: 700px) {
    .metric-grid {
      grid-template-columns: 1fr;
    }
    .workflow-strip {
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


def _render_workflow_strip(metrics: dict, verified_count: int) -> None:
    loaded = int(metrics.get("unique_companies") or 0) > 0
    prioritized = int(metrics.get("high_priority_queue") or 0) > 0
    scored = int(metrics.get("openai_scored") or 0) > 0
    reviewed = verified_count > 0

    steps = [
        ("1. Load list", loaded),
        ("2. Prioritize queue", prioritized),
        ("3. Enrich + score", scored),
        ("4. Review prospects", reviewed),
    ]
    next_index = next((index for index, (_, done) in enumerate(steps) if not done), len(steps))
    body = ["<div class='workflow-strip'>"]
    for index, (label, done) in enumerate(steps):
        state = "done" if done else "current" if index == next_index else ""
        body.append(f"<div class='workflow-step {state}'>{html.escape(label)}</div>")
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


def _run_and_store(label: str, func, *, level: str = "success"):
    with st.spinner(label):
        result = func()
    st.session_state["last_action"] = {
        "message": result.message,
        "level": level,
    }
    return result


def _run_sequence_and_store(label: str, func, *, success_message: str | None = None, level: str = "success"):
    with st.spinner(label):
        results = func()
    message = success_message or " ".join(result.message for result in results)
    st.session_state["last_action"] = {
        "message": message,
        "level": level,
    }
    return results


settings, conn = _connect()
try:
    display_database_path = settings.database_path.relative_to(PROJECT_ROOT)
except ValueError:
    display_database_path = settings.database_path

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Workflow")
    st.caption("Run the sourcing flow in order.")

    if st.button("1. Load & classify companies", type="primary", width="stretch"):
        _run_sequence_and_store(
            "Loading and classifying Manifest companies...",
            lambda: load_and_classify_companies(conn, settings),
            success_message="Manifest companies loaded and prioritized. No paid API calls were made.",
        )

    if st.button("2. Generate verified prospects", width="stretch"):
        max_enrich_for_button = int(st.session_state.get("max_enrich", settings.max_enrich))
        max_score_for_button = int(st.session_state.get("max_score", settings.max_score))
        force_refresh_for_button = bool(st.session_state.get("force_refresh", False))
        _run_sequence_and_store(
            "Generating verified prospects...",
            lambda: generate_verified_prospects(
                conn,
                settings,
                enrich_limit=max_enrich_for_button,
                score_limit=max_score_for_button,
                force=force_refresh_for_button,
            ),
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
        f"OpenAI {'configured' if settings.openai_api_key else 'missing'}"
    )

    with st.expander("How this works", expanded=False):
        st.write("Load the list, generate evidence-backed prospects, then review the verified companies.")

    with st.expander("Cost controls", expanded=False):
        st.number_input(
            "Max Tavily enrichments",
            min_value=1,
            max_value=500,
            value=settings.max_enrich,
            step=5,
            key="max_enrich",
        )
        st.number_input(
            "Max OpenAI scores",
            min_value=1,
            max_value=500,
            value=settings.max_score,
            step=5,
            key="max_score",
        )
        st.checkbox("Force refresh cached API records", value=False, key="force_refresh")
        st.caption("Normal runs reuse SQLite cache records and avoid repeat paid calls.")

    with st.expander("Advanced controls", expanded=False):
        st.caption(f"SQLite: `{display_database_path}`")
        if st.button("Load Manifest list", width="stretch"):
            _run_and_store("Loading attendee names...", lambda: load_attendees(conn, settings))

        if st.button("Classify companies", width="stretch"):
            _run_and_store("Classifying companies...", lambda: run_deterministic_classification(conn))

        if st.button("Enrich with Tavily", width="stretch"):
            max_enrich_for_button = int(st.session_state.get("max_enrich", settings.max_enrich))
            force_refresh_for_button = bool(st.session_state.get("force_refresh", False))
            _run_and_store(
                "Enriching candidates...",
                lambda: enrich_candidates(conn, settings, max_enrich_for_button, force_refresh_for_button),
            )

        if st.button("Score with OpenAI", width="stretch"):
            max_score_for_button = int(st.session_state.get("max_score", settings.max_score))
            force_refresh_for_button = bool(st.session_state.get("force_refresh", False))
            _run_and_store(
                "Scoring enriched candidates...",
                lambda: score_enriched_candidates(conn, settings, max_score_for_button, force_refresh_for_button),
            )

        if st.button("Run staged pipeline", width="stretch"):
            with st.spinner("Running staged pipeline..."):
                results = run_default_pipeline(conn, settings)
            message = " | ".join(result.message for result in results)
            st.session_state["last_action"] = {"message": message, "level": "success"}

        if st.button("Verify cache reuse", width="stretch"):
            with st.spinner("Checking SQLite cache reuse..."):
                result = verify_cache_reuse(conn, settings)
            level = "success" if result.counts.get("verified") else "warning"
            st.session_state["last_action"] = {"message": result.message, "level": level}
            if result.counts.get("verified"):
                st.success(result.message)
            else:
                st.warning(result.message)

        if st.button("Initialize database", width="stretch"):
            db.init_db(conn)
            st.session_state["last_action"] = {"message": "Database initialized.", "level": "success"}
            st.success("Database initialized.")

rows = db.dashboard_rows(conn)
frame = _rows_to_frame(rows)
metrics = db.metrics(conn)
verified_count = int(frame["is_verified_prospect"].sum()) if not frame.empty else 0

startup_only = False
priority_only = False
min_score = 0
selected_types: list[str] = []
selected_sectors: list[str] = []
selected_confidence: list[str] = []

with st.sidebar:
    if not frame.empty:
        with st.expander("Filters", expanded=False):
            startup_only = st.checkbox("Startup-likely only", value=False)
            priority_only = st.checkbox("High-priority queue only", value=False)
            min_score = st.slider("Minimum score", min_value=0, max_value=100, value=0)
            type_options = sorted([value for value in frame["company_type"].dropna().unique().tolist() if value])
            confidence_options = sorted([value for value in frame["confidence"].dropna().unique().tolist() if value])
            all_sectors = sorted({tag for tags in frame["sector_tags"] for tag in (tags or [])})
            selected_types = st.multiselect("Company type", type_options, default=[])
            selected_sectors = st.multiselect("Sector", all_sectors, default=[])
            selected_confidence = st.multiselect("Confidence", confidence_options, default=[])

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

_render_workflow_strip(metrics, verified_count)

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
            lambda: load_and_classify_companies(conn, settings),
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

    with st.expander("Pipeline details", expanded=False):
        detail_cols = st.columns(3)
        detail_cols[0].metric("Broad candidates", _format_int(metrics["candidates"]))
        detail_cols[1].metric("Tavily enriched", _format_int(metrics["enriched"]))
        detail_cols[2].metric("OpenAI scored", _format_int(metrics["openai_scored"]))

    filtered = frame[frame["display_score"] >= min_score].copy()
    if startup_only:
        filtered = filtered[filtered["is_startup_likely"] == 1]
    if priority_only:
        filtered = filtered[filtered["high_priority_enrichment"] == 1]
    if selected_types:
        filtered = filtered[filtered["company_type"].isin(selected_types)]
    if selected_confidence:
        filtered = filtered[filtered["confidence"].isin(selected_confidence)]
    if selected_sectors:
        selected = set(selected_sectors)
        filtered = filtered[filtered["sector_tags"].apply(lambda tags: bool(selected.intersection(tags or [])))]

    overview_tab, pipeline_tab, detail_tab = st.tabs(["Overview", "Ranked pipeline", "Company detail"])

    with overview_tab:
        top_prospects = verified_top_prospects(filtered, limit=10)
        st.markdown("<div class='section-label'>Verified prospects</div>", unsafe_allow_html=True)
        if top_prospects.empty:
            st.info(VERIFIED_EMPTY_STATE)
        else:
            st.dataframe(
                top_prospects[["rank", "canonical_name", "display_score", "company_type", "evidence_status", "confidence"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", width="small"),
                    "canonical_name": st.column_config.TextColumn("Company", width="medium"),
                    "display_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
                    "company_type": st.column_config.TextColumn("Type"),
                    "evidence_status": st.column_config.TextColumn("Evidence"),
                    "confidence": st.column_config.TextColumn("Confidence"),
                },
            )

        with st.expander("Pipeline analytics", expanded=False):
            chart_cols = st.columns((1.1, 1, 1))

            funnel_df = pd.DataFrame(
                {
                    "Stage": ["Unique", "High-priority", "Enriched", "OpenAI scored"],
                    "Companies": [
                        metrics["unique_companies"],
                        metrics["high_priority_queue"],
                        metrics["enriched"],
                        metrics["openai_scored"],
                    ],
                }
            )
            chart_cols[0].markdown("<div class='section-label'>Pipeline funnel</div>", unsafe_allow_html=True)
            chart_cols[0].bar_chart(funnel_df, x="Stage", y="Companies", height=220)

            chart_cols[1].markdown("<div class='section-label'>Score distribution</div>", unsafe_allow_html=True)
            chart_cols[1].bar_chart(_score_band_frame(filtered), x="Score band", y="Companies", height=220)

            chart_cols[2].markdown("<div class='section-label'>Sector mix</div>", unsafe_allow_html=True)
            chart_cols[2].bar_chart(_sector_frame(filtered), x="Sector", y="Companies", height=220)

            st.markdown("<div class='section-label'>Company types</div>", unsafe_allow_html=True)
            st.bar_chart(_type_frame(filtered), x="Type", y="Companies", height=220)

    with pipeline_tab:
        st.markdown("<div class='section-label'>Ranked pipeline</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='quiet-note'>Showing {_format_int(len(filtered))} of {_format_int(len(frame))} companies.</div>",
            unsafe_allow_html=True,
        )

        display_columns = [
            "rank",
            "canonical_name",
            "display_score",
            "evidence_status",
            "cache_status",
            "company_type",
            "is_startup_likely",
            "high_priority_enrichment",
            "sector_tags_text",
            "wittington_edge",
            "confidence",
            "primary_source_url",
        ]
        st.dataframe(
            filtered[display_columns],
            width="stretch",
            hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "canonical_name": st.column_config.TextColumn("Company", width="medium"),
                "display_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
                "evidence_status": st.column_config.TextColumn("Evidence"),
                "cache_status": st.column_config.TextColumn("Cache"),
                "company_type": st.column_config.TextColumn("Type"),
                "is_startup_likely": st.column_config.CheckboxColumn("Startup"),
                "high_priority_enrichment": st.column_config.CheckboxColumn("Priority"),
                "sector_tags_text": st.column_config.TextColumn("Sectors"),
                "wittington_edge": st.column_config.NumberColumn("WV edge", min_value=0, max_value=20),
                "confidence": st.column_config.TextColumn("Confidence"),
                "primary_source_url": st.column_config.LinkColumn("Source"),
            },
        )

        csv = filtered[display_columns + ["total_score", "rationale", "source_urls", "evidence_summary"]].to_csv(index=False)
        st.download_button(
            "Download filtered CSV",
            data=csv,
            file_name="manifest_ranked_pipeline.csv",
            mime="text/csv",
        )

    with detail_tab:
        st.markdown("<div class='section-label'>Company detail</div>", unsafe_allow_html=True)
        if filtered.empty:
            st.warning("No companies match the current filters.")
        else:
            selected_company = st.selectbox("Company", filtered["canonical_name"].tolist())
            selected_row = filtered[filtered["canonical_name"] == selected_company].iloc[0]

            detail_cols = st.columns(4)
            detail_cols[0].metric("Score", f"{int(selected_row['display_score'])}/100")
            detail_cols[1].metric("Evidence", str(selected_row.get("evidence_status") or "Baseline only"))
            detail_cols[2].metric("Cache", str(selected_row.get("cache_status") or "Not cached"))
            detail_cols[3].metric("Confidence", str(selected_row.get("confidence") or "low"))

            evidence_status = str(selected_row.get("evidence_status") or "Baseline only")
            if evidence_status == "Baseline only":
                status_text = "Unverified baseline: this company has no cached Tavily evidence or OpenAI score yet. Treat it as a screening row, not an investment prospect."
            elif evidence_status == "Enriched":
                status_text = "Enriched: Tavily evidence is cached, but OpenAI scoring has not been run yet."
            else:
                status_text = "OpenAI scored: Tavily evidence and structured scoring are cached."

            st.markdown(
                "<div class='detail-box'>"
                f"<div class='detail-title'>{html.escape(selected_row['canonical_name'])}</div>"
                f"<div class='detail-text'><strong>Status:</strong> {html.escape(status_text)}</div>"
                f"<div class='detail-text'><strong>Type:</strong> {html.escape(str(selected_row.get('company_type') or 'unknown'))}</div>"
                f"<div class='detail-text'><strong>Rationale:</strong> {html.escape(selected_row.get('rationale') or 'No rationale yet.')}</div>"
                f"<div class='detail-text'><strong>Evidence summary:</strong> {html.escape(selected_row.get('evidence_summary') or 'No evidence yet.')}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

            urls = selected_row.get("top_urls") or []
            titles = selected_row.get("top_titles") or []
            snippets = selected_row.get("top_snippets") or []
            with st.expander("Retrieved evidence", expanded=bool(urls)):
                if not urls:
                    st.caption("No external evidence cached yet.")
                for index, url in enumerate(urls):
                    title = titles[index] if index < len(titles) else url
                    snippet = snippets[index] if index < len(snippets) else ""
                    st.markdown(f"- [{title}]({url})")
                    if snippet:
                        st.caption(snippet)
