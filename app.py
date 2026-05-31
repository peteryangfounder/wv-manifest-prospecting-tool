from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import db
from src.config import PROJECT_ROOT, get_settings
from src.pipeline import (
    enrich_candidates,
    load_attendees,
    run_default_pipeline,
    run_deterministic_classification,
    score_enriched_candidates,
)


st.set_page_config(
    page_title="Manifest Prospecting Tool",
    page_icon="WV",
    layout="wide",
)


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
    frame["rank"] = frame["total_score"].rank(method="first", ascending=False).astype(int)
    return frame.sort_values(["total_score", "canonical_name"], ascending=[False, True])


def _run_and_report(label: str, func):
    with st.spinner(label):
        result = func()
    st.session_state["last_action"] = result.message
    st.success(result.message)


settings, conn = _connect()
try:
    display_database_path = settings.database_path.relative_to(PROJECT_ROOT)
except ValueError:
    display_database_path = settings.database_path

st.title("Manifest Prospecting Tool")
st.caption("Wittington Ventures sourcing pipeline")

st.write(
    "This tool converts the Manifest attendee list into a cached, ranked sourcing pipeline. "
    "Retrieval happens in code through scraping and Tavily search; OpenAI is used only for compact, structured judgment after evidence is cached."
)

with st.sidebar:
    st.header("Pipeline")
    st.caption(f"Database: `{display_database_path}`")

    tavily_ready = bool(settings.tavily_api_key)
    openai_ready = bool(settings.openai_api_key)
    st.write(f"Tavily: {'configured' if tavily_ready else 'missing key'}")
    st.write(f"OpenAI: {'configured' if openai_ready else 'missing key'}")

    max_enrich = st.number_input("Max Tavily enrichments", min_value=1, max_value=500, value=settings.max_enrich, step=5)
    max_score = st.number_input("Max OpenAI scores", min_value=1, max_value=500, value=settings.max_score, step=5)
    force_refresh = st.checkbox("Force refresh cached API records", value=False)

    if st.button("Initialize database", width="stretch"):
        db.init_db(conn)
        st.session_state["last_action"] = "Database initialized."
        st.success("Database initialized.")

    if st.button("Scrape/load Manifest list", width="stretch"):
        _run_and_report("Loading attendee names...", lambda: load_attendees(conn, settings))

    if st.button("Run deterministic classification", width="stretch"):
        _run_and_report("Classifying companies...", lambda: run_deterministic_classification(conn))

    if st.button("Enrich candidates with Tavily", width="stretch"):
        _run_and_report(
            "Enriching candidates...",
            lambda: enrich_candidates(conn, settings, int(max_enrich), force_refresh),
        )

    if st.button("Score enriched candidates with OpenAI", width="stretch"):
        _run_and_report(
            "Scoring enriched candidates...",
            lambda: score_enriched_candidates(conn, settings, int(max_score), force_refresh),
        )

    if st.button("Run default staged pipeline", width="stretch"):
        with st.spinner("Running staged pipeline..."):
            results = run_default_pipeline(conn, settings)
        st.session_state["last_action"] = " | ".join(result.message for result in results)
        for result in results:
            st.success(result.message)

    st.header("Filters")
    startup_only = st.checkbox("Startup-likely only", value=False)
    min_score = st.slider("Minimum score", min_value=0, max_value=100, value=0)

rows = db.dashboard_rows(conn)
frame = _rows_to_frame(rows)
metrics = db.metrics(conn)

last_run = metrics.get("last_run") or {}
last_api_calls = int(last_run.get("tavily_calls") or 0) + int(last_run.get("openai_calls") or 0)
last_cache_hits = int(last_run.get("cache_hits") or 0)

metric_cols = st.columns(8)
metric_cols[0].metric("Raw rows", f"{metrics['raw_companies']:,}")
metric_cols[1].metric("Unique companies", f"{metrics['unique_companies']:,}")
metric_cols[2].metric("Candidates", f"{metrics['candidates']:,}")
metric_cols[3].metric("Enriched", f"{metrics['enriched']:,}")
metric_cols[4].metric("Scored", f"{metrics['scored']:,}")
metric_cols[5].metric("OpenAI scored", f"{metrics['openai_scored']:,}")
metric_cols[6].metric("Last API calls", f"{last_api_calls:,}")
metric_cols[7].metric("Cache hits", f"{last_cache_hits:,}")

if st.session_state.get("last_action"):
    st.info(st.session_state["last_action"])

if frame.empty:
    st.warning("No companies loaded yet. Start with 'Scrape/load Manifest list' in the sidebar.")
else:
    type_options = sorted([value for value in frame["company_type"].dropna().unique().tolist() if value])
    confidence_options = sorted([value for value in frame["confidence"].dropna().unique().tolist() if value])
    all_sectors = sorted({tag for tags in frame["sector_tags"] for tag in (tags or [])})

    filter_cols = st.columns(3)
    selected_types = filter_cols[0].multiselect("Company type", type_options, default=[])
    selected_sectors = filter_cols[1].multiselect("Sector tags", all_sectors, default=[])
    selected_confidence = filter_cols[2].multiselect("Confidence", confidence_options, default=[])

    filtered = frame[frame["total_score"] >= min_score].copy()
    if startup_only:
        filtered = filtered[filtered["is_startup_likely"] == 1]
    if selected_types:
        filtered = filtered[filtered["company_type"].isin(selected_types)]
    if selected_confidence:
        filtered = filtered[filtered["confidence"].isin(selected_confidence)]
    if selected_sectors:
        selected = set(selected_sectors)
        filtered = filtered[filtered["sector_tags"].apply(lambda tags: bool(selected.intersection(tags or [])))]

    display_columns = [
        "rank",
        "canonical_name",
        "total_score",
        "company_type",
        "is_startup_likely",
        "sector_tags_text",
        "wittington_edge",
        "rationale",
        "confidence",
        "primary_source_url",
    ]
    st.subheader("Ranked pipeline")
    st.dataframe(
        filtered[display_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "canonical_name": st.column_config.TextColumn("Company", width="medium"),
            "total_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
            "sector_tags_text": st.column_config.TextColumn("Sectors"),
            "is_startup_likely": st.column_config.CheckboxColumn("Startup?"),
            "primary_source_url": st.column_config.LinkColumn("Source"),
        },
    )

    csv = filtered[display_columns + ["source_urls", "evidence_summary"]].to_csv(index=False)
    st.download_button(
        "Download filtered CSV",
        data=csv,
        file_name="manifest_ranked_pipeline.csv",
        mime="text/csv",
    )

    st.subheader("Evidence drilldown")
    selected_company = st.selectbox(
        "Company",
        filtered["canonical_name"].tolist(),
        index=0 if len(filtered) else None,
    )
    if selected_company:
        selected_row = filtered[filtered["canonical_name"] == selected_company].iloc[0]
        st.markdown(f"**Score:** {int(selected_row['total_score'])}/100")
        st.markdown(f"**Rationale:** {selected_row.get('rationale') or 'No rationale yet.'}")
        st.markdown(f"**Evidence summary:** {selected_row.get('evidence_summary') or 'No evidence yet.'}")

        urls = selected_row.get("top_urls") or []
        titles = selected_row.get("top_titles") or []
        snippets = selected_row.get("top_snippets") or []
        for index, url in enumerate(urls):
            title = titles[index] if index < len(titles) else url
            snippet = snippets[index] if index < len(snippets) else ""
            st.markdown(f"- [{title}]({url})")
            if snippet:
                st.caption(snippet)

with st.expander("Cost controls and repeatability"):
    st.write(
        "- The full attendee list is stored in SQLite and deduped by normalized company name.\n"
        "- Deterministic rules run across the full list before any paid API call.\n"
        "- Tavily is called only for candidate companies and is cached by company/provider.\n"
        "- OpenAI is called only after external search evidence exists; only compact titles, URLs, and snippets are sent.\n"
        "- Missing API keys put the app into a usable baseline mode instead of crashing."
    )

with st.expander("Setup reminder"):
    st.code(
        "python -m venv .venv\n"
        "source .venv/bin/activate\n"
        "pip install -r requirements.txt\n"
        "cp .env.example .env\n"
        "streamlit run app.py",
        language="bash",
    )
