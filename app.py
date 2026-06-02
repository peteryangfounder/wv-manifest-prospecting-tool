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
    collect_homepage_evidence,
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
    "data_confidence": "Source data confidence",
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
    "unknown": "Unclassified",
    "unknown_needs_enrichment": "Unclassified",
    "venture_backability": "Venture-scale potential",
    "warehouse_automation": "Warehouse automation",
    "wittington_edge": "Wittington edge",
    "wv_sector_fit": "Wittington sector match",
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

VERIFY_MODE_PRESETS = {
    "balanced": {
        "label": "Balanced",
        "description": "Score likely technology companies first, then unclassified companies inside the approved batch size.",
    },
    "precision-first": {
        "label": "Precision-first",
        "description": "Score only companies classified as startup or technology companies.",
    },
    "recall-first": {
        "label": "Recall-first",
        "description": "Score all API-eligible company types inside the approved batch size.",
    },
}


CUSTOM_CSS = """
<style>
  :root {
    --wv-ink: #1f2433;
    --wv-muted: #667085;
    --wv-line: #dfe4ec;
    --wv-soft: #f8fafc;
    --wv-accent: #1f6f5b;
    --wv-accent-soft: #eaf5f1;
  }
  #MainMenu,
  footer,
  header {
    visibility: hidden;
  }
  .stApp {
    background: #ffffff;
  }
  .block-container {
    max-width: 760px;
    padding-top: 1.1rem;
    padding-bottom: 2rem;
  }
  h1, h2, h3 {
    letter-spacing: 0 !important;
  }
  .wv-header {
    border-bottom: 1px solid var(--wv-line);
    margin-bottom: 0.65rem;
    padding: 0.15rem 0 0.7rem 0;
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
    color: var(--wv-ink);
    font-size: clamp(1.55rem, 2.5vw, 2rem);
    font-weight: 780;
    line-height: 1.08;
    margin: 0;
    overflow-wrap: normal;
    white-space: normal;
  }
  .wv-subtitle {
    color: var(--wv-muted);
    font-size: 0.88rem;
    margin: 0.45rem 0 0 0;
    max-width: 700px;
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
  .slide-shell {
    background: #ffffff;
    border-bottom: 1px solid #eef1f5;
    margin: 0.65rem 0 0.95rem 0;
    padding: 0.7rem 0 0.85rem 0;
  }
  .slide-label {
    color: var(--wv-accent);
    font-size: 0.78rem;
    font-weight: 760;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .slide-title {
    color: var(--wv-ink);
    font-size: clamp(1.35rem, 2.25vw, 1.8rem);
    font-weight: 780;
    line-height: 1.16;
    margin-top: 0.2rem;
  }
  .slide-copy {
    color: var(--wv-muted);
    font-size: 0.94rem;
    line-height: 1.45;
    margin: 0.42rem 0 0 0;
    max-width: 860px;
  }
  .slide-progress {
    align-items: center;
    display: grid;
    gap: 0.6rem;
    grid-template-columns: 1fr auto;
    margin: 0.35rem 0 0.75rem 0;
  }
  .slide-progress-track {
    background: #e7ebf0;
    border-radius: 999px;
    height: 0.45rem;
    overflow: hidden;
  }
  .slide-progress-fill {
    background: var(--wv-accent);
    border-radius: inherit;
    display: block;
    height: 100%;
  }
  .slide-progress-text {
    color: var(--wv-muted);
    font-size: 0.82rem;
    font-weight: 700;
    white-space: nowrap;
  }
  .slide-progress-label {
    color: #697386;
    font-size: 0.82rem;
    font-weight: 680;
    margin-top: 0.3rem;
  }
  .slide-nav-note {
    color: #8b94a5;
    font-size: 0.88rem;
    font-weight: 700;
    padding-top: 0.62rem;
    text-align: center;
  }
  .flow-grid {
    display: grid;
    gap: 0.75rem;
    grid-template-columns: 1fr;
    margin: 0.9rem 0 0.8rem 0;
  }
  .flow-step {
    background: #ffffff;
    border-left: 2px solid #d9dee7;
    min-width: 0;
    padding: 0 0 0.7rem 1.15rem;
    position: relative;
  }
  .flow-step::before {
    background: #ffffff;
    border: 2px solid #2f6b5b;
    border-radius: 999px;
    content: "";
    height: 0.72rem;
    left: -0.43rem;
    position: absolute;
    top: 0.05rem;
    width: 0.72rem;
  }
  .flow-step:first-child {
    border-left-color: #2f6b5b;
  }
  .flow-number {
    color: var(--wv-accent);
    display: block;
    font-size: 0.72rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.38rem;
  }
  .flow-title {
    color: var(--wv-ink);
    font-size: 0.9rem;
    font-weight: 780;
    line-height: 1.25;
  }
  .flow-copy {
    color: var(--wv-muted);
    font-size: 0.8rem;
    line-height: 1.35;
    margin-top: 0.22rem;
  }
  .next-step-card {
    background: #ffffff;
    border: 1px solid #dfe5ee;
    border-radius: 8px;
    margin: 0.85rem 0 1rem 0;
    padding: 0.95rem;
  }
  .next-step-label {
    color: #697386;
    font-size: 0.78rem;
    font-weight: 760;
    text-transform: uppercase;
  }
  .next-step-title {
    color: #202332;
    font-size: 1.08rem;
    font-weight: 780;
    line-height: 1.3;
    margin-top: 0.12rem;
  }
  .next-step-copy {
    color: #5d6675;
    font-size: 0.9rem;
    line-height: 1.42;
    margin-top: 0.28rem;
  }
  .mini-metric-grid {
    display: grid;
    gap: 0;
    grid-template-columns: 1fr;
    margin: 0.8rem 0 0.95rem 0;
  }
  .mini-metric {
    background: #ffffff;
    border-top: 1px solid #edf0f4;
    padding: 0.78rem 0;
  }
  .mini-metric-label {
    color: #697386;
    font-size: 0.82rem;
    font-weight: 700;
    line-height: 1.3;
  }
  .mini-metric-value {
    color: #202332;
    font-size: 1.45rem;
    font-weight: 780;
    line-height: 1.2;
    margin-top: 0.12rem;
  }
  .guided-fact-stack {
    border-top: 1px solid #edf0f4;
    margin: 0.75rem 0 1rem 0;
  }
  .guided-fact {
    border-bottom: 1px solid #edf0f4;
    padding: 0.9rem 0;
  }
  .guided-fact-label {
    color: #697386;
    font-size: 0.82rem;
    font-weight: 720;
    line-height: 1.3;
  }
  .guided-fact-value {
    color: #202332;
    font-size: 1.15rem;
    font-weight: 780;
    line-height: 1.35;
    margin-top: 0.18rem;
    overflow-wrap: anywhere;
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
    background: #ffffff;
    border-color: #9aa4b2;
  }
  .step-row.complete {
    background: #ffffff;
    border-color: #d5dbe5;
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
    background: #697386;
  }
  .step-row.active .step-number {
    background: #202332;
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
    background: var(--wv-ink) !important;
    border-color: var(--wv-ink) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-weight: 760 !important;
  }
  .stButton > button[kind="primary"]:hover,
  .stButton > button[data-testid="baseButton-primary"]:hover {
    background: #111827 !important;
    border-color: #111827 !important;
    color: #ffffff !important;
  }
  .stButton > button {
    border-radius: 8px !important;
    min-height: 2.7rem;
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
    border-top: 1px solid #edf0f4;
    margin: 0.8rem 0;
    padding: 0.9rem 0 0 0;
  }
  .cost-hero-title {
    color: #202332;
    font-size: 0.9rem;
    font-weight: 760;
    margin-bottom: 0.3rem;
    text-transform: uppercase;
  }
  .cost-hero-grid {
    display: grid;
    gap: 0;
    grid-template-columns: 1fr;
  }
  .cost-hero-label {
    color: #697386;
    font-size: 0.82rem;
    font-weight: 700;
    line-height: 1.35;
    margin-top: 0.75rem;
  }
  .cost-hero-value {
    color: #202332;
    font-size: 1.45rem;
    font-weight: 780;
    line-height: 1.2;
    margin-top: 0.18rem;
    overflow-wrap: anywhere;
  }
  .cost-hero-value.compact {
    font-size: 1.15rem;
    white-space: normal;
  }
  .billing-detail-grid {
    border-top: 1px solid #edf0f4;
    display: grid;
    gap: 0;
    grid-template-columns: 1fr;
    margin-top: 1rem;
    padding-top: 0.2rem;
  }
  .billing-detail-label {
    color: #697386;
    font-size: 0.78rem;
    font-weight: 700;
    line-height: 1.3;
    margin-top: 0.65rem;
  }
  .billing-detail-value {
    color: #202332;
    font-size: 0.88rem;
    font-weight: 720;
    line-height: 1.35;
    margin-top: 0.1rem;
    overflow-wrap: anywhere;
  }
  .billing-note {
    color: #697386;
    font-size: 0.86rem;
    line-height: 1.45;
    margin-top: 0.8rem;
  }
  .summary-card {
    background: #ffffff;
    border-top: 1px solid #edf0f4;
    margin-top: 0.9rem;
    padding: 0.85rem 0 0 0;
  }
  .summary-title {
    color: #202332;
    font-size: 1rem;
    font-weight: 760;
    margin-bottom: 0.25rem;
  }
  .summary-line {
    border-top: 1px solid #edf0f4;
    display: block;
    padding: 0.75rem 0;
  }
  .summary-label {
    color: #697386;
    display: block;
    font-size: 0.82rem;
    font-weight: 700;
    line-height: 1.3;
  }
  .summary-value {
    color: var(--wv-ink);
    display: block;
    font-size: 1rem;
    font-weight: 740;
    line-height: 1.4;
    margin-top: 0.16rem;
    text-align: left;
  }
  .status-strip {
    background: #ffffff;
    border-left: 3px solid #2f6fed;
    color: #313647;
    font-size: 0.84rem;
    line-height: 1.4;
    margin: 0.25rem 0 0.9rem 0;
    padding: 0.25rem 0 0.25rem 0.65rem;
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
    color: var(--wv-ink);
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
    grid-template-columns: minmax(12rem, 1fr) minmax(8rem, 0.65fr) minmax(14rem, 1.25fr);
    padding: 0.82rem;
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
  .prospect-support-title {
    color: #202332;
    font-size: 0.78rem;
    font-weight: 760;
    margin-bottom: 0.22rem;
    text-transform: uppercase;
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
  .route-grid {
    display: grid;
    gap: 0.75rem;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    margin: 0.5rem 0 1rem 0;
  }
  .route-card {
    background: #ffffff;
    border: 1px solid #e5e9ef;
    border-radius: 8px;
    padding: 0.82rem;
  }
  .route-kicker {
    color: #697386;
    font-size: 0.76rem;
    font-weight: 760;
    text-transform: uppercase;
  }
  .route-company {
    color: #202332;
    font-size: 0.96rem;
    font-weight: 760;
    line-height: 1.3;
    margin-top: 0.12rem;
    overflow-wrap: anywhere;
  }
  .route-meta {
    color: #5d6675;
    font-size: 0.82rem;
    line-height: 1.38;
    margin-top: 0.34rem;
    overflow-wrap: anywhere;
  }
  .route-snippet {
    color: #313647;
    font-size: 0.86rem;
    line-height: 1.42;
    margin-top: 0.45rem;
  }
  @media (max-width: 1100px) {
    .summary-grid {
      grid-template-columns: 1fr;
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
    .route-grid {
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


def _format_percent(value: int | float | None) -> str:
    return f"{float(value or 0):.0%}"


def _format_billed_total(spend: ProviderBilledSpend) -> str:
    if spend.is_complete:
        return _format_currency(spend.total_billed_usd)
    return "Unavailable"


def _format_openai_billed(snapshot: OpenAIBillingSnapshot) -> str:
    if snapshot.available and snapshot.is_project_scoped:
        return _format_money(snapshot.cost.total, snapshot.cost.currency)
    return "Unavailable"


def _cache_status(snapshot: OpenAIBillingSnapshot) -> str:
    if not snapshot.available:
        return "unavailable"
    return "cached" if snapshot.from_cache else "fresh"


def _billing_status(snapshot: OpenAIBillingSnapshot) -> str:
    if snapshot.available and snapshot.is_project_scoped:
        return "Live project data"
    if snapshot.available:
        return "Org-level only"
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


def _signal_pills(tags: list[str] | str | None, empty_label: str = "None") -> str:
    if isinstance(tags, str):
        tags = _json_list(tags)
    tags = tags or []
    if not tags:
        return f"<span class='wv-pill'>{html.escape(empty_label)}</span>"
    return "".join(f"<span class='wv-pill'>{html.escape(_clean_ui_text(str(tag)))}</span>" for tag in tags[:5])


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
    for column in [
        "sector_tags",
        "deterministic_tags",
        "top_titles",
        "top_urls",
        "top_snippets",
        "homepage_positive_signals",
        "homepage_negative_signals",
    ]:
        if column in frame.columns:
            frame[column] = frame[column].apply(_json_list)

    frame["sector_tags_text"] = frame.get("sector_tags", pd.Series(dtype=object)).apply(lambda tags: ", ".join(tags or []))
    frame["homepage_positive_signals_text"] = frame.get("homepage_positive_signals", pd.Series(dtype=object)).apply(lambda tags: ", ".join(tags or []))
    frame["homepage_negative_signals_text"] = frame.get("homepage_negative_signals", pd.Series(dtype=object)).apply(lambda tags: ", ".join(tags or []))
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
    frame["evidence_source"] = frame.get("evidence_source", pd.Series(dtype=object)).fillna("none")
    frame["evidence_source_display"] = frame["evidence_source"].apply(_humanize)
    frame["homepage_route_decision"] = frame.get("homepage_route_decision", pd.Series(dtype=object)).fillna("")
    frame["homepage_route_reason"] = frame.get("homepage_route_reason", pd.Series(dtype=object)).fillna("")
    frame["homepage_domain_status"] = frame.get("homepage_domain_status", pd.Series(dtype=object)).fillna("")
    frame["homepage_domain_confidence"] = pd.to_numeric(
        frame.get("homepage_domain_confidence", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0.0)
    frame["homepage_evidence_quality"] = pd.to_numeric(
        frame.get("homepage_evidence_quality", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0.0)
    frame["homepage_evidence_text"] = frame.get("homepage_evidence_text", pd.Series(dtype=object)).fillna("")
    frame["homepage_fetch_error"] = frame.get("homepage_fetch_error", pd.Series(dtype=object)).fillna("")
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
    frame["support_preview"] = frame["homepage_evidence_text"].apply(lambda value: _clean_ui_text(_truncate(value, 170)))
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


def _render_next_step(label: str, title: str, copy: str) -> None:
    st.markdown(
        "<div class='next-step-card'>"
        f"<div class='next-step-label'>{html.escape(_clean_ui_text(label))}</div>"
        f"<div class='next-step-title'>{html.escape(_clean_ui_text(title))}</div>"
        f"<div class='next-step-copy'>{html.escape(_clean_ui_text(copy))}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_mini_metrics(items: list[tuple[str, str]]) -> None:
    body = ["<div class='mini-metric-grid'>"]
    for label, value in items:
        body.append(
            "<div class='mini-metric'>"
            f"<div class='mini-metric-label'>{html.escape(_clean_ui_text(label))}</div>"
            f"<div class='mini-metric-value'>{html.escape(_clean_ui_text(value))}</div>"
            "</div>"
        )
    body.append("</div>")
    st.markdown("".join(body), unsafe_allow_html=True)


def _render_slide_progress(index: int, total: int) -> None:
    pct = ((index + 1) / max(1, total)) * 100
    st.markdown(
        "<div class='slide-progress'>"
        "<div class='slide-progress-track'>"
        f"<span class='slide-progress-fill' style='width:{pct:.1f}%'></span>"
        "</div>"
        f"<div class='slide-progress-text'>{index + 1} / {total}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_slide_header(label: str, title: str, copy: str) -> None:
    st.markdown(
        "<div class='slide-shell'>"
        f"<div class='slide-label'>{html.escape(_clean_ui_text(label))}</div>"
        f"<div class='slide-title'>{html.escape(_clean_ui_text(title))}</div>"
        f"<div class='slide-copy'>{html.escape(_clean_ui_text(copy))}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_flow_steps(items: list[tuple[str, str]]) -> None:
    body = ["<div class='flow-grid'>"]
    for index, (title, copy) in enumerate(items, start=1):
        body.append(
            "<div class='flow-step'>"
            f"<div class='flow-number'>STEP {index}</div>"
            f"<div class='flow-title'>{html.escape(_clean_ui_text(title))}</div>"
            f"<div class='flow-copy'>{html.escape(_clean_ui_text(copy))}</div>"
            "</div>"
        )
    body.append("</div>")
    st.markdown("".join(body), unsafe_allow_html=True)


def _route_examples_html(examples: list[dict], empty_message: str = "No cached examples yet.") -> str:
    if not examples:
        return f"<div class='empty-state'>{html.escape(empty_message)}</div>"
    body = ["<div class='route-grid'>"]
    for example in examples:
        positives = _json_list(example.get("positive_signals"))
        negatives = _json_list(example.get("negative_signals"))
        domain = example.get("resolved_url") or example.get("candidate_domain") or "No resolved domain"
        confidence = float(example.get("domain_confidence") or 0.0)
        quality = float(example.get("evidence_quality") or 0.0)
        body.append(
            "<div class='route-card'>"
            f"<div class='route-kicker'>{html.escape(_clean_ui_text(_humanize(example.get('route_decision') or 'unrouted')))}</div>"
            f"<div class='route-company'>{html.escape(_clean_ui_text(example.get('canonical_name')))}</div>"
            f"<div class='route-meta'>Domain: {html.escape(_clean_ui_text(domain))}</div>"
            f"<div class='route-meta'>Match strength: {_format_percent(confidence)} domain, {_format_percent(quality)} homepage metadata</div>"
            f"<div class='route-meta'>Positive: {_signal_pills(positives, 'None')}</div>"
            f"<div class='route-meta'>Negative: {_signal_pills(negatives, 'None')}</div>"
            f"<div class='route-snippet'>{html.escape(_clean_ui_text(_truncate(example.get('evidence_text') or example.get('fetch_error') or 'No homepage metadata available.', 220)))}</div>"
            f"<div class='route-meta'>Status note: {html.escape(_clean_ui_text(example.get('route_reason') or 'No status note recorded.'))}</div>"
            "</div>"
        )
    body.append("</div>")
    return "".join(body)


def _render_route_examples(title: str, examples: list[dict], empty_message: str = "No cached examples yet.") -> None:
    st.markdown(f"<div class='section-label'>{html.escape(_clean_ui_text(title))}</div>", unsafe_allow_html=True)
    st.markdown(_route_examples_html(examples, empty_message), unsafe_allow_html=True)


def _audit_sample_frame(samples: list[dict]) -> pd.DataFrame:
    if not samples:
        return pd.DataFrame()
    frame = pd.DataFrame(samples)
    for column in ["positive_signals", "negative_signals"]:
        if column in frame.columns:
            frame[column] = frame[column].apply(lambda value: ", ".join(_json_list(value)))
    frame["domain_confidence"] = pd.to_numeric(frame.get("domain_confidence", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    frame["domain_confidence_display"] = frame["domain_confidence"].apply(_format_percent)
    frame["evidence_text"] = frame.get("evidence_text", pd.Series(dtype=object)).fillna("")
    frame["evidence_snippet"] = frame["evidence_text"].apply(lambda value: _clean_ui_text(_truncate(value, 140)))
    return frame


def _render_cost_hero(
    *,
    provider_spend: ProviderBilledSpend,
    lifetime_openai_billing: OpenAIBillingSnapshot,
    recent_openai_billing: OpenAIBillingSnapshot,
    tavily_billing: TavilyBillingSummary,
    local_openai_estimate: float,
    total_tokens: int,
    openai_calls: int,
    last_api_calls: int,
    last_run_local_openai_estimate: float,
    model_name: str,
) -> None:
    cost_items = [
        ("All-time billed cost", _format_billed_total(provider_spend)),
        (f"OpenAI billed cost, {recent_openai_billing.window_label}", _format_openai_billed(recent_openai_billing)),
        ("Internal token-rate estimate", _format_currency(local_openai_estimate)),
    ]
    usage_items = [
        ("OpenAI tokens used", _format_int(total_tokens)),
        (
            "Tavily credits used",
            f"{_format_int(tavily_billing.credits_used)} of {_format_int(tavily_billing.included_monthly_credits)} included",
        ),
        ("OpenAI scoring calls", _format_int(openai_calls)),
        ("Last run API calls", _format_int(last_api_calls)),
        ("Last run token-rate estimate", _format_currency(last_run_local_openai_estimate)),
    ]
    detail_items = [
        ("Live billing source", lifetime_openai_billing.source_label),
        ("OpenAI project", lifetime_openai_billing.project_id or "None"),
        ("Billing start", lifetime_openai_billing.window_start_label or "Unavailable"),
        ("Last fetched", lifetime_openai_billing.fetched_at_label or "Unavailable"),
        ("Data status", _cache_status(lifetime_openai_billing)),
        ("Tavily plan", f"{tavily_billing.plan_name}, pay-as-you-go {'on' if tavily_billing.pay_as_you_go_enabled else 'off'}"),
        ("Streamlit Cloud hosting", _format_currency(provider_spend.hosting_billed_usd)),
    ]

    body = ["<div class='cost-hero'><div class='cost-hero-title'>API usage and cost</div><div class='cost-hero-grid'>"]
    for label, value in [*cost_items, *usage_items]:
        value_class = "cost-hero-value compact" if len(str(value)) > 18 else "cost-hero-value"
        body.append(
            "<div>"
            f"<div class='cost-hero-label'>{html.escape(_clean_ui_text(label))}</div>"
            f"<div class='{value_class}'>{html.escape(_clean_ui_text(value))}</div>"
            "</div>"
        )
    body.append("</div>")
    body.append("<div class='billing-detail-grid'>")
    for label, value in detail_items:
        body.append(
            "<div>"
            f"<div class='billing-detail-label'>{html.escape(_clean_ui_text(label))}</div>"
            f"<div class='billing-detail-value'>{html.escape(_clean_ui_text(value))}</div>"
            "</div>"
        )
    body.append("</div>")
    if (
        lifetime_openai_billing.available
        and lifetime_openai_billing.is_project_scoped
        and lifetime_openai_billing.cost.currency == "usd"
        and lifetime_openai_billing.cost.total == 0
        and local_openai_estimate > 0
    ):
        body.append(
            "<div class='billing-note'>Billed cost comes from live OpenAI billing data. "
            f"The internal estimate uses configured token rates for {html.escape(_clean_ui_text(model_name))} and may differ from platform billing.</div>"
        )
    if not lifetime_openai_billing.available:
        body.append(
            "<div class='billing-note'>Live OpenAI billing unavailable. Showing the internal token-rate estimate instead.</div>"
        )
    elif not lifetime_openai_billing.is_project_scoped:
        body.append(
            "<div class='billing-note'>OpenAI returned organization-level billing. The all-time billed cost above excludes org-level OpenAI spend.</div>"
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
            "Load names, merge duplicates, and remove excluded company types.",
            "Done" if stage > 1 else "Active",
        ),
        _step_row(
            2,
            "Search and score",
            "Check homepage metadata first. Run search and scoring after approval.",
            "Done" if stage > 2 else "Active" if stage == 2 else "Locked",
        ),
        _step_row(
            3,
            "Review results",
            "Review scored companies and exports.",
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


def _setting_int(settings, name: str, default: int) -> int:
    try:
        return int(_setting(settings, name, default) or default)
    except (TypeError, ValueError):
        return default


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


def _estimate_verify_run(conn, metrics: dict, settings, cap: int, mode: str) -> dict:
    homepage_candidates = db.candidates_for_homepage_evidence(
        conn,
        limit=min(cap, _setting_int(settings, "homepage_evidence_max_per_run", 100)),
        mode=mode,
        force=False,
    )
    homepage_summary = db.homepage_evidence_summary(conn, mode=mode)
    tavily_candidates = db.candidates_for_enrichment(conn, limit=cap, force=False, mode=mode)
    currently_scoreable = db.enriched_for_openai_scoring(conn, limit=cap, force=False, mode=mode)
    projected_tavily_calls = len(tavily_candidates)
    projected_openai_calls = min(cap, len(currently_scoreable) + projected_tavily_calls)
    high_signal_candidates = sum(1 for candidate in tavily_candidates if int(candidate.get("high_priority_enrichment") or 0) == 1)
    likely_tech_candidates = sum(1 for candidate in tavily_candidates if candidate.get("deterministic_type") == "likely_startup_or_tech")
    ambiguous_candidates = sum(1 for candidate in tavily_candidates if candidate.get("deterministic_type") == "unknown_needs_enrichment")
    other_likely_tech_candidates = max(0, likely_tech_candidates - high_signal_candidates)

    totals = metrics.get("run_totals") or {}
    historical_openai_calls = max(1, int(totals.get("openai_calls") or 0))
    avg_prompt_tokens = int(totals.get("prompt_tokens") or 0) / historical_openai_calls
    avg_completion_tokens = int(totals.get("completion_tokens") or 0) / historical_openai_calls
    if int(totals.get("openai_calls") or 0) <= 0:
        avg_prompt_tokens = 1_000
        avg_completion_tokens = 250

    projected_prompt_tokens = int(projected_openai_calls * avg_prompt_tokens)
    projected_completion_tokens = int(projected_openai_calls * avg_completion_tokens)
    projected_openai_estimate = (
        (projected_prompt_tokens / 1_000_000) * float(settings.openai_input_cost_per_1m_tokens or 0)
        + (projected_completion_tokens / 1_000_000) * float(settings.openai_output_cost_per_1m_tokens or 0)
    )

    existing_tavily_credits = int(totals.get("tavily_calls") or 0)
    included_tavily_credits = _setting(settings, "tavily_included_monthly_credits", 1000)
    tavily_payg_price = _setting(settings, "tavily_payg_price_per_credit_usd", 0.008)
    tavily_before = calculate_tavily_billing(
        credits_used=existing_tavily_credits,
        included_monthly_credits=included_tavily_credits,
        pay_as_you_go_enabled=_setting(settings, "tavily_pay_as_you_go_enabled", False),
        payg_price_per_credit_usd=tavily_payg_price,
        plan_name=_setting(settings, "tavily_plan_name", "Researcher"),
        shadow_price_per_credit_usd=_setting(settings, "tavily_cost_per_call_usd", 0.001),
    )
    tavily_after = calculate_tavily_billing(
        credits_used=existing_tavily_credits + projected_tavily_calls,
        included_monthly_credits=included_tavily_credits,
        pay_as_you_go_enabled=_setting(settings, "tavily_pay_as_you_go_enabled", False),
        payg_price_per_credit_usd=tavily_payg_price,
        plan_name=_setting(settings, "tavily_plan_name", "Researcher"),
        shadow_price_per_credit_usd=_setting(settings, "tavily_cost_per_call_usd", 0.001),
    )
    projected_tavily_bill = max(0.0, tavily_after.actual_billed_usd - tavily_before.actual_billed_usd)
    projected_tavily_overage = max(0, int(tavily_after.overage_credits) - int(tavily_before.overage_credits))
    projected_tavily_payg_if_enabled = projected_tavily_overage * max(0.0, float(tavily_payg_price or 0.0))
    tavily_calls_avoided = int(homepage_summary.get("score_from_homepage", 0))
    estimated_tavily_credits_saved = tavily_calls_avoided
    estimated_tavily_cost_saved = tavily_calls_avoided * float(_setting(settings, "tavily_cost_per_call_usd", 0.001) or 0.0)
    projected_total = projected_tavily_bill + projected_openai_estimate

    tavily_workers = max(1, min(_setting_int(settings, "tavily_concurrency", 12), max(1, projected_tavily_calls)))
    openai_workers = max(1, min(_setting_int(settings, "openai_concurrency", 6), max(1, projected_openai_calls)))
    tavily_seconds = projected_tavily_calls / tavily_workers * 3.0 if projected_tavily_calls else 0.0
    openai_seconds = projected_openai_calls / openai_workers * 4.0 if projected_openai_calls else 0.0
    estimated_seconds = int(tavily_seconds + openai_seconds)

    return {
        "cap": cap,
        "mode": mode,
        "projected_homepage_candidates": len(homepage_candidates),
        "api_eligible_companies": homepage_summary.get("api_eligible", 0),
        "homepage_attempted": homepage_summary.get("homepage_attempted", 0),
        "accepted_domains": homepage_summary.get("accepted_domains", 0),
        "provisional_domains": homepage_summary.get("provisional_domains", 0),
        "unresolved_domains": homepage_summary.get("unresolved_domains", 0),
        "cached_resolved_domains": homepage_summary.get("resolved_domains", 0),
        "cached_homepage_ready": homepage_summary.get("score_from_homepage", 0),
        "cached_tavily_needed": homepage_summary.get("needs_tavily", 0),
        "cached_tavily_skipped": homepage_summary.get("score_from_homepage", 0),
        "cached_homepage_data_gaps": homepage_summary.get("data_gaps", 0),
        "cached_homepage_soft_excluded": homepage_summary.get("soft_excluded", 0),
        "tavily_call_avoided_by_homepage_evidence": tavily_calls_avoided,
        "estimated_tavily_credits_saved": estimated_tavily_credits_saved,
        "estimated_tavily_cost_saved": estimated_tavily_cost_saved,
        "projected_tavily_calls": projected_tavily_calls,
        "projected_openai_calls": projected_openai_calls,
        "high_signal_candidates": high_signal_candidates,
        "other_likely_tech_candidates": other_likely_tech_candidates,
        "ambiguous_candidates": ambiguous_candidates,
        "broad_candidate_universe": db.count_candidate_universe(conn, mode=mode),
        "unique_company_universe": int(metrics.get("unique_companies") or 0),
        "projected_prompt_tokens": projected_prompt_tokens,
        "projected_completion_tokens": projected_completion_tokens,
        "projected_openai_estimate": projected_openai_estimate,
        "projected_tavily_bill": projected_tavily_bill,
        "projected_tavily_overage": projected_tavily_overage,
        "projected_tavily_payg_if_enabled": projected_tavily_payg_if_enabled,
        "projected_total": projected_total,
        "tavily_credits_after": tavily_after.credits_used,
        "tavily_free_credits_remaining_after": tavily_after.free_credits_remaining,
        "tavily_payg_enabled": tavily_after.pay_as_you_go_enabled,
        "estimated_seconds": estimated_seconds,
        "tavily_workers": tavily_workers,
        "openai_workers": openai_workers,
    }


def _format_duration(seconds: int) -> str:
    if seconds <= 0:
        return "under 1 minute"
    minutes = max(1, round(seconds / 60))
    return f"about {minutes} minute" if minutes == 1 else f"about {minutes} minutes"


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
        evidence_source = _humanize(row.get("evidence_source") or "none")
        source_url = row.get("homepage_resolved_url") or row.get("website") or row.get("primary_source_url") or ""
        snippets = row.get("top_snippets") or []
        support_snippet = row.get("support_preview") or (snippets[0] if snippets else "")
        support_snippet = _clean_ui_text(_truncate(support_snippet or row.get("evidence_summary") or "No source text available.", 160))
        evidence_confidence = (
            _format_percent(float(row.get("homepage_evidence_quality") or 0.0))
            if float(row.get("homepage_evidence_quality") or 0.0) > 0
            else _humanize(row.get("confidence") or "low")
        )
        body.append(
            "<div class='prospect-card'>"
            "<div class='prospect-main'>"
            f"<div class='prospect-rank'>Rank {rank}</div>"
            f"<div class='prospect-name'>{html.escape(_clean_ui_text(row.get('canonical_name')))}</div>"
            f"<div class='route-meta'>Source type: {html.escape(evidence_source)}</div>"
            f"<div class='route-meta'>Source URL: {_safe_link(str(source_url or ''), 'Open') if source_url else 'None'}</div>"
            "</div>"
            "<div class='prospect-score'>"
            "<div class='prospect-score-label'>Total score</div>"
            "<div class='score-cell'>"
            "<div class='score-track'>"
            f"<span class='score-fill' style='width:{max(0, min(100, score))}%'></span>"
            "</div>"
            f"<span class='score-number'>{score}</span>"
            "</div>"
            f"<div class='prospect-tags'>{_tag_pills(row.get('sector_tags'))}</div>"
            "</div>"
            "<div class='prospect-evidence'>"
            "<div class='prospect-support-title'>Source text</div>"
            f"{html.escape(support_snippet)}"
            f"<div class='route-meta'>Positive: {_signal_pills(row.get('homepage_positive_signals'), 'Not captured')}</div>"
            f"<div class='route-meta'>Negative: {_signal_pills(row.get('homepage_negative_signals'), 'Not captured')}</div>"
            f"<div class='route-meta'>Source data confidence: {html.escape(_clean_ui_text(evidence_confidence))}</div>"
            f"<div class='route-meta'>Status note: {html.escape(_clean_ui_text(row.get('homepage_route_reason') or row.get('evidence_summary') or 'No data-gap reason recorded.'))}</div>"
            "</div>"
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
        "<div class='workflow-title'>2. Scored companies</div>"
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
                f"{_format_int(metrics.get('openai_scored'))} companies scored"
                if has_verified
                else "Search and score a capped batch"
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
source_rows_shown = 30
simple_demo_view = True
demo_safe_mode = False
weights = dict(DEFAULT_WEIGHTS)
workflow_stage = _workflow_stage(metrics)

selected_model = st.session_state.get("selected_openai_model", settings.openai_model)
if selected_model not in OPENAI_MODEL_PRESETS:
    selected_model = settings.openai_model if settings.openai_model in OPENAI_MODEL_PRESETS else "gpt-4o-mini"
verify_mode = st.session_state.get("verify_mode", "balanced")
if verify_mode not in VERIFY_MODE_PRESETS:
    verify_mode = "balanced"
model_pricing = _selected_model_pricing(selected_model)
prospect_cap = int(st.session_state.get("prospect_cap", min(settings.max_score, 100)))
runtime_settings = _settings_for_run(settings, selected_model, model_pricing["input"], model_pricing["output"])

if st.session_state.pop("load_source_requested", False):
    load_result = _run_and_store("Loading attendee names...", lambda: load_attendees(conn, settings))
    classify_result = _run_and_store("Classifying source data...", lambda: run_deterministic_classification(conn))
    st.session_state["last_action"] = {
        "message": f"{load_result.message} {classify_result.message}",
        "level": "success",
    }
    frame, metrics = _load_frame_and_metrics(conn)
    st.rerun()

if st.session_state.pop("start_paid_run_requested", False):
    cap = int(prospect_cap)
    active_verify_mode = st.session_state.get("active_verify_mode", verify_mode)
    progress = st.progress(0, text=f"Refreshing source screening before verifying up to {cap:,} companies...")
    preview = st.empty()
    classify_result = run_deterministic_classification(conn)
    progress.progress(0.05, text="Reading bounded homepage metadata before paid search...")

    homepage_result = collect_homepage_evidence(
        conn,
        runtime_settings,
        cap,
        False,
        mode=active_verify_mode,
    )

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
        if total and index < total and index % 5 != 0:
            return
        latest_frame, _latest_metrics = _load_frame_and_metrics(conn)
        latest_weighted = _apply_weighted_scores(latest_frame, weights)
        latest_prospects = latest_weighted[latest_weighted["is_refined_prospect"]].head(8)
        preview.markdown(
            _prospect_cards_html(
                latest_prospects,
                "Scored companies will appear here as scoring completes.",
            ),
            unsafe_allow_html=True,
        )

    enrich_result = enrich_candidates(
        conn,
        runtime_settings,
        cap,
        False,
        progress_callback=enrichment_progress,
        mode=active_verify_mode,
    )
    score_result = score_enriched_candidates(
        conn,
        runtime_settings,
        cap,
        False,
        progress_callback=scoring_progress,
        mode=active_verify_mode,
    )
    progress.progress(1.0, text="Verification run finished.")
    level = "warning" if enrich_result.counts.get("errors") or score_result.counts.get("errors") else "success"
    run_openai_estimate = float(score_result.counts.get("estimated_cost_usd") or 0)
    run_tavily_credits = int(enrich_result.counts.get("tavily_credits_used") or enrich_result.counts.get("tavily_calls") or 0)
    run_tavily_billed = float(enrich_result.counts.get("tavily_actual_billed_usd") or 0)
    st.session_state["last_action"] = {
        "message": (
            f"Scored {score_result.counts.get('scored', 0):,} companies from a {cap:,}-company {VERIFY_MODE_PRESETS.get(active_verify_mode, VERIFY_MODE_PRESETS['balanced'])['label'].lower()} batch. "
            f"Local OpenAI token estimate: {_format_currency(run_openai_estimate)}. "
            f"Tavily credits consumed: {run_tavily_credits:,}; Tavily billed spend: {_format_currency(run_tavily_billed)}."
        ),
        "level": level,
    }
    st.session_state.pop("active_verify_mode", None)
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
billing_project_id = _setting(settings, "openai_billing_project_id", "proj_ynS2F3GVOCBbgmXvTl9Vl1Ie")
billing_start_date = _setting(settings, "openai_billing_start_date", "2026-05-31")
billing_lookback_days = _setting_int(settings, "openai_billing_lookback_days", 30)
billing_cache_ttl_seconds = _setting_int(settings, "openai_billing_cache_ttl_seconds", 300)
lifetime_openai_billing = fetch_openai_billing_snapshot(
    admin_key=_setting(settings, "openai_admin_key", None),
    project_id=billing_project_id,
    start_date=billing_start_date,
    lookback_days=billing_lookback_days,
    window_label="Wittington project lifetime to date",
    cache_ttl_seconds=billing_cache_ttl_seconds,
)
recent_openai_billing = fetch_openai_billing_snapshot(
    admin_key=_setting(settings, "openai_admin_key", None),
    project_id=billing_project_id,
    lookback_days=billing_lookback_days,
    window_label=f"last {billing_lookback_days} days",
    cache_ttl_seconds=billing_cache_ttl_seconds,
)
provider_spend = calculate_provider_billed_spend(
    tavily_billing=tavily_billing,
    openai_billing=lifetime_openai_billing,
    hosting_billed_usd=0.0,
)
local_openai_estimate = _local_openai_estimate_usd(metrics, runtime_settings)
last_run_local_openai_estimate = _last_run_local_openai_estimate_usd(last_run)

cascade_summary = db.homepage_evidence_summary(conn, mode=verify_mode) if not frame.empty else {}
tavily_avoided = int(cascade_summary.get("score_from_homepage") or 0)

pending_verify_run = _estimate_verify_run(conn, metrics, runtime_settings, int(prospect_cap), verify_mode) if workflow_stage >= 2 else None
pending_mode = pending_verify_run.get("mode") if pending_verify_run else verify_mode
broad_universe_pending = int((pending_verify_run or {}).get("broad_candidate_universe") or metrics.get("candidates") or 0)
unique_universe_pending = int((pending_verify_run or {}).get("unique_company_universe") or metrics.get("unique_companies") or 0)
projected_tavily_overage = int((pending_verify_run or {}).get("projected_tavily_overage") or 0)
tavily_payg_enabled_pending = bool((pending_verify_run or {}).get("tavily_payg_enabled"))
projected_rows = []
cascade_rows = []
if pending_verify_run:
    projected_rows = [
        ("Companies in this run", _format_int(pending_verify_run["cap"])),
        ("Eligible companies", f"{_format_int(broad_universe_pending)} of {_format_int(unique_universe_pending)} unique names"),
        ("Search calls", _format_int(pending_verify_run["projected_tavily_calls"])),
        ("Scoring calls", _format_int(pending_verify_run["projected_openai_calls"])),
        ("OpenAI tokens", f"{_format_int(pending_verify_run['projected_prompt_tokens'])} input, {_format_int(pending_verify_run['projected_completion_tokens'])} output"),
        ("Estimated provider cost", _format_currency(pending_verify_run["projected_total"])),
        ("Estimated run time", _format_duration(int(pending_verify_run["estimated_seconds"]))),
    ]
    if projected_tavily_overage > 0 and not tavily_payg_enabled_pending:
        projected_rows.extend(
            [
                ("Search credits over included plan", f"{_format_int(projected_tavily_overage)} credits, pay-as-you-go off"),
                ("Search overage if enabled", _format_currency(float(pending_verify_run.get("projected_tavily_payg_if_enabled") or 0.0))),
            ]
        )
    cascade_rows = [
        ("API-eligible companies", _format_int(pending_verify_run.get("api_eligible_companies") or broad_universe_pending)),
        ("Homepages checked", _format_int(pending_verify_run.get("homepage_attempted") or 0)),
        ("Metadata enough", _format_int(pending_verify_run.get("cached_homepage_ready") or 0)),
        ("Ready from homepage", _format_int(pending_verify_run.get("cached_tavily_skipped") or 0)),
        ("Queued for search", _format_int(pending_verify_run.get("cached_tavily_needed") or 0)),
        ("Missing data", _format_int(pending_verify_run.get("cached_homepage_data_gaps") or 0)),
    ]

slides = [
    {"key": "overview", "label": "Overview", "title": "Manifest list to scored companies.", "copy": "Load company names, remove excluded categories, check company pages, run web search when page data is incomplete, then score the remaining companies."},
    {"key": "source", "label": "Step 1", "title": "Prepare the Manifest list.", "copy": "Load attendee company names, merge duplicates, and remove rows that match excluded categories."},
    {"key": "homepage", "label": "Step 2", "title": "Check company pages.", "copy": "Read domains, page titles, descriptions, headings, and short homepage text before web search."},
    {"key": "estimate", "label": "Step 3", "title": "Approve the run.", "copy": "Review the batch size, search volume, scoring volume, runtime, tokens, and estimated cost."},
    {"key": "cost", "label": "Step 4", "title": "Track spend.", "copy": "View provider billing, included Tavily credits, and local token estimates separately."},
    {"key": "prospects", "label": "Step 5", "title": "Review scored companies.", "copy": "Sort companies by total score and inspect the page text or search results used for scoring."},
    {"key": "routing", "label": "Step 6", "title": "Review company data status.", "copy": "See which companies have usable page data and which companies need web search."},
]
slide_count = len(slides)
slide_index = int(st.session_state.get("slide_index", 0))
slide_index = max(0, min(slide_index, slide_count - 1))
st.session_state["slide_index"] = slide_index
slide = slides[slide_index]

st.markdown(
    """
    <div class="wv-header">
      <h1 class="wv-title">Manifest Prospecting Tool</h1>
      <p class="wv-subtitle">Scores Manifest attendee companies for Wittington Ventures.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
_render_slide_progress(slide_index, slide_count)
_render_slide_header(slide["label"], slide["title"], slide["copy"])

if slide["key"] == "overview":
    _render_flow_steps(
        [
            ("Source", "Load and deduplicate Manifest companies."),
            ("Screen", "Remove incumbents, investors, associations, service firms, and noisy rows."),
            ("Company page", "Read domains, titles, descriptions, and snippets."),
            ("Web search", "Use Tavily when company page data is incomplete."),
            ("Score", "Score each company against Wittington criteria."),
            ("Review", "Show scored companies and spend."),
        ],
    )
elif slide["key"] == "source":
    _render_mini_metrics(
        [
            ("Raw rows", _format_int(metrics["raw_companies"])),
            ("Unique names", _format_int(metrics["unique_companies"])),
            ("API-eligible", _format_int(candidate_count)),
            ("Provider cost", "$0.0000"),
        ]
    )
    _render_summary_card(
        "First screen",
        [
            ("Input", "Public Manifest attendee list"),
            ("Cleaned list", "Names normalized and duplicates merged"),
            ("Removed", "Incumbents, investors, associations, consulting firms, agencies, service providers, and noisy rows"),
            ("Remaining rows", f"{_format_int(candidate_count)} companies ready for page checks and search"),
        ],
    )
    if workflow_stage == 1 and st.button("Load and screen Manifest list", type="primary", use_container_width=True):
        st.session_state["load_source_requested"] = True
        st.rerun()
elif slide["key"] == "homepage":
    _render_mini_metrics(
        [
            ("Company pages checked", _format_int(cascade_summary.get("homepage_attempted") or 0)),
            ("Page data used for scoring", _format_int(cascade_summary.get("score_from_homepage") or 0)),
            ("Queued for search", _format_int(cascade_summary.get("needs_tavily") or 0)),
            ("Missing data", _format_int(cascade_summary.get("data_gaps") or 0)),
        ]
    )
    _render_summary_card(
        "Page check",
        [
            ("Reads", "Domains, page titles, descriptions, headings, and snippets"),
            ("Page data used for scoring", f"{_format_int(tavily_avoided)} companies"),
            ("Next", "Companies with incomplete page data move to web search"),
        ],
    )
    if pending_verify_run and st.button("Check one company page", type="primary", use_container_width=True):
        with st.spinner("Checking company page metadata..."):
            run_deterministic_classification(conn)
            preview_cap = max(1, min(int(pending_verify_run["cap"]), _setting_int(runtime_settings, "homepage_preview_max_per_click", 1)))
            preview_settings = replace(
                runtime_settings,
                homepage_evidence_max_per_run=preview_cap,
                homepage_fetch_timeout_seconds=min(float(_setting(runtime_settings, "homepage_fetch_timeout_seconds", 4.0) or 4.0), 1.0),
            )
            preview_result = collect_homepage_evidence(
                conn,
                preview_settings,
                preview_cap,
                False,
                mode=pending_mode,
                max_domain_attempts=1,
            )
        st.session_state["last_action"] = {
            "message": f"Checked {_format_int(preview_result.counts.get('processed'))} company page. Search and scoring were not run.",
            "level": "success",
        }
        st.rerun()
elif slide["key"] == "estimate":
    if pending_verify_run:
        _render_mini_metrics(
            [
                ("Companies", _format_int(pending_verify_run["cap"])),
                ("Search calls", _format_int(pending_verify_run["projected_tavily_calls"])),
                ("Scoring calls", _format_int(pending_verify_run["projected_openai_calls"])),
                ("Estimated cost", _format_currency(pending_verify_run["projected_total"])),
            ]
        )
        _render_summary_card("Run estimate", projected_rows)
        _render_summary_card("Company page status", cascade_rows)
        if st.button("Run search and scoring", type="primary", use_container_width=True):
            st.session_state["active_verify_mode"] = pending_mode
            st.session_state["start_paid_run_requested"] = True
            st.rerun()
    else:
        st.warning("Load the Manifest list before estimating search and scoring.")
elif slide["key"] == "cost":
    _render_mini_metrics(
        [
            ("Actual billed cost", _format_billed_total(provider_spend)),
            ("OpenAI tokens", _format_int(int(run_totals.get("total_tokens") or 0))),
            ("Search credits used", _format_int(tavily_billing.credits_used)),
            ("Scored from company pages", _format_int(tavily_avoided)),
        ]
    )
    _render_cost_hero(
        provider_spend=provider_spend,
        lifetime_openai_billing=lifetime_openai_billing,
        recent_openai_billing=recent_openai_billing,
        tavily_billing=tavily_billing,
        local_openai_estimate=local_openai_estimate,
        total_tokens=int(run_totals.get("total_tokens") or 0),
        openai_calls=int(run_totals.get("openai_calls") or 0),
        last_api_calls=last_api_calls,
        last_run_local_openai_estimate=last_run_local_openai_estimate,
        model_name=str(runtime_settings.openai_model),
    )
elif slide["key"] == "prospects":
    _render_prospect_cards(
        prospects.head(5),
        "No scored companies yet. Run search and scoring first.",
    )
elif slide["key"] == "routing":
    _render_summary_card(
        "Company data status",
        [
            ("API-eligible companies", _format_int(cascade_summary.get("api_eligible") or candidate_count)),
            ("Company pages checked", _format_int(cascade_summary.get("homepage_attempted") or 0)),
            ("Page data used for scoring", _format_int(cascade_summary.get("score_from_homepage") or 0)),
            ("Queued for web search", _format_int(cascade_summary.get("needs_tavily") or 0)),
            ("Missing data", _format_int(cascade_summary.get("data_gaps") or 0)),
        ],
    )

nav_cols = st.columns((1, 1, 1))
if nav_cols[0].button("Previous", disabled=slide_index == 0, use_container_width=True):
    st.session_state["slide_index"] = max(0, slide_index - 1)
    st.rerun()
nav_cols[1].markdown(
    f"<div class='slide-nav-note'>{slide_index + 1} / {slide_count}</div>",
    unsafe_allow_html=True,
)
if nav_cols[2].button("Next", disabled=slide_index >= slide_count - 1, type="primary", use_container_width=True):
    st.session_state["slide_index"] = min(slide_count - 1, slide_index + 1)
    st.rerun()

st.stop()
