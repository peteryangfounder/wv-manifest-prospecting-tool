from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATABASE_PATH = DATA_DIR / "prospects.db"
MANIFEST_URL = "https://manife.st/who-attends/"


WV_CRITERIA = """
Wittington Ventures invests in technology-oriented startups, primarily around
Series A/B but with flexibility earlier or later. Focus areas are commerce,
healthcare, consumer, and climate. Their strategic edge is helping startups
accelerate adoption through relationships with large Canadian brands including
Loblaw, Shoppers Drug Mart, PC Financial, PC Optimum, Choice Properties,
Holt Renfrew, Joe Fresh, and Lifemark. Strong prospects are venture-backable
technology companies where Wittington could plausibly help with validation,
commercial introductions, retail/healthcare/consumer distribution, supply-chain
adoption, or enterprise feedback.
""".strip()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@dataclass(frozen=True)
class Settings:
    database_path: Path
    manifest_url: str
    openai_api_key: str | None
    tavily_api_key: str | None
    openai_model: str
    max_enrich: int
    max_score: int
    tavily_max_results: int
    openai_input_cost_per_1m_tokens: float
    openai_output_cost_per_1m_tokens: float
    tavily_cost_per_call_usd: float
    tavily_concurrency: int = 12
    openai_concurrency: int = 6
    homepage_concurrency: int = 96
    db_commit_batch_size: int = 25
    provider_max_retries: int = 4
    provider_backoff_initial_seconds: float = 1.0
    provider_backoff_max_seconds: float = 20.0
    homepage_evidence_max_per_run: int = 100
    homepage_preview_max_per_click: int = 1
    homepage_fetch_timeout_seconds: float = 1.25
    homepage_fetch_max_bytes: int = 100_000
    homepage_max_domain_attempts: int = 3
    tavily_plan_name: str = "Researcher"
    tavily_included_monthly_credits: int = 1000
    tavily_pay_as_you_go_enabled: bool = False
    tavily_payg_price_per_credit_usd: float = 0.008
    openai_admin_key: str | None = None
    openai_billing_project_id: str | None = "proj_ynS2F3GVOCBbgmXvTl9Vl1Ie"
    openai_billing_start_date: str = "2026-05-31"
    openai_billing_lookback_days: int = 30
    openai_billing_cache_ttl_seconds: int = 300
    demo_safe_mode_default: bool = True


def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings(
        database_path=_env_path("DATABASE_PATH", DEFAULT_DATABASE_PATH),
        manifest_url=os.getenv("MANIFEST_URL", MANIFEST_URL),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        max_enrich=_env_int("MAX_ENRICH", 75),
        max_score=_env_int("MAX_SCORE", 75),
        tavily_concurrency=_env_int("TAVILY_CONCURRENCY", 12),
        openai_concurrency=_env_int("OPENAI_CONCURRENCY", 6),
        homepage_concurrency=_env_int("HOMEPAGE_CONCURRENCY", 96),
        db_commit_batch_size=_env_int("DB_COMMIT_BATCH_SIZE", 25),
        provider_max_retries=_env_int("PROVIDER_MAX_RETRIES", 4),
        provider_backoff_initial_seconds=_env_float("PROVIDER_BACKOFF_INITIAL_SECONDS", 1.0),
        provider_backoff_max_seconds=_env_float("PROVIDER_BACKOFF_MAX_SECONDS", 20.0),
        homepage_evidence_max_per_run=_env_int("HOMEPAGE_EVIDENCE_MAX_PER_RUN", 100),
        homepage_preview_max_per_click=_env_int("HOMEPAGE_PREVIEW_MAX_PER_CLICK", 1),
        homepage_fetch_timeout_seconds=_env_float("HOMEPAGE_FETCH_TIMEOUT_SECONDS", 1.25),
        homepage_fetch_max_bytes=_env_int("HOMEPAGE_FETCH_MAX_BYTES", 100_000),
        homepage_max_domain_attempts=_env_int("HOMEPAGE_MAX_DOMAIN_ATTEMPTS", 3),
        tavily_max_results=_env_int("TAVILY_MAX_RESULTS", 3),
        openai_input_cost_per_1m_tokens=_env_float("OPENAI_INPUT_COST_PER_1M_TOKENS", 0.15),
        openai_output_cost_per_1m_tokens=_env_float("OPENAI_OUTPUT_COST_PER_1M_TOKENS", 0.60),
        tavily_cost_per_call_usd=_env_float("TAVILY_COST_PER_CALL_USD", 0.001),
        tavily_plan_name=os.getenv("TAVILY_PLAN_NAME", "Researcher"),
        tavily_included_monthly_credits=_env_int("TAVILY_INCLUDED_MONTHLY_CREDITS", 1000),
        tavily_pay_as_you_go_enabled=_env_bool("TAVILY_PAY_AS_YOU_GO_ENABLED", False),
        tavily_payg_price_per_credit_usd=_env_float("TAVILY_PAYG_PRICE_PER_CREDIT_USD", 0.008),
        openai_admin_key=os.getenv("OPENAI_ADMIN_KEY") or None,
        openai_billing_project_id=os.getenv("OPENAI_BILLING_PROJECT_ID") or "proj_ynS2F3GVOCBbgmXvTl9Vl1Ie",
        openai_billing_start_date=os.getenv("OPENAI_BILLING_START_DATE", "2026-05-31"),
        openai_billing_lookback_days=_env_int("OPENAI_BILLING_LOOKBACK_DAYS", 30),
        openai_billing_cache_ttl_seconds=_env_int("OPENAI_BILLING_CACHE_TTL_SECONDS", 300),
        demo_safe_mode_default=_env_bool("DEMO_SAFE_MODE_DEFAULT", True),
    )
