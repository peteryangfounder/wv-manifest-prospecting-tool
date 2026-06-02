from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .clean import CompanyRecord


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY,
            raw_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL UNIQUE,
            canonical_name TEXT NOT NULL,
            is_duplicate INTEGER DEFAULT 0,
            duplicate_of TEXT,
            duplicate_count INTEGER DEFAULT 1,
            deterministic_type TEXT,
            deterministic_exclusion_reason TEXT,
            deterministic_tags TEXT DEFAULT '[]',
            is_candidate INTEGER DEFAULT 0,
            high_priority_enrichment INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS enrichments (
            id INTEGER PRIMARY KEY,
            company_id INTEGER NOT NULL,
            query TEXT NOT NULL,
            provider TEXT NOT NULL,
            raw_json TEXT,
            top_titles TEXT,
            top_urls TEXT,
            top_snippets TEXT,
            website TEXT,
            enriched_at TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            UNIQUE(company_id, provider),
            FOREIGN KEY(company_id) REFERENCES companies(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY,
            company_id INTEGER NOT NULL UNIQUE,
            provider TEXT NOT NULL DEFAULT 'baseline',
            company_type TEXT,
            is_startup_likely INTEGER,
            sector_tags TEXT,
            wv_sector_fit INTEGER,
            venture_backability INTEGER,
            wittington_edge INTEGER,
            stage_signal INTEGER,
            traction_signal INTEGER,
            data_confidence INTEGER,
            total_score INTEGER,
            rationale TEXT,
            evidence_summary TEXT,
            confidence TEXT,
            scored_at TEXT NOT NULL,
            raw_json TEXT,
            FOREIGN KEY(company_id) REFERENCES companies(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY,
            run_type TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            raw_count INTEGER DEFAULT 0,
            unique_count INTEGER DEFAULT 0,
            candidates_count INTEGER DEFAULT 0,
            enriched_count INTEGER DEFAULT 0,
            scored_count INTEGER DEFAULT 0,
            tavily_calls INTEGER DEFAULT 0,
            openai_calls INTEGER DEFAULT 0,
            cache_hits INTEGER DEFAULT 0,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            estimated_cost_usd REAL,
            notes TEXT
        );
        """
    )
    migrate_schema(conn)
    conn.commit()


def migrate_schema(conn: sqlite3.Connection) -> None:
    """Apply additive migrations for SQLite databases created by older app versions."""
    _ensure_column(conn, "companies", "high_priority_enrichment", "INTEGER DEFAULT 0")
    _ensure_column(conn, "runs", "prompt_tokens", "INTEGER DEFAULT 0")
    _ensure_column(conn, "runs", "completion_tokens", "INTEGER DEFAULT 0")
    _ensure_column(conn, "runs", "total_tokens", "INTEGER DEFAULT 0")


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    columns: set[str] = set()
    for row in conn.execute(f"PRAGMA table_info({table})"):
        columns.add(row["name"] if isinstance(row, sqlite3.Row) else row[1])
    return columns


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    if column not in _table_columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def start_run(conn: sqlite3.Connection, run_type: str, notes: str | None = None) -> int:
    cursor = conn.execute(
        "INSERT INTO runs (run_type, started_at, notes) VALUES (?, ?, ?)",
        (run_type, now_iso(), notes),
    )
    conn.commit()
    return int(cursor.lastrowid)


def finish_run(conn: sqlite3.Connection, run_id: int, **values: Any) -> None:
    allowed = {
        "raw_count",
        "unique_count",
        "candidates_count",
        "enriched_count",
        "scored_count",
        "tavily_calls",
        "openai_calls",
        "cache_hits",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "notes",
    }
    assignments = ["finished_at = ?"]
    params: list[Any] = [now_iso()]
    for key, value in values.items():
        if key in allowed:
            assignments.append(f"{key} = ?")
            params.append(value)
    params.append(run_id)
    conn.execute(f"UPDATE runs SET {', '.join(assignments)} WHERE id = ?", params)
    conn.commit()


def upsert_companies(conn: sqlite3.Connection, records: Iterable[CompanyRecord]) -> int:
    timestamp = now_iso()
    count = 0
    for record in records:
        conn.execute(
            """
            INSERT INTO companies (
                raw_name, normalized_name, canonical_name, duplicate_count, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(normalized_name) DO UPDATE SET
                canonical_name = excluded.canonical_name,
                duplicate_count = excluded.duplicate_count,
                updated_at = excluded.updated_at
            """,
            (
                record.raw_name,
                record.normalized_name,
                record.canonical_name,
                record.duplicate_count,
                timestamp,
                timestamp,
            ),
        )
        count += 1
    conn.commit()
    return count


def list_companies(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM companies ORDER BY canonical_name COLLATE NOCASE").fetchall()
    return [dict(row) for row in rows]


def update_deterministic_result(
    conn: sqlite3.Connection,
    company_id: int,
    deterministic_type: str,
    exclusion_reason: str | None,
    tags: list[str],
    is_candidate: bool,
    high_priority_enrichment: bool,
) -> None:
    conn.execute(
        """
        UPDATE companies
        SET deterministic_type = ?,
            deterministic_exclusion_reason = ?,
            deterministic_tags = ?,
            is_candidate = ?,
            high_priority_enrichment = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            deterministic_type,
            exclusion_reason,
            json.dumps(tags),
            1 if is_candidate else 0,
            1 if high_priority_enrichment else 0,
            now_iso(),
            company_id,
        ),
    )


def save_enrichment(conn: sqlite3.Connection, enrichment: dict[str, Any], *, commit: bool = True) -> None:
    conn.execute(
        """
        INSERT INTO enrichments (
            company_id, query, provider, raw_json, top_titles, top_urls, top_snippets,
            website, enriched_at, status, error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(company_id, provider) DO UPDATE SET
            query = excluded.query,
            raw_json = excluded.raw_json,
            top_titles = excluded.top_titles,
            top_urls = excluded.top_urls,
            top_snippets = excluded.top_snippets,
            website = excluded.website,
            enriched_at = excluded.enriched_at,
            status = excluded.status,
            error = excluded.error
        """,
        (
            enrichment["company_id"],
            enrichment["query"],
            enrichment["provider"],
            json.dumps(enrichment.get("raw_json")),
            json.dumps(enrichment.get("top_titles", [])),
            json.dumps(enrichment.get("top_urls", [])),
            json.dumps(enrichment.get("top_snippets", [])),
            enrichment.get("website"),
            now_iso(),
            enrichment["status"],
            enrichment.get("error"),
        ),
    )
    if commit:
        conn.commit()


def save_score(
    conn: sqlite3.Connection,
    company_id: int,
    score: dict[str, Any],
    provider: str,
    *,
    commit: bool = True,
) -> None:
    conn.execute(
        """
        INSERT INTO scores (
            company_id, provider, company_type, is_startup_likely, sector_tags,
            wv_sector_fit, venture_backability, wittington_edge, stage_signal,
            traction_signal, data_confidence, total_score, rationale,
            evidence_summary, confidence, scored_at, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(company_id) DO UPDATE SET
            provider = excluded.provider,
            company_type = excluded.company_type,
            is_startup_likely = excluded.is_startup_likely,
            sector_tags = excluded.sector_tags,
            wv_sector_fit = excluded.wv_sector_fit,
            venture_backability = excluded.venture_backability,
            wittington_edge = excluded.wittington_edge,
            stage_signal = excluded.stage_signal,
            traction_signal = excluded.traction_signal,
            data_confidence = excluded.data_confidence,
            total_score = excluded.total_score,
            rationale = excluded.rationale,
            evidence_summary = excluded.evidence_summary,
            confidence = excluded.confidence,
            scored_at = excluded.scored_at,
            raw_json = excluded.raw_json
        """,
        (
            company_id,
            provider,
            score.get("company_type"),
            score.get("is_startup_likely"),
            json.dumps(score.get("sector_tags", [])),
            score.get("wv_sector_fit"),
            score.get("venture_backability"),
            score.get("wittington_edge"),
            score.get("stage_signal"),
            score.get("traction_signal"),
            score.get("data_confidence"),
            score.get("total_score"),
            score.get("rationale"),
            score.get("evidence_summary"),
            score.get("confidence"),
            now_iso(),
            json.dumps(score.get("raw_json")),
        ),
    )
    if commit:
        conn.commit()


def get_successful_enrichment(conn: sqlite3.Connection, company_id: int, provider: str = "tavily") -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM enrichments WHERE company_id = ? AND provider = ? AND status = 'success'",
        (company_id, provider),
    ).fetchone()
    return dict(row) if row else None


def get_score(conn: sqlite3.Connection, company_id: int, provider: str | None = None) -> dict[str, Any] | None:
    if provider is None:
        row = conn.execute("SELECT * FROM scores WHERE company_id = ?", (company_id,)).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM scores WHERE company_id = ? AND provider = ?",
            (company_id, provider),
        ).fetchone()
    return dict(row) if row else None


def save_baseline_score_if_missing_or_baseline(
    conn: sqlite3.Connection,
    company_id: int,
    score: dict[str, Any],
    *,
    commit: bool = True,
) -> bool:
    existing = get_score(conn, company_id)
    if existing and existing.get("provider") == "openai":
        return False
    save_score(conn, company_id, score, provider="baseline", commit=commit)
    return True


def candidates_for_enrichment(
    conn: sqlite3.Connection,
    limit: int,
    force: bool = False,
    high_priority_only: bool = False,
) -> list[dict[str, Any]]:
    priority_filter = "AND c.high_priority_enrichment = 1" if high_priority_only else ""
    priority_order = """
              CASE
                WHEN c.high_priority_enrichment = 1 THEN 0
                WHEN c.deterministic_type = 'likely_startup_or_tech' THEN 1
                WHEN c.deterministic_type = 'unknown_needs_enrichment' THEN 2
                ELSE 3
              END,
    """
    if force:
        query = f"""
            SELECT c.*
            FROM companies c
            WHERE c.is_candidate = 1 {priority_filter}
            ORDER BY
              {priority_order}
              c.canonical_name COLLATE NOCASE
            LIMIT ?
        """
    else:
        query = f"""
            SELECT c.*
            FROM companies c
            LEFT JOIN enrichments e
              ON e.company_id = c.id AND e.provider = 'tavily' AND e.status = 'success'
            WHERE c.is_candidate = 1 {priority_filter} AND e.id IS NULL
            ORDER BY
              {priority_order}
              c.canonical_name COLLATE NOCASE
            LIMIT ?
        """
    rows = conn.execute(query, (limit,)).fetchall()
    return [dict(row) for row in rows]


def enriched_for_openai_scoring(
    conn: sqlite3.Connection,
    limit: int,
    force: bool = False,
    high_priority_only: bool = False,
) -> list[dict[str, Any]]:
    if force:
        score_filter = ""
    else:
        score_filter = "AND (s.id IS NULL OR s.provider != 'openai')"
    priority_filter = "AND c.high_priority_enrichment = 1" if high_priority_only else ""
    priority_order = """
          CASE
            WHEN c.high_priority_enrichment = 1 THEN 0
            WHEN c.deterministic_type = 'likely_startup_or_tech' THEN 1
            WHEN c.deterministic_type = 'unknown_needs_enrichment' THEN 2
            ELSE 3
          END,
    """

    rows = conn.execute(
        f"""
        SELECT c.*, e.top_titles, e.top_urls, e.top_snippets, e.website, e.raw_json AS enrichment_raw_json
        FROM companies c
        JOIN enrichments e ON e.company_id = c.id AND e.provider = 'tavily' AND e.status = 'success'
        LEFT JOIN scores s ON s.company_id = c.id
        WHERE c.is_candidate = 1 {priority_filter} {score_filter}
        ORDER BY
          {priority_order}
          c.canonical_name COLLATE NOCASE
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def cached_company_for_verification(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT
            c.*,
            e.top_titles,
            e.top_urls,
            e.top_snippets,
            e.website,
            e.raw_json AS enrichment_raw_json,
            s.provider AS score_provider,
            s.total_score AS cached_total_score
        FROM companies c
        JOIN enrichments e
          ON e.company_id = c.id AND e.provider = 'tavily' AND e.status = 'success'
        JOIN scores s
          ON s.company_id = c.id AND s.provider = 'openai'
        ORDER BY COALESCE(s.total_score, 0) DESC, c.canonical_name COLLATE NOCASE
        LIMIT 1
        """
    ).fetchone()
    return dict(row) if row else None


def dashboard_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    migrate_schema(conn)
    rows = conn.execute(
        """
        SELECT
            c.id,
            c.raw_name,
            c.normalized_name,
            c.canonical_name,
            c.duplicate_count,
            c.deterministic_type,
            c.deterministic_exclusion_reason,
            c.deterministic_tags,
            c.is_candidate,
            c.high_priority_enrichment,
            s.provider AS score_provider,
            s.company_type,
            s.is_startup_likely,
            s.sector_tags,
            s.wv_sector_fit,
            s.venture_backability,
            s.wittington_edge,
            s.stage_signal,
            s.traction_signal,
            s.data_confidence,
            s.total_score,
            s.rationale,
            s.evidence_summary,
            s.confidence,
            e.provider AS enrichment_provider,
            e.status AS enrichment_status,
            e.top_titles,
            e.top_urls,
            e.top_snippets,
            e.website,
            e.error AS enrichment_error
        FROM companies c
        LEFT JOIN scores s ON s.company_id = c.id
        LEFT JOIN enrichments e ON e.company_id = c.id AND e.provider = 'tavily'
        ORDER BY COALESCE(s.total_score, 0) DESC, c.canonical_name COLLATE NOCASE
        """
    ).fetchall()
    return [dict(row) for row in rows]


def metrics(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT
          COALESCE(SUM(duplicate_count), 0) AS raw_companies,
          COUNT(*) AS unique_companies,
          SUM(CASE WHEN is_candidate = 1 THEN 1 ELSE 0 END) AS candidates,
          SUM(CASE WHEN high_priority_enrichment = 1 THEN 1 ELSE 0 END) AS high_priority_queue
        FROM companies
        """
    ).fetchone()
    enrichment_count = conn.execute(
        "SELECT COUNT(*) FROM enrichments WHERE provider = 'tavily' AND status = 'success'"
    ).fetchone()[0]
    scored_count = conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    openai_count = conn.execute("SELECT COUNT(*) FROM scores WHERE provider = 'openai'").fetchone()[0]
    last_run = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    run_totals = conn.execute(
        """
        SELECT
          COALESCE(SUM(tavily_calls), 0) AS tavily_calls,
          COALESCE(SUM(openai_calls), 0) AS openai_calls,
          COALESCE(SUM(cache_hits), 0) AS cache_hits,
          COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
          COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
          COALESCE(SUM(total_tokens), 0) AS total_tokens,
          COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
        FROM runs
        """
    ).fetchone()
    cost_by_stage = conn.execute(
        """
        SELECT
          run_type,
          COALESCE(SUM(tavily_calls), 0) AS tavily_calls,
          COALESCE(SUM(openai_calls), 0) AS openai_calls,
          COALESCE(SUM(total_tokens), 0) AS total_tokens,
          COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
        FROM runs
        GROUP BY run_type
        ORDER BY run_type
        """
    ).fetchall()
    return {
        "raw_companies": int(row["raw_companies"] or 0),
        "unique_companies": int(row["unique_companies"] or 0),
        "candidates": int(row["candidates"] or 0),
        "high_priority_queue": int(row["high_priority_queue"] or 0),
        "enriched": int(enrichment_count or 0),
        "scored": int(scored_count or 0),
        "openai_scored": int(openai_count or 0),
        "last_run": dict(last_run) if last_run else None,
        "run_totals": dict(run_totals) if run_totals else {},
        "cost_by_stage": [dict(row) for row in cost_by_stage],
    }
