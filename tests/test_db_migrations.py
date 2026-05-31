from pathlib import Path

from src import db
from src.pipeline import run_deterministic_classification


def test_init_db_migrates_old_companies_table_with_high_priority_column(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "old-prospects.db")
    timestamp = db.now_iso()
    conn.executescript(
        """
        CREATE TABLE companies (
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
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        INSERT INTO companies (
            raw_name, normalized_name, canonical_name, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        ("Pickle Robot", "pickle robot", "Pickle Robot", timestamp, timestamp),
    )
    conn.commit()

    db.init_db(conn)

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(companies)")}
    assert "high_priority_enrichment" in columns
    rows = db.dashboard_rows(conn)
    assert rows[0]["high_priority_enrichment"] == 0

    result = run_deterministic_classification(conn)
    rows = db.dashboard_rows(conn)

    assert result.counts["high_priority_queue"] == 1
    assert rows[0]["high_priority_enrichment"] == 1
