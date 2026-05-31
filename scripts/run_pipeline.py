from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import db  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.pipeline import (  # noqa: E402
    enrich_candidates,
    load_attendees,
    run_default_pipeline,
    run_deterministic_classification,
    score_enriched_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Manifest prospecting pipeline.")
    parser.add_argument("--all", action="store_true", help="Run load, deterministic, Tavily, and OpenAI stages.")
    parser.add_argument("--load", action="store_true", help="Scrape/load attendee names.")
    parser.add_argument("--classify", action="store_true", help="Run deterministic classification and baseline scores.")
    parser.add_argument("--enrich", action="store_true", help="Run Tavily enrichment for candidate companies.")
    parser.add_argument("--score", action="store_true", help="Run OpenAI scoring for enriched candidates.")
    parser.add_argument("--max-enrich", type=int, default=None, help="Maximum Tavily calls for this run.")
    parser.add_argument("--max-score", type=int, default=None, help="Maximum OpenAI calls for this run.")
    parser.add_argument("--force", action="store_true", help="Refresh cached Tavily/OpenAI records for selected rows.")
    args = parser.parse_args()

    settings = get_settings()
    conn = db.connect(settings.database_path)
    db.init_db(conn)

    if args.all or not any([args.load, args.classify, args.enrich, args.score]):
        results = run_default_pipeline(conn, settings)
    else:
        results = []
        if args.load:
            results.append(load_attendees(conn, settings))
        if args.classify:
            results.append(run_deterministic_classification(conn))
        if args.enrich:
            results.append(enrich_candidates(conn, settings, args.max_enrich, args.force))
        if args.score:
            results.append(score_enriched_candidates(conn, settings, args.max_score, args.force))

    for result in results:
        print(f"[{result.stage}] {result.message}")
        print(result.counts)

    print("Metrics:", db.metrics(conn))


if __name__ == "__main__":
    main()
