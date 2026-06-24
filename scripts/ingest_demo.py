import argparse
from typing import Dict, List

import _bootstrap  # noqa: F401

from app.config import (
    COLLECTION_JOBS,
    COLLECTION_QUESTIONS,
    COLLECTION_TECH_DOCS,
    PROCESSED_DEMO_DIR,
)
from app.ingestion.ingest_pipeline import IngestTarget, ingest_target


TARGETS: List[IngestTarget] = [
    IngestTarget(PROCESSED_DEMO_DIR / "jobs.jsonl", COLLECTION_JOBS, "jobs"),
    IngestTarget(PROCESSED_DEMO_DIR / "interview_questions.jsonl", COLLECTION_QUESTIONS, "interview_questions"),
    IngestTarget(PROCESSED_DEMO_DIR / "technical_docs.jsonl", COLLECTION_TECH_DOCS, "technical_docs"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma collections from data/processed_demo/.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing docs in collections before indexing.",
    )
    args = parser.parse_args()

    totals: Dict[str, int] = {}
    for target in TARGETS:
        totals[target.collection] = ingest_target(target, rebuild=args.rebuild)

    print("Demo ingestion complete:")
    for name, count in totals.items():
        print(f"- {name}: {count} chunks")


if __name__ == "__main__":
    main()
