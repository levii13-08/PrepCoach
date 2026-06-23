"""Validate normalized datasets against schema-level expectations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from app.config import PROCESSED_DEMO_DIR, PROCESSED_DEV_DIR, PROCESSED_DIR
from app.schemas.data_models import InterviewQuestion, JobDescription, TechnicalDocManifest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGETS = {
    "full": {"jobs": 200, "questions": 1000, "topics": 9},
    "light": {"jobs": 25, "questions": 100, "topics": 9},
    "demo": {"jobs": 300, "questions": 1000, "topics": 9},
}


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_dataset(profile: str) -> None:
    if profile == "light":
        base_dir = PROCESSED_DEV_DIR
    elif profile == "demo":
        base_dir = PROCESSED_DEMO_DIR
    else:
        base_dir = PROCESSED_DIR
    targets = TARGETS[profile]

    jobs = [JobDescription.model_validate(row) for row in _load_jsonl(base_dir / "jobs.jsonl")]
    questions = [InterviewQuestion.model_validate(row) for row in _load_jsonl(base_dir / "interview_questions.jsonl")]
    technical_docs = _load_jsonl(base_dir / "technical_docs.jsonl")
    manifest = [
        TechnicalDocManifest.model_validate(row)
        for row in _load_jsonl(PROJECT_ROOT / "data" / "technical_docs" / "manifest.jsonl")
    ]

    if len(jobs) < targets["jobs"]:
        raise SystemExit(f"Expected at least {targets['jobs']} jobs, found {len(jobs)}")
    if len(questions) < targets["questions"]:
        raise SystemExit(f"Expected at least {targets['questions']} questions, found {len(questions)}")

    manifest_topics = {row.topic.value for row in manifest}
    if len(manifest_topics) < targets["topics"]:
        raise SystemExit(f"Expected at least {targets['topics']} technical topics, found {len(manifest_topics)}")

    question_topics = Counter(question.topic.value for question in questions)
    if not question_topics:
        raise SystemExit("No question topics found after normalization")

    if len(technical_docs) != len(manifest):
        raise SystemExit("technical_docs.jsonl and manifest.jsonl counts do not match")

    print(f"Validated profile={profile}")
    print(f"- jobs: {len(jobs)}")
    print(f"- questions: {len(questions)}")
    print(f"- technical topics: {len(manifest_topics)}")
    print(f"- question topics: {len(question_topics)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate normalized Phase 1 datasets.")
    parser.add_argument("--profile", choices=["full", "light", "demo"], default="full")
    args = parser.parse_args()
    validate_dataset(args.profile)


if __name__ == "__main__":
    main()
