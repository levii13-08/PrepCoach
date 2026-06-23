"""Create a curated deployment-friendly dataset for resume demos."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

from app.config import DEMO_DATASET_JOBS, DEMO_DATASET_QUESTIONS, PROCESSED_DEMO_DIR
from scripts.build_docs import DEFAULT_INPUT_DIR, DEFAULT_MANIFEST, build_docs
from scripts.filter_jobs import DEFAULT_INPUT as JOBS_INPUT
from scripts.filter_jobs import row_to_record, title_matches
from scripts.normalize_questions import DEFAULT_DS_AI_INPUT, DEFAULT_SWE_INPUT, normalize_ds_ai, normalize_swe, write_jsonl


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_demo_jobs(limit: int) -> list[dict]:
    import csv

    per_role_target = max(60, limit // 5)
    buckets: dict[str, list[dict]] = defaultdict(list)
    overflow: list[dict] = []

    with JOBS_INPUT.open("r", encoding="utf-8", errors="ignore", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if not title_matches(row.get("title", "")):
                continue
            record = row_to_record(row)
            requirements = (record.get("requirements") or "").strip()
            if not record.get("id") or len(requirements) < 120 or requirements.startswith("http"):
                continue
            role = str(record.get("role") or "SWE")
            if len(buckets[role]) < per_role_target:
                buckets[role].append(record)
            else:
                overflow.append(record)

    selected: list[dict] = []
    for role_records in buckets.values():
        selected.extend(role_records)

    seen_ids = {record["id"] for record in selected}
    for record in overflow:
        if len(selected) >= limit:
            break
        if record["id"] in seen_ids:
            continue
        selected.append(record)
        seen_ids.add(record["id"])

    return selected[:limit]


def _build_demo_questions(limit: int) -> list[dict]:
    questions = normalize_swe(DEFAULT_SWE_INPUT) + normalize_ds_ai(DEFAULT_DS_AI_INPUT)
    topic_targets = {
        "swe": 450,
        "ml": 700,
        "rag": 180,
        "system_design": 180,
        "dsa": 180,
        "oop": 120,
        "agents": 120,
        "llms": 120,
        "ai": 150,
    }
    buckets: dict[str, list[dict]] = defaultdict(list)
    overflow: list[dict] = []

    for question in questions:
        topic = str(question.get("topic") or "swe").lower()
        topic_limit = topic_targets.get(topic, 120)
        if len(buckets[topic]) < topic_limit:
            buckets[topic].append(question)
        else:
            overflow.append(question)

    selected: list[dict] = []
    for topic_records in buckets.values():
        selected.extend(topic_records)

    seen_ids = {record["id"] for record in selected}
    for record in overflow:
        if len(selected) >= limit:
            break
        if record["id"] in seen_ids:
            continue
        selected.append(record)
        seen_ids.add(record["id"])

    return selected[:limit]


def main() -> None:
    PROCESSED_DEMO_DIR.mkdir(parents=True, exist_ok=True)

    jobs = _build_demo_jobs(DEMO_DATASET_JOBS)
    _write_jsonl(jobs, PROCESSED_DEMO_DIR / "jobs.jsonl")
    print(f"Demo jobs: wrote {len(jobs)}")

    questions = _build_demo_questions(DEMO_DATASET_QUESTIONS)
    write_jsonl(questions, PROCESSED_DEMO_DIR / "interview_questions.jsonl")
    print(f"Demo questions: wrote {len(questions)}")

    doc_count = build_docs(DEFAULT_INPUT_DIR, PROCESSED_DEMO_DIR / "technical_docs.jsonl", DEFAULT_MANIFEST)
    print(f"Demo technical docs: wrote {doc_count}")


if __name__ == "__main__":
    main()
