"""Create a lightweight first-pass dataset for fast local ingestion."""

import _bootstrap  # noqa: F401

from app.config import LIGHT_DATASET_JOBS, LIGHT_DATASET_QUESTIONS
from scripts.build_docs import DEFAULT_INPUT_DIR, DEFAULT_MANIFEST, build_docs
from scripts.filter_jobs import DEFAULT_DEV_OUTPUT as JOBS_DEV_OUTPUT
from scripts.filter_jobs import DEFAULT_INPUT as JOBS_INPUT
from scripts.filter_jobs import filter_jobs
from scripts.normalize_questions import (
    DEFAULT_DEV_OUTPUT as QUESTIONS_DEV_OUTPUT,
    DEFAULT_DS_AI_INPUT,
    DEFAULT_SWE_INPUT,
    normalize_ds_ai,
    normalize_swe,
    write_jsonl,
)


def main() -> None:
    total, kept = filter_jobs(JOBS_INPUT, JOBS_DEV_OUTPUT, limit=LIGHT_DATASET_JOBS)
    print(f"Light jobs: kept {kept} from {total}")

    records = normalize_swe(DEFAULT_SWE_INPUT) + normalize_ds_ai(DEFAULT_DS_AI_INPUT)
    write_jsonl(records, QUESTIONS_DEV_OUTPUT, limit=LIGHT_DATASET_QUESTIONS)
    print(f"Light questions: wrote {min(len(records), LIGHT_DATASET_QUESTIONS)}")

    tech_output = QUESTIONS_DEV_OUTPUT.parent / "technical_docs.jsonl"
    count = build_docs(DEFAULT_INPUT_DIR, tech_output, DEFAULT_MANIFEST)
    print(f"Light technical docs: wrote {count}")


if __name__ == "__main__":
    main()
