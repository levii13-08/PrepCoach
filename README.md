# PrepCoach - Agentic Interview Preparation Coach

RAG-grounded mock interviews for ML, AI, and SWE roles.  
Canonical project root: `agentic-interview-coach/`.

## Phase status

| Phase | Status |
|-------|--------|
| 0 - Architecture and schemas | Complete |
| 1 - Data collection and validation | Implemented |
| 2 - Core RAG engine | Implemented |
| 3 - Production retrieval | Implemented |
| 4 - Question generation | Implemented |
| 5 - Multi-agent interview loop | Implemented |
| 6 - Evaluation engine | Implemented |
| 7 - Adaptive interview intelligence | Implemented |
| 8 - Skill gap analysis | Implemented |
| 9 - Coaching roadmap | Implemented |
| 10 - Persistent memory | Implemented |
| 11 - Deployment surfaces | Planned |

## Documentation

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | System design, stack, diagrams |
| [docs/agents.md](docs/agents.md) | Planner, Retriever, Interviewer, Evaluator, Coach |
| [docs/data_schema.md](docs/data_schema.md) | Jobs, questions, technical KB |
| [docs/state_schema.md](docs/state_schema.md) | `InterviewState` for LangGraph |

JSON Schema contracts: `schemas/`  
Pydantic models: `app/schemas/`

## Quick start

```bash
cd agentic-interview-coach
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python scripts\prepare_light_data.py
python scripts\validate_data.py --profile light
python scripts\ingest_dev.py --rebuild
python scripts\run_interview.py --dataset light --mock-llm --rounds 3
```

## Configuration

Copy `.env.example` to `.env`. Key variables:

- `GROQ_API_KEY` - required for real LLM-backed generation and evaluation
- `LLM_MODEL` - default `llama-3.3-70b-versatile`
- `RAG_USE_HYBRID`, `RAG_USE_MULTI_QUERY`, `RAG_USE_RERANK` - default `false` for the Phase 2 baseline
- `LIGHT_DATASET_JOBS` / `LIGHT_DATASET_QUESTIONS` - caps for the lightweight first ingestion path

## Data flow

- `scripts/filter_jobs.py` builds schema-compliant jobs JSONL.
- `scripts/normalize_questions.py` builds schema-compliant interview questions JSONL.
- `scripts/build_docs.py` builds `technical_docs.jsonl` and `data/technical_docs/manifest.jsonl`.
- `scripts/prepare_light_data.py` builds a smaller `data/processed_dev/` dataset for fast first indexing.
- `scripts/validate_data.py` validates either the `full` or `light` profile.

## Current scope

Phases 0 through 10 are implemented in code.  
Evaluation harnesses and benchmark suites are intentionally left for a later pass.  
Phase 11 deployment surfaces are still pending.
