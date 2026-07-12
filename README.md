# PrepCoach — Agentic Interview Preparation Coach

RAG-grounded mock interviews for **ML, AI, Data Science, and SWE** roles. PrepCoach retrieves context from curated job descriptions, interview questions, and technical documentation, then runs an adaptive multi-agent interview loop that evaluates your answers, identifies skill gaps, and produces a personalized learning roadmap.

**Canonical project root:** `agentic-interview-coach/`

---

## What it does

PrepCoach is not a generic chatbot interviewer. It combines retrieval-augmented generation with a **LangGraph multi-agent workflow** so every question, evaluation, and coaching recommendation is grounded in real source material.

| Capability | Description |
|------------|-------------|
| **Grounded interviews** | Questions and follow-ups cite retrieved jobs, Q&A pairs, and technical docs |
| **Adaptive difficulty** | Planner adjusts topic and difficulty based on rubric scores each round |
| **Structured evaluation** | Evaluator scores correctness, completeness, clarity, and role alignment |
| **Skill gap analysis** | Compares your performance signals against a target job description |
| **Coaching roadmap** | Generates prioritized learning steps with evidence and RAG-backed resources |
| **Session persistence** | SQLite-backed sessions you can resume from CLI, API, or UI |

### Agent workflow

```
Planner → Retriever → Interviewer → [your answer] → Evaluator → Planner …
                                                              ↓
                                                    Coach (on completion)
```

Five agents share state via `InterviewState`: **Planner** (routing), **Retriever** (RAG), **Interviewer** (question generation), **Evaluator** (rubric scoring), and **Coach** (roadmap). See [docs/agents.md](docs/agents.md) for full definitions.

---

## Tech stack

| Layer | Technology |
|-------|------------|
| LLM | [Groq API](https://groq.com/) — `llama-3.3-70b-versatile` (default) |
| Embeddings | `BAAI/bge-base-en-v1.5` via `sentence-transformers` |
| Vector store | Chroma (3 collections: jobs, questions, technical docs) |
| Agents | LangGraph + LangChain |
| Retrieval | Dense search; optional hybrid BM25, multi-query, cross-encoder rerank |
| API | FastAPI |
| UI | Streamlit |
| Memory | SQLite |

---

## Prerequisites

- **Python 3.10+**
- **Groq API key** — required for real LLM generation and evaluation ([get one free](https://console.groq.com/))
- ~2 GB disk for embeddings model cache on first ingest (Hugging Face download)

Mock mode works without a valid Groq key for smoke testing and local development.

---

## Quick start

### 1. Install

```bash
cd agentic-interview-coach
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # Windows: copy .env.example .env
```

Add your Groq key to `.env` when you want real LLM-backed interviews:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Choose a dataset profile

| Profile | Path | Jobs | Questions | Use case |
|---------|------|------|-----------|----------|
| `light` | `data/processed_dev/` | 250 | 1,200 | Fast local dev and smoke tests |
| `demo` | `data/processed_demo/` | 600 | 2,200 | **Default** — curated for demos and deployment |
| `full` | `data/processed/` | Full corpus | Full corpus | Broad evaluation and research |

Build and validate (pick one profile):

```bash
# Recommended for first run — demo profile
python scripts/prepare_demo_data.py
python scripts/validate_data.py --profile demo
python scripts/ingest_demo.py --rebuild

# Or lightweight dev profile
python scripts/prepare_light_data.py
python scripts/validate_data.py --profile light
python scripts/ingest_dev.py --rebuild
```

### 3. Run an interview

**CLI** (mock mode — no API key needed):

```bash
python scripts/run_interview.py --dataset demo --mock-llm --rounds 3 --auto-answer
```

**CLI** (real Groq LLM):

```bash
python scripts/run_interview.py --dataset demo --no-mock-llm --rounds 3 --role MLE
```

**Streamlit UI:**

```bash
streamlit run app/ui/app.py
```

**FastAPI:**

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for the interactive API reference.

---

## Usage examples

### CLI options

```bash
python scripts/run_interview.py \
  --role MLE \
  --difficulty medium \
  --topic ml \
  --rounds 5 \
  --dataset demo \
  --target-job-id <normalized_job_id> \
  --no-mock-llm
```

Supported roles: `MLE`, `AI`, `DS`, `SWE`, `GENAI`  
Supported topics: `ml`, `deep_learning`, `mlops`, `dsa`, `system_design`, `llms`, `rag`, `agents`, and more — see `app/schemas/data_models.py`.

Resume a saved session:

```bash
python scripts/run_interview.py --resume-latest --user-id alice
python scripts/run_interview.py --resume-session <session_id>
```

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/interviews/start` | Start a new interview session |
| `POST` | `/interviews/{session_id}/answer` | Submit an answer for the current round |
| `GET` | `/interviews/{session_id}` | Fetch session state |
| `GET` | `/interviews/latest` | Fetch latest session (optional `user_id`) |
| `GET` | `/interviews` | List saved sessions |

Example start request:

```json
{
  "role": "MLE",
  "max_rounds": 3,
  "difficulty": "medium",
  "dataset": "demo",
  "mock_llm": true
}
```

---

## Configuration

Copy `.env.example` to `.env`. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Required for real LLM mode |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Primary chat model |
| `LLM_MODEL_FALLBACK` | `qwen-qwq-32b` | Fallback model |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Local embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `700` / `150` | RAG chunking |
| `RAG_USE_HYBRID` | `false` in code defaults; `true` in `.env.example` | BM25 + dense fusion |
| `RAG_USE_MULTI_QUERY` | same | Query expansion |
| `RAG_USE_RERANK` | same | Cross-encoder reranking |
| `RAG_TOP_K` / `RAG_RETRIEVE_K` | `5` / `20` | Retrieval depth |
| `DEFAULT_MAX_ROUNDS` | `5` | Default session length |
| `MEMORY_DB_PATH` | `memory.sqlite3` | SQLite session store |

Enable the full Phase 3 retrieval stack by setting all three `RAG_USE_*` flags to `true` in `.env`.

---

## Project structure

```
agentic-interview-coach/
├── app/
│   ├── agents/          # LangGraph interview graph + planner
│   ├── api/             # FastAPI routes
│   ├── coaching/        # Skill gap analysis + learning roadmap
│   ├── evaluation/      # Rubric-based answer scoring
│   ├── generation/      # Structured question generation
│   ├── ingestion/       # Document loaders + ingest pipeline
│   ├── llm/             # Groq client helpers
│   ├── memory/          # SQLite session persistence
│   ├── rag/             # Chunking, Chroma, hybrid retrieval, reranker
│   ├── schemas/         # Pydantic models
│   ├── services/        # Shared orchestration (CLI / API / UI)
│   └── ui/              # Streamlit app
├── data/
│   ├── processed/       # Full normalized dataset
│   ├── processed_dev/   # Light dev dataset
│   ├── processed_demo/  # Curated demo dataset
│   └── technical_docs/  # Source technical content
├── docs/                # Architecture, agents, schemas, implementation plan
├── schemas/             # JSON Schema contracts
├── scripts/             # Data prep, ingest, validate, CLI runner
├── tests/               # Unit tests (planner, evaluator, coach, etc.)
└── vectorstore/         # Chroma persist directory (gitignored)
```

---

## Data pipeline

Raw sources are normalized into schema-compliant JSONL, validated, then indexed into Chroma:

```
filter_jobs.py / normalize_questions.py / build_docs.py
        ↓
prepare_light_data.py  OR  prepare_demo_data.py  OR  data/processed/
        ↓
validate_data.py --profile {light|demo|full}
        ↓
ingest_dev.py  OR  ingest_demo.py  OR  ingest.py --rebuild
        ↓
vectorstore/  (3 Chroma collections)
```

Before deploying or demoing, rebuild the vectorstore from the demo profile:

```bash
python scripts/ingest_demo.py --rebuild
```

---

## Development

Run unit tests:

```bash
python -m pytest tests/
```

Validate a dataset profile:

```bash
python scripts/validate_data.py --profile demo
```

Script reference: [scripts/README.md](scripts/README.md)

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | System design, stack, diagrams |
| [docs/agents.md](docs/agents.md) | Planner, Retriever, Interviewer, Evaluator, Coach |
| [docs/data_schema.md](docs/data_schema.md) | Jobs, questions, technical KB schemas |
| [docs/state_schema.md](docs/state_schema.md) | `InterviewState` for LangGraph |
| [docs/implementation_plan.md](docs/implementation_plan.md) | Phased build plan and current status |

JSON Schema contracts live in `schemas/`; Pydantic mirrors in `app/schemas/`.

---

## Known limitations

- **No auth or billing** — single-user local/demo use
- **Describe-approach mode** — no live coding sandbox
- **Groq dependency** — real mode requires a valid API key; rate limits apply
- **First ingest is slow** — embedding model download and indexing take time
- **Stale vectorstore** — re-run `ingest_demo.py --rebuild` after updating demo data

---

## License

See repository root for license information.
