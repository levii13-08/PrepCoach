# Data Schema (Phase 0)

All datasets live under `data/`. Ingestion normalizes raw files into these contracts before indexing (Phase 1–2).

JSON Schema files: `schemas/*.json`

---

## 1. Job descriptions

**Path:** `data/jobs/*.json` or `data/jobs/jobs.jsonl`  
**Collection:** `job_descriptions`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique id, e.g. `job_mle_042` |
| `role` | enum | yes | `MLE`, `AI`, `DS`, `SWE`, `GENAI` |
| `role_title` | string | yes | Human-readable title |
| `skills` | string[] | yes | Normalized skill tags |
| `requirements` | string | yes | Full JD text for embedding |
| `seniority` | enum | no | `junior`, `mid`, `senior`, `staff` |
| `company` | string | no | |
| `source` | string | yes | `kaggle`, `github`, `curated`, `normalized` |
| `source_url` | string | no | |
| `collected_at` | date | yes | ISO-8601 date |

**Example**

```json
{
  "id": "job_mle_001",
  "role": "MLE",
  "role_title": "Machine Learning Engineer",
  "skills": ["PyTorch", "RAG", "LangChain", "MLOps"],
  "requirements": "Design and deploy ML pipelines...",
  "seniority": "mid",
  "source": "curated",
  "collected_at": "2026-05-29"
}
```

**Phase 1 target:** 200–500 records, all five roles represented.

---

## 2. Interview questions

**Path:** `data/interview_questions/*.jsonl`  
**Collection:** `interview_questions`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | e.g. `iq_rag_012` |
| `question` | string | yes | Question text |
| `topic` | enum | yes | See topics below |
| `difficulty` | enum | yes | `easy`, `medium`, `hard` |
| `role_tags` | string[] | yes | Subset of role codes |
| `question_type` | enum | yes | `theory`, `coding`, `system_design`, `behavioral`, `ml`, `swe` |
| `expected_points` | string[] | yes | Rubric bullets for evaluator |
| `source` | string | yes | `curated`, `normalized`, `github` |

**Topics:** `ml`, `ml_theory`, `deep_learning`, `mlops`, `swe`, `dsa`, `oop`, `system_design`, `ai`, `llms`, `rag`, `agents`

**Example**

```json
{
  "id": "iq_rag_012",
  "question": "How does hybrid retrieval improve RAG over dense-only search?",
  "topic": "rag",
  "difficulty": "medium",
  "role_tags": ["MLE", "AI", "GENAI"],
  "question_type": "theory",
  "expected_points": [
    "dense semantic search",
    "BM25 lexical match",
    "recall improvement",
    "fusion or reranking"
  ],
  "source": "curated"
}
```

**Phase 1 target:** ≥ 1000 questions across topics.

**Legacy CSV mapping** (`input_text`, `target_text`):

- `question` ← `target_text`
- `expected_points` ← split/summarize from `input_text` (normalization script)
- `topic` ← inferred by classifier or keyword rules during Phase 1

---

## 3. Technical knowledge base

**Path:** `data/technical_docs/*.{txt,md,pdf}`  
**Manifest:** `data/technical_docs/manifest.jsonl` (optional)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `filename` | string | yes | File under `technical_docs/` |
| `topic` | enum | yes | `python`, `sql`, `pytorch`, `transformers`, `rag`, `langgraph`, `system_design`, `mlops`, `vector_databases` |
| `title` | string | yes | Display title |
| `source` | string | yes | e.g. `official_docs`, `github`, `textbook` |
| `license` | string | yes | e.g. `MIT`, `CC-BY-4.0` |
| `source_url` | string | no | |

**Phase 1 target:** ≥ 1 document per topic (9 topics minimum).

---

## 4. Chunk metadata (index time)

Every chunk stored in Chroma inherits:

| Field | Source |
|-------|--------|
| `collection` | `job_descriptions` \| `interview_questions` \| `technical_docs` |
| `doc_id` | Parent record `id` |
| `chunk_index` | 0-based |
| `role` | Jobs / questions |
| `topic` | Questions / technical |
| `difficulty` | Questions |
| `question_type` | Questions |
| `source_file` | Path or filename |

---

## 5. Evaluation output (runtime, Phase 6)

See `schemas/evaluation_score.json` and [state_schema.md](./state_schema.md).

---

## 6. Validation

Phase 1 delivers `scripts/validate_data.py` (planned) to enforce:

- Required fields present
- Enum values valid
- Minimum counts per role/topic
- No empty `requirements` / `question` strings
