"""Central configuration — aligns with docs/architecture.md (Phase 0)."""

import os
from pathlib import Path

from dotenv import load_dotenv

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DEV_DIR = DATA_DIR / "processed_dev"
PROCESSED_DEMO_DIR = DATA_DIR / "processed_demo"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
MEMORY_DB_PATH = Path(os.getenv("MEMORY_DB_PATH", PROJECT_ROOT / "memory.sqlite3"))

# --- LLM (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_MODEL_FALLBACK = os.getenv("LLM_MODEL_FALLBACK", "qwen-qwq-32b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_TEMPERATURE_EVAL = float(os.getenv("LLM_TEMPERATURE_EVAL", "0"))

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "128"))
HF_TOKEN = os.getenv("HF_TOKEN", "")

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Chroma collections ---
COLLECTION_JOBS = "job_descriptions"
COLLECTION_QUESTIONS = "interview_questions"
COLLECTION_TECH_DOCS = "technical_docs"
CHROMA_COLLECTIONS = (COLLECTION_JOBS, COLLECTION_QUESTIONS, COLLECTION_TECH_DOCS)

# --- RAG feature flags (Phase 2 baseline defaults OFF; Phase 3 can enable) ---
RAG_USE_HYBRID = os.getenv("RAG_USE_HYBRID", "false").lower() == "true"
RAG_USE_MULTI_QUERY = os.getenv("RAG_USE_MULTI_QUERY", "false").lower() == "true"
RAG_USE_RERANK = os.getenv("RAG_USE_RERANK", "false").lower() == "true"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_RETRIEVE_K = int(os.getenv("RAG_RETRIEVE_K", "20"))

# --- Data profiles ---
LIGHT_DATASET_JOBS = int(os.getenv("LIGHT_DATASET_JOBS", "250"))
LIGHT_DATASET_QUESTIONS = int(os.getenv("LIGHT_DATASET_QUESTIONS", "1200"))
DEMO_DATASET_JOBS = int(os.getenv("DEMO_DATASET_JOBS", "600"))
DEMO_DATASET_QUESTIONS = int(os.getenv("DEMO_DATASET_QUESTIONS", "2200"))

# --- Interview defaults ---
DEFAULT_MAX_ROUNDS = int(os.getenv("DEFAULT_MAX_ROUNDS", "5"))
DEFAULT_DIFFICULTY = os.getenv("DEFAULT_DIFFICULTY", "medium")

# --- Evaluation rubric weights ---
RUBRIC_WEIGHTS = {
    "correctness": 0.40,
    "completeness": 0.25,
    "clarity": 0.15,
    "alignment": 0.20,
}
