from pathlib import Path

from app.config import DATA_DIR
from app.ingestion.loaders import load_directory


def load_documents(data_type: str):
    """
    Load raw documents from `data/<data_type>/`.

    Supported:
    - TXT, MD, PDF
    - CSV
    - JSON, JSONL/NDJSON
    """
    data_path = DATA_DIR / data_type
    docs = load_directory(data_path)
    for d in docs:
        d.setdefault("metadata", {})
        d["metadata"]["type"] = data_type
    return docs


def load_all():
    docs = []
    for data_type in ["jobs", "interview_questions", "technical_docs"]:
        docs.extend(load_documents(data_type))
    return docs
