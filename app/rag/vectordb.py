import warnings
from typing import Dict, List, Optional

import chromadb
from langchain_community.vectorstores import Chroma

from app.config import (
    COLLECTION_JOBS,
    COLLECTION_QUESTIONS,
    COLLECTION_TECH_DOCS,
    INGEST_BATCH_SIZE,
    VECTORSTORE_DIR,
)
from app.rag.embeddings import embedding_model

warnings.filterwarnings(
    "ignore",
    message=r"The class `Chroma` was deprecated in LangChain",
)


def _chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(VECTORSTORE_DIR))


def create_collection(
    chunked_docs: List[Dict],
    *,
    collection_name: str,
) -> Chroma:
    if not chunked_docs:
        return load_collection(collection_name)

    texts = [d["content"] for d in chunked_docs]
    metadatas = [d.get("metadata", {}) for d in chunked_docs]

    vectorstore: Optional[Chroma] = None
    for start in range(0, len(texts), INGEST_BATCH_SIZE):
        batch_texts = texts[start : start + INGEST_BATCH_SIZE]
        batch_metas = metadatas[start : start + INGEST_BATCH_SIZE]
        if vectorstore is None:
            vectorstore = Chroma.from_texts(
                texts=batch_texts,
                embedding=embedding_model,
                metadatas=batch_metas,
                persist_directory=str(VECTORSTORE_DIR),
                collection_name=collection_name,
            )
        else:
            vectorstore.add_texts(texts=batch_texts, metadatas=batch_metas)

    return vectorstore or load_collection(collection_name)


def load_collection(collection_name: str) -> Chroma:
    return Chroma(
        embedding_function=embedding_model,
        persist_directory=str(VECTORSTORE_DIR),
        collection_name=collection_name,
    )


def reset_collection(collection_name: str) -> None:
    """Drop collection without loading the embedding model."""
    client = _chroma_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


def resolve_collection_for_type(data_type: str) -> Optional[str]:
    if data_type == "jobs":
        return COLLECTION_JOBS
    if data_type == "interview_questions":
        return COLLECTION_QUESTIONS
    if data_type == "technical_docs":
        return COLLECTION_TECH_DOCS
    return None
