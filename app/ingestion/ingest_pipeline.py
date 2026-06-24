"""Shared ingestion steps used by ingest.py and ingest_dev.py."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from app.ingestion.loaders import load_path
from app.rag.chunking import chunk_documents
from app.rag.vectordb import create_collection, reset_collection


@dataclass(frozen=True)
class IngestTarget:
    data_file: Path
    collection: str
    data_type: str


def ingest_target(target: IngestTarget, *, rebuild: bool) -> int:
    if not target.data_file.exists():
        raise FileNotFoundError(
            f"Missing processed data: {target.data_file}. "
            "Run filter_jobs.py, normalize_questions.py, and build_docs.py first."
        )

    t0 = time.perf_counter()
    raw_docs = load_path(target.data_file)
    raw_docs = [d for d in raw_docs if d.get("content", "").strip()]
    for doc in raw_docs:
        doc.setdefault("metadata", {})
        doc["metadata"]["type"] = target.data_type
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    chunked = chunk_documents(raw_docs)
    chunk_s = time.perf_counter() - t1

    if rebuild:
        t2 = time.perf_counter()
        reset_collection(target.collection)
        reset_s = time.perf_counter() - t2
    else:
        reset_s = 0.0

    t3 = time.perf_counter()
    create_collection(chunked, collection_name=target.collection)
    index_s = time.perf_counter() - t3

    print(
        f"  {target.collection}: {len(chunked)} chunks "
        f"(load {load_s:.1f}s, chunk {chunk_s:.1f}s, reset {reset_s:.1f}s, index {index_s:.1f}s)"
    )
    return len(chunked)
