from typing import List, Dict
from app.config import COLLECTION_TECH_DOCS
from app.rag.vectordb import load_collection
from app.rag.bm25_retriever import bm25_retriever


def reciprocal_rank_fusion(results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
    fused_scores = {}
    doc_map = {}

    for results in results_list:
        for rank, doc in enumerate(results):
            content = doc["content"]
            if content not in fused_scores:
                fused_scores[content] = 0
                doc_map[content] = doc
            fused_scores[content] += 1.0 / (k + rank + 1)

    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [{**doc_map[content], "score": score} for content, score in sorted_docs]


def _normalize_distance_scores(docs: List[Dict]) -> List[Dict]:
    """Convert Chroma distance scores (lower=better) to similarity (higher=better)."""
    if not docs:
        return docs
    max_score = max(d["score"] for d in docs) or 1.0
    for doc in docs:
        doc["norm_score"] = 1.0 - (doc["score"] / max_score)
    return docs


def hybrid_retrieve(
    query: str,
    k: int = 5,
    alpha: float = 0.5,
    *,
    collection: str = COLLECTION_TECH_DOCS,
) -> List[Dict]:
    vectorstore = load_collection(collection)
    vector_results = vectorstore.similarity_search_with_score(query, k=k * 2)
    vector_docs = []
    for doc, score in vector_results:
        vector_docs.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
            "source": "vector",
            "collection": collection,
        })

    bm25_retriever.set_collection(collection)
    bm25_docs = bm25_retriever.retrieve(query, k=k * 2)

    vector_docs = _normalize_distance_scores(vector_docs)
    if bm25_docs:
        max_b = max(d["score"] for d in bm25_docs) or 1
        for doc in bm25_docs:
            doc["norm_score"] = doc["score"] / max_b

    combined = {}
    for doc in vector_docs:
        combined[doc["content"]] = {
            **doc,
            "score": alpha * doc["norm_score"],
        }
    for doc in bm25_docs:
        if doc["content"] in combined:
            combined[doc["content"]]["score"] += (1 - alpha) * doc["norm_score"]
        else:
            combined[doc["content"]] = {
                **doc,
                "score": (1 - alpha) * doc["norm_score"],
            }

    sorted_docs = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return sorted_docs[:k]
