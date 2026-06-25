from typing import List, Dict

_cross_encoder = None


def _get_reranker():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank(query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
    if not documents:
        return []

    reranker = _get_reranker()
    pairs = [(query, doc["content"]) for doc in documents]
    scores = reranker.predict(pairs)

    for i, doc in enumerate(documents):
        doc["rerank_score"] = float(scores[i])

    reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
