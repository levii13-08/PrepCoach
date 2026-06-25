from typing import List, Dict, Optional


def _match_metadata_or_content(
    doc: Dict,
    metadata_key: str,
    value: str,
) -> bool:
    metadata_val = doc.get("metadata", {}).get(metadata_key)
    if metadata_val is not None:
        return metadata_val == value
    return value.lower() in doc.get("content", "").lower()


def filter_by_metadata(
    documents: List[Dict],
    type_filter: Optional[str] = None,
    filename_filter: Optional[str] = None,
    topic_filter: Optional[str] = None,
    difficulty_filter: Optional[str] = None,
    role_filter: Optional[str] = None,
) -> List[Dict]:
    filtered = documents
    if type_filter:
        filtered = [d for d in filtered if d.get("metadata", {}).get("type") == type_filter]
    if filename_filter:
        filtered = [d for d in filtered if d.get("metadata", {}).get("filename") == filename_filter]
    if topic_filter:
        filtered = [
            d for d in filtered
            if _match_metadata_or_content(d, "topic", topic_filter)
        ]
    if difficulty_filter:
        filtered = [
            d for d in filtered
            if _match_metadata_or_content(d, "difficulty", difficulty_filter)
        ]
    if role_filter:
        filtered = [
            d for d in filtered
            if role_filter.lower() in d.get("content", "").lower()
            or role_filter.lower() in d.get("metadata", {}).get("title", "").lower()
            or role_filter.lower() in d.get("metadata", {}).get("role", "").lower()
        ]
    return filtered


def filter_by_relevance(
    documents: List[Dict],
    min_score: float = 0.0,
    source_preference: Optional[str] = None,
) -> List[Dict]:
    filtered = [d for d in documents if d.get("score", 0) >= min_score]
    if source_preference:
        for doc in filtered:
            if doc.get("source") == source_preference:
                doc["score"] *= 1.2
        filtered = sorted(filtered, key=lambda x: x.get("score", 0), reverse=True)
    return filtered
