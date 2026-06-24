from typing import List, Dict

from app.config import CHUNK_OVERLAP, CHUNK_SIZE


def recursive_chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            best_sep = -1
            best_sep_len = 0
            for sep in separators:
                idx = text.rfind(sep, start, end)
                if idx > best_sep and idx >= start + chunk_size - chunk_overlap:
                    best_sep = idx
                    best_sep_len = len(sep) if sep else 0
            if best_sep > 0:
                end = best_sep + best_sep_len
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def chunk_documents(
    docs: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    chunked = []
    for doc in docs:
        chunks = recursive_chunk_text(doc["content"], chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            chunked.append({
                "content": chunk,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            })
    return chunked
