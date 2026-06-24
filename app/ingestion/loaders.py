import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pypdf import PdfReader


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_txt(path: Path) -> List[Dict[str, Any]]:
    return [
        {
            "content": _safe_read_text(path),
            "metadata": {"source": str(path), "filename": path.name},
        }
    ]


def load_md(path: Path) -> List[Dict[str, Any]]:
    # Treat markdown as plain text at ingestion time (structure comes from chunking).
    return [
        {
            "content": _safe_read_text(path),
            "metadata": {"source": str(path), "filename": path.name},
        }
    ]


def load_pdf(path: Path) -> List[Dict[str, Any]]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                {
                    "content": text,
                    "metadata": {
                        "source": str(path),
                        "filename": path.name,
                        "page": i + 1,
                    },
                }
            )
    return pages


# Fields stored in the embedding `content` — omit from Chroma metadata to avoid
# duplicating full documents on every chunk (major ingest slowdown).
_METADATA_BODY_FIELDS = frozenset(
    {
        "requirements",
        "content",
        "answer",
        "question",
        "text",
        "description",
        "input",
        "response",
        "skills_desc",
    }
)


def _clean_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    """Remove keys with empty names or None values — Chroma rejects them."""
    cleaned: Dict[str, Any] = {}
    for key, value in row.items():
        if not key or value is None or value == "":
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            cleaned[key] = ", ".join(value)
    return cleaned


def _index_metadata(record: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """Keep only small, filterable fields in vector metadata."""
    meta = {"source": str(path), "filename": path.name}
    for key, value in record.items():
        if key in _METADATA_BODY_FIELDS:
            continue
        if not key or value is None or value == "":
            continue
        if isinstance(value, (str, int, float, bool)):
            meta[key] = value
        elif isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            meta[key] = ", ".join(value)
    return _clean_metadata(meta)


def load_csv(path: Path) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = row.get("text") or row.get("content") or row.get("input_text") or row.get("description") or ""
            docs.append(
                {
                    "content": content,
                    "metadata": _index_metadata(row, path),
                }
            )
    return docs


def _iter_json_records(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item


def load_json(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(_safe_read_text(path) or "{}")
    docs: List[Dict[str, Any]] = []
    for record in _iter_json_records(raw):
        content = (
            record.get("requirements")
            or record.get("question")
            or record.get("content")
            or record.get("text")
            or json.dumps(record, ensure_ascii=False)
        )
        docs.append(
            {
                "content": str(content),
                "metadata": _index_metadata(record, path),
            }
        )
    return docs


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            content = (
                record.get("requirements")
                or record.get("question")
                or record.get("content")
                or record.get("text")
                or json.dumps(record, ensure_ascii=False)
            )
            docs.append(
                {
                    "content": str(content),
                    "metadata": _index_metadata(record, path),
                }
            )
    return docs


def load_path(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return load_txt(path)
    if suffix in (".md", ".markdown"):
        return load_md(path)
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".csv":
        return load_csv(path)
    if suffix == ".json":
        return load_json(path)
    if suffix in (".jsonl", ".ndjson"):
        return load_jsonl(path)
    return []


def load_directory(
    directory: Path,
    *,
    recursive: bool = True,
    allowed_suffixes: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    if not directory.exists():
        return []

    paths = directory.rglob("*") if recursive else directory.glob("*")
    docs: List[Dict[str, Any]] = []
    for p in paths:
        if not p.is_file():
            continue
        if allowed_suffixes is not None and p.suffix.lower() not in allowed_suffixes:
            continue
        docs.extend(load_path(p))
    return docs

