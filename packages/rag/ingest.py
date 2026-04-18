from __future__ import annotations

import hashlib
import re
from pathlib import Path

from agents.baseline.embeddings import EmbeddingClient
from rag.store import SqliteKnowledgeStore


def _iter_text_files(root: Path) -> list[Path]:
    files = list(root.rglob("*.txt"))
    files.extend(root.rglob("*.md"))
    filtered = []
    for path in files:
        if path.stem.lower() == "readme":
            continue
        filtered.append(path.resolve())
    return sorted(set(filtered))


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n")).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    clean = _normalize_text(text)
    if not clean:
        return []

    paragraphs = [part.strip() for part in clean.split("\n\n") if part.strip()]
    units: list[str] = []
    buffer = []
    for paragraph in paragraphs:
        # FAQ / heading / short title paragraphs are treated as semantic boundaries.
        if paragraph.startswith(("问：", "Q:", "#", "##")) or len(paragraph) < 28:
            if buffer:
                units.append("\n\n".join(buffer).strip())
                buffer = []
            units.append(paragraph)
            continue
        buffer.append(paragraph)
    if buffer:
        units.append("\n\n".join(buffer).strip())

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(unit) <= chunk_size:
            current = unit
            continue
        start = 0
        step = max(1, chunk_size - overlap)
        while start < len(unit):
            chunk = unit[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            start += step
        current = ""
    if current:
        chunks.append(current)
    return chunks


def _doc_id_for_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]


def _slugify_filename(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^\w\-\.]+", "_", value, flags=re.UNICODE)
    value = value.strip("._")
    return value or "document"


def _index_single_path(
    *,
    root: Path,
    path: Path,
    store: SqliteKnowledgeStore,
    chunk_size: int,
    chunk_overlap: int,
    embed_client: EmbeddingClient,
) -> tuple[str, int]:
    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    doc_id = _doc_id_for_path(path, root)
    title = path.stem
    tags = list(path.relative_to(root).parts[:-1])
    rows: list[tuple[str, str, list[str], list[float]]] = []
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}-{idx:04d}"
        rows.append((chunk_id, chunk, tags, embed_client.embed(chunk)))
    chunk_count = store.replace_document(
        doc_id=doc_id,
        title=title,
        source_path=path.relative_to(root).as_posix(),
        chunks=rows,
    )
    return doc_id, chunk_count


def ingest_documents(
    *,
    docs_dir: str,
    sqlite_path: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_client: EmbeddingClient,
) -> dict[str, int]:
    root = Path(docs_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    store = SqliteKnowledgeStore(sqlite_path)
    store.clear_all()

    files = _iter_text_files(root)
    total_chunks = 0
    for path in files:
        _, chunk_count = _index_single_path(
            root=root,
            path=path,
            store=store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_client=embed_client,
        )
        total_chunks += chunk_count
    return {"documents": len(files), "chunks": total_chunks}


def upsert_text_document(
    *,
    docs_dir: str,
    sqlite_path: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_client: EmbeddingClient,
    title: str,
    content: str,
    filename: str | None = None,
) -> dict[str, str | int]:
    root = Path(docs_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    safe_filename = _slugify_filename(filename or title)
    if not safe_filename.lower().endswith(".txt"):
        safe_filename = f"{safe_filename}.txt"
    path = root / safe_filename
    path.write_text(_normalize_text(content) + "\n", encoding="utf-8")

    store = SqliteKnowledgeStore(sqlite_path)
    doc_id, chunk_count = _index_single_path(
        root=root,
        path=path,
        store=store,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_client=embed_client,
    )
    return {
        "doc_id": doc_id,
        "title": path.stem,
        "source_path": path.relative_to(root).as_posix(),
        "chunk_count": chunk_count,
    }
