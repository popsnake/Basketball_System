from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass(frozen=True)
class KnowledgeChunkRecord:
    doc_id: str
    chunk_id: str
    title: str
    source_path: str
    text: str
    tags: list[str]


class SqliteKnowledgeStore:
    def __init__(self, sqlite_path: str) -> None:
        self._path = Path(sqlite_path)
        _ensure_parent(self._path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kb_chunks (
              chunk_id TEXT PRIMARY KEY,
              doc_id TEXT NOT NULL,
              title TEXT NOT NULL,
              source_path TEXT NOT NULL,
              text TEXT NOT NULL,
              tags_json TEXT NOT NULL,
              embedding_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_doc_id ON kb_chunks(doc_id)")
        self._conn.commit()

    def clear_all(self) -> None:
        self._conn.execute("DELETE FROM kb_chunks")
        self._conn.commit()

    def delete_document(self, doc_id: str) -> None:
        self._conn.execute("DELETE FROM kb_chunks WHERE doc_id=?", (doc_id,))
        self._conn.commit()

    def replace_document(
        self,
        *,
        doc_id: str,
        title: str,
        source_path: str,
        chunks: list[tuple[str, str, list[str], list[float]]],
    ) -> int:
        self._conn.execute("DELETE FROM kb_chunks WHERE doc_id=?", (doc_id,))
        self._conn.executemany(
            """
            INSERT INTO kb_chunks(chunk_id, doc_id, title, source_path, text, tags_json, embedding_json)
            VALUES(?,?,?,?,?,?,?)
            """,
            [
                (
                    chunk_id,
                    doc_id,
                    title,
                    source_path,
                    text,
                    json.dumps(tags, ensure_ascii=False),
                    json.dumps(embedding),
                )
                for chunk_id, text, tags, embedding in chunks
            ],
        )
        self._conn.commit()
        return len(chunks)

    def list_documents(self) -> list[dict[str, str | int]]:
        cur = self._conn.execute(
            """
            SELECT doc_id, title, source_path, COUNT(*) as chunk_count
            FROM kb_chunks
            GROUP BY doc_id, title, source_path
            ORDER BY title ASC
            """
        )
        return [
            {
                "doc_id": doc_id,
                "title": title,
                "source_path": source_path,
                "chunk_count": chunk_count,
            }
            for doc_id, title, source_path, chunk_count in cur.fetchall()
        ]

    def get_document(self, doc_id: str) -> dict[str, str | int] | None:
        cur = self._conn.execute(
            """
            SELECT doc_id, title, source_path, COUNT(*) as chunk_count
            FROM kb_chunks
            WHERE doc_id=?
            GROUP BY doc_id, title, source_path
            """,
            (doc_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        found_doc_id, title, source_path, chunk_count = row
        return {
            "doc_id": found_doc_id,
            "title": title,
            "source_path": source_path,
            "chunk_count": chunk_count,
        }

    def list_chunks(self) -> list[KnowledgeChunkRecord]:
        cur = self._conn.execute(
            "SELECT doc_id, chunk_id, title, source_path, text, tags_json FROM kb_chunks"
        )
        return [
            KnowledgeChunkRecord(
                doc_id=doc_id,
                chunk_id=chunk_id,
                title=title,
                source_path=source_path,
                text=text,
                tags=json.loads(tags_json),
            )
            for doc_id, chunk_id, title, source_path, text, tags_json in cur.fetchall()
        ]

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[KnowledgeChunkRecord, float]]:
        q = np.asarray(query_embedding, dtype=np.float64)
        cur = self._conn.execute(
            "SELECT doc_id, chunk_id, title, source_path, text, tags_json, embedding_json FROM kb_chunks"
        )
        scored: list[tuple[KnowledgeChunkRecord, float]] = []
        for doc_id, chunk_id, title, source_path, text, tags_json, emb_json in cur.fetchall():
            try:
                emb = np.asarray(json.loads(emb_json), dtype=np.float64)
                score = _cosine(q, emb)
            except Exception:
                continue
            scored.append(
                (
                    KnowledgeChunkRecord(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        title=title,
                        source_path=source_path,
                        text=text,
                        tags=json.loads(tags_json),
                    ),
                    score,
                )
            )
        scored.sort(key=lambda item: item[1], reverse=True)
        return [item for item in scored[:top_k] if item[1] > 0]
