from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KnowledgeChunkItem:
    doc_id: str
    chunk_id: str
    title: str
    source_path: str
    text: str
    tags: list[str] = field(default_factory=list)
    score: float | None = None


@dataclass(frozen=True)
class RetrievalHit:
    source: str
    text: str
    score: float
    title: str | None = None
    doc_id: str | None = None
