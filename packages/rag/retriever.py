from __future__ import annotations

from agents.baseline.embeddings import EmbeddingClient
from rag.schemas import KnowledgeChunkItem
from rag.store import SqliteKnowledgeStore


def _token_overlap_score(query: str, text: str) -> float:
    def is_cjk(ch: str) -> bool:
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF

    def to_tokens(value: str) -> set[str]:
        value = value.lower()
        if any(is_cjk(ch) for ch in value):
            return {ch for ch in value if ch.isalnum() or is_cjk(ch)}
        parts = [part for part in value.split() if part]
        if len(parts) > 1:
            return set(parts)
        return {ch for ch in value if ch.isalnum() or is_cjk(ch)}

    query_tokens = to_tokens(query)
    text_tokens = to_tokens(text)
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(text_tokens)) / len(query_tokens.union(text_tokens))
    normalized_query = query.replace("？", "").replace("?", "")
    substring_bonus = 0.5 if normalized_query and normalized_query in text else 0.0
    return overlap + substring_bonus


class KnowledgeRetriever:
    def __init__(
        self,
        sqlite_path: str,
        embed_client: EmbeddingClient,
        *,
        min_score: float = 0.025,
        max_hits_per_doc: int = 1,
        snippet_chars: int = 240,
    ) -> None:
        self._store = SqliteKnowledgeStore(sqlite_path)
        self._embed = embed_client
        self._min_score = min_score
        self._max_hits_per_doc = max_hits_per_doc
        self._snippet_chars = snippet_chars

    def search(self, query: str, top_k: int) -> list[KnowledgeChunkItem]:
        query = query.strip()
        if not query:
            return []
        query_embedding = self._embed.embed(query)
        vector_hits = self._store.search(query_embedding, max(top_k * 3, 8))
        lexical_scores = {
            record.chunk_id: score
            for record, score in self._lexical_candidates(query)
        }
        reranked: list[tuple[object, float]] = []
        seen_ids: set[str] = set()
        for record, score in vector_hits:
            combined = 0.8 * score + 0.2 * lexical_scores.get(record.chunk_id, 0.0)
            reranked.append((record, combined))
            seen_ids.add(record.chunk_id)
        for record, lexical_score in self._lexical_candidates(query):
            if record.chunk_id in seen_ids:
                continue
            reranked.append((record, lexical_score))
        reranked.sort(key=lambda item: item[1], reverse=True)
        doc_counts: dict[str, int] = {}
        out: list[KnowledgeChunkItem] = []
        for record, score in reranked:
            if score < self._min_score:
                continue
            doc_counts.setdefault(record.doc_id, 0)
            if doc_counts[record.doc_id] >= self._max_hits_per_doc:
                continue
            doc_counts[record.doc_id] += 1
            out.append(
                KnowledgeChunkItem(
                    doc_id=record.doc_id,
                    chunk_id=record.chunk_id,
                    title=record.title,
                    source_path=record.source_path,
                    text=self._make_snippet(query, record.text),
                    tags=record.tags,
                    score=score,
                )
            )
            if len(out) >= top_k:
                break
        return out

    def list_documents(self) -> list[dict[str, str | int]]:
        return self._store.list_documents()

    def _lexical_search(self, query: str, top_k: int) -> list[KnowledgeChunkItem]:
        scored = self._lexical_candidates(query)
        return [
            KnowledgeChunkItem(
                doc_id=record.doc_id,
                chunk_id=record.chunk_id,
                title=record.title,
                source_path=record.source_path,
                text=self._make_snippet(query, record.text),
                tags=record.tags,
                score=score,
            )
            for record, score in scored[:top_k]
        ]

    def _lexical_candidates(self, query: str) -> list[tuple[object, float]]:
        candidates = self._store.list_chunks()
        scored = []
        for record in candidates:
            score = _token_overlap_score(query, f"{record.title} {record.text}")
            if score > 0:
                scored.append((record, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _make_snippet(self, query: str, text: str) -> str:
        clean = " ".join(text.split())
        if len(clean) <= self._snippet_chars:
            return clean

        normalized_query = query.replace("？", "").replace("?", "").strip()
        idx = clean.find(normalized_query) if normalized_query else -1
        if idx == -1:
            for token in _query_terms(query):
                idx = clean.find(token)
                if idx != -1:
                    break
        if idx == -1:
            return clean[: self._snippet_chars].rstrip() + "..."

        half = self._snippet_chars // 2
        start = max(0, idx - half)
        end = min(len(clean), start + self._snippet_chars)
        start = max(0, end - self._snippet_chars)
        snippet = clean[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(clean):
            snippet = snippet + "..."
        return snippet


def _query_terms(query: str) -> list[str]:
    compact = "".join(ch for ch in query if ch.isalnum() or (0x4E00 <= ord(ch) <= 0x9FFF))
    if not compact:
        return []
    if any(0x4E00 <= ord(ch) <= 0x9FFF for ch in compact):
        return [compact[i : i + 2] for i in range(max(1, len(compact) - 1))]
    return [part for part in query.lower().split() if part]
