from __future__ import annotations

from agents.baseline.embeddings import EmbeddingClient
from agents.baseline.settings import BaselineAgentSettings
from rag.retriever import KnowledgeRetriever
from rag.schemas import KnowledgeChunkItem


class BaselineKnowledgeStore:
    def __init__(
        self,
        settings: BaselineAgentSettings,
        embed_client: EmbeddingClient,
    ) -> None:
        self._retriever = KnowledgeRetriever(
            settings.kb_sqlite_path,
            embed_client,
            min_score=settings.kb_min_score,
            max_hits_per_doc=settings.kb_max_hits_per_doc,
            snippet_chars=settings.kb_snippet_chars,
        )

    def search_kb(self, query: str, top_k: int) -> list[KnowledgeChunkItem]:
        return self._retriever.search(query, top_k)

    def list_documents(self) -> list[dict[str, str | int]]:
        return self._retriever.list_documents()
