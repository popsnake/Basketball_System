from __future__ import annotations

from agents.baseline.embeddings import EmbeddingClient
from agents.baseline.settings import BaselineAgentSettings
from rag.ingest import ingest_documents as ingest_rag_documents


def ingest_documents() -> dict[str, int]:
    settings = BaselineAgentSettings.from_configs()
    embed_client = EmbeddingClient(
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key,
        model=settings.embed_model,
    )
    return ingest_rag_documents(
        docs_dir=settings.kb_docs_dir,
        sqlite_path=settings.kb_sqlite_path,
        chunk_size=settings.kb_chunk_size,
        chunk_overlap=settings.kb_chunk_overlap,
        embed_client=embed_client,
    )


def main() -> None:
    result = ingest_documents()
    print(f"Indexed {result['documents']} documents into {result['chunks']} chunks.")


if __name__ == "__main__":
    main()
