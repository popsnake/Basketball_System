from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from agents.baseline.embeddings import EmbeddingClient
from agents.baseline.schemas import (
    KnowledgeDocumentCreateRequest,
    KnowledgeDocumentItem,
    KnowledgeReindexResponse,
)
from agents.baseline.settings import BaselineAgentSettings
from rag.ingest import ingest_documents, upsert_text_document
from rag.store import SqliteKnowledgeStore

router = APIRouter(prefix="/v1/kb", tags=["knowledge-base"])


def _settings_and_clients() -> tuple[BaselineAgentSettings, EmbeddingClient, SqliteKnowledgeStore]:
    settings = BaselineAgentSettings.from_configs()
    embed_client = EmbeddingClient(
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key,
        model=settings.embed_model,
    )
    store = SqliteKnowledgeStore(settings.kb_sqlite_path)
    return settings, embed_client, store


@router.get("/documents", response_model=list[KnowledgeDocumentItem])
def list_documents() -> list[KnowledgeDocumentItem]:
    _, _, store = _settings_and_clients()
    return [KnowledgeDocumentItem(**item) for item in store.list_documents()]


@router.post("/documents", response_model=KnowledgeDocumentItem)
def create_document(body: KnowledgeDocumentCreateRequest) -> KnowledgeDocumentItem:
    settings, embed_client, _ = _settings_and_clients()
    item = upsert_text_document(
        docs_dir=settings.kb_docs_dir,
        sqlite_path=settings.kb_sqlite_path,
        chunk_size=settings.kb_chunk_size,
        chunk_overlap=settings.kb_chunk_overlap,
        embed_client=embed_client,
        title=body.title,
        content=body.content,
        filename=body.filename,
    )
    return KnowledgeDocumentItem(**item)


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict[str, str]:
    settings, _, store = _settings_and_clients()
    item = store.get_document(doc_id)
    if item is None:
        raise HTTPException(status_code=404, detail="document not found")

    source_path = str(item["source_path"])
    root = Path(settings.kb_docs_dir).resolve()
    file_path = (root / source_path).resolve()
    if file_path.is_file():
        file_path.unlink()

    store.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@router.post("/documents/reindex", response_model=KnowledgeReindexResponse)
def reindex_documents() -> KnowledgeReindexResponse:
    settings, embed_client, _ = _settings_and_clients()
    result = ingest_documents(
        docs_dir=settings.kb_docs_dir,
        sqlite_path=settings.kb_sqlite_path,
        chunk_size=settings.kb_chunk_size,
        chunk_overlap=settings.kb_chunk_overlap,
        embed_client=embed_client,
    )
    return KnowledgeReindexResponse(**result)
