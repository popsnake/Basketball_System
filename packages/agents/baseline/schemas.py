from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    agent_mode: Literal["baseline", "upgraded"] = "baseline"
    top_k: int = 3
    use_kb: bool = True
    use_long_term_memory: bool = True
    mock_score_value: float | None = Field(
        default=None,
        description="模拟评分值（0~100），用于替代真实评分联调",
    )


class RetrievalHit(BaseModel):
    source: Literal["kb", "long_term"]
    text: str
    score: float
    title: str | None = None
    doc_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    citations: list[RetrievalHit] = Field(default_factory=list)
    memory_used: dict[str, int | bool] = Field(default_factory=dict)
    retrieval_summary: list[str] = Field(default_factory=list)
    mock_score_value: float | None = None
    agent_mode: Literal["baseline", "upgraded"] = "baseline"


class LongTermMemoryCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    type: Literal["preference", "fact", "goal"] = "fact"
    content: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)
    importance: int = Field(default=3, ge=1, le=5)
    expires_at: datetime | None = None


class LongTermMemoryItem(BaseModel):
    id: str
    user_id: str
    type: str
    content: str
    tags: list[str] = Field(default_factory=list)
    importance: int = 3
    created_at: datetime | None = None
    expires_at: datetime | None = None


class KnowledgeChunkItem(BaseModel):
    doc_id: str
    chunk_id: str
    title: str
    source_path: str
    text: str
    tags: list[str] = Field(default_factory=list)
    score: float | None = None


class KnowledgeDocumentCreateRequest(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)
    filename: str | None = None


class KnowledgeDocumentItem(BaseModel):
    doc_id: str
    title: str
    source_path: str
    chunk_count: int


class KnowledgeReindexResponse(BaseModel):
    documents: int
    chunks: int
