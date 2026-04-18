from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from agents.baseline.schemas import (
    ChatRequest,
    ChatResponse,
    LongTermMemoryCreateRequest,
    LongTermMemoryItem,
)
from apps.api.deps import get_baseline_agent_service, get_upgraded_agent_service
from agents.baseline.pipeline import BaselineAgentService
from agents.upgraded.graph import UpgradedAgentService

router = APIRouter(prefix="/v1", tags=["agent-baseline"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    service: BaselineAgentService = Depends(get_baseline_agent_service),
    upgraded_service: UpgradedAgentService = Depends(get_upgraded_agent_service),
) -> ChatResponse:
    if body.agent_mode == "upgraded":
        return upgraded_service.chat(body)
    return service.chat(body)


@router.post("/memory/long_term", response_model=LongTermMemoryItem)
def add_long_term_memory(
    body: LongTermMemoryCreateRequest,
    service: BaselineAgentService = Depends(get_baseline_agent_service),
) -> LongTermMemoryItem:
    return service.add_long_term_memory(body)


@router.get("/memory/long_term", response_model=list[LongTermMemoryItem])
def list_long_term_memory(
    user_id: str,
    memory_type: str | None = None,
    service: BaselineAgentService = Depends(get_baseline_agent_service),
) -> list[LongTermMemoryItem]:
    return service.list_long_term_memories(user_id, memory_type)


@router.delete("/memory/long_term/{memory_id}")
def delete_long_term_memory(
    memory_id: str,
    user_id: str,
    service: BaselineAgentService = Depends(get_baseline_agent_service),
) -> dict[str, str]:
    deleted = service.delete_long_term_memory(user_id, memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="memory not found")
    return {"status": "deleted", "id": memory_id}
