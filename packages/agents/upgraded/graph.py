from __future__ import annotations

from agents.baseline.pipeline import BaselineAgentService
from agents.baseline.schemas import ChatRequest, ChatResponse


class UpgradedAgentService:
    """
    升级版 Agent 骨架。
    当前先复用 baseline 能力，保留将来接入 router/planner/critic 的扩展点。
    """

    def __init__(self) -> None:
        self._baseline = BaselineAgentService()

    def chat(self, req: ChatRequest) -> ChatResponse:
        resp = self._baseline.chat(req)
        resp.agent_mode = "upgraded"
        return resp
