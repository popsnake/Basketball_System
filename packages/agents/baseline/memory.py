from __future__ import annotations

from collections import defaultdict, deque


class ShortTermMemoryStore:
    """基线实现：内存版滑动窗口 + 简单摘要。"""

    def __init__(self, window_size: int = 8) -> None:
        self._window_size = window_size
        self._turns: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=self._window_size))
        self._summary: dict[str, str] = {}

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        text = f"{role}: {content.strip()}"
        turns = self._turns[session_id]
        turns.append(text)
        if len(turns) == self._window_size:
            self._summary[session_id] = self._build_summary(turns)

    def get_context(self, session_id: str) -> tuple[str, list[str]]:
        return self._summary.get(session_id, ""), list(self._turns[session_id])

    @staticmethod
    def _build_summary(turns: deque[str]) -> str:
        # 基线阶段使用轻量规则摘要，后续可替换为 LLM 摘要
        if not turns:
            return ""
        joined = " | ".join(list(turns)[:4])
        return f"历史摘要: {joined}"
