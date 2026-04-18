from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None


@dataclass(frozen=True)
class SessionContext:
    summary: str
    turns: list[dict[str, Any]]  # [{"role": "...", "content": "..."}]


class RedisShortTermMemory:
    """
    生产版短期记忆：滑动窗口 + 摘要 + Redis 存储。
    - turns list: session:{id}:turns
    - summary str: session:{id}:summary
    """

    def __init__(self, redis_url: str, window_size: int, ttl_seconds: int) -> None:
        if redis is None:
            raise RuntimeError("redis package not installed")
        self._r = redis.Redis.from_url(redis_url, decode_responses=True)
        self._window_size = window_size
        self._ttl = ttl_seconds

    def append(self, session_id: str, role: str, content: str) -> None:
        key_turns = f"session:{session_id}:turns"
        item = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        pipe = self._r.pipeline()
        pipe.rpush(key_turns, item)
        pipe.ltrim(key_turns, -self._window_size, -1)
        pipe.expire(key_turns, self._ttl)
        pipe.execute()

    def get(self, session_id: str) -> SessionContext:
        key_turns = f"session:{session_id}:turns"
        key_sum = f"session:{session_id}:summary"
        turns_raw = self._r.lrange(key_turns, 0, -1) or []
        turns: list[dict[str, Any]] = []
        for t in turns_raw:
            try:
                turns.append(json.loads(t))
            except Exception:
                continue
        summary = self._r.get(key_sum) or ""
        return SessionContext(summary=summary, turns=turns)

    def set_summary(self, session_id: str, summary: str) -> None:
        key_sum = f"session:{session_id}:summary"
        self._r.setex(key_sum, self._ttl, summary)

