from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class LlmClient:
    base_url: str | None
    api_key: str | None
    model: str

    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def chat(self, system: str, user: str, context: str) -> str:
        """
        OpenAI-compatible /chat/completions 最小调用。
        未配置时由上层 fallback 生成回复。
        """
        if not self.enabled():
            raise RuntimeError("LLM not configured")

        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{context}\n\n用户问题：{user}"},
        ]
        payload = {"model": self.model, "messages": messages, "temperature": 0.3}
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
