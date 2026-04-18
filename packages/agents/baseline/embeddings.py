from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import httpx


def _stable_hash(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


@dataclass(frozen=True)
class EmbeddingClient:
    base_url: str | None
    api_key: str | None
    model: str
    dim: int = 384

    def embed(self, text: str) -> list[float]:
        text = text.strip()
        if not text:
            return [0.0] * self.dim

        # 无配置时退化为“确定性伪向量”，保证检索链路可跑通（生产环境建议配置真实 embedding）。
        if not self.base_url or not self.api_key:
            rng = np.random.default_rng(_stable_hash(text))
            v = rng.normal(size=(self.dim,)).astype(np.float64)
            v = v / (np.linalg.norm(v) + 1e-12)
            return v.tolist()

        url = self.base_url.rstrip("/") + "/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "input": text}
        with httpx.Client(timeout=15.0) as client:
            r = client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            emb = data["data"][0]["embedding"]
            return [float(x) for x in emb]
