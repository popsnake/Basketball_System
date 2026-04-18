from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_yaml(name: str) -> dict:
    path = _repo_root() / "configs" / name
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class BaselineAgentSettings:
    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    stm_window_size: int = int(os.getenv("STM_WINDOW_SIZE", "12"))
    stm_ttl_seconds: int = int(os.getenv("STM_TTL_SECONDS", "86400"))  # 24h

    # Long-term memory persistence
    ltm_sqlite_path: str = os.getenv("LTM_SQLITE_PATH", "data/ltm.sqlite3")
    kb_sqlite_path: str = os.getenv("KB_SQLITE_PATH", "data/kb.sqlite3")
    kb_docs_dir: str = os.getenv("KB_DOCS_DIR", "data/knowledge")
    kb_chunk_size: int = int(os.getenv("KB_CHUNK_SIZE", "500"))
    kb_chunk_overlap: int = int(os.getenv("KB_CHUNK_OVERLAP", "80"))
    kb_min_score: float = float(os.getenv("KB_MIN_SCORE", "0.025"))
    kb_max_hits_per_doc: int = int(os.getenv("KB_MAX_HITS_PER_DOC", "1"))
    kb_snippet_chars: int = int(os.getenv("KB_SNIPPET_CHARS", "240"))

    # LLM (OpenAI-compatible, optional)
    llm_base_url: str | None = os.getenv("LLM_BASE_URL")  # e.g. https://api.openai.com/v1
    llm_api_key: str | None = os.getenv("LLM_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4.1-mini")

    # Embeddings (OpenAI-compatible, optional)
    embed_base_url: str | None = os.getenv("EMBED_BASE_URL")
    embed_api_key: str | None = os.getenv("EMBED_API_KEY")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    @classmethod
    def from_configs(cls) -> "BaselineAgentSettings":
        rag_cfg = _load_yaml("rag.yaml")
        memory_cfg = _load_yaml("memory.yaml")
        agents_cfg = _load_yaml("agents.yaml")
        llm_cfg = agents_cfg.get("llm", {})
        return cls(
            redis_url=os.getenv("REDIS_URL", memory_cfg.get("redis_url", "redis://localhost:6379/0")),
            stm_window_size=int(os.getenv("STM_WINDOW_SIZE", str(memory_cfg.get("stm_window_size", 12)))),
            stm_ttl_seconds=int(os.getenv("STM_TTL_SECONDS", str(memory_cfg.get("stm_ttl_seconds", 86400)))),
            ltm_sqlite_path=os.getenv("LTM_SQLITE_PATH", memory_cfg.get("ltm_sqlite_path", "data/ltm.sqlite3")),
            kb_sqlite_path=os.getenv("KB_SQLITE_PATH", rag_cfg.get("kb_sqlite_path", "data/kb.sqlite3")),
            kb_docs_dir=os.getenv("KB_DOCS_DIR", rag_cfg.get("docs_dir", "data/knowledge")),
            kb_chunk_size=int(os.getenv("KB_CHUNK_SIZE", str(rag_cfg.get("chunk_size", 500)))),
            kb_chunk_overlap=int(os.getenv("KB_CHUNK_OVERLAP", str(rag_cfg.get("chunk_overlap", 80)))),
            kb_min_score=float(os.getenv("KB_MIN_SCORE", str(rag_cfg.get("min_score", 0.025)))),
            kb_max_hits_per_doc=int(os.getenv("KB_MAX_HITS_PER_DOC", str(rag_cfg.get("max_hits_per_doc", 1)))),
            kb_snippet_chars=int(os.getenv("KB_SNIPPET_CHARS", str(rag_cfg.get("snippet_chars", 240)))),
            llm_base_url=os.getenv("LLM_BASE_URL", llm_cfg.get("base_url")),
            llm_api_key=os.getenv("LLM_API_KEY", llm_cfg.get("api_key")),
            llm_model=os.getenv("LLM_MODEL", llm_cfg.get("model", "gpt-4.1-mini")),
            embed_base_url=os.getenv("EMBED_BASE_URL", rag_cfg.get("embed_base_url")),
            embed_api_key=os.getenv("EMBED_API_KEY", rag_cfg.get("embed_api_key")),
            embed_model=os.getenv("EMBED_MODEL", rag_cfg.get("embed_model", "text-embedding-3-small")),
        )
