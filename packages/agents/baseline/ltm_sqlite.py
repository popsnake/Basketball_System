from __future__ import annotations

import json
import sqlite3
from datetime import datetime, UTC
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass(frozen=True)
class LtmRecord:
    id: str
    user_id: str
    type: str
    content: str
    tags: list[str]
    importance: int
    created_at: datetime
    expires_at: datetime | None


class SqliteLongTermMemory:
    """长期记忆：SQLite 持久化，向量以 JSON 存储（基线可用，易迁移到向量库）。"""

    def __init__(self, sqlite_path: str) -> None:
        self._path = Path(sqlite_path)
        _ensure_parent(self._path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ltm (
              id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              type TEXT NOT NULL,
              content TEXT NOT NULL,
              tags_json TEXT NOT NULL,
              embedding_json TEXT NOT NULL,
              importance INTEGER NOT NULL DEFAULT 3,
              created_at TEXT NOT NULL DEFAULT '',
              expires_at TEXT
            )
            """
        )
        self._ensure_column("importance", "INTEGER NOT NULL DEFAULT 3")
        self._ensure_column("created_at", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("expires_at", "TEXT")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_ltm_user ON ltm(user_id)")
        self._conn.commit()

    def _ensure_column(self, name: str, ddl: str) -> None:
        cols = self._conn.execute("PRAGMA table_info(ltm)").fetchall()
        if any(col[1] == name for col in cols):
            return
        self._conn.execute(f"ALTER TABLE ltm ADD COLUMN {name} {ddl}")
        self._conn.commit()

    @staticmethod
    def _parse_dt(value: str | None) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(value)

    def add(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        tags: list[str],
        embedding: list[float],
        importance: int = 3,
        expires_at: datetime | None = None,
    ) -> LtmRecord:
        rid = str(uuid4())
        created_at = datetime.now(UTC)
        self._conn.execute(
            "INSERT INTO ltm(id,user_id,type,content,tags_json,embedding_json,importance,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                rid,
                user_id,
                memory_type,
                content,
                json.dumps(tags, ensure_ascii=False),
                json.dumps(embedding),
                importance,
                created_at.isoformat(),
                expires_at.isoformat() if expires_at else None,
            ),
        )
        self._conn.commit()
        return LtmRecord(
            id=rid,
            user_id=user_id,
            type=memory_type,
            content=content,
            tags=tags,
            importance=importance,
            created_at=created_at,
            expires_at=expires_at,
        )

    def list(self, user_id: str, memory_type: str | None = None) -> list[LtmRecord]:
        if memory_type:
            cur = self._conn.execute(
                "SELECT id,user_id,type,content,tags_json,importance,created_at,expires_at FROM ltm WHERE user_id=? AND type=? ORDER BY rowid DESC",
                (user_id, memory_type),
            )
        else:
            cur = self._conn.execute(
                "SELECT id,user_id,type,content,tags_json,importance,created_at,expires_at FROM ltm WHERE user_id=? ORDER BY rowid DESC",
                (user_id,),
            )
        out: list[LtmRecord] = []
        now = datetime.now(UTC)
        for rid, uid, t, content, tags_json, importance, created_at, expires_at in cur.fetchall():
            parsed_exp = self._parse_dt(expires_at)
            if parsed_exp and parsed_exp < now:
                continue
            out.append(
                LtmRecord(
                    id=rid,
                    user_id=uid,
                    type=t,
                    content=content,
                    tags=json.loads(tags_json),
                    importance=int(importance or 3),
                    created_at=self._parse_dt(created_at) or now,
                    expires_at=parsed_exp,
                )
            )
        return out

    def delete(self, user_id: str, memory_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM ltm WHERE user_id=? AND id=?", (user_id, memory_id))
        self._conn.commit()
        return cur.rowcount > 0

    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int,
        *,
        memory_type: str | None = None,
    ) -> list[tuple[LtmRecord, float]]:
        q = np.asarray(query_embedding, dtype=np.float64)
        if memory_type:
            cur = self._conn.execute(
                "SELECT id,type,content,tags_json,embedding_json,importance,created_at,expires_at FROM ltm WHERE user_id=? AND type=?",
                (user_id, memory_type),
            )
        else:
            cur = self._conn.execute(
                "SELECT id,type,content,tags_json,embedding_json,importance,created_at,expires_at FROM ltm WHERE user_id=?",
                (user_id,),
            )
        scored: list[tuple[LtmRecord, float]] = []
        now = datetime.now(UTC)
        for row in cur.fetchall():
            try:
                record_id, record_type, content, tags_json, emb_json, importance, created_at, expires_at = row
                parsed_exp = self._parse_dt(expires_at)
                if parsed_exp and parsed_exp < now:
                    continue
                emb = np.asarray(json.loads(emb_json), dtype=np.float64)
                base_score = _cosine(q, emb)
                weighted_score = base_score * (0.7 + 0.1 * int(importance or 3))
                scored.append(
                    (
                        LtmRecord(
                            id=str(record_id),
                            user_id=user_id,
                            type=str(record_type),
                            content=content,
                            tags=json.loads(tags_json),
                            importance=int(importance or 3),
                            created_at=self._parse_dt(created_at) or now,
                            expires_at=parsed_exp,
                        ),
                        weighted_score,
                    )
                )
            except Exception:
                continue
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s in scored[:top_k] if s[1] > 0]

