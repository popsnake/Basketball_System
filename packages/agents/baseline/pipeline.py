from __future__ import annotations

from agents.baseline.embeddings import EmbeddingClient
from agents.baseline.knowledge import BaselineKnowledgeStore
from agents.baseline.llm import LlmClient
from agents.baseline.schemas import (
    ChatRequest,
    ChatResponse,
    LongTermMemoryCreateRequest,
    LongTermMemoryItem,
    RetrievalHit,
)
from agents.baseline.settings import BaselineAgentSettings
from memory import build_incremental_summary
from memory.long_term import SqliteLongTermMemory
from memory.short_term import RedisShortTermMemory, ShortTermMemoryStore
from rag.rewrite import rewrite_query


class BaselineAgentService:
    """
    Baseline Agent:
    1) 短期记忆(滑动窗口 + 摘要)
    2) 查询重写
    3) 知识库 / 长期记忆检索
    4) 回答生成
    """

    def __init__(self) -> None:
        self.settings = BaselineAgentSettings.from_configs()
        self._stm_fallback = ShortTermMemoryStore(window_size=self.settings.stm_window_size)
        try:
            self.stm = RedisShortTermMemory(
                redis_url=self.settings.redis_url,
                window_size=self.settings.stm_window_size,
                ttl_seconds=self.settings.stm_ttl_seconds,
            )
        except Exception:
            self.stm = None

        self.embed = EmbeddingClient(
            base_url=self.settings.embed_base_url,
            api_key=self.settings.embed_api_key,
            model=self.settings.embed_model,
        )
        self.knowledge = BaselineKnowledgeStore(settings=self.settings, embed_client=self.embed)
        self.ltm = SqliteLongTermMemory(self.settings.ltm_sqlite_path)
        self.llm = LlmClient(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            model=self.settings.llm_model,
        )

    def chat(self, req: ChatRequest) -> ChatResponse:
        if self.stm is None:
            summary, turns = self._stm_fallback.get_context(req.session_id)
            ctx_summary, ctx_turns = summary, [{"role": "turn", "content": t} for t in turns]
        else:
            ctx = self.stm.get(req.session_id)
            ctx_summary, ctx_turns = ctx.summary, ctx.turns

        rewritten_query = rewrite_query(req.content, ctx_summary, ctx_turns)

        kb_hits: list[RetrievalHit] = []
        if req.use_kb:
            for item in self.knowledge.search_kb(rewritten_query, req.top_k):
                kb_hits.append(
                    RetrievalHit(
                        source="kb",
                        text=item.text,
                        score=item.score or 0.0,
                        title=item.title,
                        doc_id=item.doc_id,
                    )
                )

        ltm_hits: list[RetrievalHit] = []
        if req.use_long_term_memory:
            q_emb = self.embed.embed(rewritten_query)
            for record, score in self.ltm.search(req.user_id, q_emb, req.top_k):
                ltm_hits.append(
                    RetrievalHit(
                        source="long_term",
                        text=self._clip_text(record.content, 180),
                        score=score,
                        title=f"memory:{record.type}",
                        doc_id=record.id,
                    )
                )

        hits = sorted(kb_hits + ltm_hits, key=lambda item: item.score, reverse=True)

        mock_score = req.mock_score_value if req.mock_score_value is not None else 78.0
        reply = self._generate_reply(req.content, mock_score, ctx_summary, hits)

        if self.stm is None:
            self._stm_fallback.append_turn(req.session_id, "user", req.content)
            self._stm_fallback.append_turn(req.session_id, "assistant", reply)
        else:
            self.stm.append(req.session_id, "user", req.content)
            self.stm.append(req.session_id, "assistant", reply)
            self._maybe_update_summary(req.session_id)

        return ChatResponse(
            reply=reply,
            citations=hits[: req.top_k],
            memory_used={
                "short_term": bool(ctx_turns or ctx_summary),
                "long_term_hits": len(ltm_hits),
                "kb_hits": len(kb_hits),
            },
            retrieval_summary=self._build_retrieval_summary(kb_hits, ltm_hits),
            mock_score_value=mock_score,
            agent_mode="baseline",
        )

    def add_long_term_memory(self, req: LongTermMemoryCreateRequest) -> LongTermMemoryItem:
        emb = self.embed.embed(req.content)
        item = self.ltm.add(
            req.user_id,
            req.type,
            req.content,
            req.tags,
            emb,
            importance=req.importance,
            expires_at=req.expires_at,
        )
        return LongTermMemoryItem(
            id=item.id,
            user_id=item.user_id,
            type=item.type,
            content=item.content,
            tags=item.tags,
            importance=item.importance,
            created_at=item.created_at,
            expires_at=item.expires_at,
        )

    def list_long_term_memories(self, user_id: str, memory_type: str | None = None) -> list[LongTermMemoryItem]:
        items = self.ltm.list(user_id, memory_type)
        return [
            LongTermMemoryItem(
                id=i.id,
                user_id=i.user_id,
                type=i.type,
                content=i.content,
                tags=i.tags,
                importance=i.importance,
                created_at=i.created_at,
                expires_at=i.expires_at,
            )
            for i in items
        ]

    def delete_long_term_memory(self, user_id: str, memory_id: str) -> bool:
        return self.ltm.delete(user_id, memory_id)

    def _generate_reply(
        self,
        user_query: str,
        mock_score: float,
        summary: str,
        hits: list[RetrievalHit],
    ) -> str:
        refs = (
            "；".join([f"[{h.title or h.source}] {h.text}" for h in hits[:2]])
            if hits
            else "暂无命中知识片段，请明确说明信息不足。"
        )
        context = f"短期记忆摘要：{summary}\n引用片段：{refs}\n模拟评分值：{mock_score:.1f}/100"

        system = (
            "你是篮球训练问答助手。回答要简洁、可执行，并优先依据提供的引用片段。"
            "如果引用信息不足，就明确说明信息不足，不要编造。"
        )

        if self.llm.enabled():
            try:
                return self.llm.chat(system=system, user=user_query, context=context)
            except Exception:
                pass

        prefix = "结合历史对话，" if summary else ""
        return (
            f"{prefix}你的问题是：{user_query}\n"
            f"当前使用的模拟评分值：{mock_score:.1f}/100。\n"
            f"参考知识：{refs}"
        )

    def _maybe_update_summary(self, session_id: str) -> None:
        ctx = self.stm.get(session_id)
        if len(ctx.turns) < self.settings.stm_window_size:
            return
        summary = build_incremental_summary(ctx.summary, ctx.turns)
        self.stm.set_summary(session_id, summary)

    @staticmethod
    def _clip_text(text: str, limit: int) -> str:
        clean = " ".join(text.split())
        if len(clean) <= limit:
            return clean
        return clean[:limit].rstrip() + "..."

    @staticmethod
    def _build_retrieval_summary(
        kb_hits: list[RetrievalHit],
        ltm_hits: list[RetrievalHit],
    ) -> list[str]:
        summary: list[str] = []
        for hit in kb_hits[:2]:
            summary.append(f"kb:{hit.title or hit.doc_id} score={hit.score:.3f}")
        for hit in ltm_hits[:2]:
            summary.append(f"long_term:{hit.title or hit.doc_id} score={hit.score:.3f}")
        return summary
