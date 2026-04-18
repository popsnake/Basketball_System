from agents.baseline.pipeline import BaselineAgentService
from agents.baseline.schemas import ChatRequest, LongTermMemoryCreateRequest


def test_baseline_chat_with_mock_score():
    svc = BaselineAgentService()
    req = ChatRequest(
        session_id="s1",
        user_id="u1",
        content="我该如何改进投篮稳定性",
        mock_score_value=82.5,
    )
    resp = svc.chat(req)
    # 检查回复是否包含篮球相关建议，且mock_score_value正确
    assert "投篮" in resp.reply or "稳定性" in resp.reply
    assert resp.mock_score_value == 82.5


def test_long_term_memory_roundtrip():
    svc = BaselineAgentService()
    item = svc.add_long_term_memory(
        LongTermMemoryCreateRequest(
            user_id="u1",
            type="goal",
            content="希望三个月内提升三分命中率",
            tags=["训练目标"],
        )
    )
    items = svc.list_long_term_memories("u1")
    assert item.id in [i.id for i in items]
