from __future__ import annotations


def rewrite_query(content: str, summary: str, recent_turns: list[dict]) -> str:
    needs_context = any(k in content for k in ("它", "这个", "这次", "上次", "刚才"))
    if not needs_context:
        return content
    tail = []
    for turn in recent_turns[-2:]:
        role = turn.get("role", "user")
        text = turn.get("content", "")
        tail.append(f"{role}: {text}")
    ctx = summary or (" | ".join(tail) if tail else "")
    return f"{content}。上下文：{ctx}"
