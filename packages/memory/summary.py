from __future__ import annotations


def build_incremental_summary(
    previous_summary: str,
    turns: list[dict],
    *,
    max_items: int = 6,
    max_chars: int = 320,
) -> str:
    """
    轻量滚动摘要：
    - 保留历史摘要前缀
    - 从最近对话中抽取用户意图和助手建议
    - 控制长度，避免无限膨胀
    """
    snippets: list[str] = []
    if previous_summary:
        snippets.append(previous_summary.replace("历史摘要: ", "").strip())

    for turn in turns[-max_items:]:
        role = turn.get("role", "user")
        content = str(turn.get("content", "")).strip().replace("\n", " ")
        if not content:
            continue
        prefix = "用户" if role == "user" else "助手"
        snippets.append(f"{prefix}:{content[:72]}")

    merged: list[str] = []
    seen: set[str] = set()
    for item in snippets:
        if item and item not in seen:
            merged.append(item)
            seen.add(item)

    summary = " | ".join(merged)
    if len(summary) > max_chars:
        summary = summary[-max_chars:]
    return f"历史摘要: {summary}" if summary else ""
