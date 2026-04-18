"""基于关键帧角度与 readme 阈值的规则指导（不依赖 LLM）。"""

from __future__ import annotations

from typing import Any


def coaching_hints_from_keyframe(
    left_elbow_deg: float | None,
    right_elbow_deg: float | None,
    ideal_elbow: float = 42.0,
    tolerance: float = 10.0,
) -> list[str]:
    hints: list[str] = []
    low, high = ideal_elbow - tolerance, ideal_elbow + tolerance

    for name, val in (("左肘", left_elbow_deg), ("右肘", right_elbow_deg)):
        if val is None or val != val:  # nan check
            continue
        if val < low:
            hints.append(f"{name}关节弯曲偏大（关键帧约{val:.0f}°），可参考标准投篮肘角约{ideal_elbow:.0f}°（±{tolerance:.0f}°）。")
        elif val > high:
            hints.append(f"{name}伸展不足或抬臂偏高（关键帧约{val:.0f}°），建议对照肘角约{ideal_elbow:.0f}°（±{tolerance:.0f}°）调整。")

    if not hints:
        hints.append("关键帧肘部角度处于常见参考范围内，可结合整体节奏与出手点继续微调。")

    return hints


def angles_summary_dict(
    left_arm: list[float],
    left_elbow: list[float],
    right_arm: list[float],
    right_elbow: list[float],
    keyframe_index: int,
) -> dict[str, Any]:
    def clip_series(name: str, arr: list[float]) -> dict[str, float | None]:
        if not arr or keyframe_index < 0 or keyframe_index >= len(arr):
            return {"min_deg": None, "max_deg": None, "at_keyframe_deg": None}
        lo = max(0, keyframe_index - 5)
        hi = min(len(arr), keyframe_index + 6)
        seg = [x for x in arr[lo:hi] if x == x]
        at = arr[keyframe_index] if keyframe_index < len(arr) else float("nan")
        return {
            "min_deg": float(min(seg)) if seg else None,
            "max_deg": float(max(seg)) if seg else None,
            "at_keyframe_deg": float(at) if at == at else None,
        }

    return {
        "left_arm": clip_series("left_arm", left_arm),
        "left_elbow": clip_series("left_elbow", left_elbow),
        "right_arm": clip_series("right_arm", right_arm),
        "right_elbow": clip_series("right_elbow", right_elbow),
        "keyframe_index": keyframe_index,
    }
