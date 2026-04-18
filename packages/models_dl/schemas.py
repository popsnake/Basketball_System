from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ScoreSequenceOptions(BaseModel):
    return_angles: bool = True
    return_keyframe_index: bool = True


class ScoreSequenceRequest(BaseModel):
    """sequence: 每帧 75 维，与 25×3 展平顺序一致。"""

    sequence: list[list[float]] = Field(..., description="Shape (T,75)")
    fps: float | None = None
    duration_sec: float | None = None
    options: ScoreSequenceOptions = Field(default_factory=ScoreSequenceOptions)


class ScoringResult(BaseModel):
    grade: int = Field(ge=1, le=5, description="1–5 档")
    probabilities: list[float] = Field(..., min_length=5, max_length=5)
    keyframe_index: int | None = None
    angles_summary: dict[str, Any] | None = None
    coaching_hints: list[str] = Field(default_factory=list)
