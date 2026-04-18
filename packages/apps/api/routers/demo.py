from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/demo", tags=["demo"])

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class DemoVideoScoreResponse(BaseModel):
    filename: str
    content_type: str
    bytes_received: int = Field(ge=1)
    duration_sec: float | None = Field(default=None, ge=0)
    score: float = Field(ge=0, le=100)
    grade_label: str
    confidence: float = Field(ge=0, le=1)
    mock_analysis: bool = True
    summary: str
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    probabilities: list[float] = Field(default_factory=list, min_length=5, max_length=5)


def _build_rng(seed_bytes: bytes) -> np.random.Generator:
    digest = hashlib.sha256(seed_bytes).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    return np.random.default_rng(seed)


def _grade_label(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "E"


@router.post("/video_score", response_model=DemoVideoScoreResponse)
async def video_score(
    request: Request,
    x_filename: str | None = Header(default=None),
    x_video_duration: float | None = Header(default=None),
) -> DemoVideoScoreResponse:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="video body is empty")

    filename = unquote((x_filename or "uploaded-video.mp4").strip()) or "uploaded-video.mp4"
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type: {suffix}",
        )

    content_type = request.headers.get("content-type", "application/octet-stream")
    sample = body[: min(len(body), 1024 * 1024)]
    meta = f"{filename}|{len(body)}|{x_video_duration or 0}".encode("utf-8")
    rng = _build_rng(sample + meta)

    duration_sec = round(float(x_video_duration), 2) if x_video_duration is not None else None
    size_mb = len(body) / (1024 * 1024)
    duration_bonus = 4.0 if duration_sec and 1.0 <= duration_sec <= 20.0 else 0.0
    size_bonus = 3.0 if 0.5 <= size_mb <= 80.0 else -2.0
    score = float(np.clip(68.0 + rng.uniform(-8.0, 16.0) + duration_bonus + size_bonus, 52.0, 96.0))
    confidence = float(np.clip(0.6 + rng.uniform(0.05, 0.3), 0.0, 0.98))

    probs = rng.uniform(0.1, 1.0, size=5)
    probs = probs / probs.sum()

    strengths = [
        "节奏感整体稳定，出手动作看起来比较连贯。",
        "上肢发力路径比较清晰，具备继续优化的基础。",
        "本次视频质量适合做快速演示级分析。",
    ]
    improvements = [
        "建议重点观察出手瞬间肘部与手腕的协同。",
        "可以补拍一个更稳定的侧面视角，便于后续做更细的动作判断。",
        "如果想提升评分可信度，建议录制 3 到 8 秒的完整投篮片段。",
    ]

    summary = (
        f"已完成本地 demo 评分。视频大小约 {size_mb:.2f} MB，"
        f"时长 {duration_sec if duration_sec is not None else '未知'} 秒，"
        f"当前返回的是可复现的 mock 分析结果，适合联调上传、评分和问答流程。"
    )

    return DemoVideoScoreResponse(
        filename=filename,
        content_type=content_type,
        bytes_received=len(body),
        duration_sec=duration_sec,
        score=round(score, 1),
        grade_label=_grade_label(score),
        confidence=round(confidence, 2),
        summary=summary,
        strengths=strengths[:2],
        improvements=improvements,
        probabilities=[round(float(p), 4) for p in probs.tolist()],
    )
