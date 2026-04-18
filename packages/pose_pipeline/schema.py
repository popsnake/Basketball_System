from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

BODY_25_NUM_KEYPOINTS = 25
FEATURE_DIM_FLAT = BODY_25_NUM_KEYPOINTS * 3


@dataclass
class PoseFrame:
    """单帧 body_25：25×3，列为 x, y, confidence。"""

    keypoints: np.ndarray  # shape (25, 3), float

    def __post_init__(self) -> None:
        arr = np.asarray(self.keypoints, dtype=np.float64)
        if arr.shape != (BODY_25_NUM_KEYPOINTS, 3):
            raise ValueError(f"keypoints must be (25,3), got {arr.shape}")
        self.keypoints = arr


@dataclass
class PoseSequence:
    """多帧姿态序列，可由 OpenPose JSON 或 numpy 构造。"""

    frames: list[PoseFrame]
    sample_strategy: Literal["dynamic", "fixed"] = "dynamic"
    video_id: str | None = None
    fps: float | None = None
    duration_sec: float | None = None

    def to_tensor_xy_c(self) -> np.ndarray:
        """形状 (T, 25, 3)。"""
        if not self.frames:
            return np.zeros((0, BODY_25_NUM_KEYPOINTS, 3), dtype=np.float64)
        return np.stack([f.keypoints for f in self.frames], axis=0)

    def to_flat75(self) -> np.ndarray:
        """形状 (T, 75)，与 LSTM 输入一致。"""
        t = self.to_tensor_xy_c()
        if t.size == 0:
            return np.zeros((0, FEATURE_DIM_FLAT), dtype=np.float64)
        return t.reshape(t.shape[0], FEATURE_DIM_FLAT)

    @staticmethod
    def from_flat75(flat: np.ndarray) -> PoseSequence:
        a = np.asarray(flat, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != FEATURE_DIM_FLAT:
            raise ValueError(f"flat sequence must be (T,75), got {a.shape}")
        t = a.shape[0]
        frames = [
            PoseFrame(keypoints=a[i].reshape(BODY_25_NUM_KEYPOINTS, 3)) for i in range(t)
        ]
        return PoseSequence(frames=frames)
