"""序列长度对齐：与 readme 动态/定长抽帧策略配合，推理侧统一为 target_frames。"""

from __future__ import annotations

import numpy as np


def resample_or_pad_flat(sequence_75: np.ndarray, target_frames: int) -> np.ndarray:
    """
    输入 (T,75)，输出 (target_frames, 75)。
    T < target：零填充；T > target：等距下采样索引。
    """
    x = np.asarray(sequence_75, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 75:
        raise ValueError(f"expected (T,75), got {x.shape}")
    t = x.shape[0]
    if t == 0:
        return np.zeros((target_frames, 75), dtype=np.float64)
    if t == target_frames:
        return x.copy()
    if t < target_frames:
        out = np.zeros((target_frames, 75), dtype=np.float64)
        out[:t] = x
        return out
    idx = np.linspace(0, t - 1, target_frames)
    idx = np.round(idx).astype(int)
    return x[idx].copy()


def normalize_flat(sequence_75: np.ndarray, divisor: float) -> np.ndarray:
    if divisor == 0:
        raise ValueError("normalize divisor cannot be 0")
    return np.asarray(sequence_75, dtype=np.float32) / float(divisor)
