"""k = k1+k2+k3+k4（瞬时变化率之和）最大帧作为关键帧（readme）。"""

from __future__ import annotations

import numpy as np


def _instantaneous_rates(series: np.ndarray) -> np.ndarray:
    """每帧相对前一帧的绝对变化；首帧为 0。"""
    s = np.asarray(series, dtype=np.float64)
    if s.size == 0:
        return s
    r = np.zeros_like(s, dtype=np.float64)
    if s.size == 1:
        return r
    r[1:] = np.abs(np.diff(s))
    return r


def keyframe_index_from_curves(
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    c4: np.ndarray,
) -> int:
    """
    对 nan 用 nanmean 时整条曲线先向前填充简化：用 0 替代 nan 的变化率。
    """
    k1 = _instantaneous_rates(np.nan_to_num(c1, nan=0.0))
    k2 = _instantaneous_rates(np.nan_to_num(c2, nan=0.0))
    k3 = _instantaneous_rates(np.nan_to_num(c3, nan=0.0))
    k4 = _instantaneous_rates(np.nan_to_num(c4, nan=0.0))
    k_sum = k1 + k2 + k3 + k4
    if k_sum.size == 0:
        return 0
    return int(np.argmax(k_sum))
