"""OpenPose BODY_25 索引与角度曲线（与 readme 四条曲线对应）。"""

from __future__ import annotations

import numpy as np

# OpenPose BODY_25 — 与 CMU 文档一致
class BODY_25:
    NOSE = 0
    NECK = 1
    R_SHOULDER = 2
    R_ELBOW = 3
    R_WRIST = 4
    L_SHOULDER = 5
    L_ELBOW = 6
    L_WRIST = 7
    MID_HIP = 8
    R_HIP = 9
    R_KNEE = 10
    R_ANKLE = 11
    L_HIP = 12
    L_KNEE = 13
    L_ANKLE = 14


def _angle_deg_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, c_floor: float = 0.05) -> float:
    """在 p2 处由 p1-p2-p3 形成的角（度）。低置信度时用 nan。"""
    if p1[2] < c_floor or p2[2] < c_floor or p3[2] < c_floor:
        return float("nan")
    v1 = p1[:2] - p2[:2]
    v2 = p3[:2] - p2[:2]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    cos_ = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_)))


def four_angle_curves_deg(sequence_xy_c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    输入 (T,25,3)。输出四条角度曲线（度），与 readme 中 k1–k4 对应：
    左手抬臂角（颈-左肩-左肘）、左手肘角、右手抬臂角、右手肘角。
    """
    t = sequence_xy_c.shape[0]
    kpts = sequence_xy_c

    def series(idx_fn):
        out = np.full(t, np.nan, dtype=np.float64)
        for i in range(t):
            out[i] = idx_fn(kpts[i])
        return out

    neck, ls, le, lw = BODY_25.NECK, BODY_25.L_SHOULDER, BODY_25.L_ELBOW, BODY_25.L_WRIST
    rs, re, rw = BODY_25.R_SHOULDER, BODY_25.R_ELBOW, BODY_25.R_WRIST

    left_arm = series(lambda k: _angle_deg_2d(k[neck], k[ls], k[le]))
    left_elbow = series(lambda k: _angle_deg_2d(k[ls], k[le], k[lw]))
    right_arm = series(lambda k: _angle_deg_2d(k[neck], k[rs], k[re]))
    right_elbow = series(lambda k: _angle_deg_2d(k[rs], k[re], k[rw]))

    return left_arm, left_elbow, right_arm, right_elbow
