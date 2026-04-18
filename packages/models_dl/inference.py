"""加载权重、前向推理、拼装规则特征输出。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from features.coaching import angles_summary_dict, coaching_hints_from_keyframe
from features.geometry import four_angle_curves_deg
from features.keyframe import keyframe_index_from_curves
from models_dl.config import load_model_yaml, repo_root
from models_dl.preprocess import normalize_flat, resample_or_pad_flat
from models_dl.schemas import ScoringResult


class ScoringEngine:
    def __init__(self, weights_path: Path | None = None) -> None:
        self._cfg = load_model_yaml()
        paths = self._cfg["paths"]
        self._weights_path = weights_path or (repo_root() / paths["weights_h5"])
        self._model: keras.Model | None = None

    @property
    def model(self) -> keras.Model:
        if self._model is None:
            if not self._weights_path.is_file():
                raise FileNotFoundError(
                    f"模型文件不存在: {self._weights_path}，请先训练并保存到 model_data/model.h5"
                )
            self._model = tf.keras.models.load_model(str(self._weights_path))
        return self._model

    def predict_from_flat(
        self,
        sequence_75: np.ndarray,
        *,
        return_angles: bool = True,
        return_keyframe: bool = True,
    ) -> ScoringResult:
        seq_cfg = self._cfg["sequence"]
        target_t = int(seq_cfg["target_frames"])
        div = float(seq_cfg["normalize_divisor"])

        x = resample_or_pad_flat(np.asarray(sequence_75, dtype=np.float64), target_t)
        x_norm = normalize_flat(x, div)
        batch = np.expand_dims(x_norm, axis=0)

        probs = self.model.predict(batch, verbose=0)[0]
        grade = int(np.argmax(probs)) + 1

        coaching_cfg = self._cfg.get("coaching", {})
        ideal = float(coaching_cfg.get("ideal_elbow_deg", 42.0))
        tol = float(coaching_cfg.get("elbow_tolerance_deg", 10.0))

        key_idx: int | None = None
        angles_summary: dict[str, Any] | None = None
        hints: list[str] = []

        if return_angles or return_keyframe:
            raw = resample_or_pad_flat(np.asarray(sequence_75, dtype=np.float64), target_t)
            txyz = raw.reshape(target_t, 25, 3)
            c1, c2, c3, c4 = four_angle_curves_deg(txyz)
            la, le, ra, re = c1.tolist(), c2.tolist(), c3.tolist(), c4.tolist()
            key_idx = keyframe_index_from_curves(c1, c2, c3, c4)
            if return_angles:
                angles_summary = angles_summary_dict(la, le, ra, re, key_idx)
                lk = le[key_idx] if key_idx < len(le) else float("nan")
                rk = re[key_idx] if key_idx < len(re) else float("nan")
                hints = coaching_hints_from_keyframe(
                    lk if lk == lk else None,
                    rk if rk == rk else None,
                    ideal_elbow=ideal,
                    tolerance=tol,
                )

        return ScoringResult(
            grade=grade,
            probabilities=[float(p) for p in probs],
            keyframe_index=key_idx if return_keyframe else None,
            angles_summary=angles_summary if return_angles else None,
            coaching_hints=hints if return_angles else [],
        )
