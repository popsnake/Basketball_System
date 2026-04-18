"""与 oldcode/train.py 对齐的 LSTM 分类头（5 类）。"""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow import keras


def build_shot_lstm(cfg: dict[str, Any]) -> keras.Model:
    m = cfg["model"]
    seq = cfg["sequence"]
    t = int(seq["target_frames"])
    feat = int(seq["feature_dim"])

    keras.backend.clear_session()
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(t, feat)),
            keras.layers.LSTM(
                int(m["lstm_units_1"]),
                return_sequences=True,
            ),
            keras.layers.Dropout(float(m["dropout"])),
            keras.layers.Dense(int(m["dense_units"])),
            keras.layers.Dropout(float(m["dropout"])),
            keras.layers.LSTM(int(m["lstm_units_2"]), return_sequences=False),
            keras.layers.Dropout(float(m["dropout"])),
            keras.layers.Dense(int(m["num_classes"]), activation="softmax"),
        ]
    )
    return model


def compile_for_training(cfg: dict[str, Any], model: keras.Model) -> None:
    tr = cfg["training"]
    loss = tr.get("loss", "sparse_categorical_crossentropy")
    opt = tr.get("optimizer", "adam")
    model.compile(
        loss=loss,
        optimizer=opt,
        metrics=["accuracy"],
    )
