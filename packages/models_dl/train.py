"""
训练入口：从 .npz 或演示随机数据拟合并保存 model.h5。

示例 .npz 键：X_train, y_train, X_val（可选）, y_val（可选）
X_* 形状 (N, T, 75)，y_* 整数标签 0–4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from models_dl.config import load_model_yaml, repo_root
from models_dl.model import build_shot_lstm, compile_for_training


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_train(
    npz_path: Path | None,
    epochs: int | None,
    out_path: Path | None,
    demo: bool,
) -> None:
    cfg = load_model_yaml()
    tr = cfg["training"]
    ep = int(epochs if epochs is not None else tr.get("epochs_dev" if demo else "epochs", 5))
    seq = cfg["sequence"]
    t, f_ = int(seq["target_frames"]), int(seq["feature_dim"])
    div = float(seq["normalize_divisor"])
    n_classes = int(cfg["model"]["num_classes"])

    out = out_path or (repo_root() / cfg["paths"]["weights_h5"])
    _ensure_parent(out)

    if demo or npz_path is None:
        rng = np.random.default_rng(0)
        n = 32
        x = rng.uniform(0, 1500, size=(n, t, f_)).astype(np.float32)
        y = rng.integers(0, n_classes, size=(n,), dtype=np.int64)
        x_train, y_train = x[:24], y[:24]
        x_val, y_val = x[24:], y[24:]
    else:
        data = np.load(npz_path)
        x_train = np.asarray(data["X_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=np.int64).ravel()
        if "X_val" in data and "y_val" in data:
            x_val = np.asarray(data["X_val"], dtype=np.float32)
            y_val = np.asarray(data["y_val"], dtype=np.int64).ravel()
        else:
            x_val, y_val = x_train, y_train

    x_train = x_train / div
    x_val = x_val / div

    model = build_shot_lstm(cfg)
    compile_for_training(cfg, model)
    bs = int(tr.get("batch_size", 15))

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=bs,
        epochs=ep,
        verbose=1,
    )
    model.save(str(out))
    print(f"Saved model to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train LSTM shot classifier")
    p.add_argument("--npz", type=Path, default=None, help="Training npz file")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--out", type=Path, default=None, help="Output .h5 path")
    p.add_argument("--demo", action="store_true", help="Train on random data (smoke test)")
    args = p.parse_args()
    if not args.demo and args.npz is None:
        print("Use --demo for a quick run, or provide --npz path.", file=sys.stderr)
        sys.exit(2)
    run_train(args.npz, args.epochs, args.out, args.demo)


if __name__ == "__main__":
    main()
