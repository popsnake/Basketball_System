from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache
def load_model_yaml() -> dict[str, Any]:
    path = repo_root() / "configs" / "model.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
