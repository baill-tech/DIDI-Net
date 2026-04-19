from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "pretrained" / "index.json"


def get_local_model_path(alias: str) -> str:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Model index not found: {INDEX_PATH}. "
            f"Please run scripts/download_backbones_from_modelscope.py first."
        )

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)

    if alias not in index:
        raise KeyError(f"Alias '{alias}' not found in {INDEX_PATH}")

    return index[alias]["local_path"]