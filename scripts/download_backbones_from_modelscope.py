from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from modelscope import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_ROOT = PROJECT_ROOT / "pretrained"
PRETRAINED_ROOT.mkdir(parents=True, exist_ok=True)

# 统一在同一个根目录下管理，子目录按 alias 命名
# 当前确定要用的模型已经填好
# 后续模型可以继续往这里追加
MODEL_REGISTRY: Dict[str, Dict[str, Optional[str]]] = {
    # 当前要用
    "clip_scene_encoder": {
        "repo_id": "openai-mirror/clip-vit-base-patch16",
        "revision": None,
        "enabled": True,
    },
    "sd_vae": {
        "repo_id": "stabilityai/sd-vae-ft-mse",
        "revision": None,
        "enabled": True,
    },

    "sd21_base": {
        "repo_id": "stabilityai/stable-diffusion-2-1-base",
        "revision": None,
        "enabled": True,
    },
    
    "dinov2_global": {
        "repo_id": "timm/vit_large_patch14_dinov2.lvd142m",
        "revision": None,
        "enabled": True,
    },

    "sam_model": {
        "repo_id": "facebook/sam-vit-large",
        "revision": None,
        "enabled": True,
    },

    "rs_dino_or_detail_encoder": {
        "repo_id": None,
        "revision": None,
        "enabled": False,
    },
}


def download_one_model(alias: str, repo_id: str, revision: Optional[str] = None) -> str:
    """
    下载模型到统一根目录，并为该模型建立一个固定的本地软链接/指针目录：
      project/pretrained/<alias>

    snapshot_download 实际会下载到 cache_dir 下的模型缓存结构中。
    我们再把最终路径记录到 index.json，供训练脚本统一读取。
    """
    print(f"[Download] {alias} <- {repo_id}")
    local_path = snapshot_download(
        model_id=repo_id,
        revision=revision,
        cache_dir=str(PRETRAINED_ROOT),
    )
    print(f"[Done] {alias}: {local_path}")
    return local_path


def main() -> None:
    download_index = {}

    for alias, meta in MODEL_REGISTRY.items():
        enabled = bool(meta.get("enabled", False))
        repo_id = meta.get("repo_id")
        revision = meta.get("revision")

        if not enabled:
            print(f"[Skip] {alias}: enabled=False")
            continue

        if not repo_id:
            print(f"[Skip] {alias}: repo_id is empty")
            continue

        local_path = download_one_model(
            alias=alias,
            repo_id=repo_id,
            revision=revision,
        )

        download_index[alias] = {
            "repo_id": repo_id,
            "revision": revision,
            "local_path": local_path,
        }

    index_path = PRETRAINED_ROOT / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(download_index, f, ensure_ascii=False, indent=2)

    print(f"\n[Index Saved] {index_path}")
    print(json.dumps(download_index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()