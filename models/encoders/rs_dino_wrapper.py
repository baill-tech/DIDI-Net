from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import importlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RSDINOOutput:
    global_token: Tensor   # [B, 1, D]
    patch_tokens: Tensor   # [B, N, D]
    all_tokens: Tensor     # [B, 1+N, D]


def _import_dino_mc_vits(repo_root: str):
    """
    Import DINO-MC's utils.vision_transformer module the same way the repo expects.

    The repo uses absolute imports like:
        from utils.utils import trunc_normal_

    So repo_root must be on sys.path before import utils.vision_transformer.
    """
    repo_root = str(Path(repo_root).resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    return importlib.import_module("utils.vision_transformer")


def _extract_state_dict(checkpoint: Dict, checkpoint_key: str = "teacher") -> Dict[str, Tensor]:
    if checkpoint_key in checkpoint and isinstance(checkpoint[checkpoint_key], dict):
        return checkpoint[checkpoint_key]

    for key in ["teacher", "student", "model", "state_dict"]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    if isinstance(checkpoint, dict):
        return checkpoint

    raise ValueError("Unsupported checkpoint format.")


def _strip_prefixes(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    prefixes = [
        "module.",
        "backbone.",
        "encoder.",
        "student.",
        "teacher.",
        "model.",
    ]

    out = {}
    for k, v in state_dict.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    changed = True
        out[nk] = v
    return out


class DINO_MC_RSWrapper(nn.Module):
    """
    RS-domain branch implemented directly from DINO-MC repo's ViT.
    """

    def __init__(
        self,
        repo_root: str,
        checkpoint_path: str,
        arch: str = "vit_small",
        patch_size: int = 8,
        image_size: int = 224,
        checkpoint_key: str = "teacher",
        freeze: bool = True,
        normalize_input: bool = True,
        strict: bool = False,
    ) -> None:
        super().__init__()

        self.repo_root = Path(repo_root)
        self.checkpoint_path = str(checkpoint_path)
        self.arch = arch
        self.patch_size = patch_size
        self.image_size = image_size
        self.checkpoint_key = checkpoint_key
        self.normalize_input = normalize_input

        vt_file = self.repo_root / "utils" / "vision_transformer.py"
        if not vt_file.exists():
            raise FileNotFoundError(
                f"Cannot find DINO-MC vision_transformer.py at: {vt_file}"
            )

        dino_mc_vits = _import_dino_mc_vits(str(self.repo_root))

        if arch not in dino_mc_vits.__dict__:
            raise KeyError(f"Architecture '{arch}' not found in DINO-MC utils.vision_transformer")

        self.model = dino_mc_vits.__dict__[arch](
            patch_size=patch_size,
            num_classes=0,
        )

        self.hidden_size = int(getattr(self.model, "embed_dim", 384))

        #checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = _extract_state_dict(checkpoint, checkpoint_key=checkpoint_key)
        state_dict = _strip_prefixes(state_dict)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=strict)
        print(f"[DINO_MC_RSWrapper] load checkpoint from: {self.checkpoint_path}")
        print(f"[DINO_MC_RSWrapper] checkpoint_key: {checkpoint_key}")
        if len(missing) > 0:
            print(f"[DINO_MC_RSWrapper] missing keys: {len(missing)}")
        if len(unexpected) > 0:
            print(f"[DINO_MC_RSWrapper] unexpected keys: {len(unexpected)}")

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def _resize_input(self, x: Tensor) -> Tensor:
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return x

    def _normalize_input(self, x: Tensor) -> Tensor:
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=x.device,
            dtype=x.dtype,
        )[None, :, None, None]
        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=x.device,
            dtype=x.dtype,
        )[None, :, None, None]
        return (x - mean) / std

    def _forward_tokens(self, x: Tensor) -> Tensor:
        if hasattr(self.model, "prepare_tokens"):
            x = self.model.prepare_tokens(x)
            for blk in self.model.blocks:
                x = blk(x)
            x = self.model.norm(x)
            return x

        # fallback
        x = self.model.patch_embed(x)

        if getattr(self.model, "cls_token", None) is not None:
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if getattr(self.model, "pos_embed", None) is not None:
            x = x + self.model.pos_embed[:, : x.shape[1], :]

        if hasattr(self.model, "pos_drop"):
            x = self.model.pos_drop(x)

        for blk in self.model.blocks:
            x = blk(x)

        if hasattr(self.model, "norm"):
            x = self.model.norm(x)

        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self._resize_input(x)
        if self.normalize_input:
            x = self._normalize_input(x)

        tokens = self._forward_tokens(x)
        global_token = tokens[:, :1, :]
        patch_tokens = tokens[:, 1:, :]
        return global_token, patch_tokens

    @torch.no_grad()
    def forward_with_details(self, x: Tensor) -> RSDINOOutput:
        x = self._resize_input(x)
        if self.normalize_input:
            x = self._normalize_input(x)

        tokens = self._forward_tokens(x)
        return RSDINOOutput(
            global_token=tokens[:, :1, :],
            patch_tokens=tokens[:, 1:, :],
            all_tokens=tokens,
        )