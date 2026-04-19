from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import timm


@dataclass
class DINOv2Output:
    global_token: Tensor   # [B, 1, D]
    patch_tokens: Tensor   # [B, N, D]
    all_tokens: Tensor     # [B, 1+N, D]


def _load_checkpoint(path: str) -> Dict[str, Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "student", "teacher"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unsupported checkpoint format: {path}")


def _strip_common_prefixes(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    prefixes = [
        "module.",
        "backbone.",
        "encoder.",
        "student.",
        "teacher.",
        "model.",
    ]

    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if new_k.startswith(p):
                    new_k = new_k[len(p):]
                    changed = True
        cleaned[new_k] = v
    return cleaned


class DINOv2Wrapper(nn.Module):
    """
    timm-based DINOv2 wrapper for:
        timm/vit_large_patch14_dinov2.lvd142m

    Input:
        x: [B, 3, H, W], value range [0, 1]

    Output:
        global_token: [B, 1, D]
        patch_tokens: [B, N, D]
    """

    def __init__(
        self,
        backbone_name: str,
        checkpoint_path: str,
        image_size: int = 518,
        freeze: bool = True,
        normalize_input: bool = True,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.checkpoint_path = checkpoint_path
        self.image_size = image_size
        self.normalize_input = normalize_input

        self.model = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            img_size=image_size,
        )

        state_dict = _load_checkpoint(checkpoint_path)
        state_dict = _strip_common_prefixes(state_dict)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=strict)
        print(f"[DINOv2Wrapper] load checkpoint from: {checkpoint_path}")
        if len(missing) > 0:
            print(f"[DINOv2Wrapper] missing keys: {len(missing)}")
        if len(unexpected) > 0:
            print(f"[DINOv2Wrapper] unexpected keys: {len(unexpected)}")

        self.hidden_size = int(getattr(self.model, "embed_dim", 1024))

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

    def _forward_tokens_timm_vit(self, x: Tensor) -> Tensor:
        """
        Return full token sequence [B, 1+N, D] for timm VisionTransformer-like models.

        Important:
        For timm DINOv2 ViT models, `_pos_embed()` already handles prefix tokens
        (e.g. cls token). So do NOT manually prepend cls_token before calling it.
        """
        x = self.model.patch_embed(x)  # [B, N, D]

        if hasattr(self.model, "_pos_embed"):
        # timm VisionTransformer path:
        # _pos_embed() internally handles cls/reg tokens
            x = self.model._pos_embed(x)
        else:
            # generic fallback path
            if getattr(self.model, "cls_token", None) is not None:
                cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_token, x), dim=1)

            if getattr(self.model, "pos_embed", None) is not None:
                x = x + self.model.pos_embed[:, : x.shape[1], :]

            if hasattr(self.model, "pos_drop"):
                x = self.model.pos_drop(x)

        if hasattr(self.model, "patch_drop"):
            x = self.model.patch_drop(x)

        if hasattr(self.model, "norm_pre"):
            x = self.model.norm_pre(x)

        x = self.model.blocks(x)

        if hasattr(self.model, "norm"):
            x = self.model.norm(x)

        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self._resize_input(x)
        if self.normalize_input:
            x = self._normalize_input(x)

        tokens = self._forward_tokens_timm_vit(x)   # [B, 1+N, D]
        global_token = tokens[:, :1, :]
        patch_tokens = tokens[:, 1:, :]

        return global_token, patch_tokens

    @torch.no_grad()
    def forward_with_details(self, x: Tensor) -> DINOv2Output:
        x = self._resize_input(x)
        if self.normalize_input:
            x = self._normalize_input(x)

        tokens = self._forward_tokens_timm_vit(x)

        return DINOv2Output(
            global_token=tokens[:, :1, :],
            patch_tokens=tokens[:, 1:, :],
            all_tokens=tokens,
        )