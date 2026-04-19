from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import CLIPVisionModel


@dataclass
class CLIPSceneEncoderOutput:
    image_embeds: Tensor
    pooled_output: Tensor
    last_hidden_state: Tensor


class CLIPSceneEncoder(nn.Module):
    def __init__(
        self,
        local_dir: str,
        freeze: bool = True,
        output_dim: int | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.local_dir = local_dir
        self.vision_model = CLIPVisionModel.from_pretrained(self.local_dir)
        self.hidden_size = self.vision_model.config.hidden_size
        self.normalize = normalize

        # CLIP expects square inputs, typically 224
        self.image_size = int(getattr(self.vision_model.config, "image_size", 224))

        if output_dim is None or output_dim == self.hidden_size:
            self.output_proj = nn.Identity()
            self.output_dim = self.hidden_size
        else:
            self.output_proj = nn.Linear(self.hidden_size, output_dim)
            self.output_dim = output_dim

        if freeze:
            for p in self.vision_model.parameters():
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

    def _normalize_clip_input(self, x: Tensor) -> Tensor:
        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=x.device,
            dtype=x.dtype,
        )[None, :, None, None]
        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=x.device,
            dtype=x.dtype,
        )[None, :, None, None]
        return (x - mean) / std

    def forward(self, scene_img: Tensor) -> Tensor:
        x = self._resize_input(scene_img)
        x = self._normalize_clip_input(x)

        out = self.vision_model(pixel_values=x)
        pooled = out.pooler_output
        image_embeds = self.output_proj(pooled)

        if self.normalize:
            image_embeds = F.normalize(image_embeds, dim=-1)

        return image_embeds

    @torch.no_grad()
    def forward_with_details(self, scene_img: Tensor):
        x = self._resize_input(scene_img)
        x = self._normalize_clip_input(x)

        out = self.vision_model(pixel_values=x)
        pooled = out.pooler_output
        image_embeds = self.output_proj(pooled)

        if self.normalize:
            image_embeds = F.normalize(image_embeds, dim=-1)

        return {
            "image_embeds": image_embeds,
            "pooled_output": pooled,
            "last_hidden_state": out.last_hidden_state,
        }