# models/losses/recon_loss.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ReconLossOutput:
    loss_recon_total: Tensor
    loss_mask_l1: Optional[Tensor] = None
    loss_mask_l2: Optional[Tensor] = None
    loss_global_l1: Optional[Tensor] = None
    loss_global_l2: Optional[Tensor] = None


def resize_mask_to(mask: Tensor, target_hw: Tuple[int, int]) -> Tensor:
    return F.interpolate(mask, size=target_hw, mode="nearest")


def masked_l1_loss(pred: Tensor, target: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    diff = torch.abs(pred - target) * mask
    denom = mask.sum(dim=(1, 2, 3)).clamp(min=eps)
    per_sample = diff.sum(dim=(1, 2, 3)) / denom
    return per_sample.mean()


def masked_l2_loss(pred: Tensor, target: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    diff = ((pred - target) ** 2) * mask
    denom = mask.sum(dim=(1, 2, 3)).clamp(min=eps)
    per_sample = diff.sum(dim=(1, 2, 3)) / denom
    return per_sample.mean()


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss focused on defect region.
    """

    def __init__(
        self,
        use_mask_l1: bool = True,
        use_mask_l2: bool = False,
        use_global_l1: bool = False,
        use_global_l2: bool = False,
        weight_mask_l1: float = 1.0,
        weight_mask_l2: float = 1.0,
        weight_global_l1: float = 0.0,
        weight_global_l2: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_mask_l1 = use_mask_l1
        self.use_mask_l2 = use_mask_l2
        self.use_global_l1 = use_global_l1
        self.use_global_l2 = use_global_l2

        self.weight_mask_l1 = weight_mask_l1
        self.weight_mask_l2 = weight_mask_l2
        self.weight_global_l1 = weight_global_l1
        self.weight_global_l2 = weight_global_l2

    def forward(
        self,
        pred_img: Tensor,
        target_img: Tensor,
        loc_mask: Optional[Tensor] = None,
    ) -> ReconLossOutput:
        """
        pred_img:   [B,3,H,W]
        target_img: [B,3,H,W]
        loc_mask:   [B,1,H,W], optional
        """
        if pred_img.shape != target_img.shape:
            raise ValueError(f"pred_img shape {pred_img.shape} != target_img shape {target_img.shape}")

        if loc_mask is None:
            loc_mask = torch.ones(
                pred_img.size(0),
                1,
                pred_img.size(2),
                pred_img.size(3),
                device=pred_img.device,
                dtype=pred_img.dtype,
            )
        elif loc_mask.shape[-2:] != pred_img.shape[-2:]:
            loc_mask = resize_mask_to(loc_mask, pred_img.shape[-2:])

        if loc_mask.size(1) == 1 and pred_img.size(1) > 1:
            loc_mask_rgb = loc_mask.expand(-1, pred_img.size(1), -1, -1)
        else:
            loc_mask_rgb = loc_mask

        loss_mask_l1 = None
        loss_mask_l2 = None
        loss_global_l1 = None
        loss_global_l2 = None

        total = pred_img.new_tensor(0.0)

        if self.use_mask_l1:
            loss_mask_l1 = masked_l1_loss(pred_img, target_img, loc_mask_rgb)
            total = total + self.weight_mask_l1 * loss_mask_l1

        if self.use_mask_l2:
            loss_mask_l2 = masked_l2_loss(pred_img, target_img, loc_mask_rgb)
            total = total + self.weight_mask_l2 * loss_mask_l2

        if self.use_global_l1:
            loss_global_l1 = F.l1_loss(pred_img, target_img)
            total = total + self.weight_global_l1 * loss_global_l1

        if self.use_global_l2:
            loss_global_l2 = F.mse_loss(pred_img, target_img)
            total = total + self.weight_global_l2 * loss_global_l2

        return ReconLossOutput(
            loss_recon_total=total,
            loss_mask_l1=loss_mask_l1,
            loss_mask_l2=loss_mask_l2,
            loss_global_l1=loss_global_l1,
            loss_global_l2=loss_global_l2,
        )
    
def background_l1_loss(pred: Tensor, scene: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """
    pred:  [B,3,H,W]
    scene: [B,3,H,W]
    mask:  [B,1,H,W], 1 means defect region
    """
    if mask.shape[-2:] != pred.shape[-2:]:
        mask = resize_mask_to(mask, pred.shape[-2:])

    if mask.size(1) == 1 and pred.size(1) > 1:
        mask = mask.expand(-1, pred.size(1), -1, -1)

    bg_mask = 1.0 - mask
    diff = torch.abs(pred - scene) * bg_mask
    denom = bg_mask.sum(dim=(1, 2, 3)).clamp(min=eps)
    per_sample = diff.sum(dim=(1, 2, 3)) / denom
    return per_sample.mean()

def build_recon_loss() -> ReconstructionLoss:
    return ReconstructionLoss(
        use_mask_l1=True,
        use_mask_l2=False,
        use_global_l1=False,
        use_global_l2=False,
        weight_mask_l1=1.0,
        weight_mask_l2=1.0,
        weight_global_l1=0.0,
        weight_global_l2=0.0,
    )