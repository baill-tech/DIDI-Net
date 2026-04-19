# models/losses/id_loss.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class IDLossOutput:
    loss_id_total: Tensor
    loss_id_cosine: Optional[Tensor] = None
    loss_id_l2: Optional[Tensor] = None
    pred_features: Optional[Tensor] = None
    ref_features: Optional[Tensor] = None


def masked_average_pool(image: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """
    image: [B, C, H, W]
    mask:  [B, 1, H, W]
    return: [B, C]
    """
    if mask.shape[-2:] != image.shape[-2:]:
        raise ValueError("Mask spatial size must match image spatial size.")

    weighted = image * mask
    denom = mask.sum(dim=(2, 3), keepdim=False).clamp(min=eps)   # [B,1]
    pooled = weighted.sum(dim=(2, 3)) / denom                    # [B,C]
    return pooled


def apply_mask(image: Tensor, mask: Tensor) -> Tensor:
    """
    image: [B, C, H, W]
    mask:  [B, 1, H, W]
    """
    return image * mask


def resize_mask_to(mask: Tensor, target_hw: Tuple[int, int]) -> Tensor:
    return F.interpolate(mask, size=target_hw, mode="nearest")


class SimplePatchFeatureExtractor(nn.Module):
    """
    Fallback feature extractor for smoke test.
    Replace with ViT / DINO / CLIP image encoder later.
    """

    def __init__(self, in_chans: int = 3, feat_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, feat_dim)

    def forward(self, x: Tensor) -> Tensor:
        feat = self.net(x).flatten(1)
        feat = self.proj(feat)
        return feat


class IDConsistencyLoss(nn.Module):
    """
    Compare generated defect region with a reference image/template in feature space.

    Supported use cases:
    - pred_img vs target_img under loc_mask
    - pred_img vs defect_template (if pre-aligned or resized)
    """

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        mode: str = "cosine",
        detach_reference: bool = True,
        normalize_features: bool = True,
    ) -> None:
        super().__init__()
        if mode not in {"cosine", "l2", "cosine+l2"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.feature_extractor = feature_extractor or SimplePatchFeatureExtractor()
        self.mode = mode
        self.detach_reference = detach_reference
        self.normalize_features = normalize_features

    def _extract_features(self, x: Tensor) -> Tensor:
        feat = self.feature_extractor(x)
        if self.normalize_features:
            feat = F.normalize(feat, dim=-1)
        return feat

    def forward(
        self,
        pred_img: Tensor,
        ref_img: Tensor,
        mask: Optional[Tensor] = None,
    ) -> IDLossOutput:
        """
        pred_img: [B, 3, H, W]
        ref_img:  [B, 3, H, W]
        mask:     [B, 1, H, W], optional
        """
        if mask is not None:
            if mask.shape[-2:] != pred_img.shape[-2:]:
                mask = resize_mask_to(mask, pred_img.shape[-2:])

            pred_in = apply_mask(pred_img, mask)
            ref_in = apply_mask(ref_img, mask)
        else:
            pred_in = pred_img
            ref_in = ref_img

        pred_features = self._extract_features(pred_in)
        ref_features = self._extract_features(ref_in)

        if self.detach_reference:
            ref_features = ref_features.detach()

        loss_cosine = None
        loss_l2 = None

        if self.mode in {"cosine", "cosine+l2"}:
            loss_cosine = 1.0 - F.cosine_similarity(pred_features, ref_features, dim=-1).mean()

        if self.mode in {"l2", "cosine+l2"}:
            loss_l2 = F.mse_loss(pred_features, ref_features)

        if self.mode == "cosine":
            loss_total = loss_cosine
        elif self.mode == "l2":
            loss_total = loss_l2
        else:
            loss_total = loss_cosine + loss_l2

        return IDLossOutput(
            loss_id_total=loss_total,
            loss_id_cosine=loss_cosine,
            loss_id_l2=loss_l2,
            pred_features=pred_features,
            ref_features=ref_features,
        )


class TemplateMatchingIDLoss(nn.Module):
    """
    Compare pred defect region with defect template directly.

    Workflow:
    - resize template to prediction size
    - optionally use loc_mask on prediction and template_mask on template
    - extract features and compute similarity
    """

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        mode: str = "cosine",
        detach_reference: bool = True,
        normalize_features: bool = True,
    ) -> None:
        super().__init__()
        self.id_loss = IDConsistencyLoss(
            feature_extractor=feature_extractor,
            mode=mode,
            detach_reference=detach_reference,
            normalize_features=normalize_features,
        )

    def forward(
        self,
        pred_img: Tensor,          # [B,3,H,W]
        defect_template: Tensor,   # [B,3,T,T]
        loc_mask: Optional[Tensor] = None,
        template_mask: Optional[Tensor] = None,
    ) -> IDLossOutput:
        h, w = pred_img.shape[-2:]
        template_resized = F.interpolate(defect_template, size=(h, w), mode="bilinear", align_corners=False)

        if template_mask is not None:
            template_mask = resize_mask_to(template_mask, (h, w))
            template_resized = template_resized * template_mask

        if loc_mask is not None:
            pred_img = pred_img * resize_mask_to(loc_mask, (h, w))

        return self.id_loss(
            pred_img=pred_img,
            ref_img=template_resized,
            mask=None,
        )


def build_id_loss(
    feature_extractor: Optional[nn.Module] = None,
    mode: str = "cosine",
) -> IDConsistencyLoss:
    return IDConsistencyLoss(
        feature_extractor=feature_extractor,
        mode=mode,
        detach_reference=True,
        normalize_features=True,
    )