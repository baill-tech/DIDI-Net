from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DecoupledFeatures:
    pose_features: Tensor        # [B, P]
    pose_token: Tensor           # [B, 1, D]
    mask_logits: Tensor          # [B, N, D]
    mask_probs: Tensor           # [B, N, D]
    id_tokens: Tensor            # [B, N, D]
    nonid_tokens: Tensor         # [B, N, D]
    id_global: Tensor            # [B, 1, D]
    id_patch: Tensor             # [B, N-1, D]
    nonid_global: Tensor         # [B, 1, D]
    nonid_patch: Tensor          # [B, N-1, D]

    # for MI(KDE)
    id_summary: Optional[Tensor] = None       # [B, Dm]
    nonid_summary: Optional[Tensor] = None    # [B, Dm]


def compute_pose_features_from_mask(loc_mask: Tensor) -> Tensor:
    """
    Extract lightweight geometry prior from location mask.

    loc_mask: [B, 1, H, W], values in {0, 1}
    returns: [B, P]
    """
    if loc_mask.ndim != 4 or loc_mask.size(1) != 1:
        raise ValueError(f"Expected loc_mask shape [B,1,H,W], got {tuple(loc_mask.shape)}")

    bsz, _, h, w = loc_mask.shape

    ys = torch.linspace(0, 1, steps=h, device=loc_mask.device).view(1, 1, h, 1).expand(bsz, 1, h, w)
    xs = torch.linspace(0, 1, steps=w, device=loc_mask.device).view(1, 1, 1, w).expand(bsz, 1, h, w)

    mass = loc_mask.sum(dim=(2, 3), keepdim=False).clamp(min=1.0)  # [B,1]
    area_ratio = mass / float(h * w)

    cx = (loc_mask * xs).sum(dim=(2, 3)) / mass
    cy = (loc_mask * ys).sum(dim=(2, 3)) / mass

    bbox_feats = []
    for i in range(bsz):
        m = loc_mask[i, 0] > 0.5
        if m.any():
            yy, xx = torch.where(m)
            x1 = xx.min().float() / max(1, w - 1)
            x2 = xx.max().float() / max(1, w - 1)
            y1 = yy.min().float() / max(1, h - 1)
            y2 = yy.max().float() / max(1, h - 1)
            bw = (x2 - x1).unsqueeze(0)
            bh = (y2 - y1).unsqueeze(0)
            aspect = bw / bh.clamp(min=1e-6)
            bbox_feats.append(
                torch.cat(
                    [x1.unsqueeze(0), y1.unsqueeze(0), x2.unsqueeze(0), y2.unsqueeze(0), bw, bh, aspect],
                    dim=0,
                )
            )
        else:
            bbox_feats.append(torch.zeros(7, device=loc_mask.device))

    bbox_feats = torch.stack(bbox_feats, dim=0)

    pose = torch.cat(
        [
            area_ratio,   # [B,1]
            cx,           # [B,1]
            cy,           # [B,1]
            bbox_feats,   # [B,7]
        ],
        dim=1,
    )
    return pose


class PoseMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class FeatureSummaryProjector(nn.Module):
    """
    Low-dimensional projection for KDE-based MI estimation.
    """
    def __init__(self, in_dim: int, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class IDDecoupler(nn.Module):
    """
    Improved ID-aware decoupling:
    - supports external explicit pose_prior
    - keeps internal mask-derived pose prior as fallback
    - returns low-dimensional summaries for MI(KDE)
    """

    def __init__(
        self,
        token_dim: int = 1024,
        pose_dim: int = 10,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        mi_dim: int = 64,
    ) -> None:
        super().__init__()

        self.pose_dim = pose_dim
        self.pose_encoder = PoseMLP(in_dim=pose_dim, out_dim=token_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=token_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.mask_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        self.id_summary_proj = FeatureSummaryProjector(token_dim, mi_dim)
        self.nonid_summary_proj = FeatureSummaryProjector(token_dim, mi_dim)

    def forward(
        self,
        fused_tokens: Tensor,
        loc_mask: Tensor,
        pose_prior: Optional[Tensor] = None,
    ) -> DecoupledFeatures:
        """
        fused_tokens: [B, 1+N, D]
        loc_mask:     [B, 1, H, W]
        pose_prior:   [B, P] or None
        """
        if pose_prior is None:
            pose_features = compute_pose_features_from_mask(loc_mask)   # [B, P]
        else:
            if pose_prior.ndim != 2:
                raise ValueError(f"Expected pose_prior shape [B,P], got {tuple(pose_prior.shape)}")
            pose_features = pose_prior

        pose_token = self.pose_encoder(pose_features).unsqueeze(1)      # [B,1,D]

        x = torch.cat([pose_token, fused_tokens], dim=1)                # [B,2+N,D]
        for blk in self.blocks:
            x = blk(x)

        token_states = x[:, 1:, :]                                      # [B,1+N,D]

        mask_logits = self.mask_head(token_states)
        mask_probs = torch.sigmoid(mask_logits)

        id_tokens = mask_probs * fused_tokens
        nonid_tokens = (1.0 - mask_probs) * fused_tokens

        id_global = id_tokens[:, :1, :]
        id_patch = id_tokens[:, 1:, :]

        nonid_global = nonid_tokens[:, :1, :]
        nonid_patch = nonid_tokens[:, 1:, :]

        # pooled summaries for MI(KDE)
        id_summary = self.id_summary_proj(id_tokens.mean(dim=1))            # [B, Dm]
        nonid_summary = self.nonid_summary_proj(nonid_tokens.mean(dim=1))   # [B, Dm]

        return DecoupledFeatures(
            pose_features=pose_features,
            pose_token=pose_token,
            mask_logits=mask_logits,
            mask_probs=mask_probs,
            id_tokens=id_tokens,
            nonid_tokens=nonid_tokens,
            id_global=id_global,
            id_patch=id_patch,
            nonid_global=nonid_global,
            nonid_patch=nonid_patch,
            id_summary=id_summary,
            nonid_summary=nonid_summary,
        )


def _pairwise_sq_dists(x: Tensor) -> Tensor:
    """
    x: [B, D]
    return: [B, B]
    """
    x2 = (x ** 2).sum(dim=1, keepdim=True)
    dist2 = x2 + x2.t() - 2.0 * (x @ x.t())
    return dist2.clamp(min=0.0)


def _kde_log_density(x: Tensor, bandwidth: float, eps: float = 1e-8) -> Tensor:
    """
    Leave-one-out Gaussian KDE log density estimate.
    x: [B, D]
    return: [B]
    """
    bsz, dim = x.shape
    if bsz < 2:
        return torch.zeros(bsz, device=x.device, dtype=x.dtype)

    dist2 = _pairwise_sq_dists(x)                              # [B,B]
    kernel = torch.exp(-dist2 / (2.0 * bandwidth * bandwidth))

    # leave-one-out
    eye = torch.eye(bsz, device=x.device, dtype=x.dtype)
    kernel = kernel * (1.0 - eye)

    denom = max(bsz - 1, 1) * ((2.0 * math.pi * bandwidth * bandwidth) ** (dim / 2.0))
    density = kernel.sum(dim=1) / (denom + eps)
    return torch.log(density + eps)


def estimate_mutual_information_kde(
    x: Tensor,
    y: Tensor,
    bandwidth: float = 0.2,
    eps: float = 1e-8,
) -> Tensor:
    """
    KDE-based MI estimate:
        I(X;Y) = E[ log p(x,y) - log p(x) - log p(y) ]

    x, y: [B, D]
    """
    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape, got {x.shape} vs {y.shape}")

    x = F.normalize(x, dim=-1, eps=eps)
    y = F.normalize(y, dim=-1, eps=eps)
    xy = torch.cat([x, y], dim=-1)

    log_px = _kde_log_density(x, bandwidth=bandwidth, eps=eps)
    log_py = _kde_log_density(y, bandwidth=bandwidth, eps=eps)
    log_pxy = _kde_log_density(xy, bandwidth=bandwidth, eps=eps)

    mi = (log_pxy - log_px - log_py).mean()
    return mi


def decoupling_losses(
    decoupled: DecoupledFeatures,
    eps: float = 1e-6,
    kde_bandwidth: float = 0.2,
) -> Dict[str, Tensor]:
    """
    Document-aligned loss pack:
    - MI(KDE) between ID and non-ID summaries
    - mask entropy regularization
    - pose consistency auxiliary loss
    """
    if decoupled.id_summary is None or decoupled.nonid_summary is None:
        raise ValueError("DecoupledFeatures must contain id_summary and nonid_summary for MI(KDE).")

    pose_features = decoupled.pose_features
    m = decoupled.mask_probs

    # 1) Mutual information via KDE
    loss_mi_kde = estimate_mutual_information_kde(
        decoupled.id_summary,
        decoupled.nonid_summary,
        bandwidth=kde_bandwidth,
        eps=eps,
    )
    loss_mi_kde = torch.clamp(loss_mi_kde, min=0.0)

    # 2) Encourage non-trivial mask (avoid all-0 or all-1 collapse)
    loss_mask_entropy = -(
        m * torch.log(m.clamp(min=eps)) +
        (1.0 - m) * torch.log((1.0 - m).clamp(min=eps))
    ).mean()

    # 3) Keep non-ID branch related to pose / geometry
    target_pose_scalar = pose_features.mean(dim=-1, keepdim=True)         # [B,1]
    pred_pose_scalar = decoupled.nonid_global.mean(dim=-1)                # [B,1]
    loss_pose = F.mse_loss(pred_pose_scalar, target_pose_scalar)

    return {
        "loss_decouple_mi_kde": loss_mi_kde,
        "loss_decouple_mask_entropy": loss_mask_entropy,
        "loss_decouple_pose": loss_pose,
    }