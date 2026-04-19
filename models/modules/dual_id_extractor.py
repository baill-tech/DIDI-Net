from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DualIDFeatures:
    global_dino: Tensor       # [B, 1, D]
    patch_dino: Tensor        # [B, N, D]
    global_rs: Tensor         # [B, 1, D]
    patch_rs: Tensor         # [B, N, D]
    global_fused: Tensor      # [B, 1, D]
    patch_fused: Tensor       # [B, N, D]
    fused_tokens: Tensor      # [B, 1+N, D]
    fusion_weights: Optional[Tensor] = None   # [B, 1+N, D]


class MLPProjector(nn.Module):
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


class DummyVisionTokenEncoder(nn.Module):
    """
    Fallback encoder for smoke test only.
    Real runs should pass encoder_dino / encoder_rs from outside.
    """
    def __init__(self, in_chans: int = 3, embed_dim: int = 768, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x: [B, 3, H, W]
        returns:
            global_token: [B, 1, D]
            patch_tokens: [B, N, D]
        """
        bsz = x.size(0)
        feat = self.patch_embed(x)                 # [B, D, H', W']
        feat = feat.flatten(2).transpose(1, 2)     # [B, N, D]
        feat = self.norm(feat)

        cls = self.cls_token.expand(bsz, -1, -1)
        global_token = cls + feat.mean(dim=1, keepdim=True)
        return global_token, feat


def align_token_length(x: Tensor, target_len: int) -> Tensor:
    """
    x: [B, N, D]
    target_len: desired token length
    Use adaptive average pooling over token dimension instead of naive truncation.
    """
    if x.size(1) == target_len:
        return x
    x = x.transpose(1, 2)                    # [B, D, N]
    x = F.adaptive_avg_pool1d(x, target_len)
    x = x.transpose(1, 2)                    # [B, target_len, D]
    return x


class SelfAttentionHadamardFusion(nn.Module):
    """
    Document-style dynamic fusion:
    1) concatenate two source sequences
    2) use self-attention to build contextualized representations
    3) predict fusion weights
    4) fuse by Hadamard product

    fused = w ⊙ a + (1 - w) ⊙ b
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_ffn: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.use_ffn = use_ffn

        self.norm_in = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(dim)

        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
            )
            self.norm_ffn = nn.LayerNorm(dim)

        self.weight_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
        """
        a, b: [B, L, D]
        returns:
            fused:   [B, L, D]
            weights: [B, L, D]
        """
        if a.shape != b.shape:
            raise ValueError(f"Input shapes must match, got {a.shape} vs {b.shape}")

        bsz, seq_len, dim = a.shape

        joint = torch.cat([a, b], dim=1)            # [B, 2L, D]
        joint_in = self.norm_in(joint)
        joint_attn, _ = self.attn(joint_in, joint_in, joint_in, need_weights=False)
        joint = self.norm_attn(joint + joint_attn)

        if self.use_ffn:
            joint = self.norm_ffn(joint + self.ffn(joint))

        ctx_a = joint[:, :seq_len, :]
        ctx_b = joint[:, seq_len:, :]

        # self-attention-derived dynamic weight
        weights = self.weight_mlp(torch.cat([ctx_a, ctx_b], dim=-1))  # [B, L, D]

        # Hadamard dynamic fusion
        fused = weights * a + (1.0 - weights) * b
        return fused, weights


class DualIDExtractor(nn.Module):
    """
    More document-aligned executable version of dual-level ID extraction.

    Main changes vs old version:
    - keep two real branches (DINOv2 / RS branch)
    - independent token-wise MLP projectors
    - align token lengths via adaptive pooling instead of truncation
    - build full sequences first
    - use self-attention + Hadamard dynamic fusion
    """

    def __init__(
        self,
        dino_in_dim: int = 768,
        rs_in_dim: int = 768,
        fused_dim: int = 1024,
        encoder_dino: Optional[nn.Module] = None,
        encoder_rs: Optional[nn.Module] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.encoder_dino = encoder_dino or DummyVisionTokenEncoder(embed_dim=dino_in_dim)
        self.encoder_rs = encoder_rs or DummyVisionTokenEncoder(embed_dim=rs_in_dim)

        # independent MLPs for two branches
        self.proj_dino_global = MLPProjector(dino_in_dim, fused_dim)
        self.proj_dino_patch = MLPProjector(dino_in_dim, fused_dim)

        self.proj_rs_global = MLPProjector(rs_in_dim, fused_dim)
        self.proj_rs_patch = MLPProjector(rs_in_dim, fused_dim)

        # document-style dynamic fusion
        self.sequence_fusion = SelfAttentionHadamardFusion(
            dim=fused_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_ffn=True,
        )

    def forward(self, defect_template: Tensor) -> DualIDFeatures:
        """
        defect_template: [B, 3, T, T]
        """
        # 1) extract two source sequences
        g_dino_raw, p_dino_raw = self.encoder_dino(defect_template)
        g_rs_raw, p_rs_raw = self.encoder_rs(defect_template)

        # 2) independent MLP projection into shared space
        g_dino = self.proj_dino_global(g_dino_raw)   # [B,1,D]
        p_dino = self.proj_dino_patch(p_dino_raw)    # [B,N1,D]

        g_rs = self.proj_rs_global(g_rs_raw)         # [B,1,D]
        p_rs = self.proj_rs_patch(p_rs_raw)          # [B,N2,D]

        # 3) align patch token counts
        if p_dino.size(1) != p_rs.size(1):
            target_len = max(p_dino.size(1), p_rs.size(1))
            p_dino = align_token_length(p_dino, target_len)
            p_rs = align_token_length(p_rs, target_len)

        # 4) build full sequences first: [global ; patch]
        seq_dino = torch.cat([g_dino, p_dino], dim=1)   # [B,1+N,D]
        seq_rs = torch.cat([g_rs, p_rs], dim=1)         # [B,1+N,D]

        # 5) self-attention + Hadamard dynamic fusion
        fused_tokens, fusion_weights = self.sequence_fusion(seq_dino, seq_rs)

        # 6) split back to global / patch for downstream compatibility
        g_fused = fused_tokens[:, :1, :]
        p_fused = fused_tokens[:, 1:, :]

        return DualIDFeatures(
            global_dino=g_dino,
            patch_dino=p_dino,
            global_rs=g_rs,
            patch_rs=p_rs,
            global_fused=g_fused,
            patch_fused=p_fused,
            fused_tokens=fused_tokens,
            fusion_weights=fusion_weights,
        )