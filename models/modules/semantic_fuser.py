# models/modules/semantic_fuser.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class SemanticFusionOutput:
    semantic_tokens: Tensor      # [B, N_cond, D_out]
    scene_token: Tensor          # [B, 1, D_out]
    id_token: Tensor             # [B, 1, D_out]
    nonid_token: Tensor          # [B, 1, D_out]
    pose_token_proj: Tensor      # [B, 1, D_out]
    fused_global: Tensor         # [B, 1, D_out]


class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TokenGatedFusion(nn.Module):
    """
    Fuse two same-shape tokens by a learned gate.
    """

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """
        a, b: [B, 1, D]
        """
        g = self.gate(torch.cat([a, b], dim=-1))
        return g * a + (1.0 - g) * b


class SemanticFuser(nn.Module):
    """
    Semantic-level fusion module.

    Inputs:
    - scene embedding from CLIP image encoder or other scene encoder
    - id_global from IDDecoupler
    - nonid_global from IDDecoupler
    - pose_token from IDDecoupler

    Output:
    - a compact set of condition tokens for UNet conditioning
    """

    def __init__(
        self,
        scene_dim: int,
        token_dim: int = 1024,
        out_dim: int = 768,
        dropout: float = 0.0,
        include_nonid: bool = True,
        include_pose: bool = True,
        append_fused_token: bool = True,
    ) -> None:
        super().__init__()

        self.include_nonid = include_nonid
        self.include_pose = include_pose
        self.append_fused_token = append_fused_token

        self.scene_proj = MLPProjector(scene_dim, out_dim, dropout=dropout)
        self.id_proj = MLPProjector(token_dim, out_dim, dropout=dropout)
        self.nonid_proj = MLPProjector(token_dim, out_dim, dropout=dropout)
        self.pose_proj = MLPProjector(token_dim, out_dim, dropout=dropout)

        self.id_nonid_fusion = TokenGatedFusion(out_dim, dropout=dropout)
        self.global_fusion = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

        self.final_norm = nn.LayerNorm(out_dim)

    def _ensure_token(self, x: Tensor) -> Tensor:
        """
        Accept either [B, D] or [B, 1, D], return [B, 1, D].
        """
        if x.ndim == 2:
            return x.unsqueeze(1)
        if x.ndim == 3 and x.size(1) == 1:
            return x
        raise ValueError(f"Expected shape [B,D] or [B,1,D], got {tuple(x.shape)}")

    def _pool_scene(self, scene_embedding: Tensor) -> Tensor:
        """
        Support scene embedding as either:
        - [B, D]
        - [B, Ns, D]
        Return [B, D]
        """
        if scene_embedding.ndim == 2:
            return scene_embedding
        if scene_embedding.ndim == 3:
            return scene_embedding.mean(dim=1)
        raise ValueError(f"Unsupported scene_embedding shape: {tuple(scene_embedding.shape)}")

    def forward(
        self,
        scene_embedding: Tensor,
        id_global: Tensor,
        nonid_global: Tensor,
        pose_token: Tensor,
    ) -> SemanticFusionOutput:
        """
        scene_embedding: [B, D_scene] or [B, Ns, D_scene]
        id_global:       [B, 1, D_token] or [B, D_token]
        nonid_global:    [B, 1, D_token] or [B, D_token]
        pose_token:      [B, 1, D_token] or [B, D_token]
        """
        scene_embedding = self._pool_scene(scene_embedding)          # [B, D_scene]
        id_global = self._ensure_token(id_global)                   # [B, 1, D_token]
        nonid_global = self._ensure_token(nonid_global)             # [B, 1, D_token]
        pose_token = self._ensure_token(pose_token)                 # [B, 1, D_token]

        scene_token = self.scene_proj(scene_embedding).unsqueeze(1) # [B, 1, D_out]
        id_token = self.id_proj(id_global)                          # [B, 1, D_out]
        nonid_token = self.nonid_proj(nonid_global)                 # [B, 1, D_out]
        pose_token_proj = self.pose_proj(pose_token)                # [B, 1, D_out]

        if self.include_nonid:
            id_sem = self.id_nonid_fusion(id_token, nonid_token)    # [B, 1, D_out]
        else:
            id_sem = id_token

        if self.include_pose:
            fused_global = self.global_fusion(
                torch.cat([scene_token, id_sem, pose_token_proj], dim=-1)
            )
        else:
            fused_global = self.global_fusion(
                torch.cat([scene_token, id_sem, torch.zeros_like(scene_token)], dim=-1)
            )

        tokens = [scene_token, id_token]

        if self.include_nonid:
            tokens.append(nonid_token)

        if self.include_pose:
            tokens.append(pose_token_proj)

        if self.append_fused_token:
            tokens.append(fused_global)

        semantic_tokens = torch.cat(tokens, dim=1)                  # [B, N_cond, D_out]
        semantic_tokens = self.final_norm(semantic_tokens)

        return SemanticFusionOutput(
            semantic_tokens=semantic_tokens,
            scene_token=scene_token,
            id_token=id_token,
            nonid_token=nonid_token,
            pose_token_proj=pose_token_proj,
            fused_global=fused_global,
        )