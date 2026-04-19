# models/modules/id_attention.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class IDAttentionOutput:
    output: Tensor                 # [B, C, H, W]
    attn_map: Optional[Tensor]     # [B, heads, HW, N] or None
    delta: Tensor                  # [B, C, H, W]


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class IDCrossAttention(nn.Module):
    """
    Cross-attention from UNet spatial features (query) to ID patch tokens (key/value).

    Query: feature_map -> [B, HW, D]
    Key/Value: id_patch_tokens -> [B, N, D]
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        return_attention: bool = False,
    ) -> None:
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        self.return_attention = return_attention

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def _reshape_heads(self, x: Tensor) -> Tensor:
        """
        x: [B, L, H*D]
        -> [B, heads, L, D]
        """
        b, l, _ = x.shape
        x = x.view(b, l, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def _merge_heads(self, x: Tensor) -> Tensor:
        """
        x: [B, heads, L, D]
        -> [B, L, heads*D]
        """
        b, h, l, d = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, l, h * d)
        return x

    def forward(
        self,
        query_tokens: Tensor,
        context_tokens: Tensor,
        context_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        query_tokens:   [B, Lq, Cq]
        context_tokens: [B, Lc, Cc]
        context_mask:   [B, Lc] with 1 for valid, 0 for invalid

        returns:
            attended: [B, Lq, Cq]
            attn:     [B, heads, Lq, Lc] or None
        """
        q = self.to_q(query_tokens)
        k = self.to_k(context_tokens)
        v = self.to_v(context_tokens)

        q = self._reshape_heads(q)   # [B, H, Lq, Dh]
        k = self._reshape_heads(k)   # [B, H, Lc, Dh]
        v = self._reshape_heads(v)   # [B, H, Lc, Dh]

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # [B,H,Lq,Lc]

        if context_mask is not None:
            if context_mask.ndim != 2:
                raise ValueError(f"context_mask must be [B, Lc], got {tuple(context_mask.shape)}")
            mask = context_mask[:, None, None, :].to(dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn, v)                                     # [B,H,Lq,Dh]
        attended = self._merge_heads(attended)                               # [B,Lq,H*Dh]
        attended = self.to_out(attended)                                     # [B,Lq,Cq]

        if self.return_attention:
            return attended, attn
        return attended, None


class SpatialTokenAdapter(nn.Module):
    """
    Convert [B, C, H, W] <-> [B, HW, D] for attention.
    """

    def __init__(self, in_channels: int, token_dim: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.token_dim = token_dim

        self.norm = nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(in_channels, token_dim, kernel_size=1)
        self.proj_out = nn.Conv2d(token_dim, in_channels, kernel_size=1)

    def to_tokens(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """
        x: [B,C,H,W]
        returns:
            tokens: [B, HW, D]
            spatial_shape: (H, W)
        """
        x = self.norm(x)
        x = self.proj_in(x)                          # [B,D,H,W]
        b, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)       # [B,HW,D]
        return tokens, (h, w)

    def to_feature_map(self, tokens: Tensor, spatial_shape: Tuple[int, int]) -> Tensor:
        """
        tokens: [B, HW, D]
        returns: [B,C,H,W]
        """
        h, w = spatial_shape
        b, hw, d = tokens.shape
        if hw != h * w:
            raise ValueError(f"Token length {hw} does not match spatial size {h}x{w}")
        x = tokens.transpose(1, 2).reshape(b, d, h, w)
        x = self.proj_out(x)
        return x


class IDAttentionBlock(nn.Module):
    """
    Detail-level ID injection block.

    Steps:
    1) feature_map -> spatial tokens
    2) cross-attention(query=feature tokens, context=id_patch_tokens)
    3) residual add
    4) optional FFN refinement in token space
    5) project back to feature map
    """

    def __init__(
        self,
        in_channels: int,
        id_token_dim: int,
        attn_token_dim: int = 320,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        use_ffn: bool = True,
        use_residual: bool = True,
        residual_scale: float = 1.0,
        return_attention: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.id_token_dim = id_token_dim
        self.attn_token_dim = attn_token_dim
        self.use_ffn = use_ffn
        self.use_residual = use_residual
        self.residual_scale = residual_scale
        self.return_attention = return_attention

        self.adapter = SpatialTokenAdapter(in_channels=in_channels, token_dim=attn_token_dim)

        self.context_norm = nn.LayerNorm(id_token_dim)
        self.context_proj = nn.Linear(id_token_dim, attn_token_dim)

        self.attn = IDCrossAttention(
            query_dim=attn_token_dim,
            context_dim=attn_token_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            return_attention=return_attention,
        )

        self.query_norm = nn.LayerNorm(attn_token_dim)
        self.ffn = FeedForward(attn_token_dim, dropout=dropout) if use_ffn else None

    def forward(
        self,
        feature_map: Tensor,
        id_patch_tokens: Tensor,
        id_token_mask: Optional[Tensor] = None,
    ) -> IDAttentionOutput:
        """
        feature_map:     [B, C, H, W]
        id_patch_tokens: [B, N, D_id]
        id_token_mask:   [B, N], optional
        """
        residual = feature_map

        query_tokens, spatial_shape = self.adapter.to_tokens(feature_map)          # [B,HW,Dattn]
        query_tokens_in = self.query_norm(query_tokens)

        context_tokens = self.context_norm(id_patch_tokens)
        context_tokens = self.context_proj(context_tokens)                         # [B,N,Dattn]

        attn_out, attn_map = self.attn(
            query_tokens=query_tokens_in,
            context_tokens=context_tokens,
            context_mask=id_token_mask,
        )

        query_tokens = query_tokens + attn_out

        if self.ffn is not None:
            query_tokens = query_tokens + self.ffn(self.query_norm(query_tokens))

        delta = self.adapter.to_feature_map(query_tokens, spatial_shape)           # [B,C,H,W]

        if self.use_residual:
            output = residual + self.residual_scale * delta
        else:
            output = delta

        return IDAttentionOutput(
            output=output,
            attn_map=attn_map,
            delta=delta,
        )


class MultiScaleIDAttentionInjector(nn.Module):
    """
    A lightweight manager for applying several IDAttentionBlock modules
    to a list of UNet feature maps at different scales.
    """

    def __init__(
        self,
        in_channels_list: list[int],
        id_token_dim: int,
        attn_token_dim: int = 320,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        use_ffn: bool = True,
        use_residual: bool = True,
        residual_scale: float = 1.0,
        return_attention: bool = False,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([
            IDAttentionBlock(
                in_channels=c,
                id_token_dim=id_token_dim,
                attn_token_dim=attn_token_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                use_ffn=use_ffn,
                use_residual=use_residual,
                residual_scale=residual_scale,
                return_attention=return_attention,
            )
            for c in in_channels_list
        ])

    def forward(
        self,
        feature_maps: list[Tensor],
        id_patch_tokens: Tensor,
        id_token_mask: Optional[Tensor] = None,
    ) -> tuple[list[Tensor], list[Optional[Tensor]]]:
        """
        feature_maps: list of [B,C,H,W]
        returns:
            outputs:   list of [B,C,H,W]
            attn_maps: list of attention maps or None
        """
        if len(feature_maps) != len(self.blocks):
            raise ValueError(f"Expected {len(self.blocks)} feature maps, got {len(feature_maps)}")

        outputs = []
        attn_maps = []

        for fmap, block in zip(feature_maps, self.blocks):
            out = block(
                feature_map=fmap,
                id_patch_tokens=id_patch_tokens,
                id_token_mask=id_token_mask,
            )
            outputs.append(out.output)
            attn_maps.append(out.attn_map)

        return outputs, attn_maps