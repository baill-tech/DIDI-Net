from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock


@dataclass
class RealUNetDualInjectionOutput:
    sample: Tensor
    encoder_hidden_states: Tensor
    debug_info: Optional[Dict[str, Any]] = None


class IDDetailAttentionLayer(nn.Module):
    """
    Patch-level ID detail injection layer.

    hidden_states: [B, L, C]
    id_patch_tokens: [B, N, D_id]
    """

    def __init__(
        self,
        hidden_dim: int,
        id_token_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.id_token_dim = id_token_dim

        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(id_token_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(id_token_dim, hidden_dim)
        self.v_proj = nn.Linear(id_token_dim, hidden_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: Tensor, id_patch_tokens: Tensor) -> Tensor:
        residual = hidden_states

        q = self.q_proj(self.norm_q(hidden_states))
        kv = self.norm_kv(id_patch_tokens)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        detail_out, _ = self.attn(q, k, v, need_weights=False)
        detail_out = self.out_proj(detail_out)

        return residual + detail_out


class DualInjectedTransformerBlock(nn.Module):
    """
    Wrap a diffusers BasicTransformerBlock and insert ID detail attention
    after cross-attention, following the document's 'semantic path + detail path' idea.

    This wrapper targets the common SD2.1 BasicTransformerBlock structure:
        norm1 -> attn1 -> norm2 -> attn2 -> norm3 -> ff
    """

    def __init__(
        self,
        original_block: BasicTransformerBlock,
        id_token_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = original_block

        # Infer hidden dim from norm1
        hidden_dim = getattr(self.block.norm1, "normalized_shape", None)
        if isinstance(hidden_dim, (tuple, list)):
            hidden_dim = hidden_dim[0]
        elif isinstance(hidden_dim, int):
            hidden_dim = hidden_dim
        else:
            # fallback
            hidden_dim = self.block.attn1.to_q.in_features

        self.hidden_dim = hidden_dim
        self.id_detail_attn = IDDetailAttentionLayer(
            hidden_dim=hidden_dim,
            id_token_dim=id_token_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        class_labels: Optional[Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, Tensor]] = None,
        **kwargs: Any,
    ) -> Tensor:
        cross_attention_kwargs = cross_attention_kwargs or {}
        id_patch_tokens = cross_attention_kwargs.pop("id_patch_tokens", None)

        # ---- self-attention ----
        residual = hidden_states
        hidden_states = self.block.norm1(hidden_states)
        hidden_states = self.block.attn1(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = hidden_states + residual

        # ---- cross-attention (semantic path) ----
        if getattr(self.block, "attn2", None) is not None:
            residual = hidden_states
            hidden_states = self.block.norm2(hidden_states)

            hidden_states = self.block.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = hidden_states + residual

            # ---- detail injection (strictly after cross-attn) ----
            if id_patch_tokens is not None:
                hidden_states = self.id_detail_attn(hidden_states, id_patch_tokens)

        # ---- feed-forward ----
        residual = hidden_states
        hidden_states = self.block.norm3(hidden_states)
        hidden_states = self.block.ff(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class RealUNetWithDualInjection(nn.Module):
    """
    Real Stable Diffusion UNet with:
    - semantic path via encoder_hidden_states
    - detail path via ID patch injection in transformer blocks
    - real diffusion scheduler
    """

    def __init__(
        self,
        model_dir: str,
        id_token_dim: int = 1024,
        freeze_unet: bool = False,
        return_debug_info: bool = False,
    ) -> None:
        super().__init__()

        self.unet = UNet2DConditionModel.from_pretrained(model_dir, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_dir, subfolder="scheduler")
        self.cross_attention_dim = int(self.unet.config.cross_attention_dim)
        self.id_token_dim = id_token_dim
        self.return_debug_info = return_debug_info

        self._replace_transformer_blocks_with_dual_injected()

        if freeze_unet:
            for p in self.unet.parameters():
                p.requires_grad = False

    def _iter_named_modules_with_parent(
        self,
        root: nn.Module,
        prefix: str = "",
    ) -> Iterable[Tuple[nn.Module, str, nn.Module]]:
        """
        Yield (parent_module, child_name, child_module)
        """
        for child_name, child_module in root.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            yield root, child_name, child_module
            yield from self._iter_named_modules_with_parent(child_module, full_name)

    def _replace_transformer_blocks_with_dual_injected(self) -> None:
        replaced = 0
        for parent, child_name, child_module in self._iter_named_modules_with_parent(self.unet):
            if isinstance(child_module, BasicTransformerBlock):
                wrapped = DualInjectedTransformerBlock(
                    original_block=child_module,
                    id_token_dim=self.id_token_dim,
                    num_heads=8,
                    dropout=0.0,
                )
                setattr(parent, child_name, wrapped)
                replaced += 1

        print(f"[RealUNetWithDualInjection] Replaced BasicTransformerBlock count: {replaced}")

    def forward(
        self,
        noisy_latents: Tensor,
        timesteps: Tensor,
        semantic_tokens: Tensor,
        id_patch_tokens: Tensor,
        base_context: Optional[Tensor] = None,
    ) -> RealUNetDualInjectionOutput:
        if base_context is not None:
            encoder_hidden_states = torch.cat([base_context, semantic_tokens], dim=1)
        else:
            encoder_hidden_states = semantic_tokens

        out = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs={"id_patch_tokens": id_patch_tokens},
            return_dict=True,
        )

        debug_info = None
        if self.return_debug_info:
            debug_info = {
                "encoder_hidden_states_shape": tuple(encoder_hidden_states.shape),
                "noisy_latents_shape": tuple(noisy_latents.shape),
                "id_patch_tokens_shape": tuple(id_patch_tokens.shape),
            }

        return RealUNetDualInjectionOutput(
            sample=out.sample,
            encoder_hidden_states=encoder_hidden_states,
            debug_info=debug_info,
        )

    def predict_x0_from_epsilon(
        self,
        noisy_latents: Tensor,
        noise_pred: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """
        Recover x0 estimate from epsilon prediction.
        Useful for auxiliary image-space losses.
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(noisy_latents.device)
        alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_alpha_t = (1.0 - alpha_t).sqrt()

        pred_x0 = (noisy_latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t.clamp(min=1e-6)
        return pred_x0