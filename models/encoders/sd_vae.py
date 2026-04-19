from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from diffusers import AutoencoderKL


@dataclass
class SDVAEConfig:
    local_dir: str
    scaling_factor: Optional[float] = None
    freeze: bool = True
    use_mode: bool = False


class SDVAEEncoder(nn.Module):
    """
    Stable Diffusion VAE encoder wrapper.

    Input:
        x: [B, 3, H, W] in [0, 1]

    Output:
        z: [B, 4, H/8, W/8]
    """

    def __init__(self, config: SDVAEConfig) -> None:
        super().__init__()
        self.config_wrapper = config
        self.vae = AutoencoderKL.from_pretrained(self.config_wrapper.local_dir)

        if self.config_wrapper.freeze:
            for p in self.vae.parameters():
                p.requires_grad = False

        if self.config_wrapper.scaling_factor is not None:
            self.scaling_factor = self.config_wrapper.scaling_factor
        else:
            self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))

        self.use_mode = self.config_wrapper.use_mode

    def _to_vae_range(self, x: Tensor) -> Tensor:
        return x * 2.0 - 1.0

    def forward(self, x: Tensor) -> Tensor:
        x = self._to_vae_range(x)
        posterior = self.vae.encode(x).latent_dist
        z = posterior.mode() if self.use_mode else posterior.sample()
        z = z * self.scaling_factor
        return z


class SDVAEDecoder(nn.Module):
    """
    Stable Diffusion VAE decoder wrapper.

    Input:
        z: [B, 4, H/8, W/8]

    Output:
        x: [B, 3, H, W] in [0, 1]
    """

    def __init__(self, config: SDVAEConfig) -> None:
        super().__init__()
        self.config_wrapper = config
        self.vae = AutoencoderKL.from_pretrained(self.config_wrapper.local_dir)

        if self.config_wrapper.freeze:
            for p in self.vae.parameters():
                p.requires_grad = False

        if self.config_wrapper.scaling_factor is not None:
            self.scaling_factor = self.config_wrapper.scaling_factor
        else:
            self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))

    def _to_image_range(self, x: Tensor) -> Tensor:
        return (x / 2.0 + 0.5).clamp(0.0, 1.0)

    def forward(self, z: Tensor) -> Tensor:
        z = z / self.scaling_factor
        x = self.vae.decode(z).sample
        x = self._to_image_range(x)
        return x