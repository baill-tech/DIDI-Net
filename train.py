from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets.dataset import (
    MVTecDefectSynthesisDataset,
    build_records_from_manifest,
)
from models.modules.dual_id_extractor import DualIDExtractor
from models.modules.id_decoupler import IDDecoupler, decoupling_losses
from models.modules.semantic_fuser import SemanticFuser
from models.losses.id_loss import build_id_loss, TemplateMatchingIDLoss
from models.losses.recon_loss import build_recon_loss, background_l1_loss
from models.diffusion.real_unet_with_dual_injection import RealUNetWithDualInjection
from models.encoders.clip_scene_encoder import CLIPSceneEncoder
from models.encoders.sd_vae import SDVAEConfig, SDVAEEncoder, SDVAEDecoder
from models.encoders.dinov2_wrapper import DINOv2Wrapper
from models.encoders.rs_dino_wrapper import DINO_MC_RSWrapper
from models.encoders.model_paths import get_local_model_path

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =========================
# Utilities
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = move_to_device(v, device)
        else:
            out[k] = v
    return out

def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True

def freeze_unet_but_keep_injection(model: nn.Module) -> None:
    """
    Freeze the base diffusers UNet weights, but keep custom ID injection layers trainable.
    Assumes:
      model.unet is RealUNetWithDualInjection
      model.unet.unet is the inner diffusers UNet2DConditionModel
      wrapped transformer blocks expose `.id_detail_attn`
    """
    # 1) freeze the whole inner UNet
    freeze_module(model.unet.unet)

    # 2) unfreeze only custom detail injection layers
    for module in model.unet.unet.modules():
        if hasattr(module, "id_detail_attn"):
            unfreeze_module(module.id_detail_attn)

    # 3) if RealUNetWithDualInjection itself has extra projection/norm layers outside inner UNet,
    #    keep them trainable as well
    for name, p in model.unet.named_parameters():
        if ("semantic_proj" in name
            or "id_proj" in name
            or "context_norm" in name
            or "id_reducer" in name):
            p.requires_grad = True

def count_parameters(module: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def find_checkpoint_in_dir(dir_path: str) -> str:
    """
    Find a likely checkpoint file inside a local model directory.
    """
    root = Path(dir_path)
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {root}")

    candidates = []
    patterns = [
        "*.pth",
        "*.pt",
        "*.bin",
        "*.safetensors",
    ]
    for pattern in patterns:
        candidates.extend(sorted(root.glob(pattern)))

    if len(candidates) == 0:
        for pattern in patterns:
            candidates.extend(sorted(root.rglob(pattern)))

    if len(candidates) == 0:
        raise FileNotFoundError(f"No checkpoint file found under: {root}")

    preferred_keywords = [
        "model",
        "checkpoint",
        "teacher",
        "student",
        "pytorch_model",
    ]
    for kw in preferred_keywords:
        for p in candidates:
            if kw in p.name.lower():
                return str(p)

    return str(candidates[0])


# =========================
# Config
# =========================

@dataclass
class TrainConfig:
    train_manifest: str
    val_manifest: Optional[str]
    output_dir: str = "./outputs/run_001"

    image_size: int = 512
    template_size: int = 224
    bbox_expand_scale: float = 1.2
    sam_canvas_size: int = 512

    use_real_sam: bool = False
    sam_model_alias: str = "sam_model"
    sam_device: str = "cuda"

    batch_size: int = 4
    num_workers: int = 4
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    seed: int = 42
    use_masked_latent_edit: bool = True

    log_interval: int = 20
    save_interval: int = 1
    amp: bool = True

    # Module dims
    dino_in_dim: int = 1024       # DINOv2 ViT-L/14
    rs_in_dim: int = 384          # DINO-MC ViT-S/8
    token_dim: int = 1024
    scene_dim: int = 768
    context_dim: int = 1024

    # Real diffusion backbone
    sd21_model_alias: str = "sd21_base"
    freeze_unet: bool = True

    # DINOv2 branch
    freeze_dinov2: bool = True
    dinov2_model_alias: str = "dinov2_global"
    dinov2_backbone_name: str = "vit_large_patch14_dinov2.lvd142m"
    dinov2_image_size: int = 518

    # DINO-MC / RS branch
    freeze_rs_dino: bool = True
    rs_repo_root: str = "external/DINO-MC"
    rs_checkpoint_path: str = "pretrained/dino_mc/dino_mc_vits8_checkpoint.pth"
    rs_arch: str = "vit_small"
    rs_patch_size: int = 8
    rs_image_size: int = 224
    rs_checkpoint_key: str = "teacher"

    # Loss weights
    lambda_diff: float = 1.0
    lambda_recon: float = 1.0
    lambda_bg: float = 1.0
    lambda_id: float = 0.5
    lambda_template_id: float = 0.2
    lambda_decouple_mi_kde: float = 0.2
    lambda_decouple_mask_entropy: float = 0.05
    lambda_decouple_pose: float = 0.1

    # Training toggles
    freeze_dual_id_extractor: bool = False
    freeze_scene_encoder: bool = True
    freeze_vae: bool = True


# =========================
# Full training model
# =========================

class DefectSynthesisTrainingModel(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ---- real DINOv2 branch ----
        real_dinov2 = DINOv2Wrapper(
            backbone_name=cfg.dinov2_backbone_name,
            checkpoint_path=find_checkpoint_in_dir(get_local_model_path(cfg.dinov2_model_alias)),
            image_size=cfg.dinov2_image_size,
            freeze=cfg.freeze_dinov2,
            normalize_input=True,
            strict=False,
        )

        # ---- real DINO-MC / RS branch ----
        real_rs = DINO_MC_RSWrapper(
            repo_root=cfg.rs_repo_root,
            checkpoint_path=cfg.rs_checkpoint_path,
            arch=cfg.rs_arch,
            patch_size=cfg.rs_patch_size,
            image_size=cfg.rs_image_size,
            checkpoint_key=cfg.rs_checkpoint_key,
            freeze=cfg.freeze_rs_dino,
            normalize_input=True,
            strict=False,
        )

        self.dual_id_extractor = DualIDExtractor(
            dino_in_dim=cfg.dino_in_dim,
            rs_in_dim=cfg.rs_in_dim,
            fused_dim=cfg.token_dim,
            encoder_dino=real_dinov2,
            encoder_rs=real_rs,
        )

        self.id_decoupler = IDDecoupler(
            token_dim=cfg.token_dim,
            pose_dim=10,
            num_layers=4,
            num_heads=8,
            dropout=0.0,
        )

        self.scene_encoder = CLIPSceneEncoder(
            local_dir=get_local_model_path("clip_scene_encoder"),
            freeze=cfg.freeze_scene_encoder,
            output_dim=cfg.scene_dim,
            normalize=True,
        )

        self.semantic_fuser = SemanticFuser(
            scene_dim=cfg.scene_dim,
            token_dim=cfg.token_dim,
            out_dim=cfg.context_dim,
            dropout=0.0,
            include_nonid=True,
            include_pose=True,
            append_fused_token=True,
        )

        vae_cfg = SDVAEConfig(
            local_dir=get_local_model_path("sd_vae"),
            freeze=cfg.freeze_vae,
            use_mode=False,
        )
        self.vae_encoder = SDVAEEncoder(config=vae_cfg)
        self.vae_decoder = SDVAEDecoder(config=vae_cfg)

        self.unet = RealUNetWithDualInjection(
            model_dir=get_local_model_path(cfg.sd21_model_alias),
            id_token_dim=cfg.token_dim,
            freeze_unet=cfg.freeze_unet,
            return_debug_info=False,
        )

        if cfg.freeze_unet:
            freeze_unet_but_keep_injection(self)

        if cfg.freeze_dual_id_extractor:
            for p in self.dual_id_extractor.parameters():
                p.requires_grad = False

    def forward(
        self,
        batch: Dict[str, Tensor],
        base_context: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        scene_img = batch["scene_img"]
        target_img = batch["target_img"]

        # SAM-style clean template branch
        defect_template = batch["defect_template_clean"]      # [B,3,T,T]
        template_mask = batch["template_mask_clean"]          # [B,1,T,T]
        template_pose_prior = batch["template_pose_prior"]    # [B,P]
        loc_mask = batch["loc_mask"]                          # [B,1,H,W]

        # 1) dual-level ID extraction
        dual_features = self.dual_id_extractor(defect_template)

        if self.training and torch.rand(1).item() < 0.001:
            print(
                "[DualID shapes]",
                "global_dino", tuple(dual_features.global_dino.shape),
                "patch_dino", tuple(dual_features.patch_dino.shape),
                "global_rs", tuple(dual_features.global_rs.shape),
                "patch_rs", tuple(dual_features.patch_rs.shape),
                "fused_tokens", tuple(dual_features.fused_tokens.shape),
            )

        # 2) ID-aware decoupling
        # Compatible with both old and new IDDecoupler signatures
        try:
            decoupled = self.id_decoupler(
                dual_features.fused_tokens,
                loc_mask,
                pose_prior=template_pose_prior,
            )
        except TypeError:
            decoupled = self.id_decoupler(
                dual_features.fused_tokens,
                loc_mask,
            )

        # 3) semantic path
        scene_embedding = self.scene_encoder(scene_img)
        semantic_out = self.semantic_fuser(
            scene_embedding=scene_embedding,
            id_global=decoupled.id_global,
            nonid_global=decoupled.nonid_global,
            pose_token=decoupled.pose_token,
        )

        # 4) encode full-image latents
        target_latent = self.vae_encoder(target_img)
        scene_latent = self.vae_encoder(scene_img)

        # 5) downsample mask to latent space
        latent_mask = F.interpolate(
            loc_mask.float(),
            size=target_latent.shape[-2:],
            mode="nearest",
        )
        latent_mask_4 = latent_mask.expand(-1, target_latent.size(1), -1, -1)

        # 6) diffusion noise setup
        noise = torch.randn_like(target_latent)
        num_train_timesteps = self.unet.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(
            low=0,
            high=num_train_timesteps,
            size=(target_latent.size(0),),
            device=target_latent.device,
        ).long()

        if self.cfg.use_masked_latent_edit:
            noisy_target_latent = self.unet.noise_scheduler.add_noise(
                original_samples=target_latent,
                noise=noise,
                timesteps=timesteps,
            )

            # only edit inside mask; keep background latent fixed
            noisy_latent = (
                latent_mask_4 * noisy_target_latent +
                (1.0 - latent_mask_4) * scene_latent
            )
        else:
            noisy_latent = self.unet.noise_scheduler.add_noise(
                original_samples=target_latent,
                noise=noise,
                timesteps=timesteps,
            )

        # 7) UNet noise prediction
        unet_out = self.unet(
            noisy_latents=noisy_latent,
            timesteps=timesteps,
            semantic_tokens=semantic_out.semantic_tokens,
            id_patch_tokens=decoupled.id_patch,
            base_context=base_context,
        )

        noise_pred = unet_out.sample

        # 8) estimate x0
        pred_latent = self.unet.predict_x0_from_epsilon(
            noisy_latents=noisy_latent,
            noise_pred=noise_pred,
            timesteps=timesteps,
        )

        # 9) force background anchoring in latent space
        if self.cfg.use_masked_latent_edit:
            pred_latent_composite = (
                latent_mask_4 * pred_latent +
                (1.0 - latent_mask_4) * scene_latent
            )
        else:
            pred_latent_composite = pred_latent

        # 10) decode
        pred_img = self.vae_decoder(pred_latent_composite)

        if self.training and torch.rand(1).item() < 0.001:
            print(
                "[LatentMask]",
                "shape", tuple(latent_mask_4.shape),
                "mean", float(latent_mask_4.mean().item()),
                "sum", float(latent_mask_4.sum().item()),
            )

        return {
            "pred_img": pred_img,
            "pred_latent": pred_latent,
            "pred_latent_composite": pred_latent_composite,
            "target_latent": target_latent,
            "scene_latent": scene_latent,
            "noise": noise,
            "noise_pred": noise_pred,
            "timesteps": timesteps,
            "noisy_latent": noisy_latent,
            "latent_mask": latent_mask,
            "latent_mask_4": latent_mask_4,
            "dual_features": dual_features,
            "decoupled": decoupled,
            "semantic_out": semantic_out,
            "unet_out": unet_out,
            "debug_inputs": {
                "template_pose_prior": template_pose_prior,
                "template_mask_clean": template_mask,
            },
        }


# =========================
# Loss builder
# =========================

class LossManager(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recon_loss_fn = build_recon_loss()
        self.id_loss_fn = build_id_loss(mode="cosine")
        self.template_id_loss_fn = TemplateMatchingIDLoss(mode="cosine")

    def forward(
        self,
        batch: Dict[str, Tensor],
        model_out: Dict[str, Any],
        cfg: TrainConfig,
    ) -> Dict[str, Tensor]:
        pred_img = model_out["pred_img"]
        pred_latent = model_out["pred_latent_composite"]
        decoupled = model_out["decoupled"]

        scene_img = batch["scene_img"]
        target_img = batch["target_img"]

        # explicit clean-template branch
        defect_template = batch["defect_template_clean"]
        template_mask = batch["template_mask_clean"]
        loc_mask = batch["loc_mask"]

        noise_pred = model_out["noise_pred"]
        noise = model_out["noise"]

        # diffusion loss: only compute inside mask region in latent space
        if cfg.use_masked_latent_edit:
            latent_mask_4 = model_out["latent_mask_4"]
            diff_err = (noise_pred - noise) ** 2
            loss_diff = (diff_err * latent_mask_4).sum() / (latent_mask_4.sum() + 1e-6)
        else:
            loss_diff = F.mse_loss(noise_pred, noise)

        recon_out = self.recon_loss_fn(
            pred_img=pred_img,
            target_img=target_img,
            loc_mask=loc_mask,
        )

        loss_bg = background_l1_loss(
            pred=pred_img,
            scene=scene_img,
            mask=loc_mask,
        )

        id_out = self.id_loss_fn(
            pred_img=pred_img,
            ref_img=target_img,
            mask=loc_mask,
        )

        template_id_out = self.template_id_loss_fn(
            pred_img=pred_img,
            defect_template=defect_template,
            loc_mask=loc_mask,
            template_mask=template_mask,
        )

        dec_losses = decoupling_losses(decoupled)

        # compatible with both old orth loss and new MI(KDE) loss
        if "loss_decouple_mi_kde" in dec_losses:
            loss_decouple_main = dec_losses["loss_decouple_mi_kde"]
            lambda_decouple_main = getattr(cfg, "lambda_decouple_mi_kde", 0.2)
            main_name = "loss_decouple_mi_kde"
        else:
            loss_decouple_main = dec_losses["loss_decouple_orth"]
            lambda_decouple_main = getattr(cfg, "lambda_decouple_orth", 0.2)
            main_name = "loss_decouple_orth"

        total = (
            cfg.lambda_diff * loss_diff
            + cfg.lambda_recon * recon_out.loss_recon_total
            + cfg.lambda_bg * loss_bg
            + cfg.lambda_id * id_out.loss_id_total
            + cfg.lambda_template_id * template_id_out.loss_id_total
            + lambda_decouple_main * loss_decouple_main
            + cfg.lambda_decouple_mask_entropy * dec_losses["loss_decouple_mask_entropy"]
            + cfg.lambda_decouple_pose * dec_losses["loss_decouple_pose"]
        )

        out = {
            "loss_total": total,
            "loss_diff": loss_diff,
            "loss_recon": recon_out.loss_recon_total,
            "loss_bg": loss_bg,
            "loss_id": id_out.loss_id_total,
            "loss_template_id": template_id_out.loss_id_total,
            "loss_decouple_mask_entropy": dec_losses["loss_decouple_mask_entropy"],
            "loss_decouple_pose": dec_losses["loss_decouple_pose"],
        }
        out[main_name] = loss_decouple_main
        return out


# =========================
# Trainer
# =========================

class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(cfg.seed)

        self.output_dir = Path(cfg.output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        save_json(asdict(cfg), self.output_dir / "train_config.json")

        train_records = build_records_from_manifest(cfg.train_manifest)
        self.train_dataset = MVTecDefectSynthesisDataset(
            records=train_records,
            image_size=cfg.image_size,
            template_size=cfg.template_size,
            bbox_expand_scale=cfg.bbox_expand_scale,
            sam_canvas_size=cfg.sam_canvas_size,
            use_real_sam=True,
            sam_model_alias="sam_model",
            sam_device="cuda",
            return_debug_vis=False,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = None
        if cfg.val_manifest:
            val_records = build_records_from_manifest(cfg.val_manifest)
            val_dataset = MVTecDefectSynthesisDataset(
                records=val_records,
                image_size=cfg.image_size,
                template_size=cfg.template_size,
                bbox_expand_scale=cfg.bbox_expand_scale,
                sam_canvas_size=cfg.sam_canvas_size,
                use_real_sam=cfg.use_real_sam,
                sam_model_alias=cfg.sam_model_alias,
                sam_device=cfg.sam_device,
                return_debug_vis=False,
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=False,
            )

        self.model = DefectSynthesisTrainingModel(cfg).to(self.device)
        self.loss_manager = LossManager().to(self.device)

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and self.device.type == "cuda"))

        print(
            f"[Dataset] use_real_sam={cfg.use_real_sam} "
            f"sam_model_alias={cfg.sam_model_alias} "
            f"sam_device={cfg.sam_device}"
        )
        if cfg.use_real_sam and cfg.num_workers != 0:
            print(
                "[Warning] Real SAM is enabled. "
                "For online SAM inference, num_workers=0 is strongly recommended."
            )

        stats = count_parameters(self.model)
        print(f"[Model] total={stats['total']:,} trainable={stats['trainable']:,}")

    def train(self) -> None:
        global_step = 0

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()

            running: Dict[str, float] = {}
            epoch_start = time.time()

            for step, batch in enumerate(self.train_loader, start=1):
                batch = move_to_device(batch, self.device)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=(self.cfg.amp and self.device.type == "cuda")):
                    model_out = self.model(batch)
                    losses = self.loss_manager(batch, model_out, self.cfg)
                    loss_total = losses["loss_total"]

                self.scaler.scale(loss_total).backward()

                if self.cfg.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                global_step += 1

                for k, v in losses.items():
                    running[k] = running.get(k, 0.0) + float(v.detach().item())

                if step % self.cfg.log_interval == 0:
                    avg_msg = " ".join(
                        f"{k}={running[k] / self.cfg.log_interval:.4f}"
                        for k in sorted(running.keys())
                    )
                    print(
                        f"[Train] epoch={epoch}/{self.cfg.epochs} "
                        f"step={step}/{len(self.train_loader)} "
                        f"global_step={global_step} {avg_msg}"
                    )
                    running = {}

            epoch_time = time.time() - epoch_start
            print(f"[Epoch] {epoch} finished in {epoch_time:.1f}s")

            if self.val_loader is not None:
                self.validate(epoch)

            if epoch % self.cfg.save_interval == 0:
                self.save_checkpoint(epoch, global_step)

    @torch.no_grad()
    def validate(self, epoch: int) -> None:
        self.model.eval()

        sums: Dict[str, float] = {}
        count = 0

        for batch in self.val_loader:
            batch = move_to_device(batch, self.device)

            with torch.amp.autocast("cuda", enabled=(self.cfg.amp and self.device.type == "cuda")):
                model_out = self.model(batch)
                losses = self.loss_manager(batch, model_out, self.cfg)

            bsz = batch["scene_img"].size(0)
            count += bsz
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach().item()) * bsz

        if count > 0:
            msg = " ".join(f"{k}={sums[k] / count:.4f}" for k in sorted(sums.keys()))
            print(f"[Val] epoch={epoch} {msg}")

    def save_checkpoint(self, epoch: int, global_step: int) -> None:
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "config": asdict(self.cfg),
            "model": self.model.state_dict(),
            "loss_manager": self.loss_manager.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        path = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save(ckpt, path)
        print(f"[Checkpoint] saved to {path}")


# =========================
# CLI
# =========================

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/run_001")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--template_size", type=int, default=224)
    parser.add_argument("--sam_canvas_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--use_real_sam", action="store_true")
    parser.add_argument("--sam_model_alias", type=str, default="sam_model")
    parser.add_argument("--sam_device", type=str, default="cuda")
    args = parser.parse_args()

    return TrainConfig(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        image_size=args.image_size,
        template_size=args.template_size,
        sam_canvas_size=args.sam_canvas_size,
        use_real_sam=args.use_real_sam,
        sam_model_alias=args.sam_model_alias,
        sam_device=args.sam_device,
        seed=args.seed,
        amp=args.amp,
    )


def main() -> None:
    cfg = parse_args()
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()