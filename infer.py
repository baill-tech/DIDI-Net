from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train import TrainConfig, DefectSynthesisTrainingModel, move_to_device
from datasets.sam_preprocess import (
    SAMTemplatePreprocessor,
    RealSAMPredictor,
    load_rgb_image,
    load_mask_image,
)


# =========================
# Utilities
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)


def to_tensor_image(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().float()


def to_tensor_mask(mask: np.ndarray) -> torch.Tensor:
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    return torch.from_numpy(mask[None, ...]).contiguous().float()


def resize_chw_tensor(x: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    x = x.unsqueeze(0)
    if mode == "nearest":
        x = F.interpolate(x, size=(size, size), mode=mode)
    else:
        x = F.interpolate(x, size=(size, size), mode=mode, align_corners=False)
    return x.squeeze(0)


def save_tensor_image(x: torch.Tensor, path: str) -> None:
    x = x.detach().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def make_grid_numpy(images: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=1)


def tensor_to_hwc_numpy(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


# =========================
# Single-sample preprocessing
# =========================

def prepare_single_sample(
    scene_image_path: str,
    template_image_path: str,
    template_mask_path: str,
    loc_mask_path: str,
    image_size: int,
    template_size: int,
    sam_canvas_size: int,
    use_real_sam: bool,
    sam_model_alias: str,
    sam_device: str,
) -> Dict[str, Any]:
    scene_img_np = load_rgb_image(scene_image_path)
    template_img_np = load_rgb_image(template_image_path)
    template_mask_np = load_mask_image(template_mask_path)
    loc_mask_np = load_mask_image(loc_mask_path)

    scene_img = to_tensor_image(resize_image(normalize_image(scene_img_np), (image_size, image_size)))
    loc_mask = to_tensor_mask((resize_mask((loc_mask_np > 127).astype(np.float32), (image_size, image_size)) > 0.5).astype(np.float32))

    sam_predictor = None
    if use_real_sam:
        sam_predictor = RealSAMPredictor(
            model_alias=sam_model_alias,
            device=sam_device,
            multimask_output=True,
        )

    sam_preprocessor = SAMTemplatePreprocessor(
        canvas_size=sam_canvas_size,
        bbox_expand_ratio=0.08,
        morphology_ksize=5,
        use_largest_component=True,
        fill_holes_flag=True,
    )

    sam_out = sam_preprocessor.preprocess(
        image_rgb=template_img_np,
        gt_mask=template_mask_np,
        sam_predictor=sam_predictor,
    )

    defect_template_clean_512 = sam_out.clean_object
    template_mask_clean_512 = sam_out.clean_mask
    template_pose_prior = sam_out.pose_prior

    defect_template_clean = resize_chw_tensor(defect_template_clean_512, template_size, mode="bilinear")
    template_mask_clean = resize_chw_tensor(template_mask_clean_512, template_size, mode="nearest")

    sample: Dict[str, Any] = {
        "scene_img": scene_img.unsqueeze(0),
        "loc_mask": loc_mask.unsqueeze(0),

        "defect_template_clean": defect_template_clean.unsqueeze(0),
        "template_mask_clean": template_mask_clean.unsqueeze(0),
        "defect_template_clean_512": defect_template_clean_512.unsqueeze(0),
        "template_mask_clean_512": template_mask_clean_512.unsqueeze(0),
        "template_pose_prior": template_pose_prior.unsqueeze(0),

        "meta": {
            "scene_image_path": scene_image_path,
            "template_image_path": template_image_path,
            "template_mask_path": template_mask_path,
            "loc_mask_path": loc_mask_path,
            "sam_bbox_xyxy": sam_out.bbox_xyxy.tolist(),
            "sam_scale_ratio": float(sam_out.scale_ratio),
            "sam_center_offset": sam_out.center_offset.tolist(),
            "sam_used_real_model": bool(sam_predictor is not None),
        },
    }
    return sample


# =========================
# Inference
# =========================

@torch.no_grad()
def run_inference(
    model: DefectSynthesisTrainingModel,
    batch: Dict[str, Any],
    device: torch.device,
    num_inference_steps: int = 50,
    composite_background: bool = False,
) -> Dict[str, Any]:
    model.eval()
    batch = move_to_device(batch, device)

    scene_img = batch["scene_img"]                          # [1,3,H,W]
    loc_mask = batch["loc_mask"]                            # [1,1,H,W]
    defect_template = batch["defect_template_clean"]        # [1,3,T,T]
    template_mask = batch["template_mask_clean"]            # [1,1,T,T]
    template_pose_prior = batch["template_pose_prior"]      # [1,P]

    # 1) dual-level ID extraction
    dual_features = model.dual_id_extractor(defect_template)

    # 2) ID-aware decoupling
    try:
        decoupled = model.id_decoupler(
            dual_features.fused_tokens,
            loc_mask,
            pose_prior=template_pose_prior,
        )
    except TypeError:
        decoupled = model.id_decoupler(
            dual_features.fused_tokens,
            loc_mask,
        )

    # 3) semantic path
    scene_embedding = model.scene_encoder(scene_img)
    semantic_out = model.semantic_fuser(
        scene_embedding=scene_embedding,
        id_global=decoupled.id_global,
        nonid_global=decoupled.nonid_global,
        pose_token=decoupled.pose_token,
    )

    # 4) prepare background latent and latent mask
    scene_latent_ref = model.vae_encoder(scene_img)

    latent_mask = F.interpolate(
        loc_mask.float(),
        size=scene_latent_ref.shape[-2:],
        mode="nearest",
    )
    latent_mask_4 = latent_mask.expand(-1, scene_latent_ref.size(1), -1, -1)

    noise_latent = torch.randn_like(scene_latent_ref)

    if getattr(model.cfg, "use_masked_latent_edit", False):
        latents = (
            latent_mask_4 * noise_latent +
            (1.0 - latent_mask_4) * scene_latent_ref
        )
    else:
        latents = noise_latent

    scheduler = model.unet.noise_scheduler
    try:
        scheduler.set_timesteps(num_inference_steps, device=device)
    except TypeError:
        scheduler.set_timesteps(num_inference_steps)

    # 5) iterative denoising
    for t in scheduler.timesteps:
        if not torch.is_tensor(t):
            t_batch = torch.full((latents.size(0),), int(t), device=device, dtype=torch.long)
        else:
            if t.ndim == 0:
                t_batch = t.view(1).repeat(latents.size(0)).to(device=device, dtype=torch.long)
            else:
                t_batch = t.to(device=device, dtype=torch.long)

        unet_out = model.unet(
            noisy_latents=latents,
            timesteps=t_batch,
            semantic_tokens=semantic_out.semantic_tokens,
            id_patch_tokens=decoupled.id_patch,
            base_context=None,
        )
        noise_pred = unet_out.sample
        step_out = scheduler.step(noise_pred, t, latents)
        latents = step_out.prev_sample

        # hard background anchoring in latent space
        if getattr(model.cfg, "use_masked_latent_edit", False):
            latents = (
                latent_mask_4 * latents +
                (1.0 - latent_mask_4) * scene_latent_ref
            )

    # 6) final latent composite before decode
    if getattr(model.cfg, "use_masked_latent_edit", False):
        latents = (
            latent_mask_4 * latents +
            (1.0 - latent_mask_4) * scene_latent_ref
        )

    pred_img = model.vae_decoder(latents)

    # optional final hard composite in image space
    if composite_background:
        pred_img = pred_img * loc_mask + scene_img * (1.0 - loc_mask)

    return {
        "pred_img": pred_img,
        "scene_img": scene_img,
        "loc_mask": loc_mask,
        "defect_template_clean": defect_template,
        "template_mask_clean": template_mask,
        "template_pose_prior": template_pose_prior,
        "latent_mask": latent_mask,
        "latent_mask_4": latent_mask_4,
        "dual_features": dual_features,
        "decoupled": decoupled,
        "semantic_out": semantic_out,
    }


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--scene_image", type=str, required=True)
    parser.add_argument("--template_image", type=str, required=True)
    parser.add_argument("--template_mask", type=str, required=True)
    parser.add_argument("--loc_mask", type=str, required=True)

    parser.add_argument("--output_image", type=str, required=True)
    parser.add_argument("--output_grid", type=str, default=None)

    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--template_size", type=int, default=None)
    parser.add_argument("--sam_canvas_size", type=int, default=None)

    parser.add_argument("--use_real_sam", action="store_true")
    parser.add_argument("--sam_model_alias", type=str, default="sam_model")
    parser.add_argument("--sam_device", type=str, default="cuda")

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--composite_background", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg_dict = ckpt["config"]
    cfg = TrainConfig(**cfg_dict)

    if args.image_size is not None:
        cfg.image_size = args.image_size
    if args.template_size is not None:
        cfg.template_size = args.template_size
    if args.sam_canvas_size is not None:
        cfg.sam_canvas_size = args.sam_canvas_size

    if hasattr(cfg, "use_real_sam"):
        cfg.use_real_sam = args.use_real_sam
    if hasattr(cfg, "sam_model_alias"):
        cfg.sam_model_alias = args.sam_model_alias
    if hasattr(cfg, "sam_device"):
        cfg.sam_device = args.sam_device

    model = DefectSynthesisTrainingModel(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    sample = prepare_single_sample(
        scene_image_path=args.scene_image,
        template_image_path=args.template_image,
        template_mask_path=args.template_mask,
        loc_mask_path=args.loc_mask,
        image_size=cfg.image_size,
        template_size=cfg.template_size,
        sam_canvas_size=cfg.sam_canvas_size,
        use_real_sam=args.use_real_sam,
        sam_model_alias=args.sam_model_alias,
        sam_device=args.sam_device,
    )

    out = run_inference(
        model=model,
        batch=sample,
        device=device,
        num_inference_steps=args.num_inference_steps,
        composite_background=args.composite_background,
    )

    pred_img = out["pred_img"][0]
    save_tensor_image(pred_img, args.output_image)

    if args.output_grid is not None:
        scene_np = tensor_to_hwc_numpy(out["scene_img"][0])
        template_np = tensor_to_hwc_numpy(resize_chw_tensor(out["defect_template_clean"][0], cfg.image_size))
        loc_mask_np = tensor_to_hwc_numpy(out["loc_mask"][0].repeat(3, 1, 1))
        pred_np = tensor_to_hwc_numpy(pred_img)

        grid = make_grid_numpy([scene_np, template_np, loc_mask_np, pred_np])
        grid = (grid * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(grid).save(args.output_grid)

    meta_out = {
        "checkpoint": args.checkpoint,
        "scene_image": args.scene_image,
        "template_image": args.template_image,
        "template_mask": args.template_mask,
        "loc_mask": args.loc_mask,
        "output_image": args.output_image,
        "output_grid": args.output_grid,
        "num_inference_steps": args.num_inference_steps,
        "sam_used_real_model": sample["meta"]["sam_used_real_model"],
        "sam_bbox_xyxy": sample["meta"]["sam_bbox_xyxy"],
        "sam_scale_ratio": sample["meta"]["sam_scale_ratio"],
        "sam_center_offset": sample["meta"]["sam_center_offset"],
    }
    meta_path = str(Path(args.output_image).with_suffix(".json"))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    print(f"[Saved] image -> {args.output_image}")
    if args.output_grid is not None:
        print(f"[Saved] grid  -> {args.output_grid}")
    print(f"[Saved] meta  -> {meta_path}")


if __name__ == "__main__":
    main()