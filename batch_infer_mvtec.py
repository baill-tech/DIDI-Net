from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# =========================
# Utils
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


def tensor_to_hwc_numpy(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def make_grid_numpy(images: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=1)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def test_image_to_mask_name(image_path: Path) -> str:
    """
    MVTec:
      test/.../000.png -> ground_truth/.../000_mask.png
    """
    return f"{image_path.stem}_mask.png"


def choose_good_image(
    good_images: List[Path],
    defect_image: Path,
    pair_mode: str,
    index: int,
    rng: random.Random,
) -> Path:
    if len(good_images) == 0:
        raise RuntimeError("No good images found.")

    if pair_mode == "same_name_then_cyclic":
        for p in good_images:
            if p.stem == defect_image.stem:
                return p
        return good_images[index % len(good_images)]

    if pair_mode == "same_name":
        for p in good_images:
            if p.stem == defect_image.stem:
                return p
        raise RuntimeError(
            f"No same-name good image found for defect image: {defect_image.name}"
        )

    if pair_mode == "cyclic":
        return good_images[index % len(good_images)]

    if pair_mode == "random":
        return rng.choice(good_images)

    raise ValueError(f"Unsupported pair_mode: {pair_mode}")


# =========================
# Single-sample prep
# =========================

def prepare_single_sample_mvtec(
    scene_image_path: str,
    template_image_path: str,
    shared_mask_path: str,
    image_size: int,
    template_size: int,
    sam_canvas_size: int,
    sam_predictor: Optional[RealSAMPredictor],
) -> Dict[str, Any]:
    """
    shared_mask is used for:
      - template_mask
      - loc_mask

    This matches the requirement:
      use test defect image + its mask as template/source,
      generate defect on a train/good scene image only inside mask region.
    """
    scene_img_np = load_rgb_image(scene_image_path)
    template_img_np = load_rgb_image(template_image_path)
    shared_mask_np = load_mask_image(shared_mask_path)

    scene_img = to_tensor_image(resize_image(normalize_image(scene_img_np), (image_size, image_size)))
    loc_mask = to_tensor_mask(
        (resize_mask((shared_mask_np > 127).astype(np.float32), (image_size, image_size)) > 0.5).astype(np.float32)
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
        gt_mask=shared_mask_np,
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
            "shared_mask_path": shared_mask_path,
            "sam_bbox_xyxy": sam_out.bbox_xyxy.tolist(),
            "sam_scale_ratio": float(sam_out.scale_ratio),
            "sam_center_offset": sam_out.center_offset.tolist(),
            "sam_used_real_model": bool(sam_predictor is not None),
        },
    }
    return sample


# =========================
# Core inference
# =========================

@torch.no_grad()
def run_inference_mask_only(
    model: DefectSynthesisTrainingModel,
    batch: Dict[str, Any],
    device: torch.device,
    num_inference_steps: int = 50,
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

    # 3) semantic fusion
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
        latents = latent_mask_4 * noise_latent + (1.0 - latent_mask_4) * scene_latent_ref
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

        # hard latent anchoring: only edit inside mask
        latents = latent_mask_4 * latents + (1.0 - latent_mask_4) * scene_latent_ref

    # 6) final decode
    latents = latent_mask_4 * latents + (1.0 - latent_mask_4) * scene_latent_ref
    pred_img = model.vae_decoder(latents)

    # 7) hard image composite again, ensure output only changes in mask region
    pred_img = pred_img * loc_mask + scene_img * (1.0 - loc_mask)

    return {
        "pred_img": pred_img,
        "scene_img": scene_img,
        "loc_mask": loc_mask,
        "defect_template_clean": defect_template,
        "template_mask_clean": template_mask,
        "template_pose_prior": template_pose_prior,
        "dual_features": dual_features,
        "decoupled": decoupled,
        "semantic_out": semantic_out,
        "latent_mask": latent_mask,
        "latent_mask_4": latent_mask_4,
    }


# =========================
# MVTec traversal
# =========================

def collect_mvtec_jobs(
    mvtec_root: Path,
    pair_mode: str,
    seed: int,
    categories: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Traverse MVTec directory:
      mvtec_root/
        category/
          train/good/*.png
          test/<defect_type>/*.png
          ground_truth/<defect_type>/*_mask.png
    """
    rng = random.Random(seed)
    jobs: List[Dict[str, Any]] = []

    all_categories = sorted([p for p in mvtec_root.iterdir() if p.is_dir()])
    if categories is not None and len(categories) > 0:
        cat_set = set(categories)
        all_categories = [p for p in all_categories if p.name in cat_set]

    for cat_dir in all_categories:
        category = cat_dir.name
        train_good_dir = cat_dir / "train" / "good"
        test_dir = cat_dir / "test"
        gt_dir = cat_dir / "ground_truth"

        if not train_good_dir.exists() or not test_dir.exists() or not gt_dir.exists():
            continue

        good_images = list_images(train_good_dir)
        if len(good_images) == 0:
            continue

        defect_types = sorted([p for p in test_dir.iterdir() if p.is_dir() and p.name != "good"])
        for defect_type_dir in defect_types:
            defect_type = defect_type_dir.name
            gt_type_dir = gt_dir / defect_type
            if not gt_type_dir.exists():
                continue

            defect_images = list_images(defect_type_dir)
            for idx, defect_img_path in enumerate(defect_images):
                mask_name = test_image_to_mask_name(defect_img_path)
                mask_path = gt_type_dir / mask_name
                if not mask_path.exists():
                    print(f"[Skip] mask not found for {defect_img_path}")
                    continue

                good_img_path = choose_good_image(
                    good_images=good_images,
                    defect_image=defect_img_path,
                    pair_mode=pair_mode,
                    index=idx,
                    rng=rng,
                )

                jobs.append(
                    {
                        "category": category,
                        "defect_type": defect_type,
                        "defect_image_path": defect_img_path,
                        "mask_path": mask_path,
                        "good_image_path": good_img_path,
                    }
                )

    return jobs


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mvtec_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument(
        "--pair_mode",
        type=str,
        default="same_name_then_cyclic",
        choices=["same_name_then_cyclic", "same_name", "cyclic", "random"],
        help="How to choose a train/good background for each test defect image.",
    )

    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--template_size", type=int, default=None)
    parser.add_argument("--sam_canvas_size", type=int, default=None)

    parser.add_argument("--use_real_sam", action="store_true")
    parser.add_argument("--sam_model_alias", type=str, default="sam_model")
    parser.add_argument("--sam_device", type=str, default="cuda")

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_grid", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)

    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = TrainConfig(**ckpt["config"])

    if args.image_size is not None:
        cfg.image_size = args.image_size
    if args.template_size is not None:
        cfg.template_size = args.template_size
    if args.sam_canvas_size is not None:
        cfg.sam_canvas_size = args.sam_canvas_size

    model = DefectSynthesisTrainingModel(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    sam_predictor = None
    if args.use_real_sam:
        sam_predictor = RealSAMPredictor(
            model_alias=args.sam_model_alias,
            device=args.sam_device,
            multimask_output=True,
        )

    jobs = collect_mvtec_jobs(
        mvtec_root=Path(args.mvtec_root),
        pair_mode=args.pair_mode,
        seed=args.seed,
        categories=args.categories,
    )

    if args.max_samples is not None:
        jobs = jobs[: args.max_samples]

    if len(jobs) == 0:
        raise RuntimeError("No valid MVTec jobs found.")

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    grid_dir = output_dir / "grids"
    meta_dir = output_dir / "meta"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    if args.save_grid:
        grid_dir.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, Any]] = []

    for idx, job in enumerate(jobs, start=1):
        category = job["category"]
        defect_type = job["defect_type"]
        defect_img_path = job["defect_image_path"]
        mask_path = job["mask_path"]
        good_img_path = job["good_image_path"]

        key = f"{category}__{defect_type}__{defect_img_path.stem}__bg_{good_img_path.stem}"

        sample = prepare_single_sample_mvtec(
            scene_image_path=str(good_img_path),
            template_image_path=str(defect_img_path),
            shared_mask_path=str(mask_path),
            image_size=cfg.image_size,
            template_size=cfg.template_size,
            sam_canvas_size=cfg.sam_canvas_size,
            sam_predictor=sam_predictor,
        )

        out = run_inference_mask_only(
            model=model,
            batch=sample,
            device=device,
            num_inference_steps=args.num_inference_steps,
        )

        cat_img_dir = image_dir / category / defect_type
        cat_meta_dir = meta_dir / category / defect_type
        cat_img_dir.mkdir(parents=True, exist_ok=True)
        cat_meta_dir.mkdir(parents=True, exist_ok=True)

        out_img_path = cat_img_dir / f"{key}.png"
        save_tensor_image(out["pred_img"][0], str(out_img_path))

        out_grid_path = None
        if args.save_grid:
            cat_grid_dir = grid_dir / category / defect_type
            cat_grid_dir.mkdir(parents=True, exist_ok=True)

            scene_np = tensor_to_hwc_numpy(out["scene_img"][0])
            template_np = tensor_to_hwc_numpy(
                resize_chw_tensor(out["defect_template_clean"][0], cfg.image_size)
            )
            loc_mask_np = tensor_to_hwc_numpy(out["loc_mask"][0].repeat(3, 1, 1))
            pred_np = tensor_to_hwc_numpy(out["pred_img"][0])

            grid = make_grid_numpy([scene_np, template_np, loc_mask_np, pred_np])
            grid = (grid * 255.0).clip(0, 255).astype(np.uint8)

            out_grid_path = cat_grid_dir / f"{key}.png"
            Image.fromarray(grid).save(out_grid_path)

        meta = {
            "index": idx - 1,
            "category": category,
            "defect_type": defect_type,
            "defect_image_path": str(defect_img_path),
            "mask_path": str(mask_path),
            "good_image_path": str(good_img_path),
            "output_image": str(out_img_path),
            "output_grid": str(out_grid_path) if out_grid_path is not None else None,
            "num_inference_steps": args.num_inference_steps,
            "pair_mode": args.pair_mode,
            "sam_used_real_model": sample["meta"]["sam_used_real_model"],
            "sam_bbox_xyxy": sample["meta"]["sam_bbox_xyxy"],
            "sam_scale_ratio": sample["meta"]["sam_scale_ratio"],
            "sam_center_offset": sample["meta"]["sam_center_offset"],
        }

        save_json(meta, cat_meta_dir / f"{key}.json")
        summary.append(meta)

        print(f"[{idx}/{len(jobs)}] saved -> {out_img_path}")

    save_json({"results": summary}, output_dir / "summary.json")
    print(f"[Done] summary -> {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()