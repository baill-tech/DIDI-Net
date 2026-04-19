from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# =========================
# Basic IO
# =========================

def load_mask(path: str | Path) -> np.ndarray:
    mask = Image.open(path).convert("L")
    mask = np.array(mask)
    mask = (mask > 127).astype(np.uint8) * 255
    return mask


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def test_image_to_mask_name(image_path: Path) -> str:
    """
    MVTec:
      test/.../000.png -> ground_truth/.../000_mask.png
    """
    return f"{image_path.stem}_mask.png"


# =========================
# Mask geometry
# =========================

def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask_bin = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask_bin, dtype=np.uint8)
    out[labels == largest_idx] = 1
    return out * 255


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8) * 255
    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask, flood_inv)
    return filled


def clean_mask(mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = keep_largest_component(mask)
    mask = fill_holes(mask)
    mask = (mask > 0).astype(np.uint8) * 255
    return mask


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return 0, 0, w - 1, h - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def centroid_from_mask(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return float(w / 2.0), float(h / 2.0)
    return float(xs.mean()), float(ys.mean())


def area_from_mask(mask: np.ndarray) -> int:
    return int((mask > 0).sum())


def pca_angle_from_mask(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        return 0.0
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    pts = pts - pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    angle = math.degrees(math.atan2(float(major[1]), float(major[0])))
    return angle


def mask_stats(mask: np.ndarray) -> Dict[str, Any]:
    x1, y1, x2, y2 = bbox_from_mask(mask)
    cx, cy = centroid_from_mask(mask)
    return {
        "bbox_xyxy": [x1, y1, x2, y2],
        "centroid_xy": [cx, cy],
        "area": area_from_mask(mask),
        "angle_deg": pca_angle_from_mask(mask),
    }


# =========================
# Transform helpers
# =========================

def affine_warp(mask: np.ndarray, M: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    out = (out > 127).astype(np.uint8) * 255
    out = clean_mask(out)
    return out


def translate_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return affine_warp(mask, M)


def scale_mask_about_center(mask: np.ndarray, scale: float, center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    if center is None:
        center = centroid_from_mask(mask)
    cx, cy = center
    M = cv2.getRotationMatrix2D((cx, cy), angle=0.0, scale=scale)
    return affine_warp(mask, M)


def rotate_mask_about_center(mask: np.ndarray, angle_deg: float, center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    if center is None:
        center = centroid_from_mask(mask)
    cx, cy = center
    M = cv2.getRotationMatrix2D((cx, cy), angle=angle_deg, scale=1.0)
    return affine_warp(mask, M)


def scale_rotate_mask(mask: np.ndarray, scale: float, angle_deg: float, center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    if center is None:
        center = centroid_from_mask(mask)
    cx, cy = center
    M = cv2.getRotationMatrix2D((cx, cy), angle=angle_deg, scale=scale)
    return affine_warp(mask, M)


def move_mask_to_target_center(mask: np.ndarray, target_xy: Tuple[float, float]) -> np.ndarray:
    cx, cy = centroid_from_mask(mask)
    tx, ty = target_xy
    dx = tx - cx
    dy = ty - cy
    return translate_mask(mask, dx, dy)


def non_empty(mask: np.ndarray) -> bool:
    return (mask > 0).any()


def inside_margin(mask: np.ndarray, margin_ratio: float = 0.05) -> bool:
    """
    Require transformed mask bbox to stay inside image with a margin.
    """
    if not non_empty(mask):
        return False
    h, w = mask.shape
    x1, y1, x2, y2 = bbox_from_mask(mask)
    mx = int(round(w * margin_ratio))
    my = int(round(h * margin_ratio))
    return x1 >= mx and y1 >= my and x2 <= (w - 1 - mx) and y2 <= (h - 1 - my)


def clip_mask_to_valid(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8) * 255
    return clean_mask(mask)


# =========================
# Position anchors
# =========================

def build_anchor_map(h: int, w: int, margin_ratio: float = 0.18) -> Dict[str, Tuple[float, float]]:
    mx = w * margin_ratio
    my = h * margin_ratio
    cx = w / 2.0
    cy = h / 2.0

    return {
        "center": (cx, cy),
        "top": (cx, my),
        "bottom": (cx, h - my),
        "left": (mx, cy),
        "right": (w - mx, cy),
        "top_left": (mx, my),
        "top_right": (w - mx, my),
        "bottom_left": (mx, h - my),
        "bottom_right": (w - mx, h - my),
    }


# =========================
# Job collection
# =========================

def collect_mvtec_masks(
    mvtec_root: Path,
    categories: Optional[List[str]],
    samples_per_category: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    jobs: List[Dict[str, Any]] = []

    cat_dirs = sorted([p for p in mvtec_root.iterdir() if p.is_dir()])
    if categories:
        cat_set = set(categories)
        cat_dirs = [p for p in cat_dirs if p.name in cat_set]

    for cat_dir in cat_dirs:
        category = cat_dir.name
        test_dir = cat_dir / "test"
        gt_dir = cat_dir / "ground_truth"
        if not test_dir.exists() or not gt_dir.exists():
            continue

        category_samples: List[Dict[str, Any]] = []
        defect_type_dirs = sorted([p for p in test_dir.iterdir() if p.is_dir() and p.name != "good"])
        for defect_type_dir in defect_type_dirs:
            defect_type = defect_type_dir.name
            gt_type_dir = gt_dir / defect_type
            if not gt_type_dir.exists():
                continue

            defect_imgs = list_images(defect_type_dir)
            for defect_img in defect_imgs:
                mask_path = gt_type_dir / test_image_to_mask_name(defect_img)
                if not mask_path.exists():
                    continue
                category_samples.append(
                    {
                        "category": category,
                        "defect_type": defect_type,
                        "template_image_path": str(defect_img),
                        "template_mask_path": str(mask_path),
                        "sample_id": defect_img.stem,
                    }
                )

        rng.shuffle(category_samples)
        jobs.extend(category_samples[:samples_per_category])

    return jobs


# =========================
# Control generation
# =========================

def generate_position_variants(
    base_mask: np.ndarray,
    margin_ratio: float = 0.05,
) -> Dict[str, np.ndarray]:
    h, w = base_mask.shape
    anchors = build_anchor_map(h, w, margin_ratio=0.18)
    out: Dict[str, np.ndarray] = {}

    for name, target_xy in anchors.items():
        m = move_mask_to_target_center(base_mask, target_xy)
        m = clip_mask_to_valid(m)
        if non_empty(m) and inside_margin(m, margin_ratio=margin_ratio):
            out[f"position_{name}"] = m
    return out


def generate_scale_variants(
    base_mask: np.ndarray,
    scales: List[float],
    margin_ratio: float = 0.05,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    c = centroid_from_mask(base_mask)

    for s in scales:
        m = scale_mask_about_center(base_mask, scale=s, center=c)
        m = clip_mask_to_valid(m)
        if non_empty(m) and inside_margin(m, margin_ratio=margin_ratio):
            out[f"scale_{s:.2f}"] = m
    return out


def generate_rotation_variants(
    base_mask: np.ndarray,
    angles: List[float],
    margin_ratio: float = 0.05,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    c = centroid_from_mask(base_mask)

    for a in angles:
        m = rotate_mask_about_center(base_mask, angle_deg=a, center=c)
        m = clip_mask_to_valid(m)
        if non_empty(m) and inside_margin(m, margin_ratio=margin_ratio):
            sign = "neg" if a < 0 else "pos"
            out[f"rotate_{sign}_{abs(int(a)):03d}"] = m
    return out


def generate_joint_variants(
    base_mask: np.ndarray,
    joint_specs: List[Dict[str, Any]],
    margin_ratio: float = 0.05,
) -> Dict[str, np.ndarray]:
    h, w = base_mask.shape
    anchors = build_anchor_map(h, w, margin_ratio=0.18)
    out: Dict[str, np.ndarray] = {}

    base_c = centroid_from_mask(base_mask)

    for i, spec in enumerate(joint_specs):
        pos_name = spec["position"]
        scale = float(spec["scale"])
        angle = float(spec["angle"])

        m = scale_rotate_mask(base_mask, scale=scale, angle_deg=angle, center=base_c)
        target_xy = anchors[pos_name]
        m = move_mask_to_target_center(m, target_xy)
        m = clip_mask_to_valid(m)

        if non_empty(m) and inside_margin(m, margin_ratio=margin_ratio):
            out[f"joint_{i:03d}"] = m

    return out


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mvtec_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--samples_per_category", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--scales", nargs="*", type=float, default=[0.50, 0.75, 1.00, 1.25, 1.50])
    parser.add_argument("--angles", nargs="*", type=float, default=[-45, -30, -15, 0, 15, 30, 45])

    parser.add_argument("--generate_position", action="store_true")
    parser.add_argument("--generate_scale", action="store_true")
    parser.add_argument("--generate_rotation", action="store_true")
    parser.add_argument("--generate_joint", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    mvtec_root = Path(args.mvtec_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (args.generate_position or args.generate_scale or args.generate_rotation or args.generate_joint):
        args.generate_position = True
        args.generate_scale = True
        args.generate_rotation = True
        args.generate_joint = True

    jobs = collect_mvtec_masks(
        mvtec_root=mvtec_root,
        categories=args.categories,
        samples_per_category=args.samples_per_category,
        seed=args.seed,
    )

    joint_specs = [
        {"position": "top_left", "scale": 0.75, "angle": -30},
        {"position": "top_right", "scale": 1.25, "angle": 30},
        {"position": "right", "scale": 1.50, "angle": 0},
        {"position": "top", "scale": 0.50, "angle": 45},
        {"position": "left", "scale": 1.00, "angle": -45},
        {"position": "center", "scale": 1.25, "angle": 15},
    ]

    all_manifest: List[Dict[str, Any]] = []

    for idx, job in enumerate(jobs, start=1):
        category = job["category"]
        defect_type = job["defect_type"]
        sample_id = job["sample_id"]

        template_image_path = job["template_image_path"]
        template_mask_path = job["template_mask_path"]

        base_mask = load_mask(template_mask_path)
        base_mask = clean_mask(base_mask)

        sample_dir = output_dir / category / defect_type / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        template_mask_copy = sample_dir / "template_mask.png"
        save_mask(base_mask, template_mask_copy)

        variants: Dict[str, np.ndarray] = {}

        if args.generate_position:
            variants.update(generate_position_variants(base_mask))

        if args.generate_scale:
            variants.update(generate_scale_variants(base_mask, scales=args.scales))

        if args.generate_rotation:
            variants.update(generate_rotation_variants(base_mask, angles=args.angles))

        if args.generate_joint:
            variants.update(generate_joint_variants(base_mask, joint_specs=joint_specs))

        sample_manifest: List[Dict[str, Any]] = []
        for variant_name, loc_mask in variants.items():
            loc_mask_path = sample_dir / f"loc_{variant_name}.png"
            save_mask(loc_mask, loc_mask_path)

            record = {
                "category": category,
                "defect_type": defect_type,
                "sample_id": sample_id,
                "control_type": variant_name.split("_")[0],
                "variant_name": variant_name,
                "template_image_path": template_image_path,
                "template_mask_path": str(template_mask_copy),
                "loc_mask_path": str(loc_mask_path),
                "template_mask_stats": mask_stats(base_mask),
                "loc_mask_stats": mask_stats(loc_mask),
            }
            sample_manifest.append(record)
            all_manifest.append(record)

        save_json({"records": sample_manifest}, sample_dir / "manifest.json")
        print(f"[{idx}/{len(jobs)}] generated {len(sample_manifest)} masks -> {sample_dir}")

    save_json({"records": all_manifest}, output_dir / "manifest_controls.json")
    print(f"[Done] manifest -> {output_dir / 'manifest_controls.json'}")


if __name__ == "__main__":
    main()