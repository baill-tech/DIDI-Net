from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import lpips
from torch_fidelity import calculate_metrics
torch.backends.cudnn.enabled = False


# =========================
# Basic IO
# =========================

def load_rgb(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0


def save_rgb(img: np.ndarray, path: str | Path, size: int = 256) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray((img * 255.0).clip(0, 255).astype(np.uint8))
    pil = pil.resize((size, size), resample=Image.BILINEAR)
    pil.save(path)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_summary(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported summary format: {path}")


# =========================
# Path resolution
# =========================

def resolve_generated_image(record: Dict[str, Any]) -> str:
    if "output_image" in record and record["output_image"]:
        return record["output_image"]
    raise KeyError("Cannot find output_image in record.")


def resolve_reference_image(record: Dict[str, Any]) -> str:
    for k in ["defect_image_path", "template_image_path", "reference_image_path"]:
        if k in record and record[k]:
            return record[k]
    raise KeyError("Cannot find reference image path in record.")


def resolve_category(record: Dict[str, Any]) -> str:
    return record.get("category", "unknown")


def resolve_defect_type(record: Dict[str, Any]) -> str:
    return record.get("defect_type", "unknown")


# =========================
# Temporary image dirs for IS/FID
# =========================

def build_temp_image_dir(paths: List[str], size: int = 256) -> str:
    tmpdir = tempfile.mkdtemp(prefix="eval_imgs_")
    for i, p in enumerate(paths):
        img = load_rgb(p)
        save_rgb(img, Path(tmpdir) / f"{i:06d}.png", size=size)
    return tmpdir


# =========================
# IC-LPIPS
# =========================

class LPIPSVGG:
    def __init__(self, device: torch.device) -> None:
        self.metric = lpips.LPIPS(net="vgg").to(device)
        self.metric.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x = x * 2.0 - 1.0
        y = y * 2.0 - 1.0
        return float(self.metric(x, y).mean().item())


def np_hwc_to_torch_bchw(img: np.ndarray, device: torch.device, size: int = 256) -> torch.Tensor:
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x


def cluster_key_standard(record: Dict[str, Any]) -> Tuple[str, str]:
    return resolve_category(record), resolve_defect_type(record)


@torch.no_grad()
def compute_ic_lpips_standard(
    records: List[Dict[str, Any]],
    device: torch.device,
    image_size: int = 256,
) -> Dict[str, Any]:
    """
    Standard IC-LPIPS:
    cluster = (category, defect_type)
    score(cluster) = average pairwise LPIPS among generated images in that cluster
    final score = mean of cluster scores
    """
    lpips_metric = LPIPSVGG(device)
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for r in records:
        groups[cluster_key_standard(r)].append(resolve_generated_image(r))

    cluster_rows = []
    cluster_scores = []

    for key, img_paths in groups.items():
        if len(img_paths) < 2:
            continue

        pair_scores = []
        imgs = [load_rgb(p) for p in img_paths]

        for i, j in combinations(range(len(imgs)), 2):
            xi = np_hwc_to_torch_bchw(imgs[i], device, image_size)
            xj = np_hwc_to_torch_bchw(imgs[j], device, image_size)
            s = lpips_metric(xi, xj)
            pair_scores.append(s)

        score = float(np.mean(pair_scores))
        cluster_scores.append(score)
        cluster_rows.append(
            {
                "category": key[0],
                "defect_type": key[1],
                "num_images": len(img_paths),
                "num_pairs": len(pair_scores),
                "IC_LPIPS": score,
            }
        )

    overall = float(np.mean(cluster_scores)) if len(cluster_scores) > 0 else float("nan")
    return {
        "overall_ic_lpips": overall,
        "clusters": cluster_rows,
    }


# =========================
# Aggregation helpers
# =========================

def mean_std(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {"mean": float("nan"), "std": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
    }


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=256)

    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    records = read_summary(args.summary_json)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    if len(records) == 0:
        raise RuntimeError("No records found.")

    gen_paths = [resolve_generated_image(r) for r in records]
    ref_paths = [resolve_reference_image(r) for r in records]

    gen_dir = build_temp_image_dir(gen_paths, size=args.image_size)
    ref_dir = build_temp_image_dir(ref_paths, size=args.image_size)

    try:
        metrics = calculate_metrics(
            input1=gen_dir,
            input2=ref_dir,
            isc=True,
            fid=True,
            cuda=(device.type == "cuda"),
            verbose=False,
        )
    finally:
        shutil.rmtree(gen_dir, ignore_errors=True)
        shutil.rmtree(ref_dir, ignore_errors=True)

    iclpips = compute_ic_lpips_standard(
        records=records,
        device=device,
        image_size=args.image_size,
    )

    # by-category IS/FID
    by_category: Dict[str, Any] = {}
    categories = sorted(set(resolve_category(r) for r in records))
    for cat in categories:
        cat_records = [r for r in records if resolve_category(r) == cat]
        if len(cat_records) < 2:
            continue

        cat_gen = [resolve_generated_image(r) for r in cat_records]
        cat_ref = [resolve_reference_image(r) for r in cat_records]

        cat_gen_dir = build_temp_image_dir(cat_gen, size=args.image_size)
        cat_ref_dir = build_temp_image_dir(cat_ref, size=args.image_size)

        try:
            cat_metrics = calculate_metrics(
                input1=cat_gen_dir,
                input2=cat_ref_dir,
                isc=True,
                fid=True,
                cuda=(device.type == "cuda"),
                verbose=False,
            )
        finally:
            shutil.rmtree(cat_gen_dir, ignore_errors=True)
            shutil.rmtree(cat_ref_dir, ignore_errors=True)

        cat_ic_rows = [x for x in iclpips["clusters"] if x["category"] == cat]
        by_category[cat] = {
            "num_samples": len(cat_records),
            "IS_mean": float(cat_metrics["inception_score_mean"]),
            "IS_std": float(cat_metrics["inception_score_std"]),
            "FID": float(cat_metrics["frechet_inception_distance"]),
            "IC_LPIPS": mean_std([x["IC_LPIPS"] for x in cat_ic_rows]),
        }

    # save cluster CSV
    cluster_csv = output_dir / "ic_lpips_clusters.csv"
    with open(cluster_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "defect_type", "num_images", "num_pairs", "IC_LPIPS"],
        )
        writer.writeheader()
        for row in iclpips["clusters"]:
            writer.writerow(row)

    summary = {
        "overall": {
            "num_samples": len(records),
            "IS_mean": float(metrics["inception_score_mean"]),
            "IS_std": float(metrics["inception_score_std"]),
            "FID": float(metrics["frechet_inception_distance"]),
            "IC_LPIPS": float(iclpips["overall_ic_lpips"]),
        },
        "by_category": by_category,
    }

    save_json(summary, output_dir / "metrics_summary.json")

    print(f"[Saved] IC-LPIPS clusters -> {cluster_csv}")
    print(f"[Saved] summary          -> {output_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()