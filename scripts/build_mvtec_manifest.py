from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


@dataclass
class ManifestRecord:
    category: str
    defect_image_path: str
    defect_mask_path: str
    scene_image_path: str
    defect_type: str
    sample_id: str


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def stem_without_mask_suffix(name: str) -> str:
    """
    官方 ground_truth 文件通常类似:
      000_mask.png
    这里转成:
      000
    """
    if name.endswith("_mask"):
        return name[:-5]
    return name


def find_matching_good_image(train_good_dir: Path, defect_image_name: str) -> Path | None:
    """
    按用户要求，背景图固定选择 train/good 中与缺陷图同名的文件。
    """
    candidate = train_good_dir / defect_image_name
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def collect_records_for_category(category_dir: Path) -> Tuple[List[ManifestRecord], List[str]]:
    """
    遍历单个类别，生成 manifest 记录。
    返回:
      - records
      - warnings
    """
    category = category_dir.name
    warnings: List[str] = []
    records: List[ManifestRecord] = []

    train_good_dir = category_dir / "train" / "good"
    test_dir = category_dir / "test"
    gt_dir = category_dir / "ground_truth"

    if not train_good_dir.exists():
        warnings.append(f"[{category}] Missing train/good: {train_good_dir}")
        return records, warnings
    if not test_dir.exists():
        warnings.append(f"[{category}] Missing test dir: {test_dir}")
        return records, warnings
    if not gt_dir.exists():
        warnings.append(f"[{category}] Missing ground_truth dir: {gt_dir}")
        return records, warnings

    defect_types = sorted([p for p in test_dir.iterdir() if p.is_dir() and p.name != "good"])

    for defect_type_dir in defect_types:
        defect_type = defect_type_dir.name
        gt_type_dir = gt_dir / defect_type

        if not gt_type_dir.exists():
            warnings.append(f"[{category}/{defect_type}] Missing ground truth dir: {gt_type_dir}")
            continue

        defect_images = sorted([p for p in defect_type_dir.iterdir() if p.is_file() and is_image_file(p)])

        for defect_img_path in defect_images:
            # 1) scene image: train/good 中同名图
            scene_img_path = find_matching_good_image(train_good_dir, defect_img_path.name)
            if scene_img_path is None:
                warnings.append(
                    f"[{category}/{defect_type}] No matching train/good image for defect image: {defect_img_path.name}"
                )
                continue

            # 2) ground truth mask: 通常是 stem + _mask + suffix
            expected_mask_name = f"{defect_img_path.stem}_mask{defect_img_path.suffix}"
            defect_mask_path = gt_type_dir / expected_mask_name

            if not defect_mask_path.exists():
                # 尝试兜底：按 stem 对齐查找
                candidates = [
                    p for p in gt_type_dir.iterdir()
                    if p.is_file()
                    and is_image_file(p)
                    and stem_without_mask_suffix(p.stem) == defect_img_path.stem
                ]
                if len(candidates) == 1:
                    defect_mask_path = candidates[0]
                else:
                    warnings.append(
                        f"[{category}/{defect_type}] Missing unique mask for defect image: {defect_img_path.name}"
                    )
                    continue

            sample_id = f"{category}_{defect_type}_{defect_img_path.stem}"

            records.append(
                ManifestRecord(
                    category=category,
                    defect_image_path=str(defect_img_path.as_posix()),
                    defect_mask_path=str(defect_mask_path.as_posix()),
                    scene_image_path=str(scene_img_path.as_posix()),
                    defect_type=defect_type,
                    sample_id=sample_id,
                )
            )

    return records, warnings


def split_train_val(
    records: List[ManifestRecord],
    val_ratio: float,
    seed: int,
) -> Tuple[List[ManifestRecord], List[ManifestRecord]]:
    """
    按 category 分层划分 train/val。
    """
    rng = random.Random(seed)

    by_category: Dict[str, List[ManifestRecord]] = {}
    for r in records:
        by_category.setdefault(r.category, []).append(r)

    train_records: List[ManifestRecord] = []
    val_records: List[ManifestRecord] = []

    for category, items in sorted(by_category.items()):
        rng.shuffle(items)
        n_total = len(items)
        n_val = max(1, int(round(n_total * val_ratio))) if n_total > 1 else 0

        val_items = items[:n_val]
        train_items = items[n_val:]

        if len(train_items) == 0 and len(val_items) > 1:
            train_items = [val_items.pop()]

        train_records.extend(train_items)
        val_records.extend(val_items)

    return train_records, val_records


def save_manifest(records: List[ManifestRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvtec_root", type=str, default="data/mvtec")
    parser.add_argument("--train_out", type=str, default="data/manifest_train.json")
    parser.add_argument("--val_out", type=str, default="data/manifest_val.json")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict_categories", action="store_true")
    args = parser.parse_args()

    mvtec_root = Path(args.mvtec_root)
    if not mvtec_root.exists():
        raise FileNotFoundError(f"MVTec root not found: {mvtec_root}")

    all_records: List[ManifestRecord] = []
    all_warnings: List[str] = []

    categories = MVTEC_CATEGORIES if args.strict_categories else sorted(
        [p.name for p in mvtec_root.iterdir() if p.is_dir()]
    )

    for category in categories:
        category_dir = mvtec_root / category
        if not category_dir.exists():
            all_warnings.append(f"[Missing category] {category_dir}")
            continue

        records, warnings = collect_records_for_category(category_dir)
        all_records.extend(records)
        all_warnings.extend(warnings)

    if len(all_records) == 0:
        raise RuntimeError("No manifest records collected. Please check dataset structure and filenames.")

    train_records, val_records = split_train_val(
        records=all_records,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    save_manifest(train_records, Path(args.train_out))
    save_manifest(val_records, Path(args.val_out))

    print(f"Collected total records: {len(all_records)}")
    print(f"Train records: {len(train_records)}")
    print(f"Val records:   {len(val_records)}")
    print(f"Train manifest saved to: {args.train_out}")
    print(f"Val manifest saved to:   {args.val_out}")

    if all_warnings:
        warn_path = Path(args.train_out).parent / "manifest_warnings.log"
        warn_path.parent.mkdir(parents=True, exist_ok=True)
        with open(warn_path, "w", encoding="utf-8") as f:
            for w in all_warnings:
                f.write(w + "\n")
        print(f"Warnings: {len(all_warnings)}")
        print(f"Warnings saved to: {warn_path}")
    else:
        print("Warnings: 0")


if __name__ == "__main__":
    main()