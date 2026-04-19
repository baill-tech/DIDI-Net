from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

from models.encoders.model_paths import get_local_model_path


@dataclass
class SAMPreprocessOutput:
    """
    Standardized clean object/template output.

    clean_object:
        Background-removed object centered on 512x512 canvas, float32 [3,H,W] in [0,1]

    clean_mask:
        Object mask centered on 512x512 canvas, float32 [1,H,W] in [0,1]

    raw_object_mask:
        Original binary mask on original image resolution, float32 [1,H,W] in [0,1]

    bbox_xyxy:
        Bounding box on original image, (x1, y1, x2, y2)

    scale_ratio:
        Resize ratio used when pasting object crop to canvas

    center_offset:
        Top-left offset (paste_x, paste_y) on output canvas

    pose_prior:
        Simple geometric prior estimated from the raw mask
    """
    clean_object: torch.Tensor
    clean_mask: torch.Tensor
    raw_object_mask: torch.Tensor
    bbox_xyxy: torch.Tensor
    scale_ratio: float
    center_offset: torch.Tensor
    pose_prior: torch.Tensor


def load_rgb_image(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_mask_image(path: str | Path) -> np.ndarray:
    """
    Load mask as uint8 HxW in {0,255}
    """
    mask = Image.open(path).convert("L")
    mask = np.array(mask)
    mask = (mask > 127).astype(np.uint8) * 255
    return mask


def to_tensor_image(img: np.ndarray) -> torch.Tensor:
    """
    img: HWC uint8 or float32 [0,255] / [0,1]
    return: CHW float32 [0,1]
    """
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


def to_tensor_mask(mask: np.ndarray) -> torch.Tensor:
    """
    mask: HW uint8 or float32
    return: 1HW float32 [0,1]
    """
    if mask.dtype != np.float32:
        mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    return torch.from_numpy(mask[None, ...]).contiguous()


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 127).astype(np.uint8) * 255
    return mask


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    mask: uint8 HxW in {0,255}
    """
    mask_bin = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask_bin, dtype=np.uint8)
    out[labels == largest_idx] = 1
    return out * 255


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes inside foreground.
    """
    mask = ensure_binary_mask(mask)
    h, w = mask.shape

    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask, flood_inv)
    return filled


def morphological_refine(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Return xyxy on original resolution.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        h, w = mask.shape
        return 0, 0, w - 1, h - 1

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return x1, y1, x2, y2


def expand_bbox(
    bbox_xyxy: Tuple[int, int, int, int],
    image_hw: Tuple[int, int],
    expand_ratio: float = 0.08,
) -> Tuple[int, int, int, int]:
    """
    Expand bbox a little to preserve boundary context.
    """
    h, w = image_hw
    x1, y1, x2, y2 = bbox_xyxy
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1

    dx = int(round(bw * expand_ratio))
    dy = int(round(bh * expand_ratio))

    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w - 1, x2 + dx)
    y2 = min(h - 1, y2 + dy)

    return x1, y1, x2, y2


def crop_by_bbox(
    image: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox_xyxy
    crop_img = image[y1:y2 + 1, x1:x2 + 1]
    crop_mask = mask[y1:y2 + 1, x1:x2 + 1]
    return crop_img, crop_mask


def remove_background(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Background removed object image.
    image: HWC RGB
    mask:  HW uint8 in {0,255}
    """
    mask_f = (mask.astype(np.float32) / 255.0)[..., None]
    clean = image.astype(np.float32) * mask_f
    return clean.astype(np.float32)


def center_align_to_canvas(
    object_img: np.ndarray,
    object_mask: np.ndarray,
    canvas_size: int = 512,
    fill_value: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
    """
    Resize object crop with aspect ratio preserved, then paste to canvas center.

    Returns:
        canvas_img: HWC float32
        canvas_mask: HW uint8
        scale_ratio
        (paste_x, paste_y)
    """
    h, w = object_img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid object crop shape.")

    scale = min(canvas_size / h, canvas_size / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resized_img = cv2.resize(object_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(object_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas_img = np.full((canvas_size, canvas_size, 3), fill_value, dtype=np.float32)
    canvas_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    paste_x = (canvas_size - new_w) // 2
    paste_y = (canvas_size - new_h) // 2

    canvas_img[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = resized_img
    canvas_mask[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = resized_mask

    return canvas_img, canvas_mask, float(scale), (int(paste_x), int(paste_y))


def estimate_pose_prior_from_mask(mask: np.ndarray) -> torch.Tensor:
    """
    Estimate a simple geometric prior from mask.
    Current version returns a compact 10D vector:
        [area_ratio,
         cx_norm, cy_norm,
         bbox_w_norm, bbox_h_norm,
         aspect_ratio,
         angle_cos, angle_sin,
         major_len_norm, minor_len_norm]
    """
    mask_bin = (mask > 0).astype(np.uint8)
    h, w = mask_bin.shape
    area = float(mask_bin.sum())
    area_ratio = area / float(h * w + 1e-6)

    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return torch.zeros(10, dtype=torch.float32)

    x1, y1, x2, y2 = bbox_from_mask(mask_bin * 255)
    bw = float(x2 - x1 + 1)
    bh = float(y2 - y1 + 1)

    cx = float(xs.mean()) / float(w + 1e-6)
    cy = float(ys.mean()) / float(h + 1e-6)

    bbox_w_norm = bw / float(w + 1e-6)
    bbox_h_norm = bh / float(h + 1e-6)
    aspect_ratio = bw / float(bh + 1e-6)

    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    coords = coords - coords.mean(axis=0, keepdims=True)

    if coords.shape[0] < 2:
        angle_cos, angle_sin = 1.0, 0.0
        major_len_norm, minor_len_norm = 0.0, 0.0
    else:
        cov = np.cov(coords.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        major_vec = eigvecs[:, 0]
        angle = float(np.arctan2(major_vec[1], major_vec[0]))

        angle_cos = float(np.cos(angle))
        angle_sin = float(np.sin(angle))

        major_len_norm = float(np.sqrt(max(eigvals[0], 0.0))) / float(max(w, h) + 1e-6)
        minor_len_norm = float(np.sqrt(max(eigvals[1], 0.0))) / float(max(w, h) + 1e-6)

    feat = torch.tensor(
        [
            area_ratio,
            cx,
            cy,
            bbox_w_norm,
            bbox_h_norm,
            aspect_ratio,
            angle_cos,
            angle_sin,
            major_len_norm,
            minor_len_norm,
        ],
        dtype=torch.float32,
    )
    return feat


def sample_sam_points_from_mask(
    mask: np.ndarray,
    num_pos: int = 2,
    num_neg: int = 2,
) -> Tuple[List[List[float]], List[int]]:
    """
    mask: HW uint8 in {0,255}
    returns:
      point_coords: [[x,y], ...]
      point_labels: [1,1,0,0]
    """
    mask_bin = (mask > 0).astype(np.uint8)
    pos_ys, pos_xs = np.where(mask_bin > 0)
    neg_ys, neg_xs = np.where(mask_bin == 0)

    if len(pos_xs) == 0:
        raise ValueError("Cannot sample positive points from empty mask.")

    points: List[List[float]] = []
    labels: List[int] = []

    pos_count = min(num_pos, len(pos_xs))
    pos_idx = np.random.choice(len(pos_xs), size=pos_count, replace=False)
    for i in pos_idx:
        points.append([float(pos_xs[i]), float(pos_ys[i])])
        labels.append(1)

    if len(neg_xs) > 0:
        neg_count = min(num_neg, len(neg_xs))
        neg_idx = np.random.choice(len(neg_xs), size=neg_count, replace=False)
        for i in neg_idx:
            points.append([float(neg_xs[i]), float(neg_ys[i])])
            labels.append(0)

    return points, labels


def mask_to_box_xyxy(mask: np.ndarray) -> List[float]:
    x1, y1, x2, y2 = bbox_from_mask(mask)
    return [float(x1), float(y1), float(x2), float(y2)]


class RealSAMPredictor:
    """
    Real SAM predictor backed by transformers SamModel/SamProcessor.

    Supports:
      - point prompts
      - box prompts
      - point + box prompts
    """

    def __init__(
        self,
        model_alias: str = "sam_model",
        device: str = "cuda",
        multimask_output: bool = True,
    ) -> None:
        self.model_dir = get_local_model_path(model_alias)
        self.processor = SamProcessor.from_pretrained(self.model_dir)
        self.model = SamModel.from_pretrained(self.model_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.multimask_output = multimask_output

    @torch.no_grad()
    def predict_mask(
        self,
        image_rgb: np.ndarray,
        point_coords: Optional[Sequence[Sequence[float]]] = None,
        point_labels: Optional[Sequence[int]] = None,
        box_xyxy: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """
        image_rgb: HWC uint8 RGB
        point_coords: [[x, y], ...] in original image coordinates
        point_labels: [1, 0, ...] 1=positive, 0=negative
        box_xyxy: [x1, y1, x2, y2]
        return: HW uint8 mask in {0,255}
        """
        input_points = None
        input_labels = None
        input_boxes = None

        if point_coords is not None and point_labels is not None:
            # processor expects batch -> prompt_group -> point
            input_points = [[list(map(float, p)) for p in point_coords]]
            input_labels = [list(map(int, point_labels))]

        if box_xyxy is not None:
            input_boxes = [[list(map(float, box_xyxy))]]

        inputs = self.processor(
            images=image_rgb,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(
            **inputs,
            multimask_output=self.multimask_output,
        )

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]

        iou_scores = outputs.iou_scores[0].detach().cpu()

        if masks.ndim == 4:
            # [num_prompts, num_masks, H, W]
            masks_ = masks[0]
            scores_ = iou_scores[0]
        elif masks.ndim == 3:
            # [num_masks, H, W]
            masks_ = masks
            scores_ = iou_scores
        else:
            raise ValueError(f"Unexpected mask shape: {masks.shape}")

        best_idx = int(torch.argmax(scores_).item())
        best_mask = masks_[best_idx].numpy().astype(np.float32)
        best_mask = (best_mask > 0).astype(np.uint8) * 255
        return best_mask


class SAMTemplatePreprocessor:
    """
    Executable preprocessing pipeline.

    Current capabilities:
        - real SAM segmentation if sam_predictor is provided
        - fallback to GT mask proxy if sam_predictor is not provided
        - standardized O_clean generation
        - center align to 512x512 canvas
    """

    def __init__(
        self,
        canvas_size: int = 512,
        bbox_expand_ratio: float = 0.08,
        morphology_ksize: int = 5,
        use_largest_component: bool = True,
        fill_holes_flag: bool = True,
    ) -> None:
        self.canvas_size = canvas_size
        self.bbox_expand_ratio = bbox_expand_ratio
        self.morphology_ksize = morphology_ksize
        self.use_largest_component = use_largest_component
        self.fill_holes_flag = fill_holes_flag

    def generate_object_mask(
        self,
        image_rgb: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
        sam_predictor: Optional[object] = None,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Preferred path:
            Real SAM inference with prompts
        Fallback path:
            GT mask proxy
        """
        if sam_predictor is not None:
            point_coords = None
            point_labels = None
            box_xyxy = None

            if points is not None and labels is not None:
                point_coords = points.tolist() if hasattr(points, "tolist") else points
                point_labels = labels.tolist() if hasattr(labels, "tolist") else labels

            if gt_mask is not None:
                box_xyxy = mask_to_box_xyxy(gt_mask)

                if point_coords is None or point_labels is None:
                    point_coords, point_labels = sample_sam_points_from_mask(gt_mask)

            pred_mask = sam_predictor.predict_mask(
                image_rgb=image_rgb,
                point_coords=point_coords,
                point_labels=point_labels,
                box_xyxy=box_xyxy,
            )
            return ensure_binary_mask(pred_mask)

        if gt_mask is not None:
            return ensure_binary_mask(gt_mask)

        raise ValueError("Either sam_predictor or gt_mask must be provided.")

    def refine_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = ensure_binary_mask(mask)

        if self.use_largest_component:
            mask = keep_largest_connected_component(mask)

        mask = morphological_refine(mask, ksize=self.morphology_ksize)

        if self.fill_holes_flag:
            mask = fill_holes(mask)

        mask = ensure_binary_mask(mask)
        return mask

    def preprocess(
        self,
        image_rgb: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
        sam_predictor: Optional[object] = None,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> SAMPreprocessOutput:
        """
        Full pipeline:
            image -> object mask -> refine -> O_clean -> crop -> center align 512
        """
        raw_mask = self.generate_object_mask(
            image_rgb=image_rgb,
            gt_mask=gt_mask,
            sam_predictor=sam_predictor,
            points=points,
            labels=labels,
        )
        raw_mask = self.refine_mask(raw_mask)

        bbox = bbox_from_mask(raw_mask)
        bbox = expand_bbox(
            bbox_xyxy=bbox,
            image_hw=image_rgb.shape[:2],
            expand_ratio=self.bbox_expand_ratio,
        )

        crop_img, crop_mask = crop_by_bbox(image_rgb, raw_mask, bbox)
        clean_crop = remove_background(crop_img, crop_mask)

        canvas_img, canvas_mask, scale_ratio, center_offset = center_align_to_canvas(
            object_img=clean_crop,
            object_mask=crop_mask,
            canvas_size=self.canvas_size,
            fill_value=0,
        )

        pose_prior = estimate_pose_prior_from_mask(raw_mask)

        out = SAMPreprocessOutput(
            clean_object=to_tensor_image(canvas_img),
            clean_mask=to_tensor_mask(canvas_mask),
            raw_object_mask=to_tensor_mask(raw_mask),
            bbox_xyxy=torch.tensor(bbox, dtype=torch.float32),
            scale_ratio=scale_ratio,
            center_offset=torch.tensor(center_offset, dtype=torch.float32),
            pose_prior=pose_prior,
        )
        return out


def save_preprocess_output(
    output: SAMPreprocessOutput,
    clean_object_path: str | Path,
    clean_mask_path: str | Path,
    meta_path: Optional[str | Path] = None,
) -> None:
    clean_object_path = Path(clean_object_path)
    clean_mask_path = Path(clean_mask_path)

    clean_object_path.parent.mkdir(parents=True, exist_ok=True)
    clean_mask_path.parent.mkdir(parents=True, exist_ok=True)

    clean_obj = (output.clean_object.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    clean_mask = (output.clean_mask.squeeze(0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

    Image.fromarray(clean_obj).save(clean_object_path)
    Image.fromarray(clean_mask).save(clean_mask_path)

    if meta_path is not None:
        import json

        meta_path = Path(meta_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "bbox_xyxy": output.bbox_xyxy.tolist(),
            "scale_ratio": float(output.scale_ratio),
            "center_offset": output.center_offset.tolist(),
            "pose_prior": output.pose_prior.tolist(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--mask", type=str, required=False, default=None)
    parser.add_argument("--out_img", type=str, required=True)
    parser.add_argument("--out_mask", type=str, required=True)
    parser.add_argument("--out_meta", type=str, default=None)
    parser.add_argument("--use_real_sam", action="store_true")
    parser.add_argument("--sam_model_alias", type=str, default="sam_model")
    args = parser.parse_args()

    image = load_rgb_image(args.image)
    gt_mask = load_mask_image(args.mask) if args.mask is not None else None

    sam_predictor = None
    if args.use_real_sam:
        sam_predictor = RealSAMPredictor(
            model_alias=args.sam_model_alias,
            device="cuda",
            multimask_output=True,
        )

    preprocessor = SAMTemplatePreprocessor(
        canvas_size=512,
        bbox_expand_ratio=0.08,
        morphology_ksize=5,
        use_largest_component=True,
        fill_holes_flag=True,
    )

    result = preprocessor.preprocess(
        image_rgb=image,
        gt_mask=gt_mask,
        sam_predictor=sam_predictor,
    )

    print("clean_object:", tuple(result.clean_object.shape))
    print("clean_mask:", tuple(result.clean_mask.shape))
    print("raw_object_mask:", tuple(result.raw_object_mask.shape))
    print("bbox_xyxy:", result.bbox_xyxy.tolist())
    print("scale_ratio:", result.scale_ratio)
    print("center_offset:", result.center_offset.tolist())
    print("pose_prior:", result.pose_prior.tolist())

    save_preprocess_output(
        output=result,
        clean_object_path=args.out_img,
        clean_mask_path=args.out_mask,
        meta_path=args.out_meta,
    )