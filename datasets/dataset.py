from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from datasets.sam_preprocess import (
    SAMTemplatePreprocessor,
    RealSAMPredictor,
    load_rgb_image,
    load_mask_image,
)


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class SampleRecord:
    category: str
    defect_image_path: str
    defect_mask_path: str
    scene_image_path: str
    defect_type: str
    sample_id: str


def normalize_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def binarize_mask(mask: np.ndarray, threshold: int = 127, to_255: bool = True) -> np.ndarray:
    out = (mask > threshold).astype(np.uint8)
    if to_255:
        out = out * 255
    return out


def resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    size = (width, height)
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)


def to_tensor_image(img: np.ndarray) -> Tensor:
    """
    img: HWC, float32 in [0,1]
    return: CHW float32
    """
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().float()


def to_tensor_mask(mask: np.ndarray) -> Tensor:
    """
    mask: HW, float32 in [0,1] or uint8 in {0,255}
    return: 1HW float32 in [0,1]
    """
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    return torch.from_numpy(mask[None, ...]).contiguous().float()


def resize_chw_tensor(x: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    """
    x: [C,H,W] or [1,H,W]
    """
    x = x.unsqueeze(0)
    if mode == "nearest":
        x = F.interpolate(x, size=(size, size), mode=mode)
    else:
        x = F.interpolate(x, size=(size, size), mode=mode, align_corners=False)
    return x.squeeze(0)


class MVTecDefectSynthesisDataset(Dataset):
    """
    Current dataset protocol:
    - scene_img: normal image from train/good
    - target_img: defective image from test/<defect_type>
    - loc_mask: original defect mask (resized to image_size)
    - defect_template_clean: SAM-style clean template
    - template_pose_prior: simple geometric prior estimated from mask

    Backward compatibility:
    - defect_template == defect_template_clean
    - template_mask == template_mask_clean

    Notes:
    - use_real_sam=False: GT mask proxy mode
    - use_real_sam=True: real SAM segmentation mode
    - For online SAM inference, num_workers=0 is strongly recommended.
    """

    def __init__(
        self,
        records: Sequence[SampleRecord],
        image_size: int = 512,
        template_size: int = 224,
        bbox_expand_scale: float = 1.2,
        sam_canvas_size: int = 512,
        use_real_sam: bool = False,
        sam_model_alias: str = "sam_model",
        sam_device: str = "cuda",
        return_debug_vis: bool = False,
    ) -> None:
        self.records = list(records)
        self.image_size = image_size
        self.template_size = template_size
        self.bbox_expand_scale = bbox_expand_scale
        self.sam_canvas_size = sam_canvas_size
        self.use_real_sam = use_real_sam
        self.sam_model_alias = sam_model_alias
        self.sam_device = sam_device
        self.return_debug_vis = return_debug_vis

        self.sam_preprocessor = SAMTemplatePreprocessor(
            canvas_size=sam_canvas_size,
            bbox_expand_ratio=0.08,
            morphology_ksize=5,
            use_largest_component=True,
            fill_holes_flag=True,
        )

        # lazy-init to avoid pickling / CUDA issues in dataset workers
        self._real_sam_predictor: Optional[RealSAMPredictor] = None

    def __len__(self) -> int:
        return len(self.records)

    def _get_sam_predictor(self) -> Optional[RealSAMPredictor]:
        if not self.use_real_sam:
            return None

        if self._real_sam_predictor is None:
            self._real_sam_predictor = RealSAMPredictor(
                model_alias=self.sam_model_alias,
                device=self.sam_device,
                multimask_output=True,
            )
        return self._real_sam_predictor

    def _prepare_image(self, image_np: np.ndarray, out_size: int) -> torch.Tensor:
        image = normalize_image(image_np)
        image = resize_image(image, (out_size, out_size))
        return to_tensor_image(image)

    def _prepare_mask(self, mask_np: np.ndarray, out_size: int) -> torch.Tensor:
        mask = binarize_mask(mask_np, to_255=False).astype(np.float32)
        mask = resize_mask(mask, (out_size, out_size))
        mask = (mask > 0.5).astype(np.float32)
        return to_tensor_mask(mask)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self.records[index]

        # -------------------------
        # Load raw inputs
        # -------------------------
        scene_img_np = load_rgb_image(rec.scene_image_path)          # HWC uint8 RGB
        defect_img_np = load_rgb_image(rec.defect_image_path)        # HWC uint8 RGB
        defect_mask_np = load_mask_image(rec.defect_mask_path)       # HW uint8 {0,255}

        # -------------------------
        # Full-image branch
        # -------------------------
        scene_img = self._prepare_image(scene_img_np, self.image_size)   # [3,H,W]
        defect_img = self._prepare_image(defect_img_np, self.image_size) # [3,H,W]
        loc_mask = self._prepare_mask(defect_mask_np, self.image_size)   # [1,H,W]

        # -------------------------
        # SAM-style clean template preprocessing
        # use_real_sam=True -> real SAM with GT-mask-derived prompts
        # use_real_sam=False -> GT mask proxy mode
        # -------------------------
        sam_predictor = self._get_sam_predictor()

        sam_out = self.sam_preprocessor.preprocess(
            image_rgb=defect_img_np,
            gt_mask=defect_mask_np,
            sam_predictor=sam_predictor,
        )

        # 512x512 canonical canvas
        defect_template_clean_512 = sam_out.clean_object      # [3,512,512]
        template_mask_clean_512 = sam_out.clean_mask          # [1,512,512]
        template_pose_prior = sam_out.pose_prior              # [10]

        # Resize to training template size
        defect_template_clean = resize_chw_tensor(
            defect_template_clean_512,
            self.template_size,
            mode="bilinear",
        )
        template_mask_clean = resize_chw_tensor(
            template_mask_clean_512,
            self.template_size,
            mode="nearest",
        )

        sample: Dict[str, Any] = {
            # full image branch
            "scene_img": scene_img,                                # [3, H, W]
            "defect_img": defect_img,                              # [3, H, W]
            "target_img": defect_img,                              # [3, H, W]
            "loc_mask": loc_mask,                                  # [1, H, W]

            # backward-compatible names
            "defect_template": defect_template_clean,              # [3, T, T]
            "template_mask": template_mask_clean,                  # [1, T, T]

            # explicit clean-template branch
            "defect_template_clean": defect_template_clean,         # [3, T, T]
            "template_mask_clean": template_mask_clean,             # [1, T, T]
            "defect_template_clean_512": defect_template_clean_512, # [3, 512, 512]
            "template_mask_clean_512": template_mask_clean_512,     # [1, 512, 512]
            "template_pose_prior": template_pose_prior,             # [10]

            # metadata
            "category": rec.category,
            "defect_type": rec.defect_type,
            "sample_id": rec.sample_id,
            "meta": {
                "scene_image_path": rec.scene_image_path,
                "defect_image_path": rec.defect_image_path,
                "defect_mask_path": rec.defect_mask_path,
                "sam_bbox_xyxy": sam_out.bbox_xyxy.tolist(),
                "sam_scale_ratio": float(sam_out.scale_ratio),
                "sam_center_offset": sam_out.center_offset.tolist(),
                "sam_used_real_model": bool(sam_predictor is not None),
            },
        }

        if self.return_debug_vis:
            sample["debug_vis"] = {
                "scene_img_np": normalize_image(scene_img_np),
                "defect_img_np": normalize_image(defect_img_np),
                "defect_mask_np": (defect_mask_np > 127).astype(np.float32),

                "defect_template_clean_np": defect_template_clean.permute(1, 2, 0).cpu().numpy(),
                "template_mask_clean_np": template_mask_clean.squeeze(0).cpu().numpy(),

                "defect_template_clean_512_np": defect_template_clean_512.permute(1, 2, 0).cpu().numpy(),
                "template_mask_clean_512_np": template_mask_clean_512.squeeze(0).cpu().numpy(),

                "loc_mask_np": loc_mask.squeeze(0).cpu().numpy(),
                "template_pose_prior_np": template_pose_prior.cpu().numpy(),
            }

        return sample


def build_records_from_manifest(manifest_path: str) -> List[SampleRecord]:
    """
    Expected manifest format:
    [
      {
        "category": "bottle",
        "defect_image_path": "...",
        "defect_mask_path": "...",
        "scene_image_path": "...",
        "defect_type": "broken_large",
        "sample_id": "bottle_0001"
      },
      ...
    ]
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = [SampleRecord(**item) for item in data]
    return records