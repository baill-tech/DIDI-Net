# DIDI-Net

High-Fidelity Industrial Defect Image Synthesis Via Dual_Level ID Feature Learning and Decouping

## Overview

DIDI-Net is a controllable industrial defect synthesis framework designed for data augmentation in anomaly detection scenarios. Given:

- a **normal scene image**,
- a **defect source image**,
- a **defect mask / location mask**,

DIDI-Net first extracts a clean defect template, then encodes defect identity with dual visual branches, and finally injects the defect into the target scene with a diffusion-based masked editing pipeline.

## Environment

Recommended environment:

- Python 3.10+
- PyTorch with CUDA support
- Linux

Install dependencies:

```bash
conda create -n DIDI-Net python=3.10
conda activate DIDI-Net
pip install -r requirements.txt
```

## Pretrained Models

You can download the main backbones with:

```bash
python scripts/download_backbones_from_modelscope.py
```

### Additional external dependency

We use DINO-MC as the RS-DINO branch:

Please clone or copy the required DINO-MC codebase:

```bash
cd external
git clone https://github.com/WennyXY/DINO-MC.git
```
You also need to manually prepare the DINO-MC checkpoint, for example:

```text
pretrained/dino_mc/dino_mc_vits8_checkpoint.pth
```

## Dataset Preparation

This project currently targets the **MVTec AD** directory layout.
You can download it from [MVtec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)

Expected structure:

```text
data/mvtec/
├── bottle/
│   ├── train/good/
│   ├── test/broken_large/
│   └── ground_truth/broken_large/
├── cable/
├── capsule/
└── ...
```

## Get started

To build train / validation manifests:

```bash
python scripts/build_mvtec_manifest.py \
  --mvtec_root data/mvtec \
  --train_out data/manifest_train.json \
  --val_out data/manifest_val.json \
  --val_ratio 0.2
```

All the example training commands are in:

```text
run.sh
```

## Acknowledgements

This project builds upon the following open-source ecosystems:

- PyTorch
- diffusers
- transformers
- Segment Anything Model (SAM)
- DINOv2
- Stable Diffusion 2.1

## Citation

If you find this project useful in your research, please cite the corresponding paper or project page once publicly available.

## License
