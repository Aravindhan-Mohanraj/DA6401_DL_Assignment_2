# DA6401 Assignment 2 – Multi-Task Visual Perception Pipeline

## **Aravindhan Mohanraj**
## **DA25S006**

## 1. WandB Report

**WandB Link**: *https://wandb.ai/da25s006-indian-institute-of-technology-madras/DL_Assignment_2_report/reports/DA6401-DL-Assignment-2--VmlldzoxNjQ5MDcyMw?accessToken=bdrntpa26v1m36tgkmoptbf22pw3nyc4l5l0bpfqwvu2jqk4pzwx9a65szxcua6l*

**GitHub Link**: *https://github.com/Aravindhan-Mohanraj/DA6401_DL_Assignment_2.git*

## 2. Problem Statement

Build a complete visual perception pipeline using PyTorch on the **Oxford-IIIT Pet Dataset**. The pipeline performs three tasks — breed classification, bounding box localization, and semantic segmentation — using a shared VGG11 backbone, all unified into a single multi-task model.

## 3. Objective

- Implement **VGG11 from scratch** with BatchNorm and a custom Dropout layer.
- Train a **37-class breed classifier** on the Pet dataset.
- Build a **bounding box localizer** with a custom IoU loss function.
- Construct a **U-Net style segmentation** network using VGG11 as encoder.
- Integrate all three into a **unified multi-task pipeline** with a single forward pass.

---

## 4. Folder Structure

```
da6401_Assignment_2/
├── data/
│   ├── __init__.py
│   ├── pets_dataset.py          # Dataset, augmentation, stratified split
│   └── oxford-iiit-pet/         # Dataset root (not submitted)
│       ├── images/
│       ├── images_aug/          # Offline augmented images
│       └── annotations/
│           ├── trimaps/
│           ├── trimaps_aug/
│           ├── xmls/            # Bounding box annotations
│           ├── trainval.txt
│           └── test.txt
├── models/
│   ├── __init__.py
│   ├── layers.py                # CustomDropout
│   ├── vgg11.py                 # VGG11 encoder + classifier head
│   ├── classification.py        # VGG11Classifier wrapper
│   ├── localization.py          # VGG11Localizer + BBoxHead
│   ├── segmentation.py          # VGG11UNet (encoder-decoder)
│   └── multitask.py             # MultiTaskPerceptionModel
├── losses/
│   ├── __init__.py
│   └── iou_loss.py              # Custom IoU loss
├── checkpoints/                 # Trained model weights
├── notebooks/                   # WandB experiment notebooks
├── train.py                     # Training entrypoint
├── inference.py                 # Inference & evaluation
├── requirements.txt
└── README.md
```

---

## 5. Code Files

| File | Description |
|------|-------------|
| `data/pets_dataset.py` | Dataset class returning (image, label, bbox, mask). Handles offline augmentation (4 policies: spatial, color, combined, quality) and stratified train/val splitting. |
| `models/vgg11.py` | VGG11 encoder with 5 conv blocks, BatchNorm2d after every conv, and a classifier head with BatchNorm1d. Returns skip connections for U-Net. |
| `models/layers.py` | Custom Dropout using inverted dropout scaling — no `nn.Dropout` or `F.dropout`. |
| `models/classification.py` | Thin wrapper around VGG11Encoder for 37-class breed classification. |
| `models/localization.py` | BBoxHead (AdaptiveAvgPool → MLP → Sigmoid) attached to frozen/fine-tuned VGG11 encoder. Outputs normalized `[cx, cy, w, h]`. |
| `models/segmentation.py` | U-Net decoder with transposed convolutions and skip connections from the VGG11 encoder. |
| `models/multitask.py` | Downloads checkpoints via gdown, loads all 3 sub-models, runs a single forward pass producing classification logits, bbox coordinates, and segmentation mask. |
| `losses/iou_loss.py` | Custom IoU loss: converts `(cx,cy,w,h)` to corners, computes `1 - IoU`. Supports mean/sum/none reduction. |
| `train.py` | Training script for all tasks. Supports Mixup, AMP, gradient clipping, cosine annealing, staged fine-tuning, early stopping, and WandB logging. |
| `inference.py` | Evaluation and visualization — test metrics for classification, bbox overlay grids for localization, mask overlay grids for segmentation. |

---

## 6. Dataset & Augmentation

**Dataset**: Oxford-IIIT Pet Dataset — 37 breeds, ~7,349 images with class labels, bounding boxes, and trimap masks (foreground/background/boundary).

**Split**: Stratified 90/10 train/val split using `StratifiedShuffleSplit` (seed=42). Test set from `test.txt`.

**Offline Augmentation**: 4 copies per image using 4 policies:
1. **Spatial** — horizontal flip, random crop, rotation
2. **Color** — brightness/contrast/saturation jitter, CLAHE, RGB shift
3. **Combined** — spatial + color + coarse dropout
4. **Quality** — blur, noise, JPEG compression

**Online Augmentation** (during training): RandomResizedCrop, HorizontalFlip, Rotate, ColorJitter, GaussianBlur, Normalize.

All transforms are bbox-aware using Albumentations' `BboxParams(format="yolo")`.

---

## 7. How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Prepare Dataset

Download the Oxford-IIIT Pet dataset and place it under `data/oxford-iiit-pet/`.

Generate offline augmentations:
```bash
python data/pets_dataset.py augment --data_dir data/oxford-iiit-pet --num_copies 4
```

### Train Models

```bash
# Classification
python train.py --task clf --use_wandb -b 64 --clf_lr 1e-4 --clf_epochs 70

# Localization
python train.py --task loc --use_wandb -b 32 --loc_lr 1e-3 --loc_epochs 50

# Segmentation
python train.py --task seg --use_wandb --seg_classes 3 --seg_lr 1e-3 --seg_epochs 30

# All tasks sequentially
python train.py --task all --use_wandb
```

### Run Inference

```bash
# Classification — test set evaluation
python inference.py clf --mode test

# Classification — single image
python inference.py clf --mode single --image_path path/to/image.jpg

# Localization — bbox visualization grid
python inference.py loc --n 16

# Segmentation — validation grid
python inference.py seg --mode val_grid --seg_classes 3

# Segmentation — single image
python inference.py seg --mode single --image_path path/to/image.jpg
```

---

## 8. Results

| Task | Metric | Score |
|------|--------|-------|
| Classification | Macro F1 | 0.7273 |
| Localization | Val mean IoU | 0.6853 |
| Segmentation | Test mean Dice | 0.8247 |
| Segmentation | Test pixel accuracy | 0.8751 |

---


