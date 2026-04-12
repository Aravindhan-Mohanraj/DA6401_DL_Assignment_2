"""Inference and evaluation for all three tasks.

Usage:
    python inference.py clf --mode test
    python inference.py clf --mode single --image_path path/to/img.jpg
    python inference.py loc --n 16
    python inference.py seg --mode val_grid --seg_classes 3
    python inference.py seg --mode single --image_path path/to/img.jpg
"""

import argparse
import os
import pathlib
import random

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from data.pets_dataset import (
    OxfordIIITPetDataset,
    build_eval_transform,
    stratified_train_val_split,
    INPUT_DIM,
)
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


def safe_torch_load(path, map_location=None):
    """Load checkpoint with weights_only=True when supported (PyTorch >= 1.13)."""
    _ver = tuple(int(x) for x in torch.__version__.split(".")[:2] if x.isdigit())
    if _ver >= (1, 13):
        return safe_torch_load(path, map_location=map_location)
    return torch.load(path, map_location=map_location)



# Constants

N_BREEDS = 37
CLF_CKPT = os.path.join("checkpoints", "classifier.pth")
LOC_CKPT = os.path.join("checkpoints", "localizer.pth")
SEG_CKPT = os.path.join("checkpoints", "unet")

RGB_MEAN = np.array([0.485, 0.456, 0.406])
RGB_STD = np.array([0.229, 0.224, 0.225])
SEG_COLORS = np.array([[0, 200, 0], [200, 0, 0], [0, 0, 200]], dtype=np.uint8)



# Shared helpers


def _denormalize(tensor):
    """[C,H,W] float tensor → [H,W,C] numpy in [0,1]."""
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(arr * RGB_STD + RGB_MEAN, 0, 1)


def _blend_mask(img_float, mask_np, alpha=0.5):
    """Alpha-blend color-coded mask onto float image."""
    out = (img_float * 255).astype(np.uint8).copy()
    for c in range(SEG_COLORS.shape[0]):
        region = mask_np == c
        if region.any():
            out[region] = ((1 - alpha) * out[region] + alpha * SEG_COLORS[c]).astype(np.uint8)
    return out.astype(np.float32) / 255.0


def _load_ckpt(model, ckpt_path, device):
    if os.path.exists(ckpt_path):
        sd = safe_torch_load(ckpt_path, map_location=device)
        model.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
        print(f"  Loaded {ckpt_path}")
    else:
        print(f"  Warning: {ckpt_path} not found — using random weights")


def _build_val_loader(data_root, batch_sz, n_workers, seed=42):
    root = pathlib.Path(data_root)
    ann = root / "annotations" / "trainval.txt"
    _, val_recs = stratified_train_val_split(str(ann), val_ratio=0.1, rng_seed=seed)
    ds = OxfordIIITPetDataset(
        str(root), entries=val_recs,
        img_dir=root / "images", mask_dir=root / "annotations" / "trimaps",
        transform=build_eval_transform(INPUT_DIM),
    )
    return DataLoader(ds, batch_size=batch_sz, shuffle=False, num_workers=n_workers)


def _build_test_loader(data_root, batch_sz, n_workers):
    ds = OxfordIIITPetDataset(data_root, split_name="test", transform=build_eval_transform(INPUT_DIM))
    return DataLoader(ds, batch_size=batch_sz, shuffle=False, num_workers=n_workers)


def _breed_name_map(data_root):
    """Parse list.txt → {0-indexed class_id: breed_name}."""
    fpath = os.path.join(data_root, "annotations", "list.txt")
    mapping = {}
    if not os.path.exists(fpath):
        return {i: f"Breed_{i}" for i in range(N_BREEDS)}
    with open(fpath) as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            parts = ln.strip().split()
            if len(parts) < 4:
                continue
            cid = int(parts[1]) - 1
            name = "_".join(parts[0].split("_")[:-1])
            if cid not in mapping:
                mapping[cid] = name
    return mapping


def _single_iou(box_a, box_b):
    """IoU between two (cx,cy,w,h) normalized boxes."""
    a_l, a_t = box_a[0] - box_a[2] / 2, box_a[1] - box_a[3] / 2
    a_r, a_b = box_a[0] + box_a[2] / 2, box_a[1] + box_a[3] / 2
    b_l, b_t = box_b[0] - box_b[2] / 2, box_b[1] - box_b[3] / 2
    b_r, b_b = box_b[0] + box_b[2] / 2, box_b[1] + box_b[3] / 2
    iw = max(0, min(a_r, b_r) - max(a_l, b_l))
    ih = max(0, min(a_b, b_b) - max(a_t, b_t))
    inter = iw * ih
    union = (a_r - a_l) * (a_b - a_t) + (b_r - b_l) * (b_b - b_t) - inter + 1e-6
    return inter / union



# Classification inference


def infer_clf(args):
    dev = torch.device(args.device)
    breeds = _breed_name_map(args.data_root)

    net = VGG11Classifier(num_classes=N_BREEDS).to(dev)
    _load_ckpt(net, CLF_CKPT, dev)
    net.eval()

    if args.mode == "test":
        dl = _build_test_loader(args.data_root, args.batch_size, args.num_workers)
        preds, labels = [], []
        print(f"  Evaluating on {len(dl.dataset)} test images ...")
        with torch.no_grad():
            for imgs, lbl, _, _ in dl:
                out = net(imgs.to(dev))
                preds.extend(out.argmax(1).cpu().tolist())
                labels.extend(lbl.tolist())
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")
        print(f"\n  [TEST] Acc: {acc:.4f} | Macro-F1: {macro_f1:.4f}")

    elif args.mode == "single":
        if not args.image_path or not os.path.exists(args.image_path):
            print("  Error: provide a valid --image_path")
            return
        raw = Image.open(args.image_path).convert("RGB")
        tfm = build_eval_transform(INPUT_DIM)
        out = tfm(image=np.array(raw), bboxes=[], bbox_labels=[])
        tensor = out["image"].unsqueeze(0).to(dev)

        with torch.no_grad():
            logits = net(tensor)
            probs = torch.softmax(logits, dim=1)
            idx = probs.argmax(1).item()
            conf = probs[0, idx].item()

        print(f"\n  [INFERENCE] Predicted: {breeds.get(idx, 'Unknown')} (id={idx})  Confidence: {conf:.2%}")



# Localization inference


def _draw_bbox(ax, cx, cy, w, h, img_h, img_w, color, tag="", lw=2):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    rect = mpatches.FancyBboxPatch((x1, y1), w * img_w, h * img_h,
                                    boxstyle="square,pad=0", linewidth=lw,
                                    edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    if tag:
        ax.text(x1 + 3, y1 + 12, tag, color="white", fontsize=6,
                bbox=dict(facecolor=color, alpha=0.7, pad=1, linewidth=0))


def infer_loc(args):
    dev = torch.device(args.device)
    dl = _build_val_loader(args.data_root, args.batch_size, args.num_workers, args.seed)

    net = VGG11Localizer(freeze_backbone=False).to(dev)
    _load_ckpt(net, LOC_CKPT, dev)
    net.eval()

    imgs_all, preds_all, gt_all, labels_all = [], [], [], []
    with torch.no_grad():
        for imgs, lbl, boxes, _ in dl:
            pred = torch.clamp(net(imgs.to(dev)), 0.0, 1.0)
            imgs_all.append(imgs.cpu())
            preds_all.append(pred.cpu())
            gt_all.append(boxes.cpu())
            labels_all.append(lbl.cpu())

    imgs_all = torch.cat(imgs_all)
    preds_all = torch.cat(preds_all)
    gt_all = torch.cat(gt_all)
    labels_all = torch.cat(labels_all)

    n = min(args.n, len(imgs_all))
    chosen = random.sample(range(len(imgs_all)), n)

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).ravel()
    total_iou = 0.0

    for pi, ci in enumerate(chosen):
        ax = axes[pi]
        img = _denormalize(imgs_all[ci])
        h, w = img.shape[:2]
        pr = preds_all[ci].numpy()
        gt = gt_all[ci].numpy()
        iou = _single_iou(pr, gt)
        total_iou += iou

        ax.imshow(img)
        _draw_bbox(ax, pr[0], pr[1], pr[2], pr[3], h, w, "red", f"pred IoU={iou:.2f}")
        ax.set_title(f"cls={labels_all[ci].item()}  IoU={iou:.2f}", fontsize=8)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.legend(
        handles=[mpatches.Patch(color="red", label="Predicted bbox"),
                 mpatches.Patch(color="lime", label="GT bbox")],
        loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.0),
    )
    plt.suptitle(f"Localization inference (n={n})", fontsize=11)
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    plt.savefig(args.save, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {args.save}")

    all_ious = [_single_iou(preds_all[i].numpy(), gt_all[i].numpy()) for i in range(len(preds_all))]
    print(f"  Mean IoU (sampled {n}): {total_iou / n:.4f}")
    print(f"  Mean IoU (full val):    {np.mean(all_ious):.4f}")



# Segmentation inference


def _binary_to_display(binary_pred):
    """Map binary (0/1) prediction to color indices for display."""
    out = np.ones_like(binary_pred)   # bg → index 1 (red)
    out[binary_pred == 1] = 0         # fg → index 0 (green)
    return out


def _seg_predict(model, imgs, dev, nc):
    with torch.no_grad():
        logits = model(imgs.to(dev))
        if nc == 1:
            return (logits.squeeze(1) > 0).long().cpu()
        return logits.argmax(1).cpu()


def infer_seg(args):
    dev = torch.device(args.device)
    nc = args.seg_classes

    net = VGG11UNet(num_classes=nc).to(dev)
    _load_ckpt(net, f"{SEG_CKPT}_{nc}.pth", dev)
    net.eval()

    if args.mode == "single":
        if not args.image_path or not os.path.exists(args.image_path):
            print("  Error: provide a valid --image_path")
            return
        pil = Image.open(args.image_path).convert("RGB")
        tfm = build_eval_transform(INPUT_DIM)
        res = tfm(image=np.array(pil),
                  mask=np.zeros((pil.height, pil.width), dtype=np.uint8),
                  bboxes=[], bbox_labels=[])
        img_t = res["image"]
        img_np = _denormalize(img_t)

        with torch.no_grad():
            logits = net(img_t.unsqueeze(0).to(dev))
            if nc == 1:
                pr = (logits.squeeze(1) > 0).long().squeeze(0).cpu().numpy()
            else:
                pr = logits.argmax(1).squeeze(0).cpu().numpy()

        pr_vis = pr if nc == 3 else _binary_to_display(pr)

        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(_blend_mask(img_np, pr_vis)); axes[1].set_title("Prediction"); axes[1].axis("off")

        handles = [mpatches.Patch(color=SEG_COLORS[i] / 255, label=l)
                   for i, l in enumerate(["Foreground", "Background", "Boundary"][:nc if nc == 3 else 2])]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        plt.savefig(args.save, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {args.save}")

    else:  # val_grid
        dl = _build_val_loader(args.data_root, args.batch_size, args.num_workers, args.seed)

        imgs_all, gt_all, pred_all = [], [], []
        with torch.no_grad():
            for imgs, _, _, masks in dl:
                imgs_all.append(imgs)
                gt_all.append(masks)
                pred_all.append(_seg_predict(net, imgs, dev, nc))

        imgs_all = torch.cat(imgs_all)
        gt_all = torch.cat(gt_all)
        pred_all = torch.cat(pred_all)

        n = min(args.rows * args.cols, len(imgs_all))
        chosen = random.sample(range(len(imgs_all)), n)

        fig, axes = plt.subplots(args.rows, args.cols * 3, figsize=(args.cols * 9, args.rows * 3.2))
        if args.rows == 1:
            axes = axes[np.newaxis, :]
        axes = np.array(axes).reshape(args.rows, args.cols * 3)

        for ri in range(args.rows):
            for ci in range(args.cols):
                fi = ri * args.cols + ci
                ax_o = axes[ri, ci * 3]
                ax_g = axes[ri, ci * 3 + 1]
                ax_p = axes[ri, ci * 3 + 2]

                if fi >= len(chosen):
                    ax_o.axis("off"); ax_g.axis("off"); ax_p.axis("off")
                    continue

                idx = chosen[fi]
                img = _denormalize(imgs_all[idx])
                gt_np = gt_all[idx].numpy()
                pr_np = pred_all[idx].numpy()
                pr_vis = pr_np if nc == 3 else _binary_to_display(pr_np)

                ax_o.imshow(img); ax_o.axis("off")
                ax_g.imshow(_blend_mask(img, gt_np)); ax_g.axis("off")
                ax_p.imshow(_blend_mask(img, pr_vis)); ax_p.axis("off")

                if ri == 0:
                    ax_o.set_title("Original", fontsize=9, fontweight="bold")
                    ax_g.set_title("GT Overlay", fontsize=9, fontweight="bold")
                    ax_p.set_title("Pred Overlay", fontsize=9, fontweight="bold")

        handles = [mpatches.Patch(color=SEG_COLORS[i] / 255, label=l)
                   for i, l in enumerate(["Foreground", "Background", "Boundary"][:nc if nc == 3 else 2])]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9)
        plt.suptitle("Segmentation | Original — GT — Predicted", fontsize=11, y=1.01)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        plt.savefig(args.save, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {args.save}")



# CLI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for all tasks")
    sub = parser.add_subparsers(dest="task")

    # Classification
    cp = sub.add_parser("clf", help="Classification inference")
    cp.add_argument("--mode", choices=["test", "single"], default="test")
    cp.add_argument("--image_path", type=str, default=None)
    cp.add_argument("--data_root", type=str, default="./data/oxford-iiit-pet/")
    cp.add_argument("--batch_size", type=int, default=32)
    cp.add_argument("--num_workers", type=int, default=4)
    cp.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Localization
    lp = sub.add_parser("loc", help="Localization inference")
    lp.add_argument("--data_root", type=str, default="./data/oxford-iiit-pet/")
    lp.add_argument("--n", type=int, default=16)
    lp.add_argument("--batch_size", type=int, default=32)
    lp.add_argument("--num_workers", type=int, default=4)
    lp.add_argument("--seed", type=int, default=42)
    lp.add_argument("--save", type=str, default="inference/bbox_results.png")
    lp.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Segmentation
    sp = sub.add_parser("seg", help="Segmentation inference")
    sp.add_argument("--mode", choices=["val_grid", "single"], default="val_grid")
    sp.add_argument("--image_path", type=str, default=None)
    sp.add_argument("--data_root", type=str, default="./data/oxford-iiit-pet/")
    sp.add_argument("--rows", type=int, default=4)
    sp.add_argument("--cols", type=int, default=2)
    sp.add_argument("--batch_size", type=int, default=16)
    sp.add_argument("--num_workers", type=int, default=4)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--seg_classes", type=int, default=3, choices=[1, 3])
    sp.add_argument("--save", type=str, default="inference/seg_results.png")
    sp.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if args.task == "clf":
        infer_clf(args)
    elif args.task == "loc":
        infer_loc(args)
    elif args.task == "seg":
        infer_seg(args)
    else:
        parser.print_help()