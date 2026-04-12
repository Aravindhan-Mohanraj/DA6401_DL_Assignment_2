"""Training entrypoint for DA6401 Assignment-2.

Usage:
    python train.py --task clf --use_wandb -b 64 --clf_lr 5e-4 --clf_epochs 70
    python train.py --task loc --use_wandb -b 32 --loc_lr 1e-3 --loc_epochs 50
    python train.py --task seg --use_wandb --seg_classes 3 --seg_lr 1e-3 --seg_epochs 30
    python train.py --task all --use_wandb
"""

import argparse
import os
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

import wandb

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))


def safe_torch_load(path, map_location=None):
    """Load checkpoint with weights_only=True when supported (PyTorch >= 1.13)."""
    _ver = tuple(int(x) for x in torch.__version__.split(".")[:2] if x.isdigit())
    if _ver >= (1, 13):
        return torch.load(path, map_location=map_location, weights_only=True)
    return torch.load(path, map_location=map_location)

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss
from data.pets_dataset import (
    OxfordIIITPetDataset,
    build_train_transform,
    build_eval_transform,
    stratified_train_val_split,
    INPUT_DIM,
)

N_BREEDS = 37
os.makedirs("checkpoints", exist_ok=True)

CLF_CKPT = os.path.join("checkpoints", "classifier.pth")
LOC_CKPT = os.path.join("checkpoints", "localizer.pth")
SEG_CKPT = os.path.join("checkpoints", "unet")
WANDB_ENTITY = "da25s006-indian-institute-of-technology-madras"



# Utilities


def fix_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(s)


def pick_device(name):
    return torch.device(name if torch.cuda.is_available() else "cpu")


def store_ckpt(fpath, net, ep, metric_val):
    torch.save({"state_dict": net.state_dict(), "epoch": ep, "best_metric": metric_val}, fpath)
    print(f"  [ckpt] saved {fpath}  epoch={ep}  metric={metric_val:.4f}")


def transfer_encoder_weights(target_model, clf_ckpt_path, enc_name="encoder"):
    """Load encoder weights from a trained classifier checkpoint."""
    if not os.path.exists(clf_ckpt_path):
        print(f"  [transfer] {clf_ckpt_path} not found — random init")
        return
    raw = safe_torch_load(clf_ckpt_path, map_location="cpu")
    sd = raw.get("state_dict", raw)
    # Extract only encoder conv block weights (prefixed with "backbone.block")
    prefix = "backbone."
    filtered = {}
    for k, v in sd.items():
        if k.startswith(prefix + "block"):
            filtered[k[len(prefix):]] = v
    enc_module = getattr(target_model, enc_name)
    m, u = enc_module.load_state_dict(filtered, strict=False)
    print(f"  [transfer] loaded {len(filtered)} tensors | missing={len(m)} unexpected={len(u)}")


def apply_kaiming_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, 0.01)
        nn.init.constant_(module.bias, 0)


def log_wandb(d, enabled):
    if enabled:
        wandb.log(d)



# Mixup


def apply_mixup(images, targets, alpha=0.4):
    if alpha <= 0:
        return images, targets, targets, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[perm]
    return mixed, targets, targets[perm], lam


def compute_mixup_loss(loss_fn, preds, ya, yb, lam):
    return lam * loss_fn(preds, ya) + (1.0 - lam) * loss_fn(preds, yb)



# Data loading


def _gather_aug_records(base_records, data_root):
    """Expand original records to include their augmented copies from images_aug/."""
    aug_dir = data_root / "images_aug"
    out = []
    for img_id, cid, sp, br in base_records:
        if (aug_dir / f"{img_id}.jpg").exists():
            out.append((img_id, cid, sp, br))
        for suffix in range(1, 5):
            aug_id = f"{img_id}_aug{suffix}"
            if (aug_dir / f"{aug_id}.jpg").exists():
                out.append((aug_id, cid, sp, br))
    return out


def create_dataloaders(args, with_aug=True):
    root = pathlib.Path(args.data_root)
    ann = root / "annotations" / "trainval.txt"

    trn_recs, val_recs = stratified_train_val_split(str(ann), val_ratio=0.1, rng_seed=args.seed)

    if with_aug:
        trn_recs_expanded = _gather_aug_records(trn_recs, root)
        trn_img_dir = root / "images_aug"
        trn_mask_dir = root / "annotations" / "trimaps_aug"
    else:
        trn_recs_expanded = trn_recs
        trn_img_dir = root / "images"
        trn_mask_dir = root / "annotations" / "trimaps"

    ds_train = OxfordIIITPetDataset(
        str(root), entries=trn_recs_expanded,
        img_dir=trn_img_dir, mask_dir=trn_mask_dir,
        transform=build_train_transform(INPUT_DIM),
    )
    ds_val = OxfordIIITPetDataset(
        str(root), entries=val_recs,
        img_dir=root / "images", mask_dir=root / "annotations" / "trimaps",
        transform=build_eval_transform(INPUT_DIM),
    )
    ds_test = OxfordIIITPetDataset(str(root), split_name="test", transform=build_eval_transform(INPUT_DIM))

    loader_kw = dict(num_workers=args.num_workers, pin_memory=(args.device.type == "cuda"))
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kw)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, drop_last=False, **loader_kw)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, drop_last=False, **loader_kw)

    print(f"  Train: {len(ds_train)} | Val: {len(ds_val)} | Test: {len(ds_test)}")
    print(f"  Train IDs: {len(trn_recs)} originals + aug copies")
    print(f"  Val IDs: {len(val_recs)} originals only")
    return dl_train, dl_val, dl_test



# Metrics


def compute_clf_metrics(pred_list, true_list):
    p, t = np.array(pred_list), np.array(true_list)
    return {
        "accuracy": accuracy_score(t, p),
        "f1_macro": f1_score(t, p, average="macro", zero_division=0),
        "f1_micro": f1_score(t, p, average="micro", zero_division=0),
        "f1_weighted": f1_score(t, p, average="weighted", zero_division=0),
        "prec_macro": precision_score(t, p, average="macro", zero_division=0),
        "prec_micro": precision_score(t, p, average="micro", zero_division=0),
        "rec_macro": recall_score(t, p, average="macro", zero_division=0),
        "rec_micro": recall_score(t, p, average="micro", zero_division=0),
    }


def compute_iou_batch(pred_box, gt_box, eps=1e-6):
    """Mean IoU for a batch of normalized (cx,cy,w,h) boxes."""
    p_l = pred_box[:, 0] - pred_box[:, 2] / 2
    p_t = pred_box[:, 1] - pred_box[:, 3] / 2
    p_r = pred_box[:, 0] + pred_box[:, 2] / 2
    p_b = pred_box[:, 1] + pred_box[:, 3] / 2
    g_l = gt_box[:, 0] - gt_box[:, 2] / 2
    g_t = gt_box[:, 1] - gt_box[:, 3] / 2
    g_r = gt_box[:, 0] + gt_box[:, 2] / 2
    g_b = gt_box[:, 1] + gt_box[:, 3] / 2
    w_inter = torch.clamp(torch.min(p_r, g_r) - torch.max(p_l, g_l), 0)
    h_inter = torch.clamp(torch.min(p_b, g_b) - torch.max(p_t, g_t), 0)
    area_inter = w_inter * h_inter
    area_union = (p_r - p_l) * (p_b - p_t) + (g_r - g_l) * (g_b - g_t) - area_inter + eps
    return (area_inter / area_union).mean().item()


def soft_dice_loss(logits, targets, n_classes, eps=1e-6):
    """Soft multi-class Dice loss."""
    if n_classes == 1:
        prob = torch.sigmoid(logits).squeeze(1)
        inter = (prob * targets).sum((1, 2))
        denom = prob.sum((1, 2)) + targets.sum((1, 2))
    else:
        prob = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, n_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (prob * one_hot).sum(dims)
        denom = prob.sum(dims) + one_hot.sum(dims)
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


def compute_seg_metrics(pred_mask, gt_mask, n_cls, eps=1e-6):
    pf = pred_mask.cpu().numpy().ravel()
    gf = gt_mask.cpu().numpy().ravel()
    px_acc = (pred_mask == gt_mask).float().mean().item()

    dice_per_cls = []
    for c in range(n_cls):
        pc = (pred_mask == c).float()
        gc = (gt_mask == c).float()
        d = (2 * (pc * gc).sum() + eps) / (pc.sum() + gc.sum() + eps)
        dice_per_cls.append(d.item())
    avg_dice = float(np.mean(dice_per_cls))
    lbls = list(range(n_cls))

    if n_cls == 2:
        return {
            "px_accuracy": px_acc, "mean_dice": avg_dice,
            "dice_bg": dice_per_cls[0], "dice_fg": dice_per_cls[1], "dice_boundary": 0.0,
            "f1_macro": f1_score(gf, pf, average="macro", labels=lbls, zero_division=0),
            "f1_micro": f1_score(gf, pf, average="micro", labels=lbls, zero_division=0),
            "f1_weighted": f1_score(gf, pf, average="weighted", labels=lbls, zero_division=0),
            "prec_macro": precision_score(gf, pf, average="macro", labels=lbls, zero_division=0),
            "rec_macro": recall_score(gf, pf, average="macro", labels=lbls, zero_division=0),
        }

    return {
        "px_accuracy": px_acc, "mean_dice": avg_dice,
        "dice_fg": dice_per_cls[0],
        "dice_bg": dice_per_cls[1] if n_cls > 1 else 0.0,
        "dice_boundary": dice_per_cls[2] if n_cls > 2 else 0.0,
        "f1_macro": f1_score(gf, pf, average="macro", labels=lbls, zero_division=0),
        "f1_micro": f1_score(gf, pf, average="micro", labels=lbls, zero_division=0),
        "f1_weighted": f1_score(gf, pf, average="weighted", labels=lbls, zero_division=0),
        "prec_macro": precision_score(gf, pf, average="macro", labels=lbls, zero_division=0),
        "rec_macro": recall_score(gf, pf, average="macro", labels=lbls, zero_division=0),
    }



# Task 1: Classification


def run_classification(args):
    print(f"\n{'=' * 60}\nTASK 1: Classification\n{'=' * 60}")
    dev = args.device

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name="classifier_v3", entity=WANDB_ENTITY, config=vars(args), reinit=True)

    dl_trn, dl_val, dl_tst = create_dataloaders(args, with_aug=not getattr(args, "no_aug", False))

    net = VGG11Classifier(num_classes=N_BREEDS, dropout_p=args.dropout_p).to(dev)
    net.apply(apply_kaiming_init)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optim = torch.optim.AdamW(net.parameters(), lr=args.clf_lr, weight_decay=5e-5)

    warmup = LinearLR(optim, start_factor=0.1, total_iters=5)
    cosine = CosineAnnealingLR(optim, T_max=max(1, args.clf_epochs - 5))
    sched = SequentialLR(optim, schedulers=[warmup, cosine], milestones=[5])

    top_f1, top_ep = 0.0, 0
    wait = 0
    max_wait = getattr(args, "clf_patience", 25)
    amp_scaler = torch.cuda.amp.GradScaler()

    for ep in range(1, args.clf_epochs + 1):
        net.train()
        running_loss = 0.0
        ep_preds, ep_labels = [], []

        for batch_img, batch_lbl, _, _ in dl_trn:
            if np.random.rand() < 0.5:
                batch_img, ta, tb, lam = apply_mixup(batch_img, batch_lbl, alpha=0.1)
            else:
                ta = tb = batch_lbl
                lam = 1.0

            batch_img, ta, tb = batch_img.to(dev), ta.to(dev), tb.to(dev)
            optim.zero_grad()

            with torch.cuda.amp.autocast():
                out = net(batch_img)
                loss = compute_mixup_loss(loss_fn, out, ta, tb, lam)

            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            amp_scaler.step(optim)
            amp_scaler.update()

            running_loss += loss.item()
            ep_preds.extend(out.argmax(1).cpu().tolist())
            ep_labels.extend(ta.cpu().tolist())

        running_loss /= len(dl_trn)
        trn_m = compute_clf_metrics(ep_preds, ep_labels)
        sched.step()

        # validation
        net.eval()
        val_loss = 0.0
        vp, vl = [], []
        with torch.no_grad():
            for vi, vla, _, _ in dl_val:
                vi, vla = vi.to(dev), vla.to(dev)
                vo = net(vi)
                val_loss += loss_fn(vo, vla).item()
                vp.extend(vo.argmax(1).cpu().tolist())
                vl.extend(vla.cpu().tolist())
        val_loss /= len(dl_val)
        val_m = compute_clf_metrics(vp, vl)

        print(f"  Epoch {ep:3d}/{args.clf_epochs} | "
              f"train loss={running_loss:.4f} acc={trn_m['accuracy']:.4f} f1={trn_m['f1_macro']:.4f} | "
              f"val loss={val_loss:.4f} acc={val_m['accuracy']:.4f} f1={val_m['f1_macro']:.4f}")

        log_wandb({
            "epoch": ep, "best_epoch": ep if val_m["f1_macro"] > top_f1 else top_ep,
            "clf/lr": sched.get_last_lr()[0],
            "clf/train/loss": running_loss, "clf/train/accuracy": trn_m["accuracy"],
            "clf/train/f1_macro": trn_m["f1_macro"], "clf/train/f1_micro": trn_m["f1_micro"],
            "clf/train/f1_weighted": trn_m["f1_weighted"],
            "clf/train/prec_macro": trn_m["prec_macro"], "clf/train/prec_micro": trn_m["prec_micro"],
            "clf/train/rec_macro": trn_m["rec_macro"], "clf/train/rec_micro": trn_m["rec_micro"],
            "clf/val/loss": val_loss, "clf/val/accuracy": val_m["accuracy"],
            "clf/val/f1_macro": val_m["f1_macro"], "clf/val/f1_micro": val_m["f1_micro"],
            "clf/val/f1_weighted": val_m["f1_weighted"],
            "clf/val/prec_macro": val_m["prec_macro"], "clf/val/prec_micro": val_m["prec_micro"],
            "clf/val/rec_macro": val_m["rec_macro"], "clf/val/rec_micro": val_m["rec_micro"],
        }, args.use_wandb)

        if val_m["f1_macro"] > top_f1:
            wait = 0
            top_ep = ep
            top_f1 = val_m["f1_macro"]
            store_ckpt(CLF_CKPT, net, ep, top_f1)
        else:
            wait += 1
            if wait >= max_wait:
                print(f"  Early stopping at epoch {ep} with best f1_macro={top_f1:.4f}")
                break

    # test with best checkpoint
    best_sd = safe_torch_load(CLF_CKPT, map_location=dev)
    net.load_state_dict(best_sd["state_dict"])
    net.eval()
    te_loss = 0.0
    tp, tl = [], []
    with torch.no_grad():
        for ti, tla, _, _ in dl_tst:
            ti, tla = ti.to(dev), tla.to(dev)
            to = net(ti)
            te_loss += loss_fn(to, tla).item()
            tp.extend(to.argmax(1).cpu().tolist())
            tl.extend(tla.cpu().tolist())
    te_loss /= len(dl_tst)
    te_m = compute_clf_metrics(tp, tl)
    print(f"  [TEST] loss={te_loss:.4f} acc={te_m['accuracy']:.4f} f1_macro={te_m['f1_macro']:.4f}")

    log_wandb({
        "clf/test/loss": te_loss, "clf/test/accuracy": te_m["accuracy"],
        "clf/test/f1_macro": te_m["f1_macro"], "clf/test/f1_micro": te_m["f1_micro"],
        "clf/test/f1_weighted": te_m["f1_weighted"],
        "clf/test/prec_macro": te_m["prec_macro"], "clf/test/prec_micro": te_m["prec_micro"],
        "clf/test/rec_macro": te_m["rec_macro"], "clf/test/rec_micro": te_m["rec_micro"],
    }, args.use_wandb)

    if args.use_wandb:
        wandb.finish()
    return top_f1



# Task 2: Localization


def run_localization(args):
    print(f"\n{'=' * 60}\nTASK 2: Localization\n{'=' * 60}")
    dev = args.device

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name="task2-localizer", entity=WANDB_ENTITY, config=vars(args), reinit=True)

    dl_trn, dl_val, _ = create_dataloaders(args, with_aug=False)

    net = VGG11Localizer(dropout_p=args.dropout_p, freeze_backbone=False).to(dev)
    transfer_encoder_weights(net, CLF_CKPT, enc_name="encoder")

    smooth_l1 = nn.SmoothL1Loss()
    iou_crit = IoULoss(reduction="mean")

    use_amp = (dev.type == "cuda")
    amp_scaler = torch.cuda.amp.GradScaler() if use_amp else None

    s1_end = getattr(args, "loc_stage1", 10)
    s2_end = s1_end + getattr(args, "loc_stage2", 5)

    def _set_grad(module, on):
        for p in module.parameters():
            p.requires_grad = on

    def _build_optim_sched(params, lr, total_ep):
        o = torch.optim.AdamW(list(params), lr=lr, weight_decay=1e-4)
        s = CosineAnnealingLR(o, T_max=max(1, total_ep), eta_min=1e-7)
        return o, s

    # Freeze early blocks for staged fine-tuning
    _set_grad(net.encoder.block1, False)
    _set_grad(net.encoder.block2, False)
    _set_grad(net.encoder.block3, False)

    optim, sched = _build_optim_sched(filter(lambda p: p.requires_grad, net.parameters()), args.loc_lr, s1_end)

    print(f"  Stage 1 (1-{s1_end}): block4+block5+head only")
    print(f"  Stage 2 ({s1_end + 1}-{s2_end}): +block3")
    print(f"  Stage 3 ({s2_end + 1}-{args.loc_epochs}): full fine-tune")

    best_miou, best_ep, patience_cnt = 0.0, 0, 0
    max_patience = getattr(args, "loc_patience", 15)

    for ep in range(1, args.loc_epochs + 1):
        if ep == s1_end + 1:
            _set_grad(net.encoder.block3, True)
            optim, sched = _build_optim_sched(filter(lambda p: p.requires_grad, net.parameters()), args.loc_lr * 0.3, s2_end - s1_end)
            print(f"  [Epoch {ep}] Stage 2: block3 unfrozen")

        elif ep == s2_end + 1:
            _set_grad(net.encoder.block1, True)
            _set_grad(net.encoder.block2, True)
            optim, sched = _build_optim_sched(filter(lambda p: p.requires_grad, net.parameters()), args.loc_lr * 0.1, args.loc_epochs - s2_end)
            print(f"  [Epoch {ep}] Stage 3: full model")

        # train
        net.train()
        sum_reg, sum_iou, sum_tot, sum_miou = 0.0, 0.0, 0.0, 0.0

        for b_img, _, b_box, _ in dl_trn:
            b_img, b_box = b_img.to(dev), b_box.to(dev)
            optim.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = net(b_img)
                    l_reg = smooth_l1(pred, b_box)
                    l_iou = iou_crit(pred, b_box)
                    l_total = l_reg + l_iou
                amp_scaler.scale(l_total).backward()
                amp_scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                amp_scaler.step(optim)
                amp_scaler.update()
            else:
                pred = net(b_img)
                l_reg = smooth_l1(pred, b_box)
                l_iou = iou_crit(pred, b_box)
                l_total = l_reg + l_iou
                l_total.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optim.step()

            sum_reg += l_reg.item()
            sum_iou += l_iou.item()
            sum_tot += l_total.item()
            sum_miou += compute_iou_batch(pred.detach(), b_box)

        nb = len(dl_trn)
        sum_reg /= nb; sum_iou /= nb; sum_tot /= nb; sum_miou /= nb
        sched.step()

        # val
        net.eval()
        vr, vi, vt, vm = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for b_img, _, b_box, _ in dl_val:
                b_img, b_box = b_img.to(dev), b_box.to(dev)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        vp = net(b_img)
                        lr_ = smooth_l1(vp, b_box)
                        li_ = iou_crit(vp, b_box)
                        lt_ = lr_ + li_
                else:
                    vp = net(b_img)
                    lr_ = smooth_l1(vp, b_box)
                    li_ = iou_crit(vp, b_box)
                    lt_ = lr_ + li_
                vr += lr_.item(); vi += li_.item(); vt += lt_.item()
                vm += compute_iou_batch(vp, b_box)

        nv = len(dl_val)
        vr /= nv; vi /= nv; vt /= nv; vm /= nv

        print(f"  Epoch {ep:3d}/{args.loc_epochs} | "
              f"train reg={sum_reg:.4f} iou_l={sum_iou:.4f} miou={sum_miou:.4f} | "
              f"val reg={vr:.4f} iou_l={vi:.4f} miou={vm:.4f}")

        log_wandb({
            "epoch": ep, "loc/best_epoch": ep if vm > best_miou else best_ep,
            "loc/lr": sched.get_last_lr()[0],
            "loc/train/reg_loss": sum_reg, "loc/train/iou_loss": sum_iou,
            "loc/train/total_loss": sum_tot, "loc/train/mean_iou": sum_miou,
            "loc/val/reg_loss": vr, "loc/val/iou_loss": vi,
            "loc/val/total_loss": vt, "loc/val/mean_iou": vm,
        }, args.use_wandb)

        if vm > best_miou:
            best_miou, best_ep, patience_cnt = vm, ep, 0
            store_ckpt(LOC_CKPT, net, ep, best_miou)
        else:
            patience_cnt += 1
            if patience_cnt >= max_patience:
                print(f"  Early stopping at epoch {ep} (best val mIoU={best_miou:.4f})")
                break

    print(f"\n  [TEST] No bbox annotations in test split.")
    print(f"  Best val mIoU = {best_miou:.4f} (epoch {best_ep})")
    log_wandb({"loc/val/best_miou": best_miou}, args.use_wandb)

    if args.use_wandb:
        wandb.finish()
    return best_miou



# Task 3: Segmentation


def _seg_forward_loss(model, images, masks, ce_fn, nc, use_amp, device):
    """Shared forward + loss computation for segmentation (avoids code duplication)."""
    images, masks = images.to(device), masks.to(device)

    ctx = torch.cuda.amp.autocast() if use_amp else torch.no_grad.__class__()
    if not use_amp:
        # No-op context for non-amp
        logits = model(images)
    else:
        with torch.cuda.amp.autocast():
            logits = model(images)

    if nc == 1:
        tgt = (masks != 1).float()
        sq = logits.squeeze(1)
        ce = ce_fn(sq, tgt)
        dl = soft_dice_loss(torch.sigmoid(sq), tgt, n_classes=1)
        loss = ce + dl
        preds = (sq > 0).long()
        tgt_out = tgt.long().cpu()
    else:
        ce = ce_fn(logits, masks)
        dl = soft_dice_loss(logits, masks, nc)
        loss = ce + dl
        preds = logits.argmax(1)
        tgt_out = masks.cpu()

    return loss, preds, tgt_out, logits


def run_segmentation(args):
    print(f"\n{'=' * 60}\nTASK 3: Segmentation  (seg_classes={args.seg_classes})\n{'=' * 60}")
    dev = args.device
    nc = args.seg_classes

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=f"task3-unet-nc{nc}", entity=WANDB_ENTITY, config=vars(args), reinit=True)

    dl_trn, dl_val, dl_tst = create_dataloaders(args, with_aug=False)

    net = VGG11UNet(num_classes=nc, in_channels=3, dropout_p=args.dropout_p).to(dev)
    transfer_encoder_weights(net, CLF_CKPT, enc_name="encoder")

    use_amp = (dev.type == "cuda")
    amp_scaler = torch.cuda.amp.GradScaler() if use_amp else None

    if nc == 1:
        ce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(dev))
    else:
        wts = torch.tensor([1.0, 0.7, 4.5], device=dev)
        ce_fn = nn.CrossEntropyLoss(weight=wts)

    # Freeze encoder initially
    for p in net.encoder.parameters():
        p.requires_grad = False
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args.seg_lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(optim, T_max=args.seg_epochs, eta_min=1e-6)

    thaw_ep = max(1, args.seg_epochs // 3)
    best_dice, wait = 0.0, 0
    max_wait = getattr(args, "seg_patience", 20)
    metrics_nc = 2 if nc == 1 else nc

    for ep in range(1, args.seg_epochs + 1):
        if ep == thaw_ep:
            for p in net.encoder.parameters():
                p.requires_grad = True
            optim = torch.optim.AdamW(net.parameters(), lr=args.seg_lr * 0.1, weight_decay=1e-4)
            sched = CosineAnnealingLR(optim, T_max=args.seg_epochs - ep + 1, eta_min=1e-6)
            print(f"  [Epoch {ep}] encoder unfrozen")

        # train
        net.train()
        ep_loss = 0.0
        all_p, all_t = [], []

        for b_img, _, _, b_msk in dl_trn:
            optim.zero_grad(set_to_none=True)

            b_img, b_msk = b_img.to(dev), b_msk.to(dev)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = net(b_img)
                    if nc == 1:
                        tgt = (b_msk != 1).float()
                        sq = logits.squeeze(1)
                        loss = ce_fn(sq, tgt) + soft_dice_loss(torch.sigmoid(sq), tgt, 1)
                        preds = (sq > 0).long()
                        cur_tgt = tgt.long().cpu()
                    else:
                        loss = ce_fn(logits, b_msk) + soft_dice_loss(logits, b_msk, nc)
                        preds = logits.argmax(1)
                        cur_tgt = b_msk.cpu()
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                amp_scaler.step(optim)
                amp_scaler.update()
            else:
                logits = net(b_img)
                if nc == 1:
                    tgt = (b_msk != 1).float()
                    sq = logits.squeeze(1)
                    loss = ce_fn(sq, tgt) + soft_dice_loss(torch.sigmoid(sq), tgt, 1)
                    preds = (sq > 0).long()
                    cur_tgt = tgt.long().cpu()
                else:
                    loss = ce_fn(logits, b_msk) + soft_dice_loss(logits, b_msk, nc)
                    preds = logits.argmax(1)
                    cur_tgt = b_msk.cpu()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optim.step()

            ep_loss += loss.item()
            all_p.append(preds.detach().cpu())
            all_t.append(cur_tgt)

        ep_loss /= len(dl_trn)
        trn_m = compute_seg_metrics(torch.cat(all_p), torch.cat(all_t), metrics_nc)
        sched.step()

        # val
        net.eval()
        v_loss = 0.0
        all_p, all_t = [], []
        with torch.no_grad():
            for b_img, _, _, b_msk in dl_val:
                b_img, b_msk = b_img.to(dev), b_msk.to(dev)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = net(b_img)
                        if nc == 1:
                            tgt = (b_msk != 1).float()
                            sq = logits.squeeze(1)
                            loss = ce_fn(sq, tgt) + soft_dice_loss(torch.sigmoid(sq), tgt, 1)
                            preds = (sq > 0).long()
                            cur_tgt = tgt.long().cpu()
                        else:
                            loss = ce_fn(logits, b_msk) + soft_dice_loss(logits, b_msk, nc)
                            preds = logits.argmax(1)
                            cur_tgt = b_msk.cpu()
                else:
                    logits = net(b_img)
                    if nc == 1:
                        tgt = (b_msk != 1).float()
                        sq = logits.squeeze(1)
                        loss = ce_fn(sq, tgt) + soft_dice_loss(torch.sigmoid(sq), tgt, 1)
                        preds = (sq > 0).long()
                        cur_tgt = tgt.long().cpu()
                    else:
                        loss = ce_fn(logits, b_msk) + soft_dice_loss(logits, b_msk, nc)
                        preds = logits.argmax(1)
                        cur_tgt = b_msk.cpu()
                v_loss += loss.item()
                all_p.append(preds.cpu())
                all_t.append(cur_tgt)

        v_loss /= len(dl_val)
        val_m = compute_seg_metrics(torch.cat(all_p), torch.cat(all_t), metrics_nc)

        print(f"  Epoch {ep:3d}/{args.seg_epochs} | "
              f"train loss={ep_loss:.4f} dice={trn_m['mean_dice']:.4f} px_acc={trn_m['px_accuracy']:.4f} | "
              f"val loss={v_loss:.4f} dice={val_m['mean_dice']:.4f} px_acc={val_m['px_accuracy']:.4f}")

        log_wandb({
            "epoch": ep, "seg/lr": sched.get_last_lr()[0],
            "seg/train/loss": ep_loss, "seg/train/mean_dice": trn_m["mean_dice"],
            "seg/train/dice_fg": trn_m["dice_fg"], "seg/train/dice_bg": trn_m["dice_bg"],
            "seg/train/dice_boundary": trn_m["dice_boundary"], "seg/train/px_accuracy": trn_m["px_accuracy"],
            "seg/train/f1_macro": trn_m["f1_macro"], "seg/train/f1_micro": trn_m["f1_micro"],
            "seg/train/f1_weighted": trn_m["f1_weighted"],
            "seg/train/prec_macro": trn_m["prec_macro"], "seg/train/rec_macro": trn_m["rec_macro"],
            "seg/val/loss": v_loss, "seg/val/mean_dice": val_m["mean_dice"],
            "seg/val/dice_fg": val_m["dice_fg"], "seg/val/dice_bg": val_m["dice_bg"],
            "seg/val/dice_boundary": val_m["dice_boundary"], "seg/val/px_accuracy": val_m["px_accuracy"],
            "seg/val/f1_macro": val_m["f1_macro"], "seg/val/f1_micro": val_m["f1_micro"],
            "seg/val/f1_weighted": val_m["f1_weighted"],
            "seg/val/prec_macro": val_m["prec_macro"], "seg/val/rec_macro": val_m["rec_macro"],
        }, args.use_wandb)

        if val_m["mean_dice"] > best_dice:
            best_dice = val_m["mean_dice"]
            wait = 0
            store_ckpt(SEG_CKPT + "_" + str(args.seg_classes) + ".pth", net, ep, best_dice)
        else:
            wait += 1
            if wait >= max_wait:
                print(f"  Early stopping at epoch {ep} (best dice={best_dice:.4f})")
                break

    # test
    ckpt = safe_torch_load(SEG_CKPT + "_" + str(args.seg_classes) + ".pth", map_location=dev)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    te_loss = 0.0
    all_p, all_t = [], []
    with torch.no_grad():
        for b_img, _, _, b_msk in dl_tst:
            b_img, b_msk = b_img.to(dev), b_msk.to(dev)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = net(b_img)
                    if nc == 1:
                        tgt = (b_msk != 1).float()
                        sq = logits.squeeze(1)
                        loss = ce_fn(sq, tgt) + soft_dice_loss(torch.sigmoid(sq), tgt, 1)
                        preds = (sq > 0).long()
                        cur_tgt = tgt.long().cpu()
                    else:
                        loss = ce_fn(logits, b_msk) + soft_dice_loss(logits, b_msk, nc)
                        preds = logits.argmax(1)
                        cur_tgt = b_msk.cpu()
            else:
                logits = net(b_img)
                if nc == 1:
                    tgt = (b_msk != 1).float()
                    sq = logits.squeeze(1)
                    loss = ce_fn(sq, tgt) + soft_dice_loss(torch.sigmoid(sq), tgt, 1)
                    preds = (sq > 0).long()
                    cur_tgt = tgt.long().cpu()
                else:
                    loss = ce_fn(logits, b_msk) + soft_dice_loss(logits, b_msk, nc)
                    preds = logits.argmax(1)
                    cur_tgt = b_msk.cpu()
            te_loss += loss.item()
            all_p.append(preds.cpu())
            all_t.append(cur_tgt)

    te_loss /= len(dl_tst)
    te_m = compute_seg_metrics(torch.cat(all_p), torch.cat(all_t), metrics_nc)

    print(f"  [TEST] loss={te_loss:.4f} dice={te_m['mean_dice']:.4f} "
          f"px_acc={te_m['px_accuracy']:.4f} f1_macro={te_m['f1_macro']:.4f}")

    log_wandb({
        "seg/test/loss": te_loss, "seg/test/mean_dice": te_m["mean_dice"],
        "seg/test/dice_fg": te_m["dice_fg"], "seg/test/dice_bg": te_m["dice_bg"],
        "seg/test/dice_boundary": te_m["dice_boundary"], "seg/test/px_accuracy": te_m["px_accuracy"],
        "seg/test/f1_macro": te_m["f1_macro"], "seg/test/f1_micro": te_m["f1_micro"],
        "seg/test/f1_weighted": te_m["f1_weighted"],
        "seg/test/prec_macro": te_m["prec_macro"], "seg/test/rec_macro": te_m["rec_macro"],
    }, args.use_wandb)

    if args.use_wandb:
        wandb.finish()
    return best_dice



# CLI


def build_args():
    p = argparse.ArgumentParser(description="DA6401 Assignment-2 Training")
    p.add_argument("--data_root", type=str, default="./data/oxford-iiit-pet/")
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--task", type=str, default="loc", choices=["all", "clf", "loc", "seg"])
    p.add_argument("-b", "--batch_size", type=int, default=64)
    p.add_argument("-dp", "--dropout_p", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--clf_lr", type=float, default=1e-4)
    p.add_argument("--clf_epochs", type=int, default=70)
    p.add_argument("--clf_patience", type=int, default=20)

    p.add_argument("--loc_lr", type=float, default=1e-3)
    p.add_argument("--loc_epochs", type=int, default=30)
    p.add_argument("--loc_patience", type=int, default=15)
    p.add_argument("--loc_stage1", type=int, default=10)
    p.add_argument("--loc_stage2", type=int, default=5)

    p.add_argument("--seg_lr", type=float, default=1e-3)
    p.add_argument("--seg_epochs", type=int, default=30)
    p.add_argument("--seg_classes", type=int, default=3, choices=[1, 3])
    p.add_argument("--seg_patience", type=int, default=20)

    p.add_argument("--no_aug", action="store_true", help="Skip augmented data, use originals only (faster for testing)")
    p.add_argument("--wandb_project", type=str, default="DA6402-Assignment-2_v1")
    p.add_argument("--use_wandb", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = build_args()
    fix_seed(args.seed)
    args.device = pick_device(args.device)

    print(f"Device : {args.device}")
    print(f"Task   : {args.task}")
    print(f"WandB  : {args.use_wandb}")
    print(f"Seed   : {args.seed}")
    print(f"seg_classes: {args.seg_classes}")

    if args.task in ("all", "clf"):
        run_classification(args)
    if args.task in ("all", "loc"):
        run_localization(args)
    if args.task in ("all", "seg"):
        run_segmentation(args)

    print("\nDone. Checkpoints saved:")
    for ck in [CLF_CKPT, LOC_CKPT, f"{SEG_CKPT}_{args.seg_classes}.pth"]:
        print(f"  {ck}  {'✓' if os.path.exists(ck) else '(not trained)'}")