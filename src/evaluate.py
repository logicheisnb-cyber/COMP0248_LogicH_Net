import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader, Subset

import matplotlib.pyplot as plt

from .model import create_model
from .dataloader import HandGestureDataset, collate_fn, GESTURE_NAMES
from .utils import (
    set_seed,
    ensure_dir,
    bbox_cxcywh_to_xyxy,
    bbox_iou_xyxy,
    iou_score,
    dice_score,
    update_confusion_matrix,
    macro_f1_from_confusion,
)


# =========================================================
# Nested test dataset loader
# Supports:
#   dataset/test/<gesture>/<clip>/rgb/<frame>.png
#   dataset/test/<gesture>/<clip>/annotation/<frame>.png
#   dataset/test/<gesture>/<clip>/depth/<frame>.png   (for baseline / LogicH)
# RGB preprocessing matches current training:
#   image -> float32 / 255.0
# =========================================================
def load_rgb_as_tensor(rgb_path: Path, image_size: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(rgb_path).convert("RGB")
    img = img.resize((image_size[1], image_size[0]), resample=Image.BILINEAR)
    x = np.asarray(img).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)


def load_mask_png(mask_path: Path, image_size: Tuple[int, int]) -> Tuple[torch.Tensor, np.ndarray]:
    m = Image.open(mask_path).convert("L")
    m = m.resize((image_size[1], image_size[0]), resample=Image.NEAREST)
    mask01 = (np.asarray(m) > 127).astype(np.uint8)
    mask_t = torch.from_numpy(mask01.astype(np.float32)).unsqueeze(0)
    return mask_t, mask01


def load_depth_as_tensor(depth_path: Path, image_size: Tuple[int, int]) -> torch.Tensor:
    d = Image.open(depth_path).convert("L")
    d = d.resize((image_size[1], image_size[0]), resample=Image.BILINEAR)
    x = np.asarray(d).astype(np.float32)

    if x.max() > 0:
        if x.max() > 255:
            x = x / x.max()
        else:
            x = x / 255.0

    return torch.from_numpy(x).unsqueeze(0)


def bbox_from_binary_mask(mask01: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask01 > 0)
    h, w = mask01.shape

    if len(xs) == 0 or len(ys) == 0:
        return np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float32)

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    cx = (x1 + x2) / 2.0 / max(1, (w - 1))
    cy = (y1 + y2) / 2.0 / max(1, (h - 1))
    bw = (x2 - x1 + 1) / max(1, w)
    bh = (y2 - y1 + 1) / max(1, h)

    cxcywh = np.array([cx, cy, bw, bh], dtype=np.float32)
    return np.clip(cxcywh, 0.0, 1.0)


def label_from_gesture_dir(gesture_dir_name: str) -> int:
    """
    Supports:
      G01_call
      call
      G01
    """
    if "_" in gesture_dir_name:
        name = gesture_dir_name.split("_", 1)[1]
        if name in GESTURE_NAMES:
            return int(GESTURE_NAMES.index(name))

    if gesture_dir_name in GESTURE_NAMES:
        return int(GESTURE_NAMES.index(gesture_dir_name))

    m = re.match(r"G(\d+)", gesture_dir_name)
    if m:
        return int(m.group(1)) - 1

    raise ValueError(
        f"Cannot map gesture folder '{gesture_dir_name}' to a class id. "
        f"Check GESTURE_NAMES or edit label_from_gesture_dir()."
    )


class TestFolderDataset(Dataset):
    def __init__(self, test_root: str, image_size: Tuple[int, int], model_name: str = "baseline"):
        self.test_root = Path(test_root)
        self.image_size = image_size
        self.model_name = model_name.lower()

        exts = ("png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp")
        rgb_files: List[Path] = []
        for ext in exts:
            rgb_files += list(self.test_root.rglob(f"rgb/*.{ext}"))

        if not rgb_files:
            rgb_files = [p for p in self.test_root.rglob("rgb/*") if p.is_file()]

        rgb_files.sort()
        if not rgb_files:
            raise FileNotFoundError(f"No RGB frames found under {self.test_root}")

        self.rgb_files = rgb_files

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_path = self.rgb_files[idx]

        mask_path = Path(str(rgb_path).replace(os.sep + "rgb" + os.sep, os.sep + "annotation" + os.sep))
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing annotation mask: {mask_path}")

        depth_path = Path(str(rgb_path).replace(os.sep + "rgb" + os.sep, os.sep + "depth" + os.sep))
        rel_parts = rgb_path.relative_to(self.test_root).parts
        gesture_dir = rel_parts[0]
        clip_name = rel_parts[1] if len(rel_parts) > 1 else "unknown_clip"
        frame_name = rgb_path.stem

        label = label_from_gesture_dir(gesture_dir)

        rgb_t = load_rgb_as_tensor(rgb_path, self.image_size)
        mask_t, mask01 = load_mask_png(mask_path, self.image_size)
        bbox_t = torch.tensor(bbox_from_binary_mask(mask01), dtype=torch.float32)
        label_t = torch.tensor(label, dtype=torch.long)

        sample = {
            "rgb": rgb_t,
            "mask": mask_t,
            "bbox": bbox_t,
            "label": label_t,
            "meta": {
                "rgb_path": str(rgb_path),
                "mask_path": str(mask_path),
                "gesture": GESTURE_NAMES[label] if 0 <= label < len(GESTURE_NAMES) else str(label),
                "clip_name": clip_name,
                "frame_name": frame_name,
            },
        }

        if self.model_name in ("baseline", "logich"):
            if not depth_path.exists():
                raise FileNotFoundError(f"Missing depth image for RGB-D evaluation: {depth_path}")
            depth_t = load_depth_as_tensor(depth_path, self.image_size)
            sample["depth"] = depth_t
            sample["meta"]["depth_path"] = str(depth_path)

        return sample


def test_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    out = {
        "rgb": torch.stack([b["rgb"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "bbox": torch.stack([b["bbox"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "meta": [b["meta"] for b in batch],
    }

    if "depth" in batch[0]:
        out["depth"] = torch.stack([b["depth"] for b in batch], dim=0)

    return out


# =========================================================
# Overlay helpers
# =========================================================
def tensor_to_pil_for_overlay(rgb_tensor: torch.Tensor) -> Image.Image:
    x = rgb_tensor.detach().cpu().numpy()
    x = np.transpose(x, (1, 2, 0))
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    return Image.fromarray(x)


def box_xyxy_to_pixel(box_xyxy: np.ndarray, w: int, h: int):
    x1, y1, x2, y2 = box_xyxy.tolist()
    x1 = int(round(np.clip(x1, 0.0, 1.0) * (w - 1)))
    y1 = int(round(np.clip(y1, 0.0, 1.0) * (h - 1)))
    x2 = int(round(np.clip(x2, 0.0, 1.0) * (w - 1)))
    y2 = int(round(np.clip(y2, 0.0, 1.0) * (h - 1)))
    return x1, y1, x2, y2


def overlay_prediction(
    rgb_tensor: torch.Tensor,
    pred_mask_prob: torch.Tensor,
    gt_box_xyxy: torch.Tensor,
    pred_box_xyxy: torch.Tensor,
    gt_label: int,
    pred_label: int,
    save_path: str,
):
    img = tensor_to_pil_for_overlay(rgb_tensor).convert("RGBA")
    w, h = img.size

    pred_mask = (pred_mask_prob.detach().cpu().numpy() > 0.5).astype(np.uint8)

    overlay_arr = np.zeros((h, w, 4), dtype=np.uint8)
    overlay_arr[pred_mask > 0] = np.array([255, 0, 0, 90], dtype=np.uint8)
    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    img = Image.alpha_composite(img, overlay)

    draw = ImageDraw.Draw(img)

    gx1, gy1, gx2, gy2 = box_xyxy_to_pixel(gt_box_xyxy.detach().cpu().numpy(), w, h)
    px1, py1, px2, py2 = box_xyxy_to_pixel(pred_box_xyxy.detach().cpu().numpy(), w, h)

    draw.rectangle([gx1, gy1, gx2, gy2], outline=(0, 255, 0, 255), width=3)   # GT
    draw.rectangle([px1, py1, px2, py2], outline=(0, 0, 255, 255), width=3)   # Pred

    gt_name = GESTURE_NAMES[gt_label] if 0 <= gt_label < len(GESTURE_NAMES) else str(gt_label)
    pred_name = GESTURE_NAMES[pred_label] if 0 <= pred_label < len(GESTURE_NAMES) else str(pred_label)

    draw.text((8, 8), f"GT: {gt_name}", fill=(0, 255, 0, 255))
    draw.text((8, 28), f"Pred: {pred_name}", fill=(0, 0, 255, 255))

    img.convert("RGB").save(save_path)


def save_confusion_matrix_png(conf: np.ndarray, class_names: List[str], save_path: str):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(conf, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    thresh = conf.max() / 2.0 if conf.max() > 0 else 0.5
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(
                j,
                i,
                str(conf[i, j]),
                ha="center",
                va="center",
                color="white" if conf[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Build loaders
# =========================================================
def build_val_loader(args):
    """
    Rebuild validation split from dataset_root.
    Must match training seed and val_ratio.
    """
    full = HandGestureDataset(
        dataset_root=args.dataset_root,
        model_name=args.model,
        image_size=tuple(args.image_size),
        augment=False,
    )

    n_total = len(full)
    n_val = max(1, int(round(n_total * args.val_ratio)))
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError("val_ratio too large; training split becomes empty.")

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    val_indices = perm[n_train:]

    val_set = Subset(full, val_indices)

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return val_loader, val_set


def build_test_loader(args):
    test_set = TestFolderDataset(
        test_root=args.test_root,
        image_size=tuple(args.image_size),
        model_name=args.model,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=test_collate_fn,
        drop_last=False,
    )
    return test_loader, test_set


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    num_classes: int,
    save_overlay_dir: str = None,
):
    model.eval()

    all_iou = []
    all_dice = []
    all_box_iou = []
    all_det_acc = []
    all_cls_acc = []

    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)

    # Save up to 2 overlays per gesture, preferring different clips
    saved_counts = {}
    saved_clips = {}

    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        bboxes = batch["bbox"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        depth = batch["depth"].to(device, non_blocking=True)
        seg_logits, bbox_pred, cls_logits = model(rgb, depth)

        pred_mask = torch.sigmoid(seg_logits)

        miou = iou_score(pred_mask, masks)                    # (B,)
        dice = dice_score(pred_mask, masks)                  # (B,)
        gt_xyxy = bbox_cxcywh_to_xyxy(bboxes)                # (B,4)
        pr_xyxy = bbox_cxcywh_to_xyxy(bbox_pred)             # (B,4)
        box_iou = bbox_iou_xyxy(pr_xyxy, gt_xyxy)            # (B,)
        det_acc05 = (box_iou >= 0.5).float()                 # (B,)
        preds = torch.argmax(cls_logits, dim=1)              # (B,)
        cls_acc = (preds == labels).float()                  # (B,)

        all_iou.extend(miou.detach().cpu().tolist())
        all_dice.extend(dice.detach().cpu().tolist())
        all_box_iou.extend(box_iou.detach().cpu().tolist())
        all_det_acc.extend(det_acc05.detach().cpu().tolist())
        all_cls_acc.extend(cls_acc.detach().cpu().tolist())

        update_confusion_matrix(conf, labels.detach().cpu(), preds.detach().cpu())

        if save_overlay_dir is not None:
            bs = rgb.size(0)
            for i in range(bs):
                gt_label = int(labels[i].item())
                gesture_name = GESTURE_NAMES[gt_label] if 0 <= gt_label < len(GESTURE_NAMES) else f"class_{gt_label}"

                if gesture_name not in saved_counts:
                    saved_counts[gesture_name] = 0
                    saved_clips[gesture_name] = set()

                if saved_counts[gesture_name] >= 2:
                    continue

                meta_i = batch["meta"][i]
                clip_name = meta_i.get("clip_name", f"clip_{saved_counts[gesture_name]}")
                frame_name = meta_i.get("frame_name", f"{saved_counts[gesture_name]:04d}")

                # Prefer different clips for the two saved overlays
                if saved_counts[gesture_name] == 1 and clip_name in saved_clips[gesture_name]:
                    continue

                save_name = f"{gesture_name}__{clip_name}__{frame_name}.png"
                save_path = os.path.join(save_overlay_dir, save_name)

                overlay_prediction(
                    rgb_tensor=rgb[i].detach().cpu(),
                    pred_mask_prob=pred_mask[i, 0].detach().cpu(),
                    gt_box_xyxy=gt_xyxy[i].detach().cpu(),
                    pred_box_xyxy=pr_xyxy[i].detach().cpu(),
                    gt_label=gt_label,
                    pred_label=int(preds[i].item()),
                    save_path=save_path,
                )
                saved_counts[gesture_name] += 1
                saved_clips[gesture_name].add(clip_name)

    return {
        "mIoU": float(np.mean(all_iou)) if len(all_iou) > 0 else 0.0,
        "Dice": float(np.mean(all_dice)) if len(all_dice) > 0 else 0.0,
        "BoxIoU": float(np.mean(all_box_iou)) if len(all_box_iou) > 0 else 0.0,
        "DetAcc@0.5": float(np.mean(all_det_acc)) if len(all_det_acc) > 0 else 0.0,
        "ClsAcc": float(np.mean(all_cls_acc)) if len(all_cls_acc) > 0 else 0.0,
        "MacroF1": macro_f1_from_confusion(conf),
        "ConfusionMatrix": conf,
    }


# =========================================================
# Args
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", type=str, required=True, choices=["val", "test"])
    parser.add_argument("--model", type=str, required=True, choices=["baseline", "LogicH"])
    parser.add_argument("--ckpt", type=str, required=True)

    parser.add_argument("--dataset_root", type=str, default="dataset")
    parser.add_argument("--test_root", type=str, default="dataset/test")

    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--width", type=int, default=None)

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--save_overlays", action="store_true")
    parser.add_argument("--save_confusion_png", action="store_true")
    parser.add_argument("--save_confusion_npy", action="store_true")
    parser.add_argument("--save_metrics_json", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")

    return parser.parse_args()


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cpu" if args.no_cuda or (not torch.cuda.is_available()) else "cuda")

    # Save into results/<split>/<model>
    results_dir = os.path.join(args.results_root, args.split, args.model)
    ensure_dir(results_dir)

    if args.split == "val":
        loader, subset = build_val_loader(args)
        eval_root = args.dataset_root
        num_samples = len(subset)
    else:
        loader, subset = build_test_loader(args)
        eval_root = args.test_root
        num_samples = len(subset)

    model = create_model(
        model_name=args.model,
        num_classes=args.num_classes,
        width=args.width,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)

    metrics = evaluate(
        model=model,
        loader=loader,
        device=device,
        num_classes=args.num_classes,
        save_overlay_dir=results_dir if args.save_overlays else None,
    )

    conf = metrics.pop("ConfusionMatrix")

    print(f"Checkpoint : {args.ckpt}")
    print(f"Model      : {args.model}")
    print(f"Split      : {args.split}")
    print(f"Data root  : {eval_root}")
    print(f"Samples    : {num_samples}")
    print(f"Device     : {device}")
    print(
        f"{args.split.upper()} "
        f"mIoU={metrics['mIoU']:.4f} "
        f"Dice={metrics['Dice']:.4f} "
        f"BoxIoU={metrics['BoxIoU']:.4f} "
        f"DetAcc@0.5={metrics['DetAcc@0.5']:.4f} "
        f"ClsAcc={metrics['ClsAcc']:.4f} "
        f"MacroF1={metrics['MacroF1']:.4f}"
    )

    print("\nConfusion Matrix:")
    print(conf.numpy())

    if args.save_confusion_png:
        save_confusion_matrix_png(
            conf.numpy(),
            GESTURE_NAMES[:args.num_classes],
            os.path.join(results_dir, "confusion_matrix.png"),
        )

    if args.save_confusion_npy:
        np.save(os.path.join(results_dir, "confusion_matrix.npy"), conf.numpy())

    if args.save_metrics_json:
        with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    if args.save_overlays:
        print(f"\nSaved overlays to: {results_dir}")
        print("Rule: up to 2 overlays per gesture, preferring different clips.")


if __name__ == "__main__":
    main()
