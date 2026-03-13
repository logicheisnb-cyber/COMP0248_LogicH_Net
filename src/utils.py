import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# =========================================================
# Loss
# =========================================================
class DiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + eps) / (den + eps)
        return 1.0 - dice.mean()


# =========================================================
# BBox utils
# =========================================================
def bbox_cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    """
    b: (B,4) normalized (cx,cy,w,h) -> (B,4) normalized (x1,y1,x2,y2)
    """
    cx, cy, w, h = b.unbind(dim=1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=1).clamp(0.0, 1.0)


def bbox_iou_xyxy(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    a, b: (B,4) normalized xyxy
    returns IoU per sample: (B,)
    """
    ax1, ay1, ax2, ay2 = a.unbind(1)
    bx1, by1, bx2, by2 = b.unbind(1)

    ix1 = torch.max(ax1, bx1)
    iy1 = torch.max(ay1, by1)
    ix2 = torch.min(ax2, bx2)
    iy2 = torch.min(ay2, by2)

    iw = (ix2 - ix1).clamp(min=0.0)
    ih = (iy2 - iy1).clamp(min=0.0)
    inter = iw * ih

    area_a = (ax2 - ax1).clamp(min=0.0) * (ay2 - ay1).clamp(min=0.0)
    area_b = (bx2 - bx1).clamp(min=0.0) * (by2 - by1).clamp(min=0.0)
    union = area_a + area_b - inter + eps
    return inter / union


# =========================================================
# Segmentation metrics
# =========================================================
@torch.no_grad()
def dice_score(pred_mask: torch.Tensor, gt_mask: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """
    pred_mask/gt_mask: (B,1,H,W)
    returns per-sample dice: (B,)
    """
    pred = (pred_mask > thr).float()
    gt = (gt_mask > thr).float()

    inter = (pred * gt).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
    return (2.0 * inter + eps) / (den + eps)


@torch.no_grad()
def iou_score(pred_mask: torch.Tensor, gt_mask: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """
    pred_mask/gt_mask: (B,1,H,W)
    returns per-sample IoU: (B,)
    """
    pred = (pred_mask > thr).float()
    gt = (gt_mask > thr).float()

    inter = (pred * gt).sum(dim=(1, 2, 3))
    union = ((pred + gt) > 0).float().sum(dim=(1, 2, 3))
    return (inter + eps) / (union + eps)


# =========================================================
# Classification metrics
# =========================================================
def update_confusion_matrix(conf: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor) -> None:
    """
    conf: (C,C) on CPU, rows=true, cols=pred
    targets/preds: (B,) on CPU
    """
    for t, p in zip(targets.tolist(), preds.tolist()):
        conf[t, p] += 1


def macro_f1_from_confusion(conf: torch.Tensor, eps: float = 1e-9) -> float:
    tp = torch.diag(conf).float()
    fp = conf.sum(dim=0).float() - tp
    fn = conf.sum(dim=1).float() - tp
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    return float(f1.mean().item())


# =========================================================
# Helper meters
# =========================================================
@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)