import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


GESTURE_NAMES = [
    "call", "dislike", "like", "ok", "one",
    "palm", "peace", "rock", "stop", "three"
]


@dataclass
class Sample:
    rgb: torch.Tensor                    # (3,H,W)
    mask: torch.Tensor                   # (1,H,W), {0,1}
    bbox: torch.Tensor                   # (4,), normalized cxcywh
    label: torch.Tensor                  # scalar long
    depth: Optional[torch.Tensor] = None # (1,H,W) for baseline only
    meta: Optional[Dict] = None


def _list_gesture_dirs(root_rgb: str) -> List[str]:
    dirs = []
    for d in sorted(os.listdir(root_rgb)):
        full = os.path.join(root_rgb, d)
        if os.path.isdir(full) and d.upper().startswith("G"):
            dirs.append(d)
    return sorted(dirs)


def _gesture_id_from_dirname(dirname: str) -> int:
    s = dirname.upper().replace("G_", "G")
    num = int("".join([c for c in s[1:] if c.isdigit()]))
    return num - 1


def _load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_gray(path: str) -> Image.Image:
    return Image.open(path).convert("L")


def _mask_to_bbox(mask01: np.ndarray) -> Tuple[float, float, float, float]:
    """
    mask01: (H,W) in {0,1}
    return normalized bbox (cx, cy, w, h)
    """
    ys, xs = np.where(mask01 > 0.5)
    h, w = mask01.shape[:2]

    if len(xs) == 0 or len(ys) == 0:
        return 0.5, 0.5, 1e-3, 1e-3

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    bw = (x2 - x1 + 1) / w
    bh = (y2 - y1 + 1) / h
    cx = (x1 + x2 + 1) / 2.0 / w
    cy = (y1 + y2 + 1) / 2.0 / h

    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))
    bw = float(np.clip(bw, 1e-6, 1.0))
    bh = float(np.clip(bh, 1e-6, 1.0))
    return cx, cy, bw, bh


class HandGestureDataset(Dataset):
    """
    Expected structure:
      dataset/
        rgb/G01/*.png
        annotation/G01/*.png
        depth/G01/*.png   (needed for RGB-D models)
    """

    def __init__(
        self,
        dataset_root: str,
        model_name: str = "baseline",
        #image_size: Tuple[int, int] = (256, 256),
        image_size: Tuple[int, int] = (240, 320),
        #image_size: Tuple[int, int] = (480, 640),
        augment: bool = False,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.model_name = model_name.lower()
        self.image_size = image_size
        self.augment = augment

        self.root_rgb = os.path.join(dataset_root, "rgb")
        self.root_ann = os.path.join(dataset_root, "annotation")
        self.root_depth = os.path.join(dataset_root, "depth")

        if not os.path.isdir(self.root_rgb):
            raise FileNotFoundError(f"Missing folder: {self.root_rgb}")
        if not os.path.isdir(self.root_ann):
            raise FileNotFoundError(f"Missing folder: {self.root_ann}")
        if self.model_name in ("baseline", "logich") and not os.path.isdir(self.root_depth):
            raise FileNotFoundError(f"Missing folder: {self.root_depth}")

        self.samples: List[Tuple[str, str, Optional[str], int, str]] = []

        gesture_dirs = _list_gesture_dirs(self.root_rgb)
        for gdir in gesture_dirs:
            label = _gesture_id_from_dirname(gdir)

            rgb_dir = os.path.join(self.root_rgb, gdir)
            ann_dir = os.path.join(self.root_ann, gdir)
            depth_dir = os.path.join(self.root_depth, gdir)

            if not os.path.isdir(ann_dir):
                continue

            rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(".png")])
            for fname in rgb_files:
                rgb_path = os.path.join(rgb_dir, fname)
                ann_path = os.path.join(ann_dir, fname)
                if not os.path.exists(ann_path):
                    continue

                depth_path = None
                if self.model_name in ("baseline", "logich"):
                    depth_path = os.path.join(depth_dir, fname)
                    if not os.path.exists(depth_path):
                        continue

                gesture_name = GESTURE_NAMES[label] if 0 <= label < len(GESTURE_NAMES) else gdir
                self.samples.append((rgb_path, ann_path, depth_path, label, gesture_name))

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found. Check dataset structure and filenames.")

    def __len__(self) -> int:
        return len(self.samples)

    def _augment_pair(
        self,
        rgb: Image.Image,
        mask: Image.Image,
        depth: Optional[Image.Image] = None,
    ):
        # 1) horizontal flip
        if self.augment and torch.rand(()) < 0.3:
            rgb = TF.hflip(rgb)
            mask = TF.hflip(mask)
            if depth is not None:
                depth = TF.hflip(depth)

        # 2) random affine: rotation + translation + scale
        if self.augment and torch.rand(()) < 0.7:
            angle = float(torch.empty(1).uniform_(-15.0, 15.0))  # rotation in degrees

            max_dx = 0.08 * self.image_size[1]  # 8% of width
            max_dy = 0.08 * self.image_size[0]  # 8% of height
            tx = int(torch.empty(1).uniform_(-max_dx, max_dx))
            ty = int(torch.empty(1).uniform_(-max_dy, max_dy))

            scale = float(torch.empty(1).uniform_(0.8, 1.2))  # zoom out/in

            rgb = TF.affine(
                rgb,
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
            )
            if depth is not None:
                depth = TF.affine(
                    depth,
                    angle=angle,
                    translate=[tx, ty],
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                )

        # 3) color jitter for RGB only
        if self.augment:
            b = float(torch.empty(1).uniform_(0.8, 1.2))
            c = float(torch.empty(1).uniform_(0.8, 1.2))
            s = float(torch.empty(1).uniform_(0.8, 1.2))
            h = float(torch.empty(1).uniform_(-0.08, 0.08))

            rgb = TF.adjust_brightness(rgb, b)
            rgb = TF.adjust_contrast(rgb, c)
            rgb = TF.adjust_saturation(rgb, s)
            rgb = TF.adjust_hue(rgb, h)

        # 4) random blur for RGB only
        if self.augment and torch.rand(()) < 0.25:
            rgb = TF.gaussian_blur(rgb, kernel_size=[3, 3])

        return rgb, mask, depth

    def __getitem__(self, idx: int) -> Sample:
        rgb_path, ann_path, depth_path, label, gesture_name = self.samples[idx]

        target_h, target_w = self.image_size

        rgb = _load_rgb(rgb_path)
        mask = _load_gray(ann_path)
        depth = _load_gray(depth_path) if (self.model_name in ("baseline", "logich") and depth_path is not None) else None

        rgb = TF.resize(rgb, [target_h, target_w], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [target_h, target_w], interpolation=InterpolationMode.NEAREST)
        if depth is not None:
            depth = TF.resize(depth, [target_h, target_w], interpolation=InterpolationMode.BILINEAR)

        rgb, mask, depth = self._augment_pair(rgb, mask, depth)

        rgb_t = TF.to_tensor(rgb)  # [0,1]

        mask_np = np.array(mask, dtype=np.uint8)
        mask01 = (mask_np > 127).astype(np.float32)
        mask_t = torch.from_numpy(mask01).unsqueeze(0)

        bbox = _mask_to_bbox(mask01)
        bbox_t = torch.tensor(bbox, dtype=torch.float32)
        label_t = torch.tensor(label, dtype=torch.long)

        depth_t = None
        if depth is not None:
            depth_np = np.array(depth, dtype=np.float32)
            if depth_np.max() > 0:
                if depth_np.max() > 255:
                    depth_np = depth_np / depth_np.max()
                else:
                    depth_np = depth_np / 255.0
            depth_t = torch.from_numpy(depth_np).unsqueeze(0)

        meta = {
            "rgb_path": rgb_path,
            "mask_path": ann_path,
            "depth_path": depth_path,
            "gesture": gesture_name,
        }

        return Sample(
            rgb=rgb_t,
            mask=mask_t,
            bbox=bbox_t,
            label=label_t,
            depth=depth_t,
            meta=meta,
        )


def collate_fn(batch: List[Sample]) -> Dict[str, torch.Tensor]:
    out = {
        "rgb": torch.stack([b.rgb for b in batch], dim=0),
        "mask": torch.stack([b.mask for b in batch], dim=0),
        "bbox": torch.stack([b.bbox for b in batch], dim=0),
        "label": torch.stack([b.label for b in batch], dim=0),
        "meta": [b.meta for b in batch],
    }

    if batch[0].depth is not None:
        out["depth"] = torch.stack([b.depth for b in batch], dim=0)

    return out


def make_dataloaders(
    dataset_root: str,
    model_name: str = "baseline",
    image_size: Tuple[int, int] = (240, 320),
    batch_size: int = 16,
    num_workers: int = 4,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    train_dataset_full = HandGestureDataset(
        dataset_root=dataset_root,
        model_name=model_name,
        image_size=image_size,
        augment=True,
    )
    val_dataset_full = HandGestureDataset(
        dataset_root=dataset_root,
        model_name=model_name,
        image_size=image_size,
        augment=False,
    )

    n_total = len(train_dataset_full)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError("val_ratio is too large. Training split became empty.")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    train_set = Subset(train_dataset_full, train_indices)
    val_set = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader, train_set, val_set
