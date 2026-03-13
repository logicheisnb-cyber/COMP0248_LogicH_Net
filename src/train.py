import os
import argparse
import numpy as np

import torch
import torch.nn as nn

from .model import create_model
from .dataloader import make_dataloaders
from .utils import (
    set_seed,
    ensure_dir,
    DiceLoss,
    bbox_cxcywh_to_xyxy,
    bbox_iou_xyxy,
    dice_score,
    iou_score,
    update_confusion_matrix,
    macro_f1_from_confusion,
    AverageMeter,
)


# =========================================================
# Train / Validate
# =========================================================
def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    seg_bce,
    seg_dice,
    cls_ce,
    box_l1,
    lambda_seg=1.0,
    lambda_box=1.0,
    lambda_cls=1.0,
):
    model.train()

    loss_meter = AverageMeter()
    seg_meter = AverageMeter()
    box_meter = AverageMeter()
    cls_meter = AverageMeter()

    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        bbox = batch["bbox"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        seg_logits, bbox_pred, cls_logits = model(rgb, depth)

        loss_seg = seg_bce(seg_logits, mask) + seg_dice(seg_logits, mask)
        loss_box = box_l1(bbox_pred, bbox)
        loss_cls = cls_ce(cls_logits, label)

        loss = lambda_seg * loss_seg + lambda_box * loss_box + lambda_cls * loss_cls
        loss.backward()
        optimizer.step()

        bs = rgb.size(0)
        loss_meter.update(loss.item(), bs)
        seg_meter.update(loss_seg.item(), bs)
        box_meter.update(loss_box.item(), bs)
        cls_meter.update(loss_cls.item(), bs)

    return {
        "loss": loss_meter.avg,
        "seg": seg_meter.avg,
        "box": box_meter.avg,
        "cls": cls_meter.avg,
    }


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    seg_bce,
    seg_dice,
    cls_ce,
    box_l1,
    num_classes,
    lambda_seg=1.0,
    lambda_box=1.0,
    lambda_cls=1.0,
):
    model.eval()

    loss_meter = AverageMeter()
    seg_meter = AverageMeter()
    box_meter = AverageMeter()
    cls_meter = AverageMeter()

    all_iou = []
    all_dice = []
    all_box_iou = []
    all_det_acc = []
    all_cls_acc = []

    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        bbox = batch["bbox"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        seg_logits, bbox_pred, cls_logits = model(rgb, depth)

        loss_seg = seg_bce(seg_logits, mask) + seg_dice(seg_logits, mask)
        loss_box = box_l1(bbox_pred, bbox)
        loss_cls = cls_ce(cls_logits, label)
        loss = lambda_seg * loss_seg + lambda_box * loss_box + lambda_cls * loss_cls

        bs = rgb.size(0)
        loss_meter.update(loss.item(), bs)
        seg_meter.update(loss_seg.item(), bs)
        box_meter.update(loss_box.item(), bs)
        cls_meter.update(loss_cls.item(), bs)

        pred_mask = torch.sigmoid(seg_logits)
        batch_iou = iou_score(pred_mask, mask)
        batch_dice = dice_score(pred_mask, mask)

        pred_xyxy = bbox_cxcywh_to_xyxy(bbox_pred)
        gt_xyxy = bbox_cxcywh_to_xyxy(bbox)
        batch_box_iou = bbox_iou_xyxy(pred_xyxy, gt_xyxy)
        batch_det_acc = (batch_box_iou >= 0.5).float()

        pred_cls = cls_logits.argmax(dim=1)
        batch_cls_acc = (pred_cls == label).float()

        all_iou.extend(batch_iou.cpu().tolist())
        all_dice.extend(batch_dice.cpu().tolist())
        all_box_iou.extend(batch_box_iou.cpu().tolist())
        all_det_acc.extend(batch_det_acc.cpu().tolist())
        all_cls_acc.extend(batch_cls_acc.cpu().tolist())

        update_confusion_matrix(
            conf,
            label.cpu(),
            pred_cls.cpu(),
        )

    return {
        "loss": loss_meter.avg,
        "seg": seg_meter.avg,
        "box": box_meter.avg,
        "cls": cls_meter.avg,
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
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "LogicH"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="weights")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--min_lr", type=float, default=1e-5)

    parser.add_argument("--lambda_seg", type=float, default=3.0)
    parser.add_argument("--lambda_box", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=3.0)

    return parser.parse_args()


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.save_dir)

    train_loader, val_loader, train_set, val_set = make_dataloaders(
        dataset_root=args.dataset_root,
        model_name="baseline",   # dataloader always loads RGB-D data
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    model = create_model(
        model_name=args.model,
        num_classes=args.num_classes,
        width=args.width,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    seg_bce = nn.BCEWithLogitsLoss()
    seg_dice = DiceLoss()
    cls_ce = nn.CrossEntropyLoss()
    box_l1 = nn.SmoothL1Loss()

    best_miou = -1.0
    best_path = os.path.join(args.save_dir, f"best_{args.model}.pt")

    print(f"Model       : {args.model}")
    print(f"Dataset     : {args.dataset_root}")
    print(f"Train size  : {len(train_set)}")
    print(f"Val size    : {len(val_set)}")
    print(f"Device      : {device}")
    print(f"Save path   : {best_path}")
    print("-" * 100)

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            seg_bce=seg_bce,
            seg_dice=seg_dice,
            cls_ce=cls_ce,
            box_l1=box_l1,
            lambda_seg=args.lambda_seg,
            lambda_box=args.lambda_box,
            lambda_cls=args.lambda_cls,
        )

        val_stats = validate(
            model=model,
            loader=val_loader,
            device=device,
            seg_bce=seg_bce,
            seg_dice=seg_dice,
            cls_ce=cls_ce,
            box_l1=box_l1,
            num_classes=args.num_classes,
            lambda_seg=args.lambda_seg,
            lambda_box=args.lambda_box,
            lambda_cls=args.lambda_cls,
        )

        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"train loss={train_stats['loss']:.4f} "
            f"(seg={train_stats['seg']:.4f}, box={train_stats['box']:.4f}, cls={train_stats['cls']:.4f}) | "
            f"val mIoU={val_stats['mIoU']:.4f} "
            f"Dice={val_stats['Dice']:.4f} "
            f"BoxIoU={val_stats['BoxIoU']:.4f} "
            f"DetAcc@0.5={val_stats['DetAcc@0.5']:.4f} "
            f"ClsAcc={val_stats['ClsAcc']:.4f} "
            f"MacroF1={val_stats['MacroF1']:.4f}"
        )

        if val_stats["mIoU"] > best_miou:
            best_miou = val_stats["mIoU"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_mIoU": best_miou,
                    "args": vars(args),
                    "val_stats": {
                        "mIoU": val_stats["mIoU"],
                        "Dice": val_stats["Dice"],
                        "BoxIoU": val_stats["BoxIoU"],
                        "DetAcc@0.5": val_stats["DetAcc@0.5"],
                        "ClsAcc": val_stats["ClsAcc"],
                        "MacroF1": val_stats["MacroF1"],
                    },
                },
                best_path,
            )
            print(f"  -> Saved best checkpoint to {best_path} (best_mIoU={best_miou:.4f})")
        scheduler.step()


if __name__ == "__main__":
    main()