import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# =========================================================
# Common blocks
# =========================================================
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)
        self.conv3 = ConvBNReLU(out_ch, out_ch)
        self.conv4 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


def masked_avg_pool(feat: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    masked = feat * mask
    num = masked.sum(dim=(2, 3))
    den = mask.sum(dim=(2, 3)).clamp(min=eps)
    return num / den


@torch.no_grad()
def _meshgrid_xy(h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = torch.linspace(0.0, 1.0, steps=h, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, steps=w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx.view(1, 1, h, w), yy.view(1, 1, h, w)


def soft_bbox_from_mask(mask_prob: torch.Tensor, thr: float = 0.25, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert segmentation probability map into normalized bbox [cx, cy, w, h].
    """
    b, _, h, w = mask_prob.shape
    device = mask_prob.device
    dtype = mask_prob.dtype

    weights = (mask_prob - thr).clamp(min=0.0)
    ws = weights.sum(dim=(2, 3), keepdim=True)

    empty = (ws < eps).float()
    weights = weights + empty * torch.ones_like(weights)
    ws = weights.sum(dim=(2, 3), keepdim=True).clamp(min=eps)

    xx, yy = _meshgrid_xy(h, w, device, dtype)

    cx = (weights * xx).sum(dim=(2, 3), keepdim=False) / ws.squeeze(-1).squeeze(-1)
    cy = (weights * yy).sum(dim=(2, 3), keepdim=False) / ws.squeeze(-1).squeeze(-1)

    dx = (weights * (xx - cx.view(b, 1, 1, 1)).abs()).sum(dim=(2, 3), keepdim=False) / ws.squeeze(-1).squeeze(-1)
    dy = (weights * (yy - cy.view(b, 1, 1, 1)).abs()).sum(dim=(2, 3), keepdim=False) / ws.squeeze(-1).squeeze(-1)

    bw = (4.0 * dx).clamp(min=1e-3, max=1.0)
    bh = (4.0 * dy).clamp(min=1e-3, max=1.0)

    bbox = torch.cat([cx, cy, bw, bh], dim=1)
    return bbox.clamp(0.0, 1.0)


def expand_bbox_cxcywh(bbox: torch.Tensor, scale: float = 1.2) -> torch.Tensor:
    cx = bbox[:, 0:1]
    cy = bbox[:, 1:2]
    bw = (bbox[:, 2:3] * scale).clamp(min=1e-3, max=1.0)
    bh = (bbox[:, 3:4] * scale).clamp(min=1e-3, max=1.0)
    return torch.cat([cx, cy, bw, bh], dim=1)


def bbox_cxcywh_to_xyxy_norm(bbox: torch.Tensor) -> torch.Tensor:
    cx = bbox[:, 0:1]
    cy = bbox[:, 1:2]
    bw = bbox[:, 2:3]
    bh = bbox[:, 3:4]

    x1 = (cx - 0.5 * bw).clamp(0.0, 1.0)
    y1 = (cy - 0.5 * bh).clamp(0.0, 1.0)
    x2 = (cx + 0.5 * bw).clamp(0.0, 1.0)
    y2 = (cy + 0.5 * bh).clamp(0.0, 1.0)
    return torch.cat([x1, y1, x2, y2], dim=1)


def bbox_to_roi_mask(
    bbox: torch.Tensor,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    bbox: (B,4) normalized cxcywh
    return: (B,1,H,W) binary roi mask
    """
    b = bbox.shape[0]
    xyxy = bbox_cxcywh_to_xyxy_norm(bbox)

    xx, yy = _meshgrid_xy(h, w, device, dtype)
    xx = xx.expand(b, -1, -1, -1)
    yy = yy.expand(b, -1, -1, -1)

    x1 = xyxy[:, 0:1].view(b, 1, 1, 1)
    y1 = xyxy[:, 1:2].view(b, 1, 1, 1)
    x2 = xyxy[:, 2:3].view(b, 1, 1, 1)
    y2 = xyxy[:, 3:4].view(b, 1, 1, 1)

    mask = ((xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)).to(dtype)
    return mask


# =========================================================
# Lightweight RGB-D backbone blocks
# =========================================================
class DWConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand: int = 4):
        super().__init__()
        hidden = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return x + y if self.use_res else y


class DepthFiLM(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.to_gb = nn.Conv2d(ch, 2 * ch, kernel_size=1, bias=True)

    def forward(self, rgb_feat: torch.Tensor, depth_feat: torch.Tensor) -> torch.Tensor:
        gb = self.to_gb(depth_feat)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = torch.tanh(gamma)
        return rgb_feat * (1.0 + gamma) + beta


class FuseBlock(nn.Module):
    """
    Baseline RGB-D fusion:
    concatenate RGB and depth features then project.
    """
    def __init__(self, ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(ch * 2, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, fr: torch.Tensor, fd: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([fr, fd], dim=1))


# =========================================================
# Innovation modules
# =========================================================
class GatedCrossModalFusion(nn.Module):
    """
    Classification-only GCMF:
    residual gating + RGB skip.
    """
    def __init__(self, ch: int, gate_scale: float = 0.25):
        super().__init__()
        self.gate_scale = gate_scale

        self.gate = nn.Conv2d(ch * 2, ch * 2, kernel_size=1, bias=True)

        self.out = nn.Sequential(
            nn.Conv2d(ch * 2, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

        self.rgb_skip = nn.Identity()

    def forward(self, fr: torch.Tensor, fd: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([fr, fd], dim=1)

        gates = torch.tanh(self.gate(cat))
        g_rgb, g_dep = gates.chunk(2, dim=1)

        fr_refined = fr * (1.0 + self.gate_scale * g_rgb)
        fd_refined = fd * (1.0 + self.gate_scale * g_dep)

        fused = torch.cat([fr_refined, fd_refined], dim=1)
        out = self.out(fused)
        return out + self.rgb_skip(fr)


class DepthGuidedBoxRefiner(nn.Module):
    """
    Use segmentation-derived ROI + depth features to refine bbox correction.
    """
    def __init__(self, feat_ch: int, depth_ch: int, hidden_ch: int = 128, expand_scale: float = 1.2):
        super().__init__()
        self.expand_scale = expand_scale

        self.rgb_proj = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
        )
        self.dep_proj = nn.Sequential(
            nn.Conv2d(depth_ch, depth_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(depth_ch),
            nn.ReLU(inplace=True),
        )

        in_dim = feat_ch + depth_ch + 4
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_ch, hidden_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_ch // 2, 4),
        )

    def forward(self, rgb_feat: torch.Tensor, depth_feat: torch.Tensor, bbox_init: torch.Tensor) -> torch.Tensor:
        b, _, h, w = rgb_feat.shape

        bbox_roi = expand_bbox_cxcywh(bbox_init, scale=self.expand_scale)
        roi_mask = bbox_to_roi_mask(
            bbox_roi,
            h=h,
            w=w,
            device=rgb_feat.device,
            dtype=rgb_feat.dtype,
        )

        rgb_feat = self.rgb_proj(rgb_feat)
        depth_feat = self.dep_proj(depth_feat)

        rgb_vec = masked_avg_pool(rgb_feat, roi_mask)
        dep_vec = masked_avg_pool(depth_feat, roi_mask)

        box_vec = torch.cat([rgb_vec, dep_vec, bbox_init], dim=1)
        delta = self.mlp(box_vec)
        return delta


# =========================================================
# Shared RGB-D encoder
# Used by both baseline and innovation models
# =========================================================
class LiteRGBDEncoder(nn.Module):
    """
    Stable RGB-D encoder.
    Returns:
      s0: /2, s1: /4, s2: /8, s3: /16, s4: /32
      fd2, fd3, fd4: depth branch high-level features for innovation model
    """
    def __init__(self, width: int = 32):
        super().__init__()

        c1 = width
        c2 = width * 2
        c3 = width * 3
        c4 = width * 4

        self.rgb_stem0 = DWConvBNReLU(3, c1, stride=2)
        self.rgb_stem1 = DWConvBNReLU(c1, c1, stride=2)

        self.depth_stem0 = DWConvBNReLU(1, c1, stride=2)
        self.depth_stem1 = DWConvBNReLU(c1, c1, stride=2)

        self.film0 = DepthFiLM(c1)
        self.film1 = DepthFiLM(c1)

        self.rgb_stage1 = nn.Sequential(
            InvertedResidual(c1, c1, stride=1, expand=2),
            InvertedResidual(c1, c1, stride=1, expand=2),
        )
        self.rgb_stage2 = nn.Sequential(
            InvertedResidual(c1, c2, stride=2, expand=3),
            InvertedResidual(c2, c2, stride=1, expand=3),
        )
        self.rgb_stage3 = nn.Sequential(
            InvertedResidual(c2, c3, stride=2, expand=3),
            InvertedResidual(c3, c3, stride=1, expand=3),
        )
        self.rgb_stage4 = nn.Sequential(
            InvertedResidual(c3, c4, stride=2, expand=3),
            InvertedResidual(c4, c4, stride=1, expand=3),
        )

        self.depth_stage1 = nn.Sequential(
            InvertedResidual(c1, c1, stride=1, expand=2),
            InvertedResidual(c1, c1, stride=1, expand=2),
        )
        self.depth_stage2 = nn.Sequential(
            InvertedResidual(c1, c2, stride=2, expand=3),
            InvertedResidual(c2, c2, stride=1, expand=3),
        )
        self.depth_stage3 = nn.Sequential(
            InvertedResidual(c2, c3, stride=2, expand=3),
            InvertedResidual(c3, c3, stride=1, expand=3),
        )
        self.depth_stage4 = nn.Sequential(
            InvertedResidual(c3, c4, stride=2, expand=3),
            InvertedResidual(c4, c4, stride=1, expand=3),
        )

        self.fuse1 = FuseBlock(c1)
        self.fuse2 = FuseBlock(c2)
        self.fuse3 = FuseBlock(c3)
        self.fuse4 = FuseBlock(c4)

        self.out_channels = (c1, c1, c2, c3, c4)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        rgb_s0 = self.rgb_stem0(rgb)
        dep_s0 = self.depth_stem0(depth)
        s0 = self.film0(rgb_s0, dep_s0)

        rgb_s1 = self.rgb_stem1(s0)
        dep_s1 = self.depth_stem1(dep_s0)
        rgb_s1 = self.film1(rgb_s1, dep_s1)

        fr1 = self.rgb_stage1(rgb_s1)
        fd1 = self.depth_stage1(dep_s1)
        s1 = self.fuse1(fr1, fd1)

        fr2 = self.rgb_stage2(s1)
        fd2 = self.depth_stage2(fd1)
        s2 = self.fuse2(fr2, fd2)

        fr3 = self.rgb_stage3(s2)
        fd3 = self.depth_stage3(fd2)
        s3 = self.fuse3(fr3, fd3)

        fr4 = self.rgb_stage4(s3)
        fd4 = self.depth_stage4(fd3)
        s4 = self.fuse4(fr4, fd4)

        return s0, s1, s2, s3, s4, fd2, fd3, fd4


# =========================================================
# Baseline RGB-D model
# = baseline from your first version
# - standard fusion
# - bbox directly from segmentation
# =========================================================
class MultiTaskHandNetBaseline(nn.Module):
    """
    Baseline RGB-D:
    - standard fusion
    - bbox directly from segmentation
    """
    def __init__(self, num_classes: int = 10, width: int = 32, use_multiscale_cls: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.use_multiscale_cls = use_multiscale_cls

        self.encoder = LiteRGBDEncoder(width=width)
        c0, c1, c2, c3, c4 = self.encoder.out_channels

        self.up3 = UpBlock(in_ch=c4, skip_ch=c3, out_ch=c3)
        self.up2 = UpBlock(in_ch=c3, skip_ch=c2, out_ch=c2)
        self.up1 = UpBlock(in_ch=c2, skip_ch=c1, out_ch=c1)
        self.up0 = UpBlock(in_ch=c1, skip_ch=c0, out_ch=c0)

        self.seg_head = nn.Sequential(
            ConvBNReLU(c0, c0, k=3, s=1, p=1),
            ConvBNReLU(c0, c0, k=3, s=1, p=1),
            nn.Conv2d(c0, 1, kernel_size=1),
        )

        cls_dim = c2 + c3 + c4 if use_multiscale_cls else c4
        self.cls_head = nn.Sequential(
            nn.Linear(cls_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes),
        )

    def forward(self, rgb: torch.Tensor, depth: Optional[torch.Tensor] = None):
        if depth is None:
            if rgb.shape[1] != 4:
                raise ValueError("RGB-D model expects either (rgb, depth) or a 4-channel tensor (B,4,H,W).")
            x = rgb
            rgb = x[:, :3]
            depth = x[:, 3:4]

        s0, s1, s2, s3, s4, _, _, _ = self.encoder(rgb, depth)

        d3 = self.up3(s4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)

        seg_logits = self.seg_head(d0)
        seg_logits = F.interpolate(seg_logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        mask_prob = torch.sigmoid(seg_logits)

        if self.use_multiscale_cls:
            mask_d2 = F.interpolate(mask_prob, size=d2.shape[-2:], mode="bilinear", align_corners=False)
            feat_cls_d2 = masked_avg_pool(d2, mask_d2)

            mask_s3 = F.interpolate(mask_prob, size=s3.shape[-2:], mode="bilinear", align_corners=False)
            feat_cls_s3 = masked_avg_pool(s3, mask_s3)

            mask_s4 = F.interpolate(mask_prob, size=s4.shape[-2:], mode="bilinear", align_corners=False)
            feat_cls_s4 = masked_avg_pool(s4, mask_s4)

            feat_cls = torch.cat([feat_cls_d2, feat_cls_s3, feat_cls_s4], dim=1)
        else:
            mask_s4 = F.interpolate(mask_prob, size=s4.shape[-2:], mode="bilinear", align_corners=False)
            feat_cls = masked_avg_pool(s4, mask_s4)

        cls_logits = self.cls_head(feat_cls)

        # baseline bbox directly from segmentation
        bbox_pred = soft_bbox_from_mask(mask_prob, thr=0.25)

        return seg_logits, bbox_pred, cls_logits


# =========================================================
# LogicH model
# - segmentation same as baseline
# - GCMF only for classification
# - depth-guided bbox correction from segmentation ROI
# =========================================================
class MultiTaskHandNetLogicH(nn.Module):
    """
    - stable RGB-D encoder
    - baseline segmentation path
    - GCMF only for classification refinement
    - bbox correction uses segmentation-derived ROI + depth-guided local refinement
    """
    def __init__(self, num_classes: int = 10, width: int = 32, use_multiscale_cls: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.use_multiscale_cls = use_multiscale_cls

        self.encoder = LiteRGBDEncoder(width=width)
        c0, c1, c2, c3, c4 = self.encoder.out_channels

        self.up3 = UpBlock(in_ch=c4, skip_ch=c3, out_ch=c3)
        self.up2 = UpBlock(in_ch=c3, skip_ch=c2, out_ch=c2)
        self.up1 = UpBlock(in_ch=c2, skip_ch=c1, out_ch=c1)
        self.up0 = UpBlock(in_ch=c1, skip_ch=c0, out_ch=c0)

        # segmentation path same as baseline
        self.seg_head = nn.Sequential(
            ConvBNReLU(c0, c0, k=3, s=1, p=1),
            ConvBNReLU(c0, c0, k=3, s=1, p=1),
            nn.Conv2d(c0, 1, kernel_size=1),
        )

        # GCMF only for classification
        self.cls_refine_d2 = GatedCrossModalFusion(c2, gate_scale=0.25)
        self.cls_refine_s3 = GatedCrossModalFusion(c3, gate_scale=0.25)
        self.cls_refine_s4 = GatedCrossModalFusion(c4, gate_scale=0.25)

        cls_dim = c2 + c3 + c4 if use_multiscale_cls else c4
        self.cls_head = nn.Sequential(
            nn.Linear(cls_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes),
        )

        # depth-guided bbox refinement
        self.box_refiner = DepthGuidedBoxRefiner(
            feat_ch=c4,
            depth_ch=c4,
            hidden_ch=128,
            expand_scale=1.5,
        )
        self.delta_scale = 0.15

    def forward(self, rgb: torch.Tensor, depth: Optional[torch.Tensor] = None):
        if depth is None:
            if rgb.shape[1] != 4:
                raise ValueError("RGB-D model expects either (rgb, depth) or a 4-channel tensor (B,4,H,W).")
            x = rgb
            rgb = x[:, :3]
            depth = x[:, 3:4]

        s0, s1, s2, s3, s4, fd2, fd3, fd4 = self.encoder(rgb, depth)

        # segmentation path same as baseline
        d3 = self.up3(s4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)

        seg_logits = self.seg_head(d0)
        seg_logits = F.interpolate(seg_logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        mask_prob = torch.sigmoid(seg_logits)

        # classification-only GCMF
        cls_d2 = self.cls_refine_d2(d2, fd2)
        cls_s3 = self.cls_refine_s3(s3, fd3)
        cls_s4 = self.cls_refine_s4(s4, fd4)

        if self.use_multiscale_cls:
            mask_d2 = F.interpolate(mask_prob, size=cls_d2.shape[-2:], mode="bilinear", align_corners=False)
            feat_cls_d2 = masked_avg_pool(cls_d2, mask_d2)

            mask_s3 = F.interpolate(mask_prob, size=cls_s3.shape[-2:], mode="bilinear", align_corners=False)
            feat_cls_s3 = masked_avg_pool(cls_s3, mask_s3)

            mask_s4 = F.interpolate(mask_prob, size=cls_s4.shape[-2:], mode="bilinear", align_corners=False)
            feat_cls_s4 = masked_avg_pool(cls_s4, mask_s4)

            feat_cls = torch.cat([feat_cls_d2, feat_cls_s3, feat_cls_s4], dim=1)
        else:
            mask_s4 = F.interpolate(mask_prob, size=cls_s4.shape[-2:], mode="bilinear", align_corners=False)
            feat_cls = masked_avg_pool(cls_s4, mask_s4)

        cls_logits = self.cls_head(feat_cls)

        # bbox:
        # 1) initial bbox from segmentation
        # 2) ROI expansion inside refiner
        # 3) depth-guided correction
        bbox_from_mask = soft_bbox_from_mask(mask_prob, thr=0.25)

        delta = self.delta_scale * torch.tanh(
            self.box_refiner(s4, fd4, bbox_from_mask)
        )
        bbox_pred = (bbox_from_mask + delta).clamp(0.0, 1.0)

        return seg_logits, bbox_pred, cls_logits


# =========================================================
# Model Factory
# =========================================================
def create_model(model_name: str, num_classes: int = 10, width: Optional[int] = None):
    model_name = model_name.lower()

    if model_name == "baseline":
        return MultiTaskHandNetBaseline(
            num_classes=num_classes,
            width=32 if width is None else width,
            use_multiscale_cls=True,
        )

    elif model_name == "logich":
        return MultiTaskHandNetLogicH(
            num_classes=num_classes,
            width=32 if width is None else width,
            use_multiscale_cls=True,
        )

    else:
        raise ValueError(
            f"Unknown model_name: {model_name}. Use 'baseline' or 'LogicH'."
        )
