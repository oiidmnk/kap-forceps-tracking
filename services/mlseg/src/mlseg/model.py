"""Heatmap keypoint net: pretrained timm encoder + light UNet decoder, output at stride 2."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

KEYS = ("left_tip", "right_tip", "left_shadow", "right_shadow", "forceps_junction", "shadow_junction")
# horizontal mirror turns "left of the heading" into "right of the heading"
FLIP = (1, 0, 3, 2, 4, 5)


def conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class KeypointNet(nn.Module):
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True, n_out: int = len(KEYS), width: int = 64):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3, 4))
        ch = self.enc.feature_info.channels()          # strides 2,4,8,16,32
        self.lat = nn.ModuleList([nn.Conv2d(c, width, 1) for c in ch])
        self.dec = nn.ModuleList([conv(width, width) for _ in ch[:-1]])
        self.head = nn.Sequential(conv(width, width), nn.Conv2d(width, n_out, 1))
        nn.init.constant_(self.head[-1].bias, -4.0)    # sparse peaks: start with low probability

    def forward(self, x):
        f = self.enc(x)
        y = self.lat[-1](f[-1])
        for i in range(len(f) - 2, -1, -1):
            y = F.interpolate(y, size=f[i].shape[-2:], mode="bilinear", align_corners=False) + self.lat[i](f[i])
            y = self.dec[i](y)
        return self.head(y)                             # logits, [B, K, H/2, W/2]


def decode(logits: torch.Tensor, win: int = 3):
    """logits [B,K,H,W] -> xy [B,K,2] (heatmap px, sub-pixel centroid around the peak), score [B,K]."""
    p = torch.sigmoid(logits.float())
    B, K, H, W = p.shape
    score, idx = p.flatten(2).max(-1)
    py, px = idx // W, idx % W
    r = win
    ys = torch.arange(-r, r + 1, device=p.device)
    out = torch.zeros(B, K, 2, device=p.device)
    pad = F.pad(p, (r, r, r, r))
    for b in range(B):
        for k in range(K):
            patch = pad[b, k, py[b, k]:py[b, k] + 2 * r + 1, px[b, k]:px[b, k] + 2 * r + 1]
            w = patch.sum().clamp_min(1e-6)
            out[b, k, 0] = px[b, k] + (patch.sum(0) * ys).sum() / w
            out[b, k, 1] = py[b, k] + (patch.sum(1) * ys).sum() / w
    return out, score
