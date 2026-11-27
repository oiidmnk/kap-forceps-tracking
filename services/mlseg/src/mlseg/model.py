"""Heatmap keypoint net: pretrained timm encoder + light UNet decoder, output at stride 2."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

KEYS = ("left_tip", "right_tip", "left_shadow", "right_shadow", "forceps_junction", "shadow_junction")
# Heatmap channels: one per forceps keypoint, plus two multi-peak channels for the rod (a symmetric capsule, so its
# two end points are unordered): both rod ends in one channel, both ends of the rod's shadow in another.
CHANNELS = KEYS + ("rod_end", "rod_shadow_end")
# Label rows (keypoints) -> channel; the rod channels carry two rows each.
ROW_CH = (0, 1, 2, 3, 4, 5, 6, 6, 7, 7)
ROWS = KEYS + ("rod_a", "rod_b", "rod_shadow_a", "rod_shadow_b")
# horizontal mirror turns "left of the heading" into "right of the heading" (rows)
FLIP = (1, 0, 3, 2, 4, 5, 6, 7, 8, 9)


def conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class KeypointNet(nn.Module):
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True, n_out: int = len(CHANNELS), width: int = 64):
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


def decode_rows(logits: torch.Tensor, nms: int = 6):
    """logits [B,C,H,W] -> per label row: xy [B,R,2] (heatmap px), score [B,R]. Single-point channels give their
    peak; a rod channel gives its two strongest peaks at least `nms` heatmap px apart (unordered)."""
    xy, sc = decode(logits)
    C = logits.shape[1]
    if C <= len(KEYS):
        return xy, sc
    p = torch.sigmoid(logits[:, len(KEYS):].float())
    second = p.clone()
    B, M, H, W = p.shape
    pk = xy[:, len(KEYS):].round().long()
    yy = torch.arange(H, device=p.device)[:, None]
    xx = torch.arange(W, device=p.device)[None, :]
    for b in range(B):
        for m in range(M):
            x, y = pk[b, m]
            second[b, m][((xx - x) ** 2 + (yy - y) ** 2) <= nms ** 2] = 0
    xy2, sc2 = decode(torch.logit(second.clamp(1e-6, 1 - 1e-6)))
    rows_xy = [xy[:, :len(KEYS)]]
    rows_sc = [sc[:, :len(KEYS)]]
    for m in range(M):
        rows_xy.append(torch.stack([xy[:, len(KEYS) + m], xy2[:, m]], 1))
        rows_sc.append(torch.stack([sc[:, len(KEYS) + m], sc2[:, m]], 1))
    return torch.cat(rows_xy, 1), torch.cat(rows_sc, 1)
