"""LabelMe point frames -> augmented (image, heatmap) pairs."""
from __future__ import annotations

import glob
import json
import math
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .model import FLIP, KEYS
from .relight import DiscRelight
from .shadowreloc import ShadowBank

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def load_frames(folder: str, kind: str | None = None):
    """gen_pseudo.py output -> list of (name, bgr, pts [K,2] px, state [K]); state 1 present / 0 absent / -1 ignore."""
    out = []
    for jf in sorted(glob.glob(os.path.join(folder, "*.json"))):
        d = json.load(open(jf))
        if kind and d["kind"] != kind:
            continue
        img = cv2.imread(jf[:-4] + "png", cv2.IMREAD_COLOR)
        out.append((os.path.basename(jf)[:-5], img, np.array(d["pts"], np.float32), np.array(d["state"], np.int8)))
    return out


def to_tensor(img: np.ndarray) -> torch.Tensor:
    x = (img[..., ::-1].astype(np.float32) - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def heatmaps(pts: np.ndarray, size: int, sigma: float, valid=None) -> np.ndarray:
    """valid[i] False -> empty map (used for absent keypoints and for ones rotated out of the frame)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    hm = np.zeros((len(pts), size, size), np.float32)
    for i, (x, y) in enumerate(pts):
        if valid is not None and not valid[i]:
            continue
        hm[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        hm[i, int(round(y)), int(round(x))] = 1.0       # exactly one positive pixel per keypoint (focal loss)
    return hm


def affine(src: int, dst: int, angle: float, scale: float, shift, flip: bool) -> np.ndarray:
    """2x3 matrix mapping a src x src image into dst x dst with rotation about the image centre."""
    c = src / 2
    m = cv2.getRotationMatrix2D((c, c), angle, scale * dst / src)
    m[0, 2] += dst / 2 - c + shift[0]
    m[1, 2] += dst / 2 - c + shift[1]
    if flip:
        f = np.array([[-1, 0, dst], [0, 1, 0]], np.float32)
        m = (np.vstack([f, [0, 0, 1]]) @ np.vstack([m, [0, 0, 1]]))[:2]
    return m


class Aug:
    def __init__(self, size=768, stride=2, sigma=3.0, train=True, relight: DiscRelight | None = None, relight_p=0.5,
                 bank: ShadowBank | None = None, reloc_p=0.4):
        self.size, self.stride, self.sigma, self.train = size, stride, sigma, train
        self.relight, self.relight_p = relight, relight_p
        self.bank, self.reloc_p = bank, reloc_p

    def __call__(self, img, pts, rng: np.random.Generator, state=None, name=None):
        S, n = img.shape[0], self.size
        if self.train and self.bank is not None and name in self.bank and rng.random() < self.reloc_p:
            img, pts, state = self.bank.apply(name, rng)
        if self.train and self.relight is not None and rng.random() < self.relight_p:
            st = np.ones(len(pts)) if state is None else state
            tips = [q for q in (2, 3) if st[q] == 1] or [q for q in range(4) if st[q] == 1]
            if tips and rng.random() < 0.8:          # disc under a shadow tip: the hard case
                c = pts[tips[rng.integers(len(tips))]] + rng.uniform(-45, 45, 2)
            else:
                c = rng.uniform(150, S - 150, 2)
            img = self.relight.apply(img, c, rng.uniform(0.8, 1.4), rng.uniform(0, 360), rng.uniform(0.7, 1.0))
        if self.train:
            flip = bool(rng.random() < 0.5)
            m = affine(S, n, rng.uniform(-180, 180), rng.uniform(0.75, 1.3), rng.uniform(-0.06, 0.06, 2) * n, flip)
        else:
            flip = False
            m = affine(S, n, 0, 1, (0, 0), False)
        out = cv2.warpAffine(img, m, (n, n), flags=cv2.INTER_AREA if S > n else cv2.INTER_LINEAR)
        p = np.concatenate([pts, np.ones((len(pts), 1), np.float32)], 1) @ m.T
        state = np.ones(len(pts), np.int8) if state is None else state
        if flip:
            p, state = p[list(FLIP)], state[list(FLIP)]
        if self.train:
            out = self.photometric(out, rng)
        inside = (p >= 0).all(1) & (p < n - self.stride).all(1)
        hm = heatmaps(p / self.stride, n // self.stride, self.sigma, inside & (state == 1))
        # loss mask per channel: ignored keypoints are masked; a present one rotated out of frame is a true negative
        mask = (state != -1).astype(np.float32)
        return out, hm, p.astype(np.float32), mask

    @staticmethod
    def photometric(img, rng):
        x = img.astype(np.float32)
        x = x * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20)
        x = x * rng.uniform(0.9, 1.1, 3)                          # per-channel gain
        if rng.random() < 0.3:
            x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 2.0))
        if rng.random() < 0.3:
            x = x + rng.normal(0, rng.uniform(1, 6), x.shape)
        g = rng.uniform(0.8, 1.25)
        return (255 * (np.clip(x, 0, 255) / 255) ** g).astype(np.uint8)


class PointDataset(Dataset):
    def __init__(self, frames, aug: Aug, repeat=1, seed=0):
        self.frames, self.aug, self.repeat, self.seed = frames, aug, repeat, seed

    def __len__(self):
        return len(self.frames) * self.repeat

    def __getitem__(self, i):
        name, img, pts, state = self.frames[i % len(self.frames)]
        rng = np.random.default_rng((self.seed, i, torch.initial_seed() % 2**31))
        out, hm, p, mask = self.aug(img, pts, rng, state, name)
        return to_tensor(out), torch.from_numpy(hm), torch.from_numpy(p), torch.from_numpy(mask)
