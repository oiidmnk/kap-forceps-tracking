"""Pixel masks for the forceps (grey, unsaturated) and its shadow (dark relative to the background)."""
from __future__ import annotations

import cv2
import numpy as np

from .config import Settings


def disc(size: int, radius_frac: float) -> np.ndarray:
    d = np.zeros((size, size), np.uint8)
    cv2.circle(d, (size // 2, size // 2), int(radius_frac * size), 1, -1)
    return d


def compute_masks(bgr: np.ndarray, ref_v: np.ndarray, from_model: bool, cfg: Settings):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    val = cv2.GaussianBlur(hsv[..., 2].astype(np.float32), (0, 0), 1.0)
    ratio = val / np.maximum(ref_v, 1)
    inside = disc(bgr.shape[0], cfg.mask_disc_radius) > 0
    forceps = ((sat < cfg.forceps_sat_max) & (val < cfg.forceps_val_max) & inside).astype(np.uint8)
    thr = cfg.shadow_ratio
    if not from_model:    # no background: ratio is relative to a bright envelope, so re-centre it
        thr = cfg.fallback_rel * float(np.median(ratio[inside & (sat >= cfg.forceps_sat_max)]))
    shadow = ((ratio < thr) & (sat >= cfg.forceps_sat_max) & inside).astype(np.uint8)
    if not from_model:    # thin vessels leak into the mask
        shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return forceps, shadow, ratio
