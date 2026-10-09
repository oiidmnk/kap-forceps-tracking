"""Pixel masks for the forceps (grey, unsaturated) and its shadow (dark relative to the background)."""
from __future__ import annotations

import cv2
import numpy as np

from .config import Settings


def disc(size: int, radius_frac: float, eye=None) -> np.ndarray:
    """Circular region of interest: the detected eye circle (scaled), else centred with a default radius."""
    d = np.zeros((size, size), np.uint8)
    if eye is not None:
        cv2.circle(d, (int(eye[0]), int(eye[1])), int(eye[2] * radius_frac), 1, -1)
    else:
        cv2.circle(d, (size // 2, size // 2), int(radius_frac * size), 1, -1)
    return d


def colour_angle(bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    """Angle (deg) between pixel and background colour vectors: ~0 for a shadow (same colour, darker),
    large for the forceps (different colour), whatever its saturation."""
    p = cv2.GaussianBlur(bgr.astype(np.float32), (0, 0), 1.5)
    cos = (p * ref_bgr).sum(2) / np.maximum(np.linalg.norm(p, axis=2) * np.linalg.norm(ref_bgr, axis=2), 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def compute_masks(bgr: np.ndarray, ref_v: np.ndarray, from_model: bool, cfg: Settings, eye=None, ref_bgr=None):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    val = cv2.GaussianBlur(hsv[..., 2].astype(np.float32), (0, 0), 1.0)
    ratio = val / np.maximum(ref_v, 1)
    inside = disc(bgr.shape[0], cfg.mask_disc_eye if eye is not None else cfg.mask_disc_radius, eye) > 0
    forceps = ((sat < cfg.forceps_sat_max) & (val < cfg.forceps_val_max) & inside).astype(np.uint8)
    if ref_bgr is not None and cfg.forceps_angle > 0:      # chroma cue (needs the colour background)
        ang = (colour_angle(bgr, ref_bgr) > cfg.forceps_angle) & (ratio < 0.97) & inside
        ang = cv2.morphologyEx(ang.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        forceps = forceps | ang
    thr = cfg.shadow_ratio
    if not from_model:    # no background: ratio is relative to a bright envelope, so re-centre it
        sel = inside & (sat >= cfg.forceps_sat_max)
        thr = cfg.fallback_rel * float(np.median(ratio[sel])) if sel.any() else 0.0
    shadow = ((ratio < thr) & (sat >= cfg.forceps_sat_max) & inside & (forceps == 0)).astype(np.uint8)
    if not from_model:    # thin vessels leak into the mask
        shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return forceps, shadow, ratio
