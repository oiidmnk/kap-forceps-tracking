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


def isolate_tool(mask: np.ndarray, core: int = 13, half_width: int = 56, min_elong: float = 3.0) -> np.ndarray:
    """Shadow masks get glued to vessels and blotchy background, so the largest blob is no longer tool-shaped.
    The thick shaft survives an opening; its axis then defines a strip in which we keep the original mask
    (jaws included, being thin and attached to the shaft). Without a thick elongated shaft there is no shadow
    (vessel debris alone must not pass as a tool): returns an empty mask.
    The strip width matters: narrower cuts jaw tips, wider lets vessels back in (tuned on the sample video)."""
    small = cv2.resize(mask * 255, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) > 127          # half-res: 4x cheaper
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core // 2 | 1, core // 2 | 1))
    opened = cv2.morphologyEx(small.astype(np.uint8), cv2.MORPH_OPEN, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    best, best_score = None, 0.0
    for i in np.argsort(st[1:, cv2.CC_STAT_AREA])[::-1][:6] + 1:
        if st[i, cv2.CC_STAT_AREA] < 75:
            break
        ys, xs = np.nonzero(lab == i)
        P = 2 * np.stack([xs, ys], 1).astype(np.float32)
        c = P.mean(0)
        ev, evec = np.linalg.eigh(np.cov((P - c).T))
        elong = float(np.sqrt(ev[1] / max(ev[0], 1e-9)))
        score = elong * float(np.sqrt(ev[1]))             # long *and* extended, not just a thin sliver
        if elong >= min_elong and score > best_score:
            best, best_score = (c, evec[:, 1]), score
    if best is None:
        return np.zeros_like(mask)
    c, d = best
    ys, xs = np.nonzero(mask)
    off = np.stack([xs, ys], 1).astype(np.float32) - c
    keep = np.abs(off[:, 0] * d[1] - off[:, 1] * d[0]) <= half_width
    out = np.zeros_like(mask)
    out[ys[keep], xs[keep]] = 1
    return out

