"""Classical rod / rod-shadow helpers that produce the labels for the learned model (no reference background).

The rod is a saturated red capsule (G/R ~0.03 vs. ~0.35 for retina and vessels): `find_rod` is reliable on its own.
Its shadow is a dark capsule - dark relative to the frame's own local brightness once a morphological closing has
removed the thin vessels. A global search for it confuses it with the forceps shaft and the forceps shadow, so
`fit_shadow` only snaps end points read by eye (+-15 px) onto the capsule. Both are two end points (cap apexes along
the long axis), unordered.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Capsule:
    ends: np.ndarray        # [2,2] px
    width: float
    length: float
    fill: float             # component area / (length*width): ~0.9 for a clean capsule

    @property
    def centre(self):
        return self.ends.mean(0)


def _capsule(mask: np.ndarray) -> Capsule | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return None
    p = np.stack([xs, ys], 1).astype(np.float32)
    c = p.mean(0)
    u, s, vt = np.linalg.svd(p - c, full_matrices=False)
    d = vt[0]
    t = (p - c) @ d
    n = (p - c) @ vt[1]
    lo, hi = np.percentile(t, 0.3), np.percentile(t, 99.7)
    w = float(np.percentile(n, 98) - np.percentile(n, 2))
    L = float(hi - lo)
    return Capsule(np.stack([c + lo * d, c + hi * d]), w, L, len(xs) / max(L * w, 1.0))


def rod_mask(bgr: np.ndarray) -> np.ndarray:
    b, g, r = (bgr[..., i].astype(np.float32) for i in range(3))
    m = ((g < 0.13 * r + 4) & (r > 45) & (r > 1.5 * b)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def darkness(bgr: np.ndarray, vessel: int = 13, bg: int = 121) -> np.ndarray:
    """V / local background V after removing dark structures thinner than `vessel` px (vessels)."""
    v = bgr.max(2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (vessel, vessel))
    vc = cv2.morphologyEx(v, cv2.MORPH_CLOSE, k).astype(np.float32)
    small = cv2.resize(vc, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    kb = max(3, bg // 8 | 1)
    ref = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb)))   # local bright level
    ref = cv2.GaussianBlur(ref, (0, 0), kb / 2)
    ref = cv2.resize(ref, v.shape[::-1], interpolation=cv2.INTER_LINEAR)
    return vc / np.maximum(ref, 1.0)


def retina_mask(bgr: np.ndarray) -> np.ndarray:
    v = cv2.GaussianBlur(bgr.max(2), (0, 0), 9)
    m = (v > 45).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return m
    m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    return cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))


def find_rod(bgr: np.ndarray) -> Capsule | None:
    m = rod_mask(bgr)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    if st[i, cv2.CC_STAT_AREA] < 150:
        return None
    return _capsule(lab == i)


def rod_occluded(bgr: np.ndarray, rod: Capsule, gap: int = 14) -> bool:
    """An end next to the forceps (it may continue underneath), or the rod split in two by the forceps."""
    grey = cv2.dilate(_grey_mask(bgr), np.ones((5, 5), np.uint8))
    h, w = grey.shape
    for x, y in rod.ends.astype(int):
        if grey[max(0, y - gap):y + gap + 1, max(0, x - gap):x + gap + 1].mean() > 0.08:
            return True
    n, _, st, _ = cv2.connectedComponentsWithStats(rod_mask(bgr))
    areas = sorted(st[1:, cv2.CC_STAT_AREA], reverse=True)
    return len(areas) > 1 and areas[1] > 80


def _capsule_kernels(L: float, w: float, theta: float, ring: float):
    """inside / surround masks of a capsule (length L between the cap apexes, width w) at angle theta, centred."""
    half = int(np.ceil(L / 2 + w / 2 + ring + 2))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    c, s = np.cos(theta), np.sin(theta)
    t, n = xx * c + yy * s, -xx * s + yy * c
    seg = np.clip(t, -(L - w) / 2, (L - w) / 2)                 # distance to the capsule's core segment
    d = np.hypot(t - seg, n)
    inside = (d < 0.38 * w).astype(np.float32)                  # the shadow's dark core (its edge is soft)
    out = ((d > 0.75 * w) & (d < 0.75 * w + ring)).astype(np.float32)
    return inside, out


def fit_shadow(bgr: np.ndarray, a, b, width: float, shift: float = 16, dang: float = 12, scale: float = 0.5):
    """Snap a rough pair of shadow end points (read by eye, +-15 px) onto the dark capsule.
    -> (ends [2,2] or None, score). Rod/forceps pixels are 'unknown' (normalised convolution)."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    dk = darkness(bgr)
    unknown = cv2.dilate(rod_mask(bgr) | _grey_mask(bgr), np.ones((7, 7), np.uint8))
    valid = ((retina_mask(bgr) > 0) & (unknown == 0)).astype(np.float32)
    D = np.clip(1.0 - dk, 0, 0.7) * valid
    L0 = float(np.hypot(*(b - a)))
    th0 = float(np.arctan2(*(b - a)[::-1]))
    c0 = (a + b) / 2
    r = int(L0 / 2 + width + shift + 20)
    x0, y0 = int(c0[0]) - r, int(c0[1]) - r
    pad = ((max(0, -y0), max(0, y0 + 2 * r + 1 - D.shape[0])), (max(0, -x0), max(0, x0 + 2 * r + 1 - D.shape[1])))
    Dp, Wp = np.pad(D, pad), np.pad(valid, pad)
    ox, oy = x0 + pad[1][0], y0 + pad[0][0]
    Dc = cv2.resize(Dp[oy:oy + 2 * r + 1, ox:ox + 2 * r + 1], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    Wc = cv2.resize(Wp[oy:oy + 2 * r + 1, ox:ox + 2 * r + 1], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    H = Dc.shape[0]
    cc = (H - 1) / 2
    yy, xx = np.mgrid[0:H, 0:H]
    near = np.hypot(xx - cc, yy - cc) <= shift * scale
    w = max(width, 10) * scale
    best = (-9.0, None)
    for L in L0 * np.array([0.85, 0.92, 1.0, 1.08, 1.16]):
        for th in th0 + np.deg2rad(np.arange(-dang, dang + 0.1, 3)):
            ki, ko = _capsule_kernels(L * scale, w, th, max(3.0, 10 * scale))
            si, wi = cv2.filter2D(Dc, -1, ki), cv2.filter2D(Wc, -1, ki)
            so, wo = cv2.filter2D(Dc, -1, ko), cv2.filter2D(Wc, -1, ko)
            ok = near & (wi > 0.4 * ki.sum()) & (wo > 0.3 * ko.sum())
            sc = np.where(ok, si / np.maximum(wi, 1e-3) - so / np.maximum(wo, 1e-3), -9)
            k = int(np.argmax(sc))
            if sc.flat[k] > best[0]:
                best = (float(sc.flat[k]), (np.array([x0 + (k % H) / scale, y0 + (k // H) / scale], np.float32), th, L))
    if best[1] is None:
        return None, best[0]
    c, th, L = best[1]
    ends = _refine_ends(D, valid, c, th, L, width)
    return (None if ends is None else np.stack(ends)), best[0]


def _refine_ends(D, valid, c, th, L, w):
    """Cap apexes from the deficit profile along the axis: where it falls to half between core and surround."""
    d = np.array([np.cos(th), np.sin(th)], np.float32)
    nrm = np.array([-d[1], d[0]], np.float32)
    ts = np.arange(-L, L + 0.5, 1.0, dtype=np.float32)
    off = np.linspace(-0.25 * w, 0.25 * w, 5, dtype=np.float32)
    pts = c[None, None] + ts[:, None, None] * d + off[None, :, None] * nrm
    mx = pts[..., 0].astype(np.float32)
    my = pts[..., 1].astype(np.float32)
    prof = cv2.remap(D, mx, my, cv2.INTER_LINEAR, borderValue=0)
    pw = cv2.remap(valid, mx, my, cv2.INTER_LINEAR, borderValue=0)
    prof = (prof * pw).sum(1) / np.maximum(pw.sum(1), 1e-3)
    seen = pw.sum(1) > 0.5 * len(off)
    core = np.abs(ts) < 0.3 * L
    if not (core & seen).any():
        return None
    hi = np.median(prof[core & seen])
    lo = np.median(prof[(np.abs(ts) > 0.8 * L) & seen]) if ((np.abs(ts) > 0.8 * L) & seen).any() else 0.0
    half = (hi + lo) / 2
    ends = []
    for sgn in (-1, 1):
        tt = None
        for k in (np.where(ts >= 0)[0] if sgn > 0 else np.where(ts <= 0)[0][::-1]):     # walk outward from the centre
            if not seen[k]:
                continue                         # hidden under rod/forceps: keep walking
            if prof[k] < half:
                tt = ts[k]
                break
        if tt is None:
            return None
        ends.append(c + tt * d)
    return ends


def _grey_mask(bgr: np.ndarray) -> np.ndarray:
    """Forceps: unsaturated grey/dark metal."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 1] < 90) & (hsv[..., 2] > 40)).astype(np.uint8)
