"""Jaw geometry: tips + junction of a two-jaw tool from a binary mask."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CW, CH, TX, TY = 700, 600, 60, 300     # canvas for the tool-aligned (tip-left) view
SCAN = 320                             # how far from the tip we look for the fork


@dataclass
class Jaws:
    ok: bool
    reason: str = ""
    left_tip: tuple[float, float] | None = None
    right_tip: tuple[float, float] | None = None
    junction: tuple[float, float] | None = None
    angle_deg: float = 0.0           # tool direction (tip -> base) in image
    jaw_len: float = 0.0
    spread: float = 0.0


def _largest(mask: np.ndarray):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None
    return (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)


def _runs(col: np.ndarray):
    d = np.diff(np.concatenate([[0], col, [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def _rot(angle: float, c):
    m = cv2.getRotationMatrix2D(c, angle, 1.0)
    m[0, 2] += TX - c[0]
    m[1, 2] += TY - c[1]
    return m


def _axis(R: np.ndarray, x0: int):
    pts = []
    for x in range(x0 + 90, x0 + 260, 3):
        rr = _runs(R[:, x].astype(np.uint8))
        if rr:
            a, b = max(rr, key=lambda r: r[1] - r[0])
            pts.append((x, (a + b) / 2))
    if len(pts) < 8:
        return None
    pts = np.array(pts)
    return np.polyfit(pts[:, 0], pts[:, 1], 1)


def analyze(mask: np.ndarray, min_area: int = 800) -> Jaws:
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cc = _largest(m)
    if cc is None or cc.sum() < min_area:
        return Jaws(False, "tool not visible")
    h, w = cc.shape
    ys, xs = np.nonzero(cc)
    P = np.stack([xs, ys], 1).astype(np.float32)
    c = P.mean(0)
    Q = P - c
    evals, evecs = np.linalg.eigh(Q.T @ Q)
    d = evecs[:, -1]                                # principal axis (2x2 eigenproblem, cheap)
    if np.sqrt(evals[-1] / max(evals[0], 1e-9)) < 4:     # a tool is long and thin, not a blob
        return Jaws(False, "blob is not tool-shaped")
    proj = (P - c) @ d
    # the tool enters from the rim, so the tip is the end closer to the image centre
    centre = np.array([w / 2, h / 2])
    if np.linalg.norm(c + d * proj.max() - centre) < np.linalg.norm(c + d * proj.min() - centre):
        d, proj = -d, -proj
    tip0 = P[np.argmin(proj)]
    c0 = (float(tip0[0]), float(tip0[1]))
    ang = float(np.degrees(np.arctan2(d[1], d[0])))
    k = None
    for _ in range(3):                       # refine the axis angle on the straight shaft
        M = _rot(ang, c0)
        R = cv2.warpAffine(cc * 255, M, (CW, CH), flags=cv2.INTER_NEAREST) > 127
        if not R.any():
            return Jaws(False, "tool out of view")
        x0 = int(np.nonzero(R.any(0))[0].min())
        k = _axis(R, x0)
        if k is None:
            return Jaws(False, "shaft too short")
        ang += float(np.degrees(np.arctan(k[0])))
    Minv = cv2.invertAffineTransform(M)
    to_img = lambda x, y: tuple(float(v) for v in Minv @ np.array([x, y, 1.0]))

    runs = {x: _runs(R[:, x].astype(np.uint8)) for x in range(x0, min(x0 + SCAN, CW))}
    runs = {x: r for x, r in runs.items()}
    two = sorted(x for x in runs if len(runs[x]) >= 2)
    if not two:
        return Jaws(False, "no fork (jaws closed?)", angle_deg=ang)
    blocks = [[two[0]]]
    for x in two[1:]:
        (blocks[-1].append(x) if x - blocks[-1][-1] <= 3 else blocks.append([x]))
    blk = max(blocks, key=len)
    if len(blk) < 8:
        return Jaws(False, "fork too short", angle_deg=ang)
    xj = blk[-1]
    kg = np.polyfit(blk, [(runs[x][0][1] + runs[x][1][0]) / 2 for x in blk], 1)

    def jaw_tip(sign):
        pts = sorted((x, (a + b) / 2) for x in range(x0, xj + 1) for a, b in runs[x]
                     if ((a + b) / 2 - (kg[0] * x + kg[1])) * sign > 0)
        if len(pts) < 3:
            return None
        return pts[0][0], float(np.mean([p[1] for p in pts[:3]]))

    up, dn = jaw_tip(-1), jaw_tip(+1)
    if up is None or dn is None:
        return Jaws(False, "jaw missing", angle_deg=ang)
    jx = xj + 1
    # in the tip-left view "left" of the heading is +y (image down)
    return Jaws(True, "", left_tip=to_img(*dn), right_tip=to_img(*up),
                junction=to_img(jx, float(np.polyval(kg, jx))), angle_deg=ang,
                jaw_len=float(jx - min(up[0], dn[0])), spread=float(dn[1] - up[1]))
