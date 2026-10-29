"""Optic-disc relighting augmentation.
A shadow darkens whatever lies below it multiplicatively, so a bright disc under a shadow is `img * gain`, where
`gain` is the disc's brightness/colour relative to the surrounding retina. The gain is cut out of the empty background
image and multiplied onto training frames, preferably at the shadow tips (the failure case: tips on the disc)."""
from __future__ import annotations

import cv2
import numpy as np


class DiscRelight:
    def __init__(self, background_bgr: np.ndarray):
        bg = background_bgr
        hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 0] > 8) & (hsv[..., 0] < 30) & (hsv[..., 1] > 80) & (hsv[..., 2] > 150)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        cx, cy = cen[k]
        r = float(np.sqrt(st[k, cv2.CC_STAT_AREA] / np.pi))
        core = cv2.dilate((lab == k).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        surround = cv2.GaussianBlur(cv2.inpaint(bg, core * 255, 9, cv2.INPAINT_TELEA).astype(np.float32), (0, 0), 25)
        gain = np.clip(bg.astype(np.float32) / np.maximum(surround, 1), 0.5, 3.0)
        R = int(round(r * 3))                                    # patch half size
        x0, y0 = int(cx) - R, int(cy) - R
        self.gain = gain[y0:y0 + 2 * R, x0:x0 + 2 * R].copy()
        yy, xx = np.mgrid[-R:R, -R:R]
        d = np.sqrt(xx ** 2 + yy ** 2)
        self.soft = np.clip((R - d) / (0.35 * R), 0, 1).astype(np.float32)      # fades to 1.0 (no change) at the border
        self.R = R

    def apply(self, img: np.ndarray, centre, scale: float = 1.0, angle: float = 0.0, strength: float = 1.0) -> np.ndarray:
        """Multiply the disc gain onto `img` (uint8 BGR) centred at `centre` (x, y); returns a new image."""
        R = self.R
        M = cv2.getRotationMatrix2D((R, R), angle, scale)
        M[0, 2] += centre[0] - R
        M[1, 2] += centre[1] - R
        h, w = img.shape[:2]
        g = cv2.warpAffine(self.gain - 1.0, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        s = cv2.warpAffine(self.soft, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        out = img.astype(np.float32) * (1.0 + strength * g * s[..., None])
        return np.clip(out, 0, 255).astype(np.uint8)
