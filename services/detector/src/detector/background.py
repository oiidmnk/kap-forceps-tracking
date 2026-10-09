"""Background (tool-free retina) model: static file, or learned online from live footage."""
from __future__ import annotations

import cv2
import numpy as np


def value_channel(bgr: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32), (0, 0), 1.5)


def local_max_value(bgr: np.ndarray, window: int = 161) -> np.ndarray:
    """Single-frame fallback: bright envelope of V (tools/shadows are darker than their surroundings)."""
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2]
    small = cv2.resize(v, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    k = max(3, window // 4) | 1
    env = cv2.GaussianBlur(cv2.dilate(small, np.ones((k, k), np.uint8)).astype(np.float32), (0, 0), k / 3)
    return cv2.resize(env, (v.shape[1], v.shape[0]), interpolation=cv2.INTER_LINEAR)


def find_eye_disc(bgr: np.ndarray):
    """Retina circle (cx, cy, r) in px: the largest *bright*, saturated blob. The lens ring around it is darker,
    so the brightness threshold is relative to the frame's bright end (p98) instead of absolute."""
    hsv = cv2.cvtColor(cv2.resize(bgr, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    vmin = max(60.0, 0.68 * float(np.percentile(v, 98)))
    m = ((hsv[..., 1] > 90) & (v > vmin)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    cnt, _ = cv2.findContours((lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8),
                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (x, y), r = cv2.minEnclosingCircle(max(cnt, key=cv2.contourArea))
    return 4 * x, 4 * y, 4 * r


class Background:
    """Holds the reference V channel. States: `static` (loaded), `learning` (bootstrapping), `none`."""

    def __init__(self, path: str | None = None, size: int = 1080):
        self.size = size
        self.v: np.ndarray | None = None
        self.bgr: np.ndarray | None = None          # float32 colour background (for chromaticity tests)
        self.state = "none"
        self.disc: tuple[float, float, float] | None = None   # eye circle found in the background
        self._samples: list[np.ndarray] = []
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.set_image(img)

    def set_image(self, bgr: np.ndarray) -> None:
        if bgr.shape[0] != self.size or bgr.shape[1] != self.size:
            bgr = cv2.resize(bgr, (self.size, self.size))
        self.v = value_channel(bgr)
        self.bgr = cv2.GaussianBlur(bgr.astype(np.float32), (0, 0), 1.5)
        self.disc = find_eye_disc(bgr)
        self.state = "static"

    def reset(self) -> None:
        self.v, self.bgr, self.state, self.disc, self._samples = None, None, "none", None, []

    # -- online learning -------------------------------------------------
    def add_sample(self, bgr: np.ndarray, n_needed: int = 30) -> bool:
        """Collect frames; once enough exist build the background as a per-pixel bright percentile."""
        self.state = "learning"
        self._samples.append(bgr.copy())
        if len(self._samples) < n_needed:
            return False
        stack = np.stack(self._samples)
        self.set_image(np.percentile(stack, 80, axis=0).astype(np.uint8))
        self._samples = []
        return True

    def adapt(self, bgr: np.ndarray, tool_mask: np.ndarray, alpha: float = 0.02) -> None:
        """Slowly follow illumination changes, only where no tool/shadow is."""
        if self.v is None:
            return
        self._n = getattr(self, "_n", 0) + 1
        if self._n % 3:                          # 10 updates/s are plenty for slow drift (and 3x cheaper)
            return
        free = cv2.erode(1 - tool_mask, np.ones((9, 9), np.uint8))
        cv2.accumulateWeighted(value_channel(bgr), self.v, 3 * alpha, free)
        cv2.accumulateWeighted(bgr, self.bgr, 3 * alpha, free)

    def reference(self, bgr: np.ndarray) -> tuple[np.ndarray, bool]:
        """Returns (reference V, is_background_model)."""
        if self.v is not None:
            return self.v, True
        return local_max_value(bgr), False
