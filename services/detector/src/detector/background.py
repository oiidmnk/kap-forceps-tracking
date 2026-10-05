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


class Background:
    """Holds the reference V channel. States: `static` (loaded), `learning` (bootstrapping), `none`."""

    def __init__(self, path: str | None = None, size: int = 1080):
        self.size = size
        self.v: np.ndarray | None = None
        self.state = "none"
        self._samples: list[np.ndarray] = []
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.set_image(img)

    def set_image(self, bgr: np.ndarray) -> None:
        if bgr.shape[0] != self.size or bgr.shape[1] != self.size:
            bgr = cv2.resize(bgr, (self.size, self.size))
        self.v = value_channel(bgr)
        self.state = "static"

    def reset(self) -> None:
        self.v, self.state, self._samples = None, "none", []

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
        v = value_channel(bgr)
        free = cv2.erode(1 - tool_mask, np.ones((9, 9), np.uint8)) > 0
        self.v[free] += alpha * (v[free] - self.v[free])

    def reference(self, bgr: np.ndarray) -> tuple[np.ndarray, bool]:
        """Returns (reference V, is_background_model)."""
        if self.v is not None:
            return self.v, True
        return local_max_value(bgr), False
