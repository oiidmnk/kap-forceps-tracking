"""Learned drop-in for `detector.pipeline.Detector`: same `.detect()` / `.roi` / `_prepare()` contract."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from detector.config import Settings
from detector.pipeline import Detection, Detector, RodPose, ToolPose

from .infer import Segmenter

TOOLS = (("forceps", ("left_tip", "right_tip")), ("shadow", ("left_tip", "right_tip")))


@dataclass
class MLToolPose(ToolPose):
    scores: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {**super().as_dict(), "scores": {k: round(float(v), 3) for k, v in self.scores.items()}}


def pair_rod_shadow(ends: np.ndarray, shadow: np.ndarray) -> np.ndarray:
    """Order the two (unordered) shadow ends so that shadow[i] is cast by ends[i]. A shadow is the rod pushed away
    from the light, so both ends move by about the same offset; the wrong pairing differs by ~2x the rod."""
    keep = np.linalg.norm((shadow[0] - ends[0]) - (shadow[1] - ends[1]))
    swap = np.linalg.norm((shadow[1] - ends[0]) - (shadow[0] - ends[1]))
    return shadow if keep <= swap else shadow[::-1]


class _NoBackground:
    """The learned model needs no empty-retina reference; satisfies the live worker's background hooks."""
    state, v, bgr, disc = "static", None, None, None

    def reset(self): pass
    def add_sample(self, *_a, **_k): return True
    def adapt(self, *_a, **_k): pass
    def set_image(self, *_a, **_k): pass
    def reference(self, bgr): return None, False


class MLDetector(Detector):
    def __init__(self, ckpt: str, cfg=None, device: str | None = None, min_score: float = 0.3,
                 shadow_min_score: float = 0.1, rod_min_score: float = 0.3, rod_shadow_min_score: float = 0.2,
                 shadow_hold_s: float = 1.0):
        super().__init__(cfg or Settings(background_path=""))
        self.bg = _NoBackground()
        self.seg = Segmenter(ckpt, device, min_score)
        self.shadow_min = shadow_min_score         # low confidence != wrong position for shadow tips (measured), see README
        self.rod_min, self.rod_shadow_min = rod_min_score, rod_shadow_min_score
        # The forceps shadow vanishes for a moment when it runs into the rod's shadow (grasping): keep the last
        # tip -> shadow offset for this long instead of dropping the forceps (reported with reason "held").
        self.shadow_hold_s = shadow_hold_s
        self._offset: tuple[np.ndarray, float] | None = None

    def detect(self, frame: np.ndarray, adapt: bool = False, prepared=None) -> Detection:
        t0 = cv2.getTickCount()
        work, back = prepared or self._prepare(frame)
        pts, score = (a[0] for a in self.seg([work]))
        poses = []
        for t, (_, keys) in enumerate(TOOLS):
            p = {}
            sc = {}
            for j, k in enumerate(keys):
                i = 2 * t + j
                sc[k] = float(score[i])
                if score[i] >= (self.shadow_min if t else self.seg.min_score):
                    p[k] = tuple(float(v) for v in back(pts[i]))
            jn = None
            if len(score) > 4:                              # 6-channel model: junction channel 4 (forceps) / 5 (shadow)
                sc["junction"] = float(score[4 + t])
                if score[4 + t] >= (self.shadow_min if t else self.seg.min_score) and len(p) == 2:
                    jn = tuple(float(v) for v in back(pts[4 + t]))
            ok = len(p) == 2
            poses.append(MLToolPose(ok, "" if ok else ("tool not visible" if not p else "one jaw tip missing"),
                                    p.get("left_tip"), p.get("right_tip"), jn, sc))
        poses[1] = self._hold_shadow(poses[0], poses[1])
        rod = self._rod(pts, score, back) if len(score) >= 10 else None
        self.view = work if self.roi is not None else frame
        size = work.shape[1::-1] if self.roi is not None else frame.shape[1::-1]
        ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
        return Detection(poses[0], poses[1], size, "learned", ms, rod)

    def _hold_shadow(self, forceps: MLToolPose, shadow: MLToolPose) -> MLToolPose:
        now = time.monotonic()
        if forceps.ok and shadow.ok and shadow.reason != "held":
            self._offset = (np.subtract([shadow.left_tip, shadow.right_tip], [forceps.left_tip, forceps.right_tip]), now)
            return shadow
        if not forceps.ok or self._offset is None or now - self._offset[1] > self.shadow_hold_s:
            return shadow
        off = self._offset[0]
        tip = lambda p, o: (float(p[0] + o[0]), float(p[1] + o[1]))
        return MLToolPose(True, "held", tip(forceps.left_tip, off[0]), tip(forceps.right_tip, off[1]), None,
                          shadow.scores)

    def _rod(self, pts, score, back) -> RodPose:
        """Rows 6,7: rod ends; 8,9: rod-shadow ends (two peaks per channel). A part is reported when both its
        peaks pass the threshold; the shadow is paired to the rod ends."""
        sc = {"rod": float(score[6:8].min()), "rod_shadow": float(score[8:10].min())}
        ends = np.array([back(p) for p in pts[6:8]], np.float64) if sc["rod"] >= self.rod_min else None
        sh = np.array([back(p) for p in pts[8:10]], np.float64) if sc["rod_shadow"] >= self.rod_shadow_min else None
        if ends is not None and sh is not None:
            sh = pair_rod_shadow(ends, sh)
        tup = lambda a: None if a is None else tuple(tuple(float(v) for v in p) for p in a)
        return RodPose(tup(ends), tup(sh), sc)
