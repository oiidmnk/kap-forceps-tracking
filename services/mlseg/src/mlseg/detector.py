"""Learned drop-in for `detector.pipeline.Detector`: same `.detect()` / `.roi` / `_prepare()` contract."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from detector.config import Settings
from detector.pipeline import Detection, Detector, ToolPose

from .infer import Segmenter

TOOLS = (("forceps", ("left_tip", "right_tip")), ("shadow", ("left_tip", "right_tip")))


@dataclass
class MLToolPose(ToolPose):
    scores: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {**super().as_dict(), "scores": {k: round(float(v), 3) for k, v in self.scores.items()}}


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
                 shadow_min_score: float = 0.1):
        super().__init__(cfg or Settings(background_path=""))
        self.bg = _NoBackground()
        self.seg = Segmenter(ckpt, device, min_score)
        self.shadow_min = shadow_min_score         # low confidence != wrong position for shadow tips (measured), see README

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
        self.view = work if self.roi is not None else frame
        size = work.shape[1::-1] if self.roi is not None else frame.shape[1::-1]
        ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
        return Detection(poses[0], poses[1], size, "learned", ms)
