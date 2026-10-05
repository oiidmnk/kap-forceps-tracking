"""Frame -> Detection: tips of forceps and shadow plus their jaw junctions, in original pixel coords."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .background import Background
from .config import Settings
from .geometry import Jaws, analyze
from .masks import compute_masks, disc

POINTS = ("left_tip", "right_tip", "junction")
# colours (BGR)
COL = {"left_tip": (255, 255, 0), "right_tip": (255, 0, 255), "junction": (0, 255, 255)}
LINE = {"forceps": (0, 255, 0), "shadow": (255, 160, 0)}


@dataclass
class ToolPose:
    ok: bool
    reason: str = ""
    left_tip: tuple[float, float] | None = None
    right_tip: tuple[float, float] | None = None
    junction: tuple[float, float] | None = None

    def as_dict(self) -> dict:
        return {"ok": self.ok, "reason": self.reason,
                **{k: (None if getattr(self, k) is None else [round(getattr(self, k)[0], 1), round(getattr(self, k)[1], 1)])
                   for k in POINTS}}


@dataclass
class Detection:
    forceps: ToolPose
    shadow: ToolPose
    frame_size: tuple[int, int] = (0, 0)
    background: str = "none"
    ms: float = 0.0

    def as_dict(self) -> dict:
        return {"forceps": self.forceps.as_dict(), "shadow": self.shadow.as_dict(),
                "frame_size": list(self.frame_size), "background": self.background, "ms": round(self.ms, 1)}

    def orchestrator_points(self) -> dict | None:
        """The four points the orchestrator needs, or None unless all are present."""
        if not (self.forceps.ok and self.shadow.ok):
            return None
        return {"left_tip_px": list(self.forceps.left_tip), "right_tip_px": list(self.forceps.right_tip),
                "left_shadow_px": list(self.shadow.left_tip), "right_shadow_px": list(self.shadow.right_tip)}


class Detector:
    def __init__(self, cfg: Settings | None = None, background: Background | None = None):
        self.cfg = cfg or Settings()
        self.bg = background or Background(self.cfg.background_path, self.cfg.work_size)
        self.analysis_disc = disc(self.cfg.work_size, self.cfg.disc_radius)

    # --- geometry of the work frame <-> original frame ---
    def _prepare(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        s = min(h, w)
        ox, oy = (w - s) // 2, (h - s) // 2
        sq = frame[oy:oy + s, ox:ox + s]
        n = self.cfg.work_size
        work = sq if s == n else cv2.resize(sq, (n, n), interpolation=cv2.INTER_AREA)
        scale = s / n
        return work, (lambda p: (p[0] * scale + ox, p[1] * scale + oy))

    def detect(self, frame: np.ndarray, adapt: bool = False) -> Detection:
        t0 = cv2.getTickCount()
        work, back = self._prepare(frame)
        ref, from_model = self.bg.reference(work)
        forceps_m, shadow_m, _ = compute_masks(work, ref, from_model, self.cfg)
        res = {}
        for name, m in (("forceps", forceps_m), ("shadow", shadow_m)):
            j: Jaws = analyze(m * self.analysis_disc)
            res[name] = (ToolPose(True, "", *(back(getattr(j, k)) for k in POINTS)) if j.ok
                         else ToolPose(False, j.reason))
        if adapt and from_model:
            self.bg.adapt(work, cv2.dilate(forceps_m | shadow_m, np.ones((15, 15), np.uint8)))
        ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
        return Detection(res["forceps"], res["shadow"], frame.shape[1::-1], self.bg.state, ms)


def annotate(frame: np.ndarray, det: Detection) -> np.ndarray:
    out = frame.copy()
    r = max(4, frame.shape[1] // 160)
    for tool in ("forceps", "shadow"):
        p: ToolPose = getattr(det, tool)
        if not p.ok:
            continue
        j = tuple(map(int, p.junction))
        for k in ("left_tip", "right_tip"):
            cv2.line(out, j, tuple(map(int, getattr(p, k))), LINE[tool], 1, cv2.LINE_AA)
            cv2.circle(out, tuple(map(int, getattr(p, k))), r, COL[k], 2, cv2.LINE_AA)
        cv2.drawMarker(out, j, COL["junction"], cv2.MARKER_CROSS, 2 * r + 4, 2)
    return out
