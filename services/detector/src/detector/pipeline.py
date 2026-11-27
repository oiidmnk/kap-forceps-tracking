"""Frame -> Detection: tips of forceps and shadow plus their jaw junctions, in original pixel coords."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .background import Background
from .config import Settings
from .geometry import Jaws, analyze
from .masks import compute_masks, disc, isolate_tool

POINTS = ("left_tip", "right_tip", "junction")
# colours (BGR)
COL = {"left_tip": (255, 255, 0), "right_tip": (255, 0, 255), "junction": (0, 255, 255)}
LINE = {"forceps": (0, 255, 0), "shadow": (255, 160, 0)}
ROD_COL, ROD_SHADOW_COL = (80, 255, 80), (255, 160, 0)


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
class RodPose:
    """The rod's two end points and the matching ends of its shadow (`shadow[i]` is cast by `ends[i]`)."""
    ends: tuple[tuple[float, float], tuple[float, float]] | None = None
    shadow: tuple[tuple[float, float], tuple[float, float]] | None = None
    scores: dict = field(default_factory=dict)
    contact: bool = False               # forceps touching the rod (set by the live worker, see contact.RodContact)

    @property
    def ok(self) -> bool:
        return self.ends is not None and self.shadow is not None

    def as_dict(self) -> dict:
        r = lambda pts: None if pts is None else [[round(p[0], 1), round(p[1], 1)] for p in pts]
        return {"ok": self.ok, "ends": r(self.ends), "shadow": r(self.shadow), "contact": self.contact,
                "scores": {k: round(float(v), 3) for k, v in self.scores.items()}}


@dataclass
class Detection:
    forceps: ToolPose
    shadow: ToolPose
    frame_size: tuple[int, int] = (0, 0)
    background: str = "none"
    ms: float = 0.0
    rod: RodPose | None = None          # learned backend with rod channels only

    def as_dict(self) -> dict:
        d = {"forceps": self.forceps.as_dict(), "shadow": self.shadow.as_dict(),
             "frame_size": list(self.frame_size), "background": self.background, "ms": round(self.ms, 1)}
        if self.rod is not None:
            d["rod"] = self.rod.as_dict()
        return d

    def orchestrator_points(self) -> dict | None:
        """Forceps tips + their shadows when all four are present, rod ends + their shadows when all four are
        present (plus ``rod_contact``); None if neither is complete."""
        pts = {}
        if self.forceps.ok and self.shadow.ok:
            pts.update({"left_tip_px": list(self.forceps.left_tip), "right_tip_px": list(self.forceps.right_tip),
                        "left_shadow_px": list(self.shadow.left_tip), "right_shadow_px": list(self.shadow.right_tip)})
        if self.rod is not None and self.rod.ok:
            pts.update({"rod_a_px": list(self.rod.ends[0]), "rod_b_px": list(self.rod.ends[1]),
                        "rod_a_shadow_px": list(self.rod.shadow[0]), "rod_b_shadow_px": list(self.rod.shadow[1]),
                        "rod_contact": bool(self.rod.contact)})
        return pts or None


class Detector:
    def __init__(self, cfg: Settings | None = None, background: Background | None = None):
        self.cfg = cfg or Settings()
        self.bg = background or Background(self.cfg.background_path, self.cfg.work_size)
        self._disc_cache: tuple = ("unset", None)
        self.roi: tuple[float, float, float] | None = None   # eye circle (cx, cy, r) in raw frame px
        self.view: np.ndarray | None = None                  # image the last detection refers to

    # --- geometry of the work frame <-> original frame ---
    def normalize(self, frame: np.ndarray) -> np.ndarray:
        """Content of the eye circle, everything outside black, as a work_size x work_size square (1:1)."""
        cx, cy, r = self.roi
        side = max(2, int(round(2 * r)))
        x0, y0 = int(round(cx - r)), int(round(cy - r))
        h, w = frame.shape[:2]
        canvas = np.zeros((side, side, 3), np.uint8)
        fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
        if fx1 > fx0 and fy1 > fy0:
            canvas[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] = frame[fy0:fy1, fx0:fx1]
        circle = np.zeros((side, side), np.uint8)
        cv2.circle(circle, (side // 2, side // 2), side // 2, 255, -1)
        canvas[circle == 0] = 0
        n = self.cfg.work_size
        return cv2.resize(canvas, (n, n), interpolation=cv2.INTER_AREA)

    def _prepare(self, frame: np.ndarray):
        """-> (work image, mapper from work coords to output coords).
        With an eye circle the output space IS the normalised square; without, the original frame."""
        if self.roi is not None:
            return self.normalize(frame), (lambda p: (float(p[0]), float(p[1])))
        h, w = frame.shape[:2]
        s = min(h, w)
        ox, oy = (w - s) // 2, (h - s) // 2
        sq = frame[oy:oy + s, ox:ox + s]
        n = self.cfg.work_size
        work = sq if s == n else cv2.resize(sq, (n, n), interpolation=cv2.INTER_AREA)
        scale = s / n
        return work, (lambda p: (p[0] * scale + ox, p[1] * scale + oy))

    def detect(self, frame: np.ndarray, adapt: bool = False, prepared=None) -> Detection:
        """`prepared` = (work, mapper) from _prepare(), if the caller already normalised the frame."""
        t0 = cv2.getTickCount()
        work, back = prepared or self._prepare(frame)
        ref, from_model = self.bg.reference(work)
        eye = self.bg.disc if from_model else None
        forceps_m, shadow_m, _ = compute_masks(work, ref, from_model, self.cfg, eye, self.bg.bgr if from_model else None)
        if self._disc_cache[0] != eye:
            self._disc_cache = (eye, disc(self.cfg.work_size, self.cfg.disc_eye if eye else self.cfg.disc_radius, eye))
        analysis = self._disc_cache[1]
        res = {}
        for name, m in (("forceps", forceps_m), ("shadow", shadow_m)):
            m = m * analysis
            j: Jaws = analyze(isolate_tool(m) if name == "shadow" else m)
            res[name] = (ToolPose(True, "", *(back(getattr(j, k)) for k in POINTS)) if j.ok
                         else ToolPose(False, j.reason))
        if adapt and from_model:
            self.bg.adapt(work, cv2.dilate(forceps_m | shadow_m, np.ones((15, 15), np.uint8)))
        ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
        self.view = work if self.roi is not None else frame
        size = work.shape[1::-1] if self.roi is not None else frame.shape[1::-1]
        return Detection(res["forceps"], res["shadow"], size, self.bg.state, ms)


def annotate(frame: np.ndarray, det: Detection) -> np.ndarray:
    out = frame.copy()
    r = max(4, frame.shape[1] // 160)
    for tool in ("forceps", "shadow"):
        p: ToolPose = getattr(det, tool)
        if not p.ok:
            continue
        j = tuple(map(int, p.junction)) if p.junction is not None else None   # learned backends have no junction
        for k in ("left_tip", "right_tip"):
            if j is not None:
                cv2.line(out, j, tuple(map(int, getattr(p, k))), LINE[tool], 1, cv2.LINE_AA)
            # thin ring: not seen in this frame, carried over (learned backend, see MLDetector._hold_shadow)
            cv2.circle(out, tuple(map(int, getattr(p, k))), r, COL[k], 1 if p.reason == "held" else 2, cv2.LINE_AA)
        if j is not None:
            cv2.drawMarker(out, j, COL["junction"], cv2.MARKER_CROSS, 2 * r + 4, 2)
    rod = det.rod
    if rod is not None:
        for pts, col in ((rod.ends, ROD_COL), (rod.shadow, ROD_SHADOW_COL)):
            if pts is None:
                continue
            a, b = (tuple(map(int, p)) for p in pts)
            cv2.line(out, a, b, col, 3 if rod.contact and col == ROD_COL else 1, cv2.LINE_AA)
            for p in (a, b):
                cv2.circle(out, p, r, col, 2, cv2.LINE_AA)
        if rod.ok:                                         # which shadow end belongs to which rod end
            for e, s in zip(rod.ends, rod.shadow):
                cv2.line(out, tuple(map(int, e)), tuple(map(int, s)), (200, 200, 200), 1, cv2.LINE_AA)
        if rod.contact and rod.ends is not None:
            c = np.mean(rod.ends, axis=0).astype(int)
            cv2.putText(out, "CONTACT", (int(c[0]) + 2 * r, int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROD_COL, 2,
                        cv2.LINE_AA)
    return out
