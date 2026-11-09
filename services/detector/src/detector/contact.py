"""Forceps touching the rod, decided in the image (independent of the 3D calibration).

A tip that lies on the rod in the image may hover above it; it touches when its shadow also meets the rod's shadow at
the same place along the rod (same image point + same shadow -> same 3D point), or when the rod moves along with the
tips (the rod only moves when something pushes it). Grasping hides the forceps shadow inside the rod's shadow, so
contact, once seen, is kept while the tips stay on the rod and the shadows do not clearly separate.
Time is counted in frames: a file source plays slower than real time when detection cannot keep up.
"""
from __future__ import annotations

import numpy as np

from .pipeline import Detection

ROD_ASPECT = 7.5            # rod length / width in the image (the capsule in the rod footage)


def _on_segment(p, a, b) -> tuple[float, float]:
    """-> (distance of p to segment a-b, position 0..1 of its closest point)."""
    v = b - a
    t = float(np.clip((p - a) @ v / max(float(v @ v), 1e-9), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * v))), t


class RodContact:
    """Per-frame contact with hysteresis. Distances in px of the detection frame, scaled with the rod's width.

    enter_shadow / leave_shadow: forceps shadow to rod shadow (x rod width) to start / end contact
    on_margin / off_margin: tips to rod axis beyond its half width (px) to count as on / off the rod
    move_px: rod speed per frame (moving with the tips) that counts as being pushed
    confirm: evidence needed in this many frames in a row to start contact
    release: a reason to end contact must hold this many frames in a row (one noisy frame does not end a grasp)
    hold: contact without fresh evidence ends after this many frames once the tips are no longer on the rod; also how
        long contact survives the rod or the forceps not being found
    """

    def __init__(self, enter_shadow: float = 1.3, leave_shadow: float = 2.5, on_margin: float = 12.0,
                 off_margin: float = 24.0, move_px: float = 1.3, confirm: int = 2, release: int = 3, hold: int = 15):
        self.enter_shadow, self.leave_shadow = enter_shadow, leave_shadow
        self.on_margin, self.off_margin, self.move_px = on_margin, off_margin, move_px
        self.confirm, self.release, self.hold = confirm, release, hold
        self.reset()

    def reset(self) -> None:
        self.state, self.streak, self.leaving, self.since_evidence, self.missing = False, 0, 0, 0, 0
        self._last: tuple[np.ndarray, np.ndarray] | None = None
        self._v_rod = self._v_tips = np.zeros(2)

    def __call__(self, det: Detection) -> bool:
        rod, forceps = det.rod, det.forceps
        if rod is None or rod.ends is None or not forceps.ok:
            self.missing += 1                      # a grasped rod is often hidden for a moment: keep the state a while
            self._last, self.streak = None, 0
            if self.missing > self.hold:
                self.reset()
            return self.state
        self.missing = 0
        a, b = (np.asarray(p, np.float64) for p in rod.ends)
        tips = np.asarray([forceps.left_tip, forceps.right_tip], np.float64)
        mid = tips.mean(0)
        width = max(float(np.linalg.norm(b - a)) / ROD_ASPECT, 8.0)
        dist, t = min((_on_segment(p, a, b) for p in (tips[0], tips[1], mid)), key=lambda x: x[0])

        centre = (a + b) / 2                       # independent of the (unordered) end labels
        if self._last is not None:                 # px per frame, smoothed over a few frames
            self._v_rod = 0.5 * self._v_rod + 0.5 * (centre - self._last[0])
            self._v_tips = 0.5 * self._v_tips + 0.5 * (mid - self._last[1])
        self._last = (centre, mid)
        speed = float(np.linalg.norm(self._v_rod))
        pushed = speed > self.move_px and self._v_rod @ self._v_tips > 0.5 * speed * np.linalg.norm(self._v_tips)

        shadow_gap = None
        if rod.shadow is not None and det.shadow.ok and det.shadow.reason != "held":
            sa, sb = (np.asarray(p, np.float64) for p in rod.shadow)
            fs = np.asarray([det.shadow.left_tip, det.shadow.right_tip], np.float64).mean(0)
            shadow_gap = float(np.linalg.norm(fs - (sa + t * (sb - sa))))

        on = dist < width / 2 + self.on_margin
        evidence = on and (pushed or (shadow_gap is not None and shadow_gap < self.enter_shadow * width))
        self.streak = self.streak + 1 if evidence else 0
        self.since_evidence = 0 if evidence else self.since_evidence + 1
        leaving = (dist > width / 2 + self.off_margin
                   or (shadow_gap is not None and shadow_gap > self.leave_shadow * width)
                   or (not on and self.since_evidence > self.hold))
        self.leaving = self.leaving + 1 if leaving and not evidence else 0
        if not self.state:
            self.state = self.streak >= self.confirm
        elif self.leaving >= self.release:
            self.state = False
        return self.state
