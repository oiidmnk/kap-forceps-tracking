"""Shadow relocation augmentation.
A cast shadow is multiplicative: frame = background * gain. For labelled frames we cut the real shadow's gain out
(using the empty-retina background of that video), divide it away (leaving the forceps alone) and multiply it back
at a new place, rotated/scaled, close to the forceps tips - exactly the configuration that is rare in the footage
(shadow tip a few px from the forceps tip). The shadow tip labels move with the same affine map."""
from __future__ import annotations

import cv2
import numpy as np


class ShadowBank:
    def __init__(self, frames, backgrounds, max_items: int = 450, seed: int = 0, max_resid: float = 0.035):
        """frames: load_frames() output (name, img, pts, state). Keeps those with both shadow tips labelled.
        backgrounds: empty-retina image(s), one per video; each frame uses the one it matches best (median |diff|)."""
        backgrounds = backgrounds if isinstance(backgrounds, (list, tuple)) else [backgrounds]
        prep = [(b, cv2.GaussianBlur(b.astype(np.float32), (0, 0), 1.5), cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32))
                for b in backgrounds]
        self.items, self.resid = {}, []
        cand = [f for f in frames if f[3][2] == 1 and f[3][3] == 1]
        seen, uniq = set(), []
        for f in cand:
            if f[0] not in seen:
                seen.add(f[0]); uniq.append(f)
        real = [f for f in uniq if f[0].startswith("frame_")]
        rest = [f for f in uniq if not f[0].startswith("frame_")]
        np.random.default_rng(seed).shuffle(rest)
        for name, img, pts, st in real + rest[:max(0, max_items - len(real))]:
            small = cv2.resize(img, (108, 108), interpolation=cv2.INTER_AREA).astype(np.float32)
            errs = [np.median(np.abs(small - cv2.resize(b, (108, 108), interpolation=cv2.INTER_AREA).astype(np.float32))) for b, _, _ in prep]
            background_bgr, bg, bgs = prep[int(np.argmin(errs))]
            f = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
            g = np.clip(f / np.maximum(bg, 1), 0.15, 1.2)                   # per-channel gain
            ratio = g.mean(-1)
            sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
            cand = ((ratio < 0.94) & (sat > 0.75 * bgs)).astype(np.uint8)      # darker, still reddish (a shadow, not the grey tool)
            cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            n, lab, stt, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
            ids = {int(lab[int(np.clip(p[1], 0, 1079)), int(np.clip(p[0], 0, 1079))]) for p in pts[2:4]} - {0}
            if not ids:                                                          # tips sit on a thin edge: search nearby
                for p in pts[2:4]:
                    x, y = int(p[0]), int(p[1])
                    near = lab[max(0, y - 6):y + 7, max(0, x - 6):x + 7]
                    ids |= {int(v) for v in np.unique(near)} - {0}
            if not ids:
                continue
            m = np.isin(lab, list(ids)).astype(np.uint8)
            m = cv2.dilate(m, np.ones((5, 5), np.uint8))
            soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.0)
            gain = 1.0 + (g - 1.0) * soft[..., None]                              # 1 outside the shadow, soft edge
            far = (cv2.dilate(m, np.ones((41, 41), np.uint8)) == 0) & (sat > 0.75 * bgs)       # neither shadow nor the grey tool
            resid = float(np.median(np.abs(ratio[far] - 1.0))) if far.any() else 9.0           # ghosts / misalignment
            self.resid.append(resid)
            if resid > max_resid and not name.startswith("frame_"):
                continue
            f0 = np.clip(img.astype(np.float32) / np.maximum(gain, 0.15), 0, 255)  # shadow-free frame (forceps untouched)
            self.items[name] = (f0.astype(np.uint8), gain.astype(np.float16), pts.copy(), st.copy())

    def __contains__(self, name):
        return name in self.items

    def apply(self, name, rng: np.random.Generator):
        """-> (img, pts, state) with the shadow moved; tips of channels 2,3 (and junction 5) follow."""
        f0, gain, pts, st = self.items[name]
        fm = (pts[0] + pts[1]) / 2
        sm = (pts[2] + pts[3]) / 2
        target = fm + (sm - fm) * rng.uniform(0.04, 1.0) + rng.normal(0, 10, 2)    # up to touching the forceps tips
        ang, sc = rng.uniform(-18, 18), rng.uniform(0.9, 1.15)
        M = cv2.getRotationMatrix2D((float(sm[0]), float(sm[1])), ang, sc)
        M[:, 2] += target - sm
        g2 = cv2.warpAffine(gain.astype(np.float32), M, (f0.shape[1], f0.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(1.0, 1.0, 1.0))
        out = np.clip(f0.astype(np.float32) * g2, 0, 255).astype(np.uint8)
        p2 = pts.copy()
        p2[[2, 3, 5]] = (np.concatenate([pts[[2, 3, 5]], np.ones((3, 1), np.float32)], 1) @ M.T)
        return out, p2, st.copy()
