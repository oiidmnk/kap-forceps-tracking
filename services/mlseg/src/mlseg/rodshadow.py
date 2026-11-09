"""Rod-shadow overlay augmentation.
The forceps shadow drops out where it runs into the rod's shadow (the forceps grasps the rod), a configuration the
labels hardly contain. Real rod shadows (labelled rod-shadow ends) are cut out as a brightness factor of the frame's
own darkness map (rodcv.darkness: vessel-free V / local bright level) and laid over the forceps shadow tips of other
training frames. Two shadows of one light do not multiply - the overlap is as dark as the darker one - so the new
factor is min(own, pasted) / own. The forceps itself stays untouched; the forceps labels do not move, the pasted
rod-shadow ends become labels (the rod rows are unknown: its rod is not in that frame)."""
from __future__ import annotations

import cv2
import numpy as np

from .rodcv import _grey_mask, darkness, find_rod, rod_mask


class RodShadowBank:
    def __init__(self, frames, max_items: int = 300, seed: int = 0):
        """frames: load_frames() output; uses those with both rod-shadow ends (rows 8, 9) labelled."""
        cand = [f for f in frames if len(f[3]) >= 10 and (f[3][8:10] == 1).all()]
        seen, uniq = set(), []
        for f in cand:
            if f[0] not in seen:
                seen.add(f[0]); uniq.append(f)
        np.random.default_rng(seed).shuffle(uniq)
        self.items = []
        for name, img, pts, st in uniq[:max_items]:
            rod = find_rod(img)
            w = rod.width if rod is not None else 20.0
            a, b = pts[8].astype(np.float32), pts[9].astype(np.float32)
            L = float(np.linalg.norm(b - a))
            if L < 5:
                continue
            r = int(L / 2 + 2 * w + 8)
            c = (a + b) / 2
            x0, y0 = int(c[0]) - r, int(c[1]) - r
            if x0 < 0 or y0 < 0 or x0 + 2 * r >= img.shape[1] or y0 + 2 * r >= img.shape[0]:
                continue
            dk = darkness(img)[y0:y0 + 2 * r, x0:x0 + 2 * r]
            crop = img[y0:y0 + 2 * r, x0:x0 + 2 * r]
            cover = cv2.dilate(rod_mask(crop) | _grey_mask(crop), np.ones((7, 7), np.uint8)) > 0   # rod/forceps on top
            yy, xx = np.mgrid[0:2 * r, 0:2 * r].astype(np.float32) + 0.5
            p = np.stack([xx + x0, yy + y0], -1)
            d = b - a
            t = np.clip(((p - a) @ d) / (d @ d), 0, 1)
            dist = np.linalg.norm(p - (a + t[..., None] * d), axis=-1)
            core = 0.5 * w
            soft = np.clip((core + 0.6 * w - dist) / (0.6 * w), 0, 1)          # 1 in the capsule, fades out
            inner = (dist < core) & ~cover
            if inner.sum() < 20:
                continue
            level = float(np.median(dk[inner]))
            dk = np.where(cover, level, dk)                                    # shadow continues under rod/forceps
            dk = np.maximum(dk, 0.85 * level)          # no extra dark spots (other shadows crossing it in the source)
            gain = 1.0 - (1.0 - np.clip(dk, 0.2, 1.0)) * soft
            self.items.append((gain.astype(np.float16), np.stack([a, b]) - c))

    def __len__(self):
        return len(self.items)

    def apply(self, img, pts, state, rng: np.random.Generator):
        """-> (img, pts, state) with a rod shadow across the forceps shadow tips (rows 2, 3)."""
        gain, ends = self.items[rng.integers(len(self.items))]
        r = gain.shape[0] / 2
        tips = pts[2:4]
        u = rng.uniform(0, 1)
        target = tips[0] + u * (tips[1] - tips[0]) + rng.normal(0, 8, 2)        # crosses the shadow tips
        ang, sc = rng.uniform(0, 360), rng.uniform(0.8, 1.25)
        M = cv2.getRotationMatrix2D((r, r), ang, sc)
        # slide the capsule along its own axis so the tips land anywhere between its middle and its ends
        along = M[:, :2] @ (ends[1] - ends[0]) / 2 * rng.uniform(-0.8, 0.8)
        M[:, 2] += target - along - r
        h, w = img.shape[:2]
        g = cv2.warpAffine(gain.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR, borderValue=1.0)
        own = np.clip(darkness(img), 0.05, 1.0)
        f = np.minimum(own, g) / own
        f[cv2.dilate(_grey_mask(img), np.ones((5, 5), np.uint8)) > 0] = 1.0     # the forceps is above the shadow
        out = np.clip(img.astype(np.float32) * f[..., None], 0, 255).astype(np.uint8)
        p2, st2 = pts.copy(), state.copy()
        c = np.array([r, r], np.float32)
        p2[8:10] = (np.concatenate([ends + c, np.ones((2, 1), np.float32)], 1) @ M.T)
        st2[8:10] = 1
        st2[6:8] = -1
        return out, p2, st2
