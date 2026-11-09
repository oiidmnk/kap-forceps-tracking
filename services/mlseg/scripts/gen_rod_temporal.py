"""Forceps-shadow labels across short gaps (gen_temporal.py for the rod footage, output for gen_rod.py --extra).
The forceps shadow drops out where it runs into the rod's shadow. Confident before and after a gap -> in between the
model's own (unsure) peak if it follows the motion, else the interpolation when the shadow hardly moved.

python scripts/gen_rod_temporal.py --ckpt runs/x/last.pt --video rod.mp4 --holdout 2400:3200 6000:6800 --out temporal.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mlseg.infer import Segmenter
from mlseg.model import ROWS

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--video", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--holdout", nargs="*", default=[])
ap.add_argument("--max-gap", type=int, default=45)
ap.add_argument("--hi", type=float, default=0.7)
ap.add_argument("--lo", type=float, default=0.03)
ap.add_argument("--fit", type=float, default=10.0, help="max px between the unsure peak and the interpolation")
ap.add_argument("--still", type=float, default=12.0, help="max shadow motion over the gap to use the interpolation")
ap.add_argument("--every", type=int, default=3)
a = ap.parse_args()

hold = [tuple(int(v) for v in r.split(":")) for r in a.holdout]
seg = Segmenter(a.ckpt)
cap = cv2.VideoCapture(a.video)
P, S, buf = [], [], []
while True:
    ok, f = cap.read()
    if ok:
        buf.append(f)
    if len(buf) == 16 or (not ok and buf):
        p, s = seg(buf)
        P.extend(p), S.extend(s)
        buf = []
    if not ok:
        break
P, S = np.array(P), np.array(S)
n = len(P)
tips = S[:, 0:2].min(1) > a.hi
conf = (S[:, 2:4].min(1) > a.hi) & tips
out, stats = {}, {"frames": n, "confident": int(conf.sum()), "peak": 0, "interp": 0}
i = 0
while i < n:
    if conf[i]:
        i += 1
        continue
    j = i
    while j < n and not conf[j]:
        j += 1
    if 0 < i and j < n and j - i <= a.max_gap:
        before, after = P[i - 1, 2:4], P[j, 2:4]
        moved = np.linalg.norm(after - before, axis=-1).max()
        for k in range(i, j, a.every):
            if not tips[k] or any(lo <= k < hi for lo, hi in hold):
                continue
            w = (k - (i - 1)) / (j - (i - 1))
            interp = (1 - w) * before + w * after
            if S[k, 2:4].min() > a.lo and np.linalg.norm(P[k, 2:4] - interp, axis=-1).max() < a.fit:
                xy, kind = P[k, 2:4], "peak"
            elif moved < a.still:
                xy, kind = interp, "interp"
            else:
                continue
            rows = np.zeros((len(ROWS), 2))
            state = [-1] * len(ROWS)
            rows[0:2], rows[2:4] = P[k, 0:2], xy
            state[0:4] = [1, 1, 1, 1]
            out[str(k)] = {"rows": rows.tolist(), "state": state}
            stats[kind] += 1
    i = j
json.dump(out, open(a.out, "w"))
print(stats)
