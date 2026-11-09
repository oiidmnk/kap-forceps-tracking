"""Self-training for the rod shadow: the model's shadow ends are a guess, rodcv.fit_shadow checks them.
A frame's shadow becomes a label only when the capsule fit lands within --agree px of both predicted ends (an
independent check: the fit only sees the darkness map). Output feeds gen_rod.py --extra.

python scripts/gen_rod_selftrain.py --ckpt runs/x/last.pt --video rod.mp4 --holdout 2400:3200 6000:6800 --out extra.json
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
from mlseg.rodcv import find_rod, fit_shadow

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--video", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--holdout", nargs="*", default=[])
ap.add_argument("--step", type=int, default=3)
ap.add_argument("--lo", type=float, default=0.05, help="min shadow score to try the fit")
ap.add_argument("--agree", type=float, default=5.0)
ap.add_argument("--min-fit", type=float, default=0.12, help="min capsule contrast of the fit")
a = ap.parse_args()

hold = [tuple(int(v) for v in r.split(":")) for r in a.holdout]
seg = Segmenter(a.ckpt)
cap = cv2.VideoCapture(a.video)
out, stats, buf, k = {}, {"tried": 0, "ok": 0, "frames": 0}, [], 0


def flush():
    global buf
    if not buf:
        return
    P, S = seg([f for _, f in buf])
    for (i, f), p, s in zip(buf, P, S):
        stats["frames"] += 1
        if s[8:10].min() < a.lo:
            continue
        rod = find_rod(f)
        stats["tried"] += 1
        ends, score = fit_shadow(f, p[8], p[9], rod.width if rod else 20, shift=8, dang=6)
        if ends is None or score < a.min_fit:
            continue
        d = min(np.linalg.norm(ends - p[8:10], axis=1).max(), np.linalg.norm(ends[::-1] - p[8:10], axis=1).max())
        if d > a.agree:
            continue
        rows = np.zeros((len(ROWS), 2)).tolist()
        state = [-1] * len(ROWS)
        rows[8], rows[9] = ends[0].tolist(), ends[1].tolist()
        state[8] = state[9] = 1
        out[str(i)] = {"rows": rows, "state": state, "score": round(float(s[8:10].min()), 3)}
        stats["ok"] += 1
    buf = []


while True:
    ok = cap.grab()
    if not ok:
        break
    if k % a.step == 0 and not any(lo <= k < hi for lo, hi in hold):
        buf.append((k, cap.retrieve()[1]))
        if len(buf) == 16:
            flush()
    k += 1
flush()
json.dump(out, open(a.out, "w"))
print(stats)
