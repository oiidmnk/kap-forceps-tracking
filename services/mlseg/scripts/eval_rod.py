"""Rod / rod-shadow accuracy on held-out frames (gen_rod.py --val-out) + detection rates on held-out video blocks.

python scripts/eval_rod.py --ckpt runs/x/last.pt --val ../data/rod1_val --video rod.mp4 --holdout 2400:3200 6000:6800 \
    [--sheet runs/x/rod_val.jpg]
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mlseg.data import load_frames
from mlseg.evaluate import match_pairs
from mlseg.infer import Segmenter
from mlseg.rodcv import find_rod, fit_shadow

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--val", required=True)
ap.add_argument("--video", default="")
ap.add_argument("--holdout", nargs="*", default=[])
ap.add_argument("--step", type=int, default=5)
ap.add_argument("--rod-thr", type=float, default=0.3)
ap.add_argument("--shadow-thr", type=float, nargs="*", default=[0.1, 0.2, 0.3])
ap.add_argument("--sheet", default="")
a = ap.parse_args()

seg = Segmenter(a.ckpt)
val = load_frames(a.val)
P, S = [], []
for i in range(0, len(val), 16):
    p, s = seg([f[1] for f in val[i:i + 16]])
    P.append(p), S.append(s)
P, S = np.concatenate(P), np.concatenate(S)
gt = np.stack([f[2] for f in val])
st = np.stack([f[3] for f in val])
P = match_pairs(P, gt)
err = np.linalg.norm(P - gt, axis=-1)


def part(rows, name, thr):
    have = (st[:, rows] == 1).all(1)
    found = S[:, rows].min(1) >= thr
    e = err[have][:, rows]
    ok = found[have]
    print(f"{name:12s} thr {thr:.2f}: labelled {have.sum():3d}  found {ok.mean():.2f}  "
          f"mean {e[ok].mean() if ok.any() else float('nan'):.1f} px  median {np.median(e[ok]) if ok.any() else float('nan'):.1f} px  "
          f"<=5px {(e[ok] <= 5).mean() if ok.any() else 0:.2f}  max {e[ok].max() if ok.any() else float('nan'):.1f}")
    absent = (st[:, rows] == 0).all(1)
    if absent.any():
        print(f"{'':12s} false alarms on {absent.sum()} absent: {(found & absent).sum()}")


part([6, 7], "rod", a.rod_thr)
for t in a.shadow_thr:
    part([8, 9], "rod shadow", t)

if a.sheet:
    COL = {6: (80, 255, 80), 7: (80, 255, 80), 8: (255, 160, 0), 9: (255, 160, 0)}
    tiles = []
    for (name, img, pts, sta), p, s, e in zip(val, P, S, err):
        if not (sta[8:10] == 1).all():
            continue
        im = img.copy()
        for r in range(6, 10):
            if sta[r] == 1:
                cv2.drawMarker(im, tuple(int(v) for v in pts[r]), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
            if s[r] >= 0.1:
                cv2.circle(im, tuple(int(v) for v in p[r]), 9, COL[r], 2)
        cv2.putText(im, f"{name[-6:]} rod {s[6:8].min():.2f} sh {s[8:10].min():.2f} err {e[8:10].max():.0f}px",
                    (20, 50), 0, 1.2, (255, 255, 255), 3)
        tiles.append(cv2.resize(im, (540, 540)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(a.sheet, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))

if a.video:
    hold = [tuple(int(v) for v in r.split(":")) for r in a.holdout]
    cap = cv2.VideoCapture(a.video)
    k, buf, rows, agree = 0, [], [], []

    def run(buf):
        p, s = seg(buf)
        rows.append(s)
        for f, pp, ss in zip(buf, p, s):        # shadow found: does an independent capsule fit land on it?
            if ss[8:10].min() >= min(a.shadow_thr):
                rod = find_rod(f)
                e, _ = fit_shadow(f, pp[8], pp[9], rod.width if rod else 20, shift=8, dang=6)
                agree.append(np.inf if e is None else min(np.linalg.norm(e - pp[8:10], axis=1).max(),
                                                          np.linalg.norm(e[::-1] - pp[8:10], axis=1).max()))

    while True:
        ok = cap.grab()
        if not ok:
            break
        if k % a.step == 0 and any(lo <= k < hi for lo, hi in hold):
            buf.append(cap.retrieve()[1])
            if len(buf) == 16:
                run(buf); buf = []
        k += 1
    if buf:
        run(buf)
    sc = np.concatenate(rows)
    agree = np.array(agree)
    rod = sc[:, 6:8].min(1) >= a.rod_thr
    print(f"held-out video frames {len(sc)}: rod found {rod.mean():.2f}", end="")
    for t in a.shadow_thr:
        print(f" | rod+shadow (thr {t}) {(rod & (sc[:, 8:10].min(1) >= t)).mean():.2f}", end="")
    print(f"\nshadow found on {len(agree)} frames: capsule fit within 5 px {(agree <= 5).mean():.2f}, "
          f"within 10 px {(agree <= 10).mean():.2f} (fit fails where the shadow runs under rod/forceps)")
