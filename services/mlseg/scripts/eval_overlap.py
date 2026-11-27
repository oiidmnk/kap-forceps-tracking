"""Forceps shadow inside the rod's shadow (grasping). There the shadow tips are not visible at all - two shadows of one
light merge - so there is no hand label; two measurements instead, both on held-out frames of the rod video:

* synthetic: frames where --ref sees the forceps shadow confidently get a real rod shadow (held-out labelled one) laid
  across the shadow tips (rodshadow.RodShadowBank); truth = --ref's tips on the untouched frame.
* real: frames where the forceps tips are found and lie within 60 px of a found rod shadow: how often is the forceps
  shadow (both tips) reported.

python scripts/eval_overlap.py --ckpt runs/x/last.pt --ref runs/r3/last.pt --video rod.mp4 --val ../data/rod1_val \
    --holdout 2400:3200 6000:6800
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mlseg.data import load_frames
from mlseg.infer import Segmenter
from mlseg.rodshadow import RodShadowBank

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", nargs="+", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--video", required=True)
ap.add_argument("--val", required=True)
ap.add_argument("--holdout", nargs="+", required=True)
ap.add_argument("--step", type=int, default=3)
ap.add_argument("--thr", type=float, default=0.1)
a = ap.parse_args()

hold = [tuple(int(v) for v in r.split(":")) for r in a.holdout]
cap = cv2.VideoCapture(a.video)
frames, k = [], 0
while True:
    ok = cap.grab()
    if not ok:
        break
    if k % a.step == 0 and any(lo <= k < hi for lo, hi in hold):
        frames.append(cap.retrieve()[1])
    k += 1


def run(seg, imgs):
    P, S = [], []
    for i in range(0, len(imgs), 16):
        p, s = seg(imgs[i:i + 16])
        P.append(p), S.append(s)
    return np.concatenate(P), np.concatenate(S)


ref = Segmenter(a.ref)
P0, S0 = run(ref, frames)
base = np.where((S0[:, 0:4].min(1) > 0.7))[0]                     # forceps and its shadow clearly visible
bank = RodShadowBank(load_frames(a.val))
rng = np.random.default_rng(0)
synth, truth = [], []
for i in base:
    pts = np.zeros((10, 2), np.float32)
    st = np.zeros(10, np.int8)
    pts[:4], st[:4] = P0[i, :4], 1
    img, _, _ = bank.apply(frames[i], pts, st, rng)
    synth.append(img), truth.append(P0[i, 2:4])
truth = np.array(truth)


def seg_dist(p, a_, b_):
    d = b_ - a_
    t = np.clip(((p - a_) @ d) / max(float(d @ d), 1e-6), 0, 1)
    return np.linalg.norm(p - (a_ + t * d))


print(f"held-out frames {len(frames)}; synthetic overlap on {len(synth)} (rod shadows from {len(bank)} labelled frames)")
for ck in a.ckpt:
    seg = Segmenter(ck)
    P, S = run(seg, synth)
    found = S[:, 2:4].min(1) >= a.thr
    err = np.minimum(np.linalg.norm(P[:, 2:4] - truth, axis=-1).max(1),
                     np.linalg.norm(P[:, 2:4][:, ::-1] - truth, axis=-1).max(1))
    Pr, Sr = run(seg, frames)
    tips = Sr[:, 0:2].min(1) >= 0.3
    rodsh = Sr[:, 8:10].min(1) >= 0.1
    near = np.array([min(seg_dist(Pr[i, q], Pr[i, 8], Pr[i, 9]) for q in (0, 1)) < 60 for i in range(len(Pr))])
    real = tips & rodsh & near
    print(f"{ck}: synthetic found {found.mean():.2f}  error median {np.median(err[found]):.1f} px  <=5 px "
          f"{(err[found] <= 5).mean():.2f}  <=10 px {(err[found] <= 10).mean():.2f} | real overlap frames {real.sum()}: "
          f"forceps shadow found {(Sr[real, 2:4].min(1) >= a.thr).mean():.2f} | all forceps frames {tips.sum()}: "
          f"{(Sr[tips, 2:4].min(1) >= a.thr).mean():.2f}")
