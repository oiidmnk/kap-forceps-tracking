"""Training frames from rod footage (rod = red capsule, its shadow = dark capsule; see mlseg/rodcv.py).

Rows as model.ROWS. Forceps rows come from a forceps model (confident -> present, clearly nothing -> absent, else
ignored); rod rows from the colour finder (ignored when an end touches the forceps: it may be hidden); rod-shadow
rows from hand-read end points (snapped by rodcv.fit_shadow), ignored where nobody looked.

python scripts/gen_rod.py --video rod.mp4 --ckpt p18.pt --shadows shadows.json --out data/rod1 \
    --holdout 2400:3200 6000:6800 --val-out data/rod1_val
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
from mlseg.rodcv import find_rod, rod_occluded

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--ckpt", required=True, help="forceps model for rows 0-5")
ap.add_argument("--shadows", required=True, help='{frame: [[x,y],[x,y]] | "absent" | null}')
ap.add_argument("--extra", nargs="*", default=[], help="{frame: {'rows': ..., 'state': ...}} files (self-training); "
                "they fill rows the other sources left unknown")
ap.add_argument("--out", required=True)
ap.add_argument("--val-out", default="")
ap.add_argument("--holdout", nargs="*", default=[], help="frame ranges a:b never used for training")
ap.add_argument("--step", type=int, default=3)
ap.add_argument("--hi", type=float, default=0.7)
a = ap.parse_args()

shadows = json.load(open(a.shadows))
extra = {}
for path in a.extra:                       # later files fill rows earlier ones left unknown
    for k, e in json.load(open(path)).items():
        if k not in extra:
            extra[k] = {"rows": e["rows"], "state": list(e["state"])}
            continue
        for i, s in enumerate(e["state"]):
            if s != -1 and extra[k]["state"][i] == -1:
                extra[k]["rows"][i], extra[k]["state"][i] = e["rows"][i], s
hold = [tuple(int(v) for v in r.split(":")) for r in a.holdout]
in_hold = lambda k: any(lo <= k < hi for lo, hi in hold)
seg = Segmenter(a.ckpt)
for d in (a.out, a.val_out):
    if d:
        os.makedirs(d, exist_ok=True)
tag = os.path.splitext(os.path.basename(a.video))[0]
stats = {"train": 0, "val": 0, "rod": 0, "rod_ign": 0, "rod_abs": 0, "shadow": 0, "forceps": 0}


def forceps_rows(p, sc):
    pts = np.zeros((6, 2), np.float32)
    st = np.full(6, -1, np.int8)
    for ix, jn in (((0, 1), 4), ((2, 3), 5)):
        s2 = sc[list(ix)]
        if (s2 > a.hi).all():
            for q in ix:
                pts[q], st[q] = p[q], 1
            if sc[jn] > a.hi:
                pts[jn], st[jn] = p[jn], 1
        elif (s2 < 0.05).all():
            st[list(ix)] = 0
            st[jn] = 0
    return pts, st


def rows(k, img, p, sc):
    pts = np.zeros((len(ROWS), 2), np.float32)
    st = np.full(len(ROWS), -1, np.int8)
    pts[:6], st[:6] = forceps_rows(p, sc)
    rod = find_rod(img)
    if rod is None:
        st[6:8] = 0
        stats["rod_abs"] += 1
    elif rod_occluded(img, rod):
        stats["rod_ign"] += 1
    else:
        pts[6:8], st[6:8] = rod.ends, 1
        stats["rod"] += 1
    sh = shadows.get(str(k))
    if sh == "absent" or (rod is None and sh is None):
        st[8:10] = 0
    elif isinstance(sh, list):
        pts[8:10], st[8:10] = np.array(sh, np.float32), 1
        stats["shadow"] += 1
    if str(k) in extra:
        e = extra[str(k)]
        for i, (xy, s) in enumerate(zip(e["rows"], e["state"])):
            if s != -1 and st[i] == -1:
                pts[i], st[i] = xy, s
    stats["forceps"] += int(st[0] == 1)
    return pts, st


def save(folder, name, img, pts, st, kind):
    cv2.imwrite(f"{folder}/{name}.png", img)
    json.dump({"pts": pts.tolist(), "state": [int(s) for s in st], "kind": kind}, open(f"{folder}/{name}.json", "w"))


want = {int(k) for k in shadows} | {int(k) for k in extra}
cap = cv2.VideoCapture(a.video)
buf, i = [], 0


def flush():
    global buf
    if not buf:
        return
    P, S = seg([f for _, f in buf])
    for (k, f), p, sc in zip(buf, P, S):
        pts, st = rows(k, f, p, sc)
        if in_hold(k):
            if a.val_out and (str(k) in shadows or k % 15 == 0):
                save(a.val_out, f"{tag}_{k:06d}", f, pts, st, "real")
                stats["val"] += 1
            continue
        if (st == -1).all():
            continue
        save(a.out, f"{tag}_{k:06d}", f, pts, st, "pseudo")
        stats["train"] += 1
    buf = []


while True:
    ok = cap.grab()
    if not ok:
        break
    if i % a.step == 0 or i in want or (in_hold(i) and i % 15 == 0):
        _, f = cap.retrieve()
        buf.append((i, f))
        if len(buf) == 16:
            flush()
    i += 1
flush()
print(stats)
