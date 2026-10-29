"""Closed jaws: the two tips coincide, but the labelled frames only have open jaws, so the model reports one tip.
If the forceps tips are <close px apart (jaws closed) and one shadow tip is confident, label both shadow channels
at that point. Adds frames to a gen_pseudo dataset."""
import argparse, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mlseg.infer import Segmenter

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--videos", nargs="+", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--close", type=float, default=14.0)
ap.add_argument("--hi", type=float, default=0.6); ap.add_argument("--every", type=int, default=3)
a = ap.parse_args()
seg = Segmenter(a.ckpt); stats = {}
for vp in a.videos:
    cap = cv2.VideoCapture(vp); P, S, buf = [], [], []
    def flush():
        global buf
        if buf:
            p, s = seg(buf); P.extend(p); S.extend(s); buf = []
    while True:
        ok, f = cap.read()
        if not ok: break
        buf.append(f)
        if len(buf) == 16: flush()
    flush(); P, S = np.array(P), np.array(S)
    fc = (S[:, 0] > a.hi) & (S[:, 1] > a.hi) & (np.linalg.norm(P[:, 0] - P[:, 1], axis=-1) < a.close)
    one = ((S[:, 2] > a.hi) ^ (S[:, 3] > a.hi)) & (np.maximum(S[:, 2], S[:, 3]) > a.hi) & (np.minimum(S[:, 2], S[:, 3]) < 0.5)
    pick = np.flatnonzero(fc & one)[::a.every]
    tag = "c" + os.path.basename(vp).split("_")[0][:6].replace(".", ""); cap = cv2.VideoCapture(vp); got = 0
    for k in pick:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k)); ok, f = cap.read()
        if not ok: continue
        q = 2 if S[k, 2] > S[k, 3] else 3
        pts = np.zeros((6, 2), np.float32); st = [-1] * 6
        for j in (0, 1): pts[j], st[j] = P[k, j], 1
        for j in (2, 3): pts[j], st[j] = P[k, q], 1
        name = f"{tag}_{k:06d}"; cv2.imwrite(f"{a.out}/{name}.png", f)
        json.dump({"pts": pts.tolist(), "state": st, "kind": "pseudo"}, open(f"{a.out}/{name}.json", "w")); got += 1
    stats[os.path.basename(vp)[:20]] = {"frames": len(P), "closed_forceps": int(fc.sum()), "one_shadow_tip": int((fc & one).sum()), "added": got}
print(stats)
