"""Hard positives from time: a shadow tip that is confident in the frames before and after a short gap and whose
(unsure) peak in between fits the motion is a verified positive. Adds frames to an existing gen_pseudo dataset."""
import argparse, glob, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mlseg.infer import Segmenter

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--videos", nargs="+", required=True)
ap.add_argument("--out", required=True, help="dataset dir to add to (gen_pseudo output)")
ap.add_argument("--max-gap", type=int, default=24); ap.add_argument("--lo", type=float, default=0.08)
ap.add_argument("--hi", type=float, default=0.7); ap.add_argument("--fit", type=float, default=12.0)
ap.add_argument("--every", type=int, default=2)
a = ap.parse_args()
seg = Segmenter(a.ckpt)
stats = {}
for vp in a.videos:
    cap = cv2.VideoCapture(vp); frames = []
    P, S = [], []
    buf = []
    def flush():
        global buf
        if buf:
            p, s = seg(buf); P.extend(p); S.extend(s); buf = []
    while True:
        ok, f = cap.read()
        if not ok: break
        buf.append(f); frames.append(None)
        if len(buf) == 16: flush()
    flush()
    P, S = np.array(P), np.array(S); n = len(P)
    conf = np.minimum(S[:, 2], S[:, 3]) > a.hi
    tag = "t" + os.path.basename(vp).split("_")[0][:6].replace(".", "")
    picks = []
    i = 0
    while i < n:
        if conf[i]: i += 1; continue
        j = i
        while j < n and not conf[j]: j += 1
        gap = j - i
        if i > 0 and j < n and gap <= a.max_gap and np.linalg.norm(P[i - 1, 2:4] - P[j, 2:4], axis=-1).max() < 12 * (gap + 1):
            for k in range(i, j):
                w = (k - (i - 1)) / (j - (i - 1))
                I = (1 - w) * P[i - 1, 2:4] + w * P[j, 2:4]
                if S[k, 2:4].min() > a.lo and np.linalg.norm(P[k, 2:4] - I, axis=-1).max() < a.fit:
                    picks.append(k)
        i = j
    picks = picks[::a.every]
    cap = cv2.VideoCapture(vp); got = 0
    for k in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
        if not ok: continue
        pts = np.zeros((6, 2), np.float32); st = [-1] * 6
        for q in (2, 3): pts[q], st[q] = P[k, q], 1
        for q in (0, 1):
            if S[k, q] > a.hi: pts[q], st[q] = P[k, q], 1
        name = f"{tag}_{k:06d}"
        cv2.imwrite(f"{a.out}/{name}.png", f)
        json.dump({"pts": pts.tolist(), "state": st, "kind": "pseudo"}, open(f"{a.out}/{name}.json", "w")); got += 1
    stats[os.path.basename(vp)[:20]] = {"frames": n, "shadow_conf": int(conf.sum()), "gap_positives": got}
print(stats)
