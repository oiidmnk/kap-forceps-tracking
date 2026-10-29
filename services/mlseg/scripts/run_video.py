"""Run model (and optionally the classical detector) over a video; save predictions + annotated clip."""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mlseg.infer import Segmenter

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--detector-src", default="")
ap.add_argument("--background", default="")
ap.add_argument("--clip", type=int, default=600, help="frames in the annotated clip (0 = none)")
a = ap.parse_args()

seg = Segmenter(a.ckpt)
det = None
if a.detector_src:
    sys.path.insert(0, a.detector_src)
    from detector.config import Settings
    from detector.pipeline import Detector
    det = Detector(Settings(background_path=a.background))

COL = [(255, 255, 0), (255, 0, 255), (0, 200, 255), (0, 120, 255)]
cap = cv2.VideoCapture(a.video)
n_tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
vw = cv2.VideoWriter(a.out + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30 / a.step, (720, 720)) if a.clip else None
P, S, D, idxs = [], [], [], []
buf, ids, i, t_model, t_det = [], [], 0, 0.0, 0.0


def flush():
    global buf, ids, t_model, t_det
    if not buf:
        return
    t = time.time()
    p, s = seg(buf)
    t_model += time.time() - t
    for f, k, pp, ss in zip(buf, ids, p, s):
        P.append(pp), S.append(ss), idxs.append(k)
        row = np.full((4, 2), np.nan, np.float32)
        if det is not None:
            t = time.time()
            d = det.detect(f)
            t_det += time.time() - t
            for j, (tool, key) in enumerate([("forceps", "left_tip"), ("forceps", "right_tip"), ("shadow", "left_tip"), ("shadow", "right_tip")]):
                v = getattr(getattr(d, tool), key)
                if v is not None:
                    row[j] = v
        D.append(row)
        if vw is not None and len(P) <= a.clip:
            im = f.copy()
            for j in range(4):
                if ss[j] > seg.min_score:
                    cv2.circle(im, tuple(int(round(v)) for v in pp[j]), 9, COL[j], 2, cv2.LINE_AA)
                if det is not None and not np.isnan(row[j, 0]):
                    cv2.drawMarker(im, tuple(int(round(v)) for v in row[j]), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
            cv2.putText(im, f"#{k}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            vw.write(cv2.resize(im, (720, 720)))
    buf, ids = [], []


while True:
    ok, f = cap.read()
    if not ok:
        break
    if i % a.step == 0:
        buf.append(f), ids.append(i)
        if len(buf) == 16:
            flush()
    i += 1
flush()
if vw is not None:
    vw.release()
np.savez(a.out + ".npz", idx=np.array(idxs), pts=np.array(P), score=np.array(S), det=np.array(D))
n = len(idxs)
print(f"frames {n}  model {t_model / n * 1000:.1f} ms/frame" + (f"  detector {t_det / n * 1000:.1f} ms/frame" if det else ""))
