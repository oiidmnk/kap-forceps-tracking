"""Build the 6-keypoint training set: real labels (+ detector junctions) and confident pseudo-labels from videos.
State per keypoint: 1 = present, 0 = absent (negative), -1 = ignore (masked in the loss)."""
import argparse, glob, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mlseg.infer import Segmenter

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--material", required=True)
ap.add_argument("--detector-src", required=True); ap.add_argument("--background", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--step", type=int, default=6)
ap.add_argument("--agree", type=float, default=3.0); ap.add_argument("--lo", type=float, default=0.2, help="unsure-but-verified keypoints: min model score")
ap.add_argument("--verify", type=float, default=8.0, help="max px between model peak and detector tip to count as verified"); ap.add_argument("--hi", type=float, default=0.7)
a = ap.parse_args()
sys.path.insert(0, a.detector_src)
from detector.config import Settings
from detector.pipeline import Detector
det = Detector(Settings(background_path=a.background))
seg = Segmenter(a.ckpt)
os.makedirs(a.out, exist_ok=True)


def detect(f):
    d = det.detect(f)
    return {t: (getattr(d, t) if getattr(d, t).ok else None) for t in ("forceps", "shadow")}


def junction_state(d, tool, tips):
    """detector junction is trusted only where its tips match ours."""
    p = d[tool]
    if p is None or tips is None:
        return None
    e = [np.hypot(*(np.array(getattr(p, k)) - t)) for k, t in zip(("left_tip", "right_tip"), tips)]
    return np.array(p.junction, np.float32) if max(e) < a.agree else None


def save(name, img, pts, st, kind):
    cv2.imwrite(f"{a.out}/{name}.png", img)
    json.dump({"pts": np.nan_to_num(np.array(pts)).tolist(), "state": [int(s) for s in st], "kind": kind}, open(f"{a.out}/{name}.json", "w"))


stats = {"real": 0, "pseudo": 0, "neg": 0, "skip": 0, "junc": 0}
# real labelled frames: tips are ground truth, junction from the detector where it agrees
for jf in sorted(glob.glob(f"{a.material}/frame_*.json")):
    d = json.load(open(jf)); lab = {s["label"]: s["points"][0] for s in d["shapes"]}
    img = cv2.imread(jf[:-4] + "png")
    pts = np.zeros((6, 2), np.float32); st = [1, 1, 1, 1, -1, -1]
    for i, k in enumerate(("left_tip", "right_tip", "left_shadow", "right_shadow")):
        pts[i] = lab[k]
    dd = detect(img)
    for j, (tool, ix) in enumerate((("forceps", (0, 1)), ("shadow", (2, 3)))):
        jn = junction_state(dd, tool, pts[list(ix)])
        if jn is not None:
            pts[4 + j], st[4 + j] = jn, 1; stats["junc"] += 1
    save(os.path.basename(jf)[:-5], img, pts, st, "real"); stats["real"] += 1

for vp in sorted(glob.glob(f"{a.material}/*.mp4")):
    tag = "v" + str(abs(hash(os.path.basename(vp))) % 1000)
    cap = cv2.VideoCapture(vp); i = 0; buf = []
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % a.step == 0: buf.append((i, f))
        if len(buf) == 16 or (not ok):
            pass
        i += 1
    for s in range(0, len(buf), 16):
        chunk = buf[s:s + 16]
        P, S = seg([f for _, f in chunk])
        for (k, f), p, sc in zip(chunk, P, S):
            dd = detect(f)
            pts = np.zeros((6, 2), np.float32); st = [-1] * 6
            for t, ix in enumerate(((0, 1), (2, 3))):
                sc2 = sc[list(ix)]
                if (sc2 > a.hi).all():
                    for q in ix: pts[q], st[q] = p[q], 1
                    jn = junction_state(dd, ("forceps", "shadow")[t], p[list(ix)])
                    if jn is not None: pts[4 + t], st[4 + t] = jn, 1; stats["junc"] += 1
                elif (sc2 > a.lo).all() and dd[("forceps", "shadow")[t]] is not None and max(
                        np.hypot(*(np.array(getattr(dd[("forceps", "shadow")[t]], k)) - p[q]))
                        for k, q in zip(("left_tip", "right_tip"), ix)) < a.verify:
                    # model unsure, but the independent classical detector agrees on the spot: a hard positive
                    for q in ix: pts[q], st[q] = p[q], 1
                    stats["hard"] = stats.get("hard", 0) + 1
                    jn = junction_state(dd, ("forceps", "shadow")[t], p[list(ix)])
                    if jn is not None: pts[4 + t], st[4 + t] = jn, 1
                elif (sc2 < 0.05).all() and dd[("forceps", "shadow")[t]] is None:
                    for q in ix: st[q] = 0
                    st[4 + t] = 0
            if all(x == -1 for x in st): stats["skip"] += 1; continue
            stats["neg" if all(x == 0 for x in st) else "pseudo"] += 1
            save(f"{tag}_{k:06d}", f, pts, st, "pseudo")
print(stats)
