"""Evaluate against the LabelMe annotations in data/annotated. Usage: evaluate.py [--no-background]"""
import argparse, glob, json
import cv2, numpy as np
from detector.background import Background
from detector.config import Settings
from detector.pipeline import Detector

ap = argparse.ArgumentParser()
ap.add_argument("--no-background", action="store_true", help="single-frame fallback (no background model)")
ap.add_argument("--dir", default="data/annotated")
a = ap.parse_args()
cfg = Settings(); det = Detector(cfg, Background("data/background.png" if not a.no_background else None))
errs, fails = {}, 0
for jf in sorted(glob.glob(f"{a.dir}/frame_*.json")):
    gt = {s["label"]: np.array(s["points"][0]) for s in json.load(open(jf))["shapes"]}
    d = det.detect(cv2.imread(jf.replace(".json", ".png")))
    for tool, (l, r) in {"forceps": ("left_tip", "right_tip"), "shadow": ("left_shadow", "right_shadow")}.items():
        p = getattr(d, tool)
        if not p.ok: fails += 1; print(jf, tool, "FAIL", p.reason); continue
        for key, lab in (("left_tip", l), ("right_tip", r)):
            errs.setdefault((tool, key), []).append(np.linalg.norm(np.array(getattr(p, key)) - gt[lab]))
for k, v in errs.items(): print(k, "n=%d mean=%.1f med=%.1f max=%.1f px" % (len(v), np.mean(v), np.median(v), np.max(v)))
print("failures:", fails)
