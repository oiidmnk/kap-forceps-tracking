"""Shadow mask before / after the vessel fixes, on the sample video.

Variants (all followed by the same tip/junction geometry, detector.geometry.analyze):
  A  darker than the average retina (one global threshold, no background image)
  B  ratio to the tool-free background, largest blob        (before commit 518d623)
  C  B + only the strip around the thick shaft (isolate_tool) (current)

Backgrounds: the static median image (data/background.png) and one learned like
in live use (80th percentile of 30 frames over the first 8 s while the tools move).

Measured on the 43 labelled frames (shadow tips: found, median error, <= 10 px)
and on every frame of data/video/forceps_lev1.mp4 (shadow found on frames with
a tool, frames 0-969; false shadow on the empty retina, frames 990-1260).
Output: results.json next to this file.

    PYTHONPATH=services/detector/src python report/figures/vessels/benchmark.py   # from the repo root
"""
import glob
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from detector.background import Background
from detector.config import Settings
from detector.geometry import analyze
from detector.masks import compute_masks, disc, isolate_tool
from detector.pipeline import Detector

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TOOL_FRAMES = range(0, 970)
EMPTY_FRAMES = range(990, 1261)


def shadow_masks(work, d, cfg):
    """The three shadow-mask variants for one normalised frame."""
    ref, from_model = d.bg.reference(work)
    eye = d.bg.disc if from_model else None
    forceps, shadow, _ = compute_masks(work, ref, from_model, cfg, eye, d.bg.bgr if from_model else None)
    analysis = disc(cfg.work_size, cfg.disc_eye if eye else cfg.disc_radius, eye)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1], cv2.GaussianBlur(hsv[..., 2].astype(np.float32), (0, 0), 1.0)
    retina = (analysis > 0) & (sat >= cfg.forceps_sat_max) & (forceps == 0)
    naive = ((val < cfg.shadow_ratio * np.median(val[retina])) & retina).astype(np.uint8)
    ratio = shadow * analysis
    return {"A": naive, "B": ratio, "C": isolate_tool(ratio)}


def tips(mask):
    j = analyze(mask)
    return (j.left_tip, j.right_tip) if j.ok else None


def pair_error(pred, true):
    same = [math.dist(pred[0], true[0]), math.dist(pred[1], true[1])]
    swap = [math.dist(pred[0], true[1]), math.dist(pred[1], true[0])]
    return min(same, swap, key=sum)


def make_detector(kind, frames):
    cfg = Settings(background_path=str(ROOT / "data/background.png") if kind == "static" else "")
    d = Detector(cfg)
    if kind == "learned":
        d.bg = Background(None, cfg.work_size)
        for k in range(0, 240, 8):                     # 30 samples over the first 8 s
            d.bg.add_sample(d._prepare(frames[k])[0])
    return d, cfg


def main():
    cap = cv2.VideoCapture(str(ROOT / "data/video/forceps_lev1.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    labelled = {}
    for f in sorted(glob.glob(str(ROOT / "data/annotated/*.json"))):
        pts = {s["label"]: s["points"][0] for s in json.load(open(f))["shapes"]}
        labelled[f[:-5] + ".png"] = [pts["left_shadow"], pts["right_shadow"]]

    results = {}
    for kind in ("static", "learned"):
        d, cfg = make_detector(kind, frames)
        res = {v: {"err": [], "tool_found": 0, "empty_false": 0} for v in "ABC"}
        for path, true in labelled.items():
            for v, m in shadow_masks(d._prepare(cv2.imread(path))[0], d, cfg).items():
                p = tips(m)
                res[v]["err"] += [math.inf, math.inf] if p is None else pair_error(p, true)
        for idx, key in ((TOOL_FRAMES, "tool_found"), (EMPTY_FRAMES, "empty_false")):
            for k in idx:
                for v, m in shadow_masks(d._prepare(frames[k])[0], d, cfg).items():
                    res[v][key] += tips(m) is not None
        out = {}
        for v, r in res.items():
            e = np.array(r["err"])
            out[v] = {"labelled_found": float(np.mean(np.isfinite(e[::2]))),
                      "labelled_median_px": float(np.median(e)),
                      "labelled_within_10px": float(np.mean(e <= 10)),
                      "tool_frames_found": r["tool_found"] / len(TOOL_FRAMES),
                      "empty_frames_false": r["empty_false"] / len(EMPTY_FRAMES)}
            print(f"{kind:8s} {v}  " + "  ".join(f"{k} {x:.3g}" for k, x in out[v].items()), flush=True)
        results[kind] = out
    results["n"] = {"labelled": len(labelled), "tool_frames": len(TOOL_FRAMES), "empty_frames": len(EMPTY_FRAMES)}
    json.dump(results, open(HERE / "results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
