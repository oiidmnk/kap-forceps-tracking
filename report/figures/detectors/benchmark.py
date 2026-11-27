"""Classical vs. learned detector on this Mac (Apple M3 Pro).

  speed     ms per frame, from the 1080 px frame to the points (detect()), on the 43 labelled frames
  accuracy  forceps / shadow tips on the 43 labelled frames (data/annotated). They come from the same video
            as the learned model's training frames, so its numbers are optimistic.
  coverage  share of frames in which forceps and shadow (both tips each) are found, on every 5th frame
            (0-4000, a tool is in view) of data/video/bober2.mp4: a forceps recording without the rod that
            neither detector was developed or trained on (not in the repository; copy it there)

The learned detector runs without its 1 s shadow hold, so every frame counts on its own; the classical one
learns its background from the first 8 s of each video, as in live use.
Output: results.json next to this file.

    PYTHONPATH=services/detector/src:services/mlseg/src .venv-ml/bin/python report/figures/detectors/benchmark.py
"""
import glob
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from detector.background import Background
from detector.config import Settings
from detector.pipeline import Detector
from mlseg.detector import MLDetector

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CKPT = ROOT / "services/mlseg/weights/last.pt"
OTHER_VIDEO = ROOT / "data/video/bober2.mp4"
OTHER_FRAMES = list(range(0, 4000, 5))


def read_video(path, wanted):
    wanted, out = set(wanted), {}
    cap = cv2.VideoCapture(str(path))
    i = 0
    while wanted - out.keys():
        ok, f = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = f
        i += 1
    return out


def classical(first_frames):
    d = Detector(Settings(background_path=""))
    d.bg = Background(None, 1080)
    for f in first_frames:
        d.bg.add_sample(d._prepare(f)[0])
    return d


def pair_error(pred, true):
    same = [math.dist(pred[0], true[0]), math.dist(pred[1], true[1])]
    swap = [math.dist(pred[0], true[1]), math.dist(pred[1], true[0])]
    return min(same, swap, key=sum)


def timed(d, frames, sync=lambda: None, reps=3):
    for f in frames[:5]:
        d.detect(f)
    sync()
    t = time.perf_counter()
    for _ in range(reps):
        for f in frames:
            d.detect(f)
    sync()
    return (time.perf_counter() - t) / (reps * len(frames)) * 1000


def main():
    labelled = []
    for jf in sorted(glob.glob(str(ROOT / "data/annotated/*.json"))):
        p = {s["label"]: s["points"][0] for s in json.load(open(jf))["shapes"]}
        labelled.append((cv2.imread(jf[:-5] + ".png"), [p["left_tip"], p["right_tip"]],
                         [p["left_shadow"], p["right_shadow"]]))
    lev1_first = list(read_video(ROOT / "data/video/forceps_lev1.mp4", range(0, 240, 8)).values())
    other = read_video(OTHER_VIDEO, list(range(0, 240, 8)) + OTHER_FRAMES)
    other_first = [other[k] for k in range(0, 240, 8)]

    mps_sync = torch.mps.synchronize if torch.backends.mps.is_available() else (lambda: None)
    dets = {
        "classical (CPU)": (lambda first: classical(first), lambda: None),
        "learned (GPU)": (lambda first: MLDetector(str(CKPT), device="mps", shadow_hold_s=0), mps_sync),
        "learned (CPU)": (lambda first: MLDetector(str(CKPT), device="cpu", shadow_hold_s=0), lambda: None),
    }
    frames = [f for f, _, _ in labelled]
    results = {}
    for name, (make, sync) in dets.items():
        r = {"ms_per_frame": timed(make(lev1_first), frames, sync, reps=3 if "CPU" not in name or "classical" in name else 1)}
        if name != "learned (CPU)":                      # same outputs as on the GPU
            d = make(lev1_first)
            for key, idx in (("forceps", 1), ("shadow", 2)):
                errs = []
                for item in labelled:
                    pose = getattr(d.detect(item[0]), key)
                    errs += pair_error((pose.left_tip, pose.right_tip), item[idx]) if pose.ok else [math.inf] * 2
                errs = np.array(errs)
                r[f"{key}_median_px"] = float(np.median(errs))
                r[f"{key}_within_10px"] = float(np.mean(errs <= 10))
            d = make(other_first)
            both = [(det.forceps.ok, det.shadow.ok) for det in (d.detect(other[k]) for k in OTHER_FRAMES)]
            r["other_forceps_found"] = float(np.mean([a for a, _ in both]))
            r["other_both_found"] = float(np.mean([a and b for a, b in both]))
        results[name] = r
        print(name, {k: round(v, 3) for k, v in r.items()}, flush=True)
    results["n"] = {"labelled": len(labelled), "other_video": OTHER_VIDEO.name, "other_frames": len(OTHER_FRAMES), "device": "Apple M3 Pro"}
    json.dump(results, open(HERE / "results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
