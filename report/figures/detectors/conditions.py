"""Where is the learned detector better? Shadow found rate of both detectors, split by situation.

Situations (no ground truth; the shadow position for the split comes from the learned detector if it found
the shadow, else from the classical one, so the split does not favour either detector):
  - distance between forceps tips and shadow tips, in bins (the shadow approaches the tip before contact)
  - shadow tips over the bright optic disc (background brightness at the shadow above its 99th percentile)
  - a red rod in view: the blocks of data/video/rod.mp4 the learned model was not trained on

Data: every 5th frame (0-4000) of data/video/bober2.mp4, which neither detector was developed or trained on,
and every 2nd frame of rod.mp4's held-out blocks (2400-3200, 6000-6800). Classical detector with the background
learned from the first 8 s of each video; learned detector without its shadow hold.
Output: conditions.json next to this file.

    PYTHONPATH=services/detector/src:services/mlseg/src .venv-ml/bin/python report/figures/detectors/conditions.py
"""
import json
from pathlib import Path

import numpy as np

from benchmark import CKPT, ROOT, classical, read_video
from mlseg.detector import MLDetector

HERE = Path(__file__).resolve().parent
BINS = [0, 40, 80, 160, 320, 10_000]          # px, mean distance tip -> shadow tip of the same side


def run(video, frames):
    data = read_video(video, list(range(0, 240, 8)) + frames)
    cl = classical([data[k] for k in range(0, 240, 8)])
    ml = MLDetector(str(CKPT), device="mps", shadow_hold_s=0)
    bg = cl.bg.v
    disc_level = np.percentile(bg[bg > 0], 99)
    rows = []
    for k in frames:
        c, m = cl.detect(data[k]), ml.detect(data[k])
        ref = m if m.shadow.ok and m.forceps.ok else c if c.shadow.ok and c.forceps.ok else None
        row = {"frame": k, "classical": bool(c.shadow.ok), "learned": bool(m.shadow.ok),
               "forceps_any": bool(c.forceps.ok or m.forceps.ok)}
        if ref is not None:
            tips = np.array([ref.forceps.left_tip, ref.forceps.right_tip])
            shadow = np.array([ref.shadow.left_tip, ref.shadow.right_tip])
            same = np.linalg.norm(tips - shadow, axis=1).mean()
            swap = np.linalg.norm(tips - shadow[::-1], axis=1).mean()
            row["distance"] = float(min(same, swap))
            xy = np.clip(shadow.mean(0).astype(int), 0, bg.shape[0] - 1)
            row["over_disc"] = bool(bg[xy[1], xy[0]] > disc_level)
        rows.append(row)
    return rows


def rate(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def main():
    unseen = run(ROOT / "data/video/bober2.mp4", list(range(0, 4000, 5)))
    rod = run(ROOT / "data/video/rod.mp4", list(range(2400, 3200, 2)) + list(range(6000, 6800, 2)))
    with_shadow = [r for r in unseen if "distance" in r]
    out = {"all frames": {"n": len(unseen), "classical": rate(unseen, "classical"), "learned": rate(unseen, "learned")}}
    for lo, hi in zip(BINS, BINS[1:]):
        sel = [r for r in with_shadow if lo <= r["distance"] < hi]
        out[f"distance {lo}-{hi} px"] = {"n": len(sel), "classical": rate(sel, "classical"), "learned": rate(sel, "learned")}
    for name, flag in (("shadow over optic disc", True), ("shadow elsewhere", False)):
        sel = [r for r in with_shadow if r["over_disc"] == flag]
        out[name] = {"n": len(sel), "classical": rate(sel, "classical"), "learned": rate(sel, "learned")}
    sel = [r for r in rod if r["forceps_any"]]
    out["red rod in view"] = {"n": len(sel), "classical": rate(sel, "classical"), "learned": rate(sel, "learned")}
    for k, v in out.items():
        print(f"{k:28s} n={v['n']:4d}  classical {v['classical']:.0%}  learned {v['learned']:.0%}")
    json.dump({"summary": out, "unseen": unseen, "rod": rod}, open(HERE / "conditions.json", "w"))


if __name__ == "__main__":
    main()
