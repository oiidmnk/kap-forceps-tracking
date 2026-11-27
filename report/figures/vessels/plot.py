"""Why vessels are a problem for the shadow mask, and how the classical detector removes them.

Runs the detector's own mask code on one labelled frame, with the background
learned like in live use (see benchmark.py), and shows four steps: the frame,
a naive "darker than the average retina" mask, the ratio to the background,
and the strip around the shaft (masks.isolate_tool). Panels (c) and (d) mark
the shadow tips the geometry finds on that mask, (a) the hand labels.
Output: ../vessels.pdf

    PYTHONPATH=services/detector/src python report/figures/vessels/plot.py [frame]   # from the repo root
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bench  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FRAME = sys.argv[1] if len(sys.argv) > 1 else "frame_000044"
ORANGE = np.array([227, 114, 34]) / 255
BLUE, GREEN = "#0065BD", "#A2AD00"


def overlay(rgb, mask, colour, alpha=0.75):
    out = rgb * 0.55
    out[mask > 0] = (1 - alpha) * out[mask > 0] + alpha * colour
    return out


def main():
    cap = cv2.VideoCapture(str(ROOT / "data/video/forceps_lev1.mp4"))
    frames = [cap.read()[1] for _ in range(240)]
    d, cfg = bench.make_detector("learned", frames)
    work = d._prepare(cv2.imread(str(ROOT / f"data/annotated/{FRAME}.png")))[0]
    masks = bench.shadow_masks(work, d, cfg)
    pts = {s["label"]: s["points"][0] for s in json.load(open(ROOT / f"data/annotated/{FRAME}.json"))["shapes"]}
    label = np.array([pts["left_shadow"], pts["right_shadow"]])
    found = {v: bench.tips(masks[v]) for v in "BC"}

    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    allp = np.vstack([label] + [np.array(p) for p in found.values() if p])
    cx, cy = allp.mean(0).astype(int)
    half = int(max(230, np.abs(allp - [cx, cy]).max() + 70))
    x0, y0 = int(np.clip(cx - half, 0, 1080 - 2 * half)), int(np.clip(cy - half, 0, 1080 - 2 * half))
    crop = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))

    panels = [("Frame, labelled tips", rgb, label, GREEN),
              ("Global threshold", overlay(rgb, masks["A"], ORANGE), None, None),
              ("Step 1: ratio", overlay(rgb, masks["B"], ORANGE), found["B"], BLUE),
              ("Step 2: strip", overlay(rgb, masks["C"], ORANGE), found["C"], BLUE)]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 1.85))
    for i, (ax, (title, img, points, colour)) in enumerate(zip(axes, panels)):
        ax.imshow(np.clip(img[crop], 0, 1))
        if points is not None:
            for p in points:
                ax.plot(p[0] - x0, p[1] - y0, "o", ms=3.8, mfc="white", mec=colour, mew=1.3)
        ax.set_xlim(0, 2 * half), ax.set_ylim(2 * half, 0)
        ax.set_title(f"({'abcd'[i]}) {title}", fontsize=7.5, loc="left")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(pad=0.2, w_pad=0.4)
    fig.savefig(HERE.parent / "vessels.pdf", bbox_inches="tight", dpi=300)
    for v in "BC":
        p = found[v]
        print(v, "no shadow" if p is None else f"error {max(bench.pair_error(p, label.tolist())):.1f} px")


if __name__ == "__main__":
    main()
