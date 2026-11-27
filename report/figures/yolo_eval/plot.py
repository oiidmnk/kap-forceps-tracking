"""Old YOLO pose models vs. the classical detector on the labelled simulator frames.

Inputs: yolo.json (run_yolo.py, checkpoints from the old auge_segment and
kap-forceps-tracking projects), classic.json (run_classic.py) and
yolo26s_val_frames.txt (the 15 of the 43 frames that YOLO26s was not trained on).
Output: ../yolo_vs_classical.pdf

    python report/figures/yolo_eval/plot.py   # from the repository root
"""
import glob
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BLUE, BLUE_LIGHT, ORANGE, ORANGE_LIGHT, GRAY = "#0065BD", "#64A0C8", "#E37222", "#F1B48A", "#999999"
CONF = 0.05      # a detection below this confidence counts as missed
TOL = 10.0       # px, on the 1080 px frame

# (key in yolo.json, label, training data); the classical detector is added on top
MODELS = [
    ("yolo11s_eyemodel", "YOLO11s-pose", "eye model"),
    ("yolo26n_eyemodel", "YOLO26n-pose", "eye model"),
    ("yolo26n_synth_a", "YOLO26n-pose (a)", "synthetic"),
    ("yolo26n_synth_b", "YOLO26n-pose (b)", "synthetic"),
    ("yolo26n_synth_c", "YOLO26n-pose (c)", "synthetic"),
    ("yolo11n_synth_d", "YOLO11n-pose", "synthetic"),
    ("yolo26x_synth_a", "YOLO26x-pose (a)", "synthetic"),
    ("yolo26x_synth_b", "YOLO26x-pose (b)", "synthetic"),
    ("yolo26s_sim", "YOLO26s-pose", "simulator*"),
]


def ground_truth():
    gt = {}
    for f in glob.glob(str(ROOT / "data/annotated/*.json")):
        d = {s["label"]: s["points"][0] for s in json.load(open(f))["shapes"]}
        gt[os.path.basename(f)[:-5] + ".png"] = {"tips": [d["left_tip"], d["right_tip"]],
                                                 "shadows": [d["left_shadow"], d["right_shadow"]]}
    return gt


def pair_errors(pred, true):
    """Errors of a left/right pair; the sides may be swapped (the projects label them differently)."""
    same = [math.dist(pred[0], true[0]), math.dist(pred[1], true[1])]
    swap = [math.dist(pred[0], true[1]), math.dist(pred[1], true[0])]
    return min(same, swap, key=sum)


def score(get, gt, frames, key):
    """Share of keypoints within TOL px and the median error; a missed frame counts as infinite."""
    errs = []
    for frame in frames:
        pred = get(frame, key)
        errs += [math.inf, math.inf] if pred is None else pair_errors(pred, gt[frame][key])
    errs = np.array(errs)
    return float(np.mean(errs <= TOL)), float(np.median(errs))


def main():
    gt = ground_truth()
    all_frames = sorted(gt)
    val15 = sorted(open(HERE / "yolo26s_val_frames.txt").read().split())
    yolo = json.load(open(HERE / "yolo.json"))
    classic = json.load(open(HERE / "classic.json"))["classical"]

    def classic_get(frame, key):
        p = classic.get(frame)
        return None if p is None else (p["kpts"][:2] if key == "tips" else p["kpts"][2:])

    rows = [("Classical detector", "no training", classic_get, all_frames)]
    for key, label, data in MODELS:
        frames = val15 if key == "yolo26s_sim" else all_frames

        def get(frame, k, per=yolo[key]["frames"]):
            p = per[frame][k]
            return p["kpts"] if p and p["conf"] >= CONF else None

        rows.append((label, data, get, frames))

    results = [(label, data, score(get, gt, frames, "tips"), score(get, gt, frames, "shadows"))
               for label, data, get, frames in rows]
    for label, data, (tw, tm), (sw, sm) in results:
        print(f"{label:20s} {data:12s} tips {tw:4.0%} (median {tm:7.1f})  shadows {sw:4.0%} (median {sm:7.1f})")

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    y = np.arange(len(results))[::-1].astype(float)
    h = 0.36
    for yi, (label, data, (tw, _), (sw, _)) in zip(y, results):
        classical = data == "no training"
        ax.barh(yi + h / 2, tw, h, color=BLUE if classical else ORANGE, label=None)
        ax.barh(yi - h / 2, sw, h, color=BLUE_LIGHT if classical else ORANGE_LIGHT, label=None)
        for v, off in ((tw, h / 2), (sw, -h / 2)):
            ax.text(v + 0.01, yi + off, f"{v:.0%}", va="center", fontsize=6.5, color="#444444")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{label}  ·  {data}" for label, data, _, _ in results])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.axhline(y[0] - 0.6, color=GRAY, lw=0.6)
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_xlabel(f"Keypoints within {TOL:g} px of the label")
    ax.grid(axis="x", color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    handles = [matplotlib.patches.Patch(color=ORANGE, label="Forceps tips"),
               matplotlib.patches.Patch(color=ORANGE_LIGHT, label="Shadow tips")]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(HERE.parent / "yolo_vs_classical.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
