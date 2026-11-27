"""Where the learned detector is better: shadow found rate vs. tip-to-shadow distance, plus one example frame.

Inputs: conditions.json (conditions.py). Output: ../detectors_distance.pdf

    PYTHONPATH=services/detector/src:services/mlseg/src:report/figures/detectors \
        .venv-ml/bin/python report/figures/detectors/plot.py      # from the repository root
"""
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark import CKPT, ROOT, classical, read_video
from mlseg.detector import MLDetector

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, GRAY = "#0065BD", "#E37222", "#999999"
BINS = [(0, 80), (80, 160), (160, 320), (320, 10_000)]
EXAMPLE = 1860          # bober2.mp4: instrument high above the retina, faint shadow


def main():
    rows = [r for r in json.load(open(HERE / "conditions.json"))["unseen"] if "distance" in r]
    labels, cl, ml, n = [], [], [], []
    for lo, hi in BINS:
        sel = [r for r in rows if lo <= r["distance"] < hi]
        labels.append(f"{lo}–{hi}" if hi < 10_000 else f"> {lo}")
        cl.append(np.mean([r["classical"] for r in sel]))
        ml.append(np.mean([r["learned"] for r in sel]))
        n.append(len(sel))

    data = read_video(ROOT / "data/video/bober2.mp4", list(range(0, 240, 8)) + [EXAMPLE])
    det_c = classical([data[k] for k in range(0, 240, 8)])
    det_m = MLDetector(str(CKPT), device="mps", shadow_hold_s=0)
    frame = data[EXAMPLE]
    work = det_m._prepare(frame)[0]
    c, m = det_c.detect(frame), det_m.detect(frame)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, im) = plt.subplots(1, 2, figsize=(6.3, 2.6), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(BINS))
    w = 0.38
    ax.bar(x - w / 2, cl, w, color=BLUE, label="Classical")
    ax.bar(x + w / 2, ml, w, color=ORANGE, label="Learned")
    for i in range(len(BINS)):
        ax.text(x[i] - w / 2, cl[i] + 0.02, f"{cl[i]:.0%}", ha="center", fontsize=6.5)
        ax.text(x[i] + w / 2, ml[i] + 0.02, f"{ml[i]:.0%}", ha="center", fontsize=6.5)
        ax.text(x[i], -0.17, f"n = {n[i]}", ha="center", fontsize=6.5, color="#666666",
                transform=ax.get_xaxis_transform())
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_ylabel("Frames with shadow found")
    ax.set_xlabel("Distance between forceps and shadow tips (px)", labelpad=12)
    ax.set_title("(a) Shadow found vs. distance", fontsize=7.5, loc="left")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.13), ncol=2, fontsize=7)
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)

    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    pts = [p for p in (m.shadow.left_tip, m.shadow.right_tip, m.shadow.junction, m.forceps.left_tip, m.forceps.right_tip) if p]
    pts = np.array(pts)
    cx, cy = pts.mean(0).astype(int)
    half = int(max(200, np.abs(pts - [cx, cy]).max() + 60))
    x0, y0 = int(np.clip(cx - half, 0, 1080 - 2 * half)), int(np.clip(cy - half, 0, 1080 - 2 * half))
    im.imshow(rgb[y0:y0 + 2 * half, x0:x0 + 2 * half])
    for tool, marker in (("forceps", "o"), ("shadow", "o")):
        p = getattr(m, tool)
        if p.ok:
            for q in (p.left_tip, p.right_tip):
                im.plot(q[0] - x0, q[1] - y0, marker, ms=4.5, mfc="white", mec=ORANGE, mew=1.4)
    im.set_xlim(0, 2 * half), im.set_ylim(2 * half, 0)
    im.set_xticks([]), im.set_yticks([])
    for s in im.spines.values():
        s.set_visible(False)
    im.set_title("(b) Faint shadow, found only by the learned detector", fontsize=7.5, loc="left")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(HERE.parent / "detectors_distance.pdf", bbox_inches="tight", dpi=300)
    print("classical shadow:", c.shadow.ok, c.shadow.reason, "| learned shadow:", m.shadow.ok)
    print(list(zip(labels, n, [round(v, 2) for v in cl], [round(v, 2) for v in ml])))


if __name__ == "__main__":
    main()
