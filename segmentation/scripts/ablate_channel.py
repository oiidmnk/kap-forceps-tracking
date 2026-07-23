"""Honest test: is LAB a* actually better, or is the connected-component
cleanup doing all the work? Threshold grayscale / value / saturation / a* with
the SAME Otsu + SAME cleanup, and quantify separability, so nothing is doing
extra work behind the scenes."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent / "out"


def keep_largest(mask, keep=1, min_area_frac=0.0003):
    h, w = mask.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = sorted(((i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)),
                   key=lambda t: t[1], reverse=True)
    out = np.zeros_like(mask)
    for lid, area in areas[:keep]:
        if area >= min_area_frac * h * w:
            out[labels == lid] = 255
    return out


def otsu_inv(chan):
    thr, mask = cv2.threshold(chan, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thr, mask


def separability(chan, thr):
    """How cleanly does Otsu's threshold split this channel? Returns the
    between-class variance ratio (Otsu's own objective, normalized) — higher
    means a more genuinely bimodal 'tool vs tissue' split rather than an
    arbitrary cut through a unimodal blob."""
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256]).ravel()
    p = hist / hist.sum()
    idx = np.arange(256)
    w0 = p[:int(thr) + 1].sum()
    w1 = 1 - w0
    if w0 == 0 or w1 == 0:
        return 0.0
    mu0 = (idx[:int(thr) + 1] * p[:int(thr) + 1]).sum() / w0
    mu1 = (idx[int(thr) + 1:] * p[int(thr) + 1:]).sum() / w1
    between = w0 * w1 * (mu0 - mu1) ** 2
    total = ((idx - (idx * p).sum()) ** 2 * p).sum()
    return between / total if total else 0.0


def process(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        print("unreadable", path)
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    channels = {
        "gray": gray,
        "value(HSV V)": hsv[:, :, 2],
        "saturation": hsv[:, :, 1],
        "LAB a*": lab[:, :, 1],
    }

    cols = len(channels)
    fig, axes = plt.subplots(3, cols, figsize=(5 * cols, 12))
    for j, (name, chan) in enumerate(channels.items()):
        # saturation is the only one where "tool = low", others tool=dark too;
        # a* tool=neutral(low). All use BINARY_INV so tool should come out white.
        thr, raw = otsu_inv(chan)
        clean = keep_largest(raw, keep=1)
        sep = separability(chan, thr)
        tool_px = int((clean > 0).sum())

        axes[0, j].imshow(chan, cmap="gray")
        axes[0, j].set_title(f"{name}\nOtsu thr={thr:.0f}  separability={sep:.2f}", fontsize=10)
        axes[1, j].imshow(raw, cmap="gray")
        axes[1, j].set_title("raw Otsu mask (no cleanup)", fontsize=9)
        axes[2, j].imshow(clean, cmap="gray")
        axes[2, j].set_title(f"+ largest-component ({tool_px}px)", fontsize=9)
        for i in range(3):
            axes[i, j].axis("off")
    fig.suptitle(f"{path.name} — same Otsu + same cleanup, only the channel differs")
    fig.tight_layout()
    out = OUT_DIR / f"{path.stem}_ablate.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


def main():
    args = sys.argv[1:] or [str(Path(__file__).parent / "images/forceps_sample.jpg")]
    for a in args:
        process(Path(a))


if __name__ == "__main__":
    main()
