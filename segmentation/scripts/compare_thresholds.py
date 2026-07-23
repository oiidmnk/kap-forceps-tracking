"""Side-by-side comparison of dynamic-thresholding candidates for forceps/shadow
segmentation, run against real frames before wiring anything into the YOLO
preprocessing pipeline.

Usage:
    .venv/bin/python compare_thresholds.py images/forceps_sample.jpg
    .venv/bin/python compare_thresholds.py images/*.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent / "out"


def _odd(value: float) -> int:
    value = int(round(value))
    return value if value % 2 == 1 else value + 1


def keep_largest_components(mask: np.ndarray, keep: int = 1, min_area_frac: float = 0.0003) -> np.ndarray:
    """Strip speckle noise: keep only the `keep` largest connected components
    above a minimum area (as a fraction of the image), so a clean 'this is
    the tool' blob survives instead of hundreds of stray pixels."""
    height, width = mask.shape[:2]
    min_area = min_area_frac * height * width
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    areas.sort(key=lambda item: item[1], reverse=True)
    cleaned = np.zeros_like(mask)
    for label_id, area in areas[:keep]:
        if area >= min_area:
            cleaned[labels == label_id] = 255
    return cleaned


def otsu_gray(img: np.ndarray) -> np.ndarray:
    """Baseline: single global threshold on grayscale. Expected to fail —
    included so you can see *why* a naive threshold isn't enough here."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def adaptive_saturation(img: np.ndarray, block_frac: float = 0.03, c: int = 5) -> np.ndarray:
    """Forceps candidate: the metal shaft/tips are near-gray (low saturation)
    against the strongly saturated orange/red fundus. Adaptive, not global,
    because illumination (vignette, light-pipe falloff) drifts across the
    frame, so a fixed saturation cutoff won't hold everywhere. block_frac is
    relative to image size so the neighborhood stays 'instrument-scale'
    rather than 'vessel-scale' regardless of input resolution."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    block_size = _odd(block_frac * min(img.shape[:2]))
    mask = cv2.adaptiveThreshold(
        sat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c
    )
    return keep_largest_components(mask, keep=3)


def lab_a_channel(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forceps candidate #2: LAB a* is the green-red opponent axis. Fundus
    tissue sits high on a* (red); metal sits near-neutral. Returns the raw
    channel (for visual inspection) and a cleaned Otsu threshold (largest
    connected component only, since at any instant there's one tool blob)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    _, raw_mask = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = keep_largest_components(raw_mask, keep=1)
    return a_channel, mask


def local_contrast_shadow(img: np.ndarray, sigma_frac: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """Shadow candidate: divide out a heavily-blurred version of the
    lightness channel to cancel the slowly-varying vignette/illumination
    field, then threshold what's left. A genuine shadow is a broad, soft
    local darkening relative to its own neighborhood -- this is a cheap
    illumination-invariant proxy (poor man's version of the log-chromaticity
    shadow-removal trick) that doesn't require calibrating a color model.
    sigma_frac is relative to image size: too small and thin vessels register
    as 'locally darker than neighborhood' too; it needs to sit above vessel
    scale and below shadow-blob scale."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float32) + 1.0
    sigma = sigma_frac * min(img.shape[:2])
    background = cv2.GaussianBlur(lightness, (0, 0), sigmaX=sigma, sigmaY=sigma) + 1.0
    ratio = lightness / background  # ~1.0 where locally normal, <1.0 where darker than surroundings
    ratio_u8 = np.clip(ratio * 128, 0, 255).astype(np.uint8)
    _, raw_mask = cv2.threshold(ratio_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # thin vessels survive the blur/ratio test too; open with a kernel wider
    # than a vessel but narrower than a shadow blob to strip them out
    kernel_size = _odd(0.01 * min(img.shape[:2]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    mask = keep_largest_components(opened, keep=2)
    return ratio_u8, mask


def combined_overlay(img: np.ndarray, instrument_mask: np.ndarray, shadow_mask: np.ndarray) -> np.ndarray:
    """Sanity-check overlay: green = instrument candidate, red = shadow
    candidate, on top of the original frame."""
    overlay = img.copy()
    overlay[instrument_mask > 0] = (0.4 * overlay[instrument_mask > 0] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[shadow_mask > 0] = (0.4 * overlay[shadow_mask > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    return overlay


def process(path: Path) -> None:
    img = cv2.imread(str(path))
    if img is None:
        print(f"skip (unreadable): {path}")
        return

    otsu_mask = otsu_gray(img)
    sat_mask = adaptive_saturation(img)
    a_channel, a_mask = lab_a_channel(img)
    ratio_img, shadow_mask = local_contrast_shadow(img)

    # LAB a* is the reliable instrument detector; the local-contrast detector
    # fires on the instrument too (it's also locally dark), so subtract the
    # known instrument region (dilated a touch to cover its boundary) to
    # isolate the cast-shadow-only candidate
    instrument_dilated = cv2.dilate(a_mask, np.ones((15, 15), np.uint8))
    shadow_only = cv2.bitwise_and(shadow_mask, cv2.bitwise_not(instrument_dilated))
    overlay = combined_overlay(img, a_mask, shadow_only)

    panels = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "original"),
        (otsu_mask, "1. Otsu on grayscale (baseline / expected to fail)"),
        (sat_mask, "2. adaptive threshold on HSV saturation (instrument)"),
        (a_channel, "3a. LAB a* channel (raw)"),
        (a_mask, "3b. LAB a* Otsu (instrument)"),
        (ratio_img, "4a. local-contrast ratio (raw)"),
        (shadow_mask, "4b. local-contrast Otsu (shadow)"),
        (cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), "5. combined overlay (green=tool, red=shadow)"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, (panel, title) in zip(axes.flat, panels):
        cmap = "gray" if panel.ndim == 2 else None
        ax.imshow(panel, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(path.name)
    fig.tight_layout()

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{path.stem}_compare.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        args = [str(p) for p in (Path(__file__).parent / "images").glob("*")]
    for arg in args:
        process(Path(arg))


if __name__ == "__main__":
    main()
