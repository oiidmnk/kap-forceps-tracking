"""Raw local-contrast Otsu mask ONLY -- no morphology, no component filtering.

Divide the LAB lightness channel by a heavily-blurred copy of itself (an
estimate of the slowly-varying illumination field), then Otsu-threshold the
ratio. Regions locally darker than their surroundings -- the forceps and its
cast shadow -- come out white. Nothing is removed afterward, so faint shadow
pixels are preserved (at the cost of keeping vessels).

CLI (single image, directory, or glob):
    python raw_local_contrast.py images/forceps_sample.jpg
    python raw_local_contrast.py images/ --out out/raw
    python raw_local_contrast.py "clips/*.jpg" --masks-only --sigma-frac 0.08

Import:
    from raw_local_contrast import raw_local_contrast_mask
    mask = raw_local_contrast_mask(bgr_image)
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def raw_local_contrast_mask(image: np.ndarray, sigma_frac: float = 0.08) -> np.ndarray:
    """Return a uint8 {0,255} mask of locally-dark regions (tool + shadow).

    sigma_frac is the Gaussian sigma of the illumination estimate as a
    fraction of the smaller image dimension, so the same value works at any
    resolution. Larger -> more of the shadow's own darkness is preserved but
    more low-frequency shading leaks in; smaller -> vessels register more.
    """
    if image is None or image.size == 0:
        raise ValueError("empty image")
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32) + 1.0
    sigma = sigma_frac * min(image.shape[:2])
    background = cv2.GaussianBlur(lightness, (0, 0), sigmaX=sigma, sigmaY=sigma) + 1.0
    ratio_u8 = np.clip((lightness / background) * 128, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(ratio_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.6) -> np.ndarray:
    out = image.copy()
    sel = mask > 0
    out[sel] = (alpha * np.array(color) + (1 - alpha) * out[sel]).astype(np.uint8)
    return out


def find_images(source: str) -> list[Path]:
    path = Path(source)
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    matches = sorted(Path(m) for m in glob.glob(source))
    if matches:
        return [p for p in matches if p.suffix.lower() in IMAGE_EXTS]
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]
    return []


def run_batch(args: argparse.Namespace) -> None:
    images = find_images(args.source)
    if not images:
        print(f"no images found for: {args.source}")
        return
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"skip (unreadable): {path}")
            continue
        mask = raw_local_contrast_mask(image, sigma_frac=args.sigma_frac)
        cv2.imwrite(str(out_dir / f"{path.stem}_mask.png"), mask)
        if not args.masks_only:
            cv2.imwrite(str(out_dir / f"{path.stem}_overlay.png"), overlay_mask(image, mask))
        print(f"{path.name}: mask coverage {100.0 * (mask > 0).mean():5.2f}%")
    print(f"\ndone -> {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="image file, directory, or glob (quote globs)")
    p.add_argument("--out", default="out/raw", help="output directory")
    p.add_argument("--sigma-frac", type=float, default=0.08, dest="sigma_frac")
    p.add_argument("--masks-only", action="store_true", help="skip writing overlays")
    return p


if __name__ == "__main__":
    run_batch(build_parser().parse_args())
