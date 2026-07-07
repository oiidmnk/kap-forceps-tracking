"""Config-driven image preprocessing shared by training and inference tools."""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from scripts.common import IMAGE_EXTENSIONS, REPO_ROOT

DEFAULT_PREPROCESS_CONFIG = REPO_ROOT / "configs" / "preprocessing.yaml"


@dataclass(frozen=True)
class CropTransform:
    source_width: int
    source_height: int
    x: int
    y: int
    width: int
    height: int

    @property
    def is_identity(self) -> bool:
        return (
            self.x == 0
            and self.y == 0
            and self.width == self.source_width
            and self.height == self.source_height
        )


@dataclass(frozen=True)
class PreprocessResult:
    image: np.ndarray
    transform: CropTransform


def load_preprocess_presets(config_path: Path = DEFAULT_PREPROCESS_CONFIG) -> dict[str, dict]:
    with config_path.open() as file:
        config = yaml.safe_load(file) or {}
    presets = config.get("presets")
    if not isinstance(presets, dict) or not presets:
        raise ValueError(f"{config_path} must define a non-empty 'presets' mapping")
    return presets


def load_preprocess_preset(
    name: str,
    config_path: Path = DEFAULT_PREPROCESS_CONFIG,
) -> dict:
    presets = load_preprocess_presets(config_path)
    if name not in presets:
        available = ", ".join(sorted(presets))
        raise ValueError(f"unknown preprocessing preset '{name}'; available: {available}")
    return presets[name]


def find_images(source: str | Path) -> list[Path]:
    source_text = str(source)
    path = Path(source_text)
    if path.is_dir():
        return sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
    matches = sorted(Path(match) for match in glob.glob(source_text))
    if matches:
        return [
            item
            for item in matches
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ]
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    return []


def _roi_geometry(image: np.ndarray, roi: dict) -> tuple[int, int, int]:
    height, width = image.shape[:2]
    center = roi.get("center", [0.5, 0.5])
    radius = float(roi.get("radius", 0.48))
    if len(center) != 2:
        raise ValueError("roi.center must contain [x, y]")
    if radius <= 0:
        raise ValueError("roi.radius must be greater than 0")
    return (
        int(round(float(center[0]) * width)),
        int(round(float(center[1]) * height)),
        int(round(radius * min(width, height))),
    )


def _crop_to_roi(image: np.ndarray, roi: dict) -> PreprocessResult:
    source_height, source_width = image.shape[:2]
    center_x, center_y, radius = _roi_geometry(image, roi)
    x1 = max(0, center_x - radius)
    y1 = max(0, center_y - radius)
    x2 = min(source_width, center_x + radius)
    y2 = min(source_height, center_y + radius)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI crop does not intersect the image")
    transform = CropTransform(
        source_width=source_width,
        source_height=source_height,
        x=x1,
        y=y1,
        width=x2 - x1,
        height=y2 - y1,
    )
    return PreprocessResult(image=image[y1:y2, x1:x2].copy(), transform=transform)


def _apply_bilateral(image: np.ndarray, config: dict) -> np.ndarray:
    if not config.get("enabled", False):
        return image
    return cv2.bilateralFilter(
        image,
        d=int(config.get("diameter", 5)),
        sigmaColor=float(config.get("sigma_color", 30)),
        sigmaSpace=float(config.get("sigma_space", 30)),
    )


def _apply_clahe(image: np.ndarray, config: dict) -> np.ndarray:
    if not config.get("enabled", False):
        return image
    tile_grid = config.get("tile_grid", [8, 8])
    if len(tile_grid) != 2:
        raise ValueError("clahe.tile_grid must contain [width, height]")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(config.get("clip_limit", 2.0)),
        tileGridSize=(int(tile_grid[0]), int(tile_grid[1])),
    )
    enhanced = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((enhanced, channel_a, channel_b)), cv2.COLOR_LAB2BGR)


def _apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be greater than 0")
    if abs(gamma - 1.0) < 1e-9:
        return image
    lookup = np.array(
        [((value / 255.0) ** gamma) * 255.0 for value in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, lookup)


def _compress_highlights(image: np.ndarray, config: dict) -> np.ndarray:
    if not config.get("enabled", False):
        return image
    threshold = float(config.get("threshold", 235))
    strength = float(config.get("strength", 0.5))
    if not 0.0 <= strength <= 1.0:
        raise ValueError("highlight_compression.strength must be within [0, 1]")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = lab[:, :, 0]
    mask = lightness > threshold
    lightness[mask] = threshold + (lightness[mask] - threshold) * strength
    lab[:, :, 0] = np.clip(lightness, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _apply_sharpen(image: np.ndarray, config: dict) -> np.ndarray:
    if not config.get("enabled", False):
        return image
    amount = float(config.get("amount", 0.3))
    sigma = float(config.get("sigma", 1.0))
    if amount < 0:
        raise ValueError("sharpen.amount must be non-negative")
    if sigma <= 0:
        raise ValueError("sharpen.sigma must be greater than 0")
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def _apply_roi_mask(image: np.ndarray, roi: dict) -> np.ndarray:
    center_x, center_y, radius = _roi_geometry(image, roi)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1, lineType=cv2.LINE_AA)
    fill = np.asarray(roi.get("fill", [0, 0, 0]), dtype=np.uint8)
    if fill.shape != (3,):
        raise ValueError("roi.fill must contain three BGR values")
    background = np.empty_like(image)
    background[:] = fill
    foreground = cv2.bitwise_and(image, image, mask=mask)
    background = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))
    return cv2.add(foreground, background)


def apply_preprocessing(image: np.ndarray, preset: dict) -> PreprocessResult:
    if image is None or image.size == 0:
        raise ValueError("cannot preprocess an empty image")

    source_height, source_width = image.shape[:2]
    transform = CropTransform(source_width, source_height, 0, 0, source_width, source_height)
    roi = preset.get("roi", {})
    roi_mode = roi.get("mode", "none")
    if roi_mode not in {"none", "mask", "crop"}:
        raise ValueError("roi.mode must be one of: none, mask, crop")
    if roi_mode == "crop":
        cropped = _crop_to_roi(image, roi)
        output = cropped.image
        transform = cropped.transform
    else:
        output = image.copy()

    output = _apply_bilateral(output, preset.get("bilateral", {}))
    output = _apply_clahe(output, preset.get("clahe", {}))
    output = _apply_gamma(output, float(preset.get("gamma", 1.0)))
    output = _compress_highlights(output, preset.get("highlight_compression", {}))
    output = _apply_sharpen(output, preset.get("sharpen", {}))
    if roi_mode == "mask":
        output = _apply_roi_mask(output, roi)
    return PreprocessResult(image=output, transform=transform)
