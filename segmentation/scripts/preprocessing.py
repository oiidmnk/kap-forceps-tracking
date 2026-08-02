"""Config-driven image preprocessing shared by training and inference tools."""

from __future__ import annotations

import glob
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
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
    preset = deepcopy(presets[name])
    reference_match = preset.get("reference_match", {})
    reference = reference_match.get("reference")
    if reference:
        reference_path = Path(reference)
        if not reference_path.is_absolute():
            reference_path = (config_path.parent / reference_path).resolve()
        reference_match["reference"] = str(reference_path)
    return preset


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


def _read_reference_image(path: Path) -> np.ndarray:
    source = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if source is None:
        raise FileNotFoundError(f"reference normalization image not found: {path}")
    if source.ndim == 2:
        return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    if source.shape[2] == 4:
        alpha = source[:, :, 3:4].astype(np.float32) / 255.0
        return np.clip(source[:, :, :3].astype(np.float32) * alpha, 0, 255).astype(
            np.uint8
        )
    return source[:, :, :3]


def _nonblack_mask(image: np.ndarray, threshold: float) -> np.ndarray:
    return np.max(image, axis=2) > threshold


def _robust_lab_statistics(
    image: np.ndarray,
    mask: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    if int(np.count_nonzero(mask)) < 32:
        raise ValueError("reference matching requires at least 32 non-black pixels")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab[mask]
    centers = np.median(pixels, axis=0).astype(np.float32)
    lower = np.percentile(pixels, lower_percentile, axis=0)
    upper = np.percentile(pixels, upper_percentile, axis=0)
    spans = np.maximum(upper - lower, 1.0).astype(np.float32)
    return centers, spans


def _background_candidate_mask(
    image: np.ndarray,
    valid_mask: np.ndarray,
    config: dict,
) -> np.ndarray:
    if not config.get("enabled", False):
        return valid_mask

    minimum_saturation = float(config.get("minimum_saturation", 35.0))
    minimum_value = float(config.get("minimum_value", 50.0))
    if not 0.0 <= minimum_saturation <= 255.0:
        raise ValueError(
            "reference_match.background_only.minimum_saturation must be within [0, 255]"
        )
    if not 0.0 <= minimum_value <= 255.0:
        raise ValueError(
            "reference_match.background_only.minimum_value must be within [0, 255]"
        )
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (
        valid_mask
        & (hsv[:, :, 1] >= minimum_saturation)
        & (hsv[:, :, 2] >= minimum_value)
    )


def _background_blend_alpha(
    image: np.ndarray,
    valid_mask: np.ndarray,
    candidate_mask: np.ndarray,
    config: dict,
) -> np.ndarray:
    if not config.get("enabled", False):
        return valid_mask.astype(np.float32)

    dilation_kernel = int(config.get("dilation_kernel", 9))
    if dilation_kernel < 1:
        raise ValueError(
            "reference_match.background_only.dilation_kernel must be at least 1"
        )
    if dilation_kernel % 2 == 0:
        dilation_kernel += 1
    feather_sigma = float(config.get("feather_sigma", 2.5))
    if feather_sigma < 0:
        raise ValueError(
            "reference_match.background_only.feather_sigma must be non-negative"
        )

    protected = (valid_mask & ~candidate_mask).astype(np.uint8) * 255
    if dilation_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_kernel, dilation_kernel),
        )
        protected = cv2.dilate(protected, kernel)
    background = (
        valid_mask & (protected == 0)
    ).astype(np.float32)
    if feather_sigma > 0:
        background = cv2.GaussianBlur(
            background,
            (0, 0),
            feather_sigma,
        )
    background[~valid_mask] = 0.0
    return np.clip(background, 0.0, 1.0)


@lru_cache(maxsize=8)
def _cached_reference_statistics(
    path_text: str,
    modified_time_ns: int,
    lower_percentile: float,
    upper_percentile: float,
    black_threshold: float,
    background_only: bool,
    minimum_saturation: float,
    minimum_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    del modified_time_ns
    reference = _read_reference_image(Path(path_text))
    valid_mask = _nonblack_mask(reference, black_threshold)
    mask = _background_candidate_mask(
        reference,
        valid_mask,
        {
            "enabled": background_only,
            "minimum_saturation": minimum_saturation,
            "minimum_value": minimum_value,
        },
    )
    return _robust_lab_statistics(
        reference,
        mask,
        lower_percentile,
        upper_percentile,
    )


def _apply_reference_match(image: np.ndarray, config: dict) -> np.ndarray:
    if not config.get("enabled", False):
        return image

    reference_value = config.get("reference")
    if not reference_value:
        raise ValueError("reference_match.reference is required when enabled")
    reference_path = Path(reference_value)
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"reference normalization image not found: {reference_path}"
        )

    lower_percentile = float(config.get("lower_percentile", 5.0))
    upper_percentile = float(config.get("upper_percentile", 95.0))
    if not 0.0 <= lower_percentile < 50.0 < upper_percentile <= 100.0:
        raise ValueError(
            "reference_match percentiles must satisfy "
            "0 <= lower < 50 < upper <= 100"
        )
    strength = float(config.get("strength", 1.0))
    if not 0.0 <= strength <= 1.0:
        raise ValueError("reference_match.strength must be within [0, 1]")
    minimum_scale = float(config.get("minimum_scale", 0.35))
    maximum_scale = float(config.get("maximum_scale", 3.0))
    if minimum_scale <= 0 or maximum_scale < minimum_scale:
        raise ValueError(
            "reference_match scales must satisfy 0 < minimum_scale <= maximum_scale"
        )
    black_threshold = float(config.get("black_threshold", 8.0))
    if not 0.0 <= black_threshold <= 255.0:
        raise ValueError("reference_match.black_threshold must be within [0, 255]")

    background_only = config.get("background_only", {})
    background_only_enabled = bool(background_only.get("enabled", False))
    minimum_saturation = float(background_only.get("minimum_saturation", 35.0))
    minimum_value = float(background_only.get("minimum_value", 50.0))
    source_mask = _nonblack_mask(image, black_threshold)
    source_background_mask = _background_candidate_mask(
        image,
        source_mask,
        background_only,
    )
    source_centers, source_spans = _robust_lab_statistics(
        image,
        source_background_mask,
        lower_percentile,
        upper_percentile,
    )
    reference_centers, reference_spans = _cached_reference_statistics(
        str(reference_path.resolve()),
        reference_path.stat().st_mtime_ns,
        lower_percentile,
        upper_percentile,
        black_threshold,
        background_only_enabled,
        minimum_saturation,
        minimum_value,
    )

    source_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    scales = np.clip(
        reference_spans / source_spans,
        minimum_scale,
        maximum_scale,
    )
    matched_lab = (
        source_lab - source_centers.reshape(1, 1, 3)
    ) * scales.reshape(1, 1, 3) + reference_centers.reshape(1, 1, 3)
    blended_lab = source_lab + (matched_lab - source_lab) * strength
    matched = cv2.cvtColor(
        np.clip(blended_lab, 0, 255).astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )
    blend_alpha = _background_blend_alpha(
        image,
        source_mask,
        source_background_mask,
        background_only,
    )[:, :, None]
    output = np.clip(
        image.astype(np.float32) * (1.0 - blend_alpha)
        + matched.astype(np.float32) * blend_alpha,
        0,
        255,
    ).astype(np.uint8)
    if config.get("preserve_black", True):
        output[~source_mask] = image[~source_mask]
    return output


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

    output = _apply_reference_match(output, preset.get("reference_match", {}))
    output = _apply_bilateral(output, preset.get("bilateral", {}))
    output = _apply_clahe(output, preset.get("clahe", {}))
    output = _apply_gamma(output, float(preset.get("gamma", 1.0)))
    output = _compress_highlights(output, preset.get("highlight_compression", {}))
    output = _apply_sharpen(output, preset.get("sharpen", {}))
    if roi_mode == "mask":
        output = _apply_roi_mask(output, roi)
    return PreprocessResult(image=output, transform=transform)
