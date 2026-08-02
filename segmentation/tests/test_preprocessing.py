from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.preprocessing import (
    apply_preprocessing,
    load_preprocess_preset,
)


def robust_lab_stats(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.max(image, axis=2) > 8
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab[mask]
    center = np.median(pixels, axis=0)
    span = np.percentile(pixels, 95, axis=0) - np.percentile(
        pixels,
        5,
        axis=0,
    )
    return center, span


def make_gradient(base: tuple[int, int, int], contrast: float) -> np.ndarray:
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float32)
    signal = ((xx + yy) / 190.0 - 0.5)[:, :, None]
    color = np.asarray(base, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(color + signal * contrast, 0, 255).astype(np.uint8)


def test_reference_match_normalizes_color_and_contrast(tmp_path: Path) -> None:
    reference = make_gradient((32, 92, 178), 110.0)
    source = make_gradient((88, 70, 112), 38.0)
    reference_path = tmp_path / "reference.png"
    assert cv2.imwrite(str(reference_path), reference)
    preset = {
        "roi": {"mode": "none"},
        "reference_match": {
            "enabled": True,
            "reference": str(reference_path),
            "lower_percentile": 5,
            "upper_percentile": 95,
            "minimum_scale": 0.1,
            "maximum_scale": 8.0,
            "black_threshold": 8,
        },
    }

    result = apply_preprocessing(source, preset).image
    reference_center, reference_span = robust_lab_stats(reference)
    result_center, result_span = robust_lab_stats(result)

    assert result_center == pytest.approx(reference_center, abs=3.0)
    assert result_span == pytest.approx(reference_span, abs=4.0)


def test_reference_match_preserves_black_pixels(tmp_path: Path) -> None:
    reference = make_gradient((28, 86, 170), 80.0)
    source = make_gradient((70, 66, 105), 40.0)
    source[:12] = 0
    reference_path = tmp_path / "reference.png"
    assert cv2.imwrite(str(reference_path), reference)
    preset = {
        "roi": {"mode": "none"},
        "reference_match": {
            "enabled": True,
            "reference": str(reference_path),
            "preserve_black": True,
        },
    }

    result = apply_preprocessing(source, preset).image

    assert np.all(result[:12] == 0)
    assert np.any(result[20:] > 0)


def test_background_only_reference_match_preserves_instrument_and_shadow(
    tmp_path: Path,
) -> None:
    reference = make_gradient((28, 88, 178), 90.0)
    source = make_gradient((62, 76, 132), 38.0)
    source[28:68, 40:56] = (112, 112, 112)
    source[32:64, 60:78] = (28, 31, 35)
    original_instrument = source[36:60, 44:52].copy()
    original_shadow = source[38:58, 64:74].copy()
    original_background = source[10:22, 10:22].copy()
    reference_path = tmp_path / "reference.png"
    assert cv2.imwrite(str(reference_path), reference)
    preset = {
        "roi": {"mode": "none"},
        "reference_match": {
            "enabled": True,
            "reference": str(reference_path),
            "minimum_scale": 0.1,
            "maximum_scale": 8.0,
            "background_only": {
                "enabled": True,
                "minimum_saturation": 35,
                "minimum_value": 50,
                "dilation_kernel": 9,
                "feather_sigma": 2.0,
            },
        },
    }

    result = apply_preprocessing(source, preset).image

    assert np.array_equal(result[36:60, 44:52], original_instrument)
    assert np.array_equal(result[38:58, 64:74], original_shadow)
    assert not np.array_equal(result[10:22, 10:22], original_background)


def test_retina_reference_preset_resolves_repository_reference() -> None:
    preset = load_preprocess_preset("retina_reference")

    reference_path = Path(preset["reference_match"]["reference"])
    assert reference_path.is_absolute()
    assert reference_path.name == "retina.png"
    assert reference_path.is_file()
