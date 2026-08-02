"""Recover backgrounds and photometric statistics from unlabeled scope video."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class RealismProfile:
    channel_mean: tuple[float, float, float]
    channel_std: tuple[float, float, float]
    laplacian_variance: float
    noise_std: float
    downscale_min: float = 0.72
    jpeg_quality_min: int = 62
    jpeg_quality_max: int = 92
    motion_blur_max: int = 5

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def read(cls, path: Path) -> "RealismProfile":
        payload = json.loads(path.read_text())
        payload["channel_mean"] = tuple(payload["channel_mean"])
        payload["channel_std"] = tuple(payload["channel_std"])
        return cls(**payload)


def resize_cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    scale = max(width / source_width, height / source_height)
    resized = cv2.resize(
        image,
        (max(width, math.ceil(source_width * scale)), max(height, math.ceil(source_height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    x = max(0, (resized.shape[1] - width) // 2)
    y = max(0, (resized.shape[0] - height) // 2)
    return resized[y : y + height, x : x + width].copy()


def sample_video_frames(
    video_path: Path,
    *,
    width: int,
    height: int,
    sample_count: int,
) -> list[np.ndarray]:
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open realism video: {video_path}")
    total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if total < 2:
        capture.release()
        raise ValueError("realism video must contain at least two readable frames")

    indices = np.linspace(0, total - 1, min(sample_count, total), dtype=int)
    frames: list[np.ndarray] = []
    try:
        for index in np.unique(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(resize_cover(frame, width, height))
    finally:
        capture.release()
    if len(frames) < 2:
        raise ValueError("could not sample enough readable frames from realism video")
    return frames


def estimate_realism_profile(frames: list[np.ndarray]) -> RealismProfile:
    if not frames:
        raise ValueError("at least one frame is required")
    means: list[np.ndarray] = []
    deviations: list[np.ndarray] = []
    laplacians: list[float] = []
    noise_levels: list[float] = []
    for frame in frames:
        visible = np.max(frame, axis=2) > 8
        pixels = frame[visible] if np.any(visible) else frame.reshape(-1, 3)
        means.append(np.mean(pixels, axis=0))
        deviations.append(np.std(pixels, axis=0))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacians.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
        residual = gray.astype(np.float32) - cv2.GaussianBlur(
            gray.astype(np.float32), (0, 0), 1.0
        )
        robust_sigma = float(np.median(np.abs(residual)) / 0.6745)
        noise_levels.append(float(np.clip(robust_sigma * 0.35, 0.2, 6.0)))
    return RealismProfile(
        channel_mean=tuple(float(value) for value in np.median(means, axis=0)),
        channel_std=tuple(float(max(1.0, value)) for value in np.median(deviations, axis=0)),
        laplacian_variance=max(1.0, float(np.median(laplacians))),
        noise_std=float(np.median(noise_levels)),
    )


def recover_video_backgrounds(
    frames: list[np.ndarray],
    output_dir: Path,
    *,
    count: int,
    seed: int | None = None,
) -> list[Path]:
    """Remove moving dark foreground with robust temporal upper quantiles."""
    if count < 1:
        raise ValueError("background count must be positive")
    if len(frames) < 2:
        raise ValueError("at least two frames are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    group_size = min(len(frames), max(12, math.ceil(len(frames) * 0.88)))
    paths: list[Path] = []
    for index in range(count):
        if group_size == len(frames):
            selected = frames
        else:
            chosen = rng.choice(len(frames), size=group_size, replace=False)
            selected = [frames[int(frame_index)] for frame_index in chosen]
        stack = np.stack(selected).copy()
        # The instrument and its shadow are predominantly darker than exposed
        # retina. A high temporal quantile removes them even when they linger
        # over the same area for a substantial fraction of the clip.
        quantile_index = min(len(stack) - 1, round((len(stack) - 1) * 0.90))
        stack.partition(quantile_index, axis=0)
        background = stack[quantile_index].copy()
        # Suppress isolated temporal-compositing speckles without erasing vessels.
        background = cv2.medianBlur(background, 3)
        path = output_dir / f"video_background_{index:03d}.png"
        if not cv2.imwrite(str(path), background):
            raise RuntimeError(f"failed to write recovered video background: {path}")
        paths.append(path)
    return paths


def _motion_blur(image: np.ndarray, size: int, angle: float) -> np.ndarray:
    if size <= 1:
        return image
    size = size + 1 if size % 2 == 0 else size
    center = size // 2
    direction = np.array([math.cos(angle), math.sin(angle)]) * center
    kernel = np.zeros((size, size), dtype=np.float32)
    cv2.line(
        kernel,
        (round(center - direction[0]), round(center - direction[1])),
        (round(center + direction[0]), round(center + direction[1])),
        1.0,
        1,
        cv2.LINE_AA,
    )
    kernel /= max(float(np.sum(kernel)), 1e-6)
    return cv2.filter2D(image, -1, kernel)


def apply_realism_degradation(
    image: np.ndarray,
    profile: RealismProfile,
    rng: np.random.Generator,
) -> np.ndarray:
    """Match color/detail statistics and add video-like acquisition artifacts."""
    result = image.astype(np.float32)
    visible = np.max(image, axis=2) > 8
    pixels = result[visible] if np.any(visible) else result.reshape(-1, 3)
    current_mean = np.mean(pixels, axis=0)
    current_std = np.maximum(np.std(pixels, axis=0), 1.0)
    target_mean = np.asarray(profile.channel_mean) * rng.uniform(0.94, 1.06)
    target_std = np.asarray(profile.channel_std) * rng.uniform(0.88, 1.12)
    transferred = (result - current_mean) * (target_std / current_std) + target_mean
    strength = rng.uniform(0.58, 0.88)
    result = np.clip(result * (1.0 - strength) + transferred * strength, 0, 255).astype(np.uint8)

    scale = rng.uniform(profile.downscale_min, 1.0)
    if scale < 0.995:
        height, width = result.shape[:2]
        small = cv2.resize(
            result,
            (max(2, round(width * scale)), max(2, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        result = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)

    target_laplacian = profile.laplacian_variance * rng.uniform(0.75, 1.30)
    candidates = [result]
    for sigma in (0.45, 0.75, 1.1, 1.55, 2.1, 2.8, 3.6):
        candidates.append(cv2.GaussianBlur(result, (0, 0), sigma))
    result = min(
        candidates,
        key=lambda candidate: abs(
            float(
                cv2.Laplacian(
                    cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY), cv2.CV_32F
                ).var()
            )
            - target_laplacian
        ),
    )

    if profile.motion_blur_max > 1 and rng.random() < 0.35:
        size = int(rng.integers(2, profile.motion_blur_max + 1))
        result = _motion_blur(result, size, rng.uniform(0, math.tau))
    noise = rng.normal(0, profile.noise_std, size=result.shape[:2] + (1,))
    result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    quality = int(rng.integers(profile.jpeg_quality_min, profile.jpeg_quality_max + 1))
    ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is not None:
            result = decoded
    return result


def build_video_realism_assets(
    video_path: Path,
    output_dir: Path,
    *,
    width: int,
    height: int,
    sample_count: int,
    background_count: int,
    seed: int | None = None,
) -> tuple[list[Path], RealismProfile]:
    frames = sample_video_frames(
        video_path,
        width=width,
        height=height,
        sample_count=sample_count,
    )
    profile = estimate_realism_profile(frames)
    backgrounds = recover_video_backgrounds(
        frames,
        output_dir / "backgrounds",
        count=background_count,
        seed=seed,
    )
    profile.write(output_dir / "realism_profile.json")
    return backgrounds, profile
