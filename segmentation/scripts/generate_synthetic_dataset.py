#!/usr/bin/env python3
"""Generate synthetic forceps pose training data.

The generated labels follow YOLO pose format with two objects per image:

    0 cx cy w h tip_left_x tip_left_y v tip_right_x tip_right_y v jaw_root_x jaw_root_y v
    1 cx cy w h shadow_left_x shadow_left_y v shadow_right_x shadow_right_y v shadow_root_x shadow_root_y v
"""

from __future__ import annotations

import argparse
import os
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_labels import CLASS_COLORS

FORCEPS_KEYPOINT_NAMES = ("tip_left", "tip_right", "jaw_root")
SHADOW_KEYPOINT_NAMES = ("shadow_left", "shadow_right", "shadow_root")
FORCEPS_CLASS_ID = 0
SHADOW_CLASS_ID = 1
EXPECTED_POSE_LABEL_COLUMNS = 14


@dataclass(frozen=True)
class Pose:
    tip_polygons: list[np.ndarray]
    shadow_polygons: list[np.ndarray]
    variation: Optional["RenderVariation"] = None


@dataclass(frozen=True)
class RenderVariation:
    forceps_roll_degrees: float
    shadow_roll_degrees: float
    shadow_scale: float
    tip_scale: float
    shadow_softness: float
    shadow_opacity: float
    image_rotation_degrees: float = 0.0


@dataclass(frozen=True)
class GenerationTask:
    index: int
    seed: int
    count: int
    out_dir: Path
    width: int
    height: int
    background: Optional[Path]
    background_rotation: float
    axis_roll: float
    prefix: str
    start_index: int
    val_fraction: float
    preview: int
    preview_dir: Path
    shadow_axis_roll: float = 180.0
    shadow_scale_min: float = 0.90
    shadow_scale_max: float = 1.90
    tip_scale_min: float = 0.85
    tip_scale_max: float = 1.85
    backgrounds: tuple[Path, ...] = ()
    shadow_opacity_min: float = 0.30
    shadow_opacity_max: float = 0.55
    shadow_blur_min: float = 3.0
    shadow_blur_max: float = 18.0
    circular_mask: bool = True
    image_rotations: tuple[float, ...] = (0.0,)


def unit(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)


def ellipse_polygon(
    center: np.ndarray,
    radius_x: float,
    radius_y: float,
    angle: float = 0.0,
    points: int = 14,
) -> np.ndarray:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    coords: list[list[float]] = []
    for theta in np.linspace(0, math.tau, points, endpoint=False):
        x = radius_x * math.cos(theta)
        y = radius_y * math.sin(theta)
        coords.append(
            [
                float(center[0] + x * cos_a - y * sin_a),
                float(center[1] + x * sin_a + y * cos_a),
            ]
        )
    return np.array(coords, dtype=np.float32)


def oriented_box(center: np.ndarray, direction: np.ndarray, length: float, width: float) -> np.ndarray:
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    half_l = length / 2.0
    half_w = width / 2.0
    return np.array(
        [
            center - direction * half_l - normal * half_w,
            center + direction * half_l - normal * half_w,
            center + direction * half_l + normal * half_w,
            center - direction * half_l + normal * half_w,
        ],
        dtype=np.float32,
    )


def keypoint_polygon(center: np.ndarray, radius: float = 1.8) -> np.ndarray:
    return np.array(
        [
            center + [-radius, -radius],
            center + [radius, -radius],
            center + [radius, radius],
            center + [-radius, radius],
        ],
        dtype=np.float32,
    )


def point_segment_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    segment_length_sq = float(np.dot(segment, segment))
    if segment_length_sq <= 1e-6:
        return np.linalg.norm(points - start, axis=1)
    t = np.clip(((points - start) @ segment) / segment_length_sq, 0.0, 1.0)
    closest = start + t[:, None] * segment
    return np.linalg.norm(points - closest, axis=1)


def shadow_clears_forceps(
    shadow_points: np.ndarray,
    forceps_segments: list[tuple[np.ndarray, np.ndarray]],
    min_distance: float,
) -> bool:
    for start, end in forceps_segments:
        if np.any(point_segment_distances(shadow_points, start, end) < min_distance):
            return False
    return True


def blend_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    mix = (mask.astype(np.float32) / 255.0)[:, :, None] * alpha
    color_arr = np.array(color, dtype=np.float32)
    image[:] = np.clip(image.astype(np.float32) * (1.0 - mix) + color_arr * mix, 0, 255).astype(np.uint8)


def draw_soft_line(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    alpha: float,
    blur: int,
) -> None:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.round(points).astype(np.int32)
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_AA)
    if blur > 0:
        blur = blur + 1 if blur % 2 == 0 else blur
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    blend_mask(image, mask, color, alpha)


def draw_soft_polygon(
    image: np.ndarray,
    polygon: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
    blur: int = 0,
) -> None:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], color=255, lineType=cv2.LINE_AA)
    if blur > 0:
        blur = blur + 1 if blur % 2 == 0 else blur
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    blend_mask(image, mask, color, alpha)


def motion_blur_mask(mask: np.ndarray, direction: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return mask
    size = size + 1 if size % 2 == 0 else size
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    center = size // 2
    endpoint = direction * center
    kernel_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.line(
        kernel_mask,
        (int(round(center - endpoint[0])), int(round(center - endpoint[1]))),
        (int(round(center + endpoint[0])), int(round(center + endpoint[1]))),
        255,
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    kernel = kernel_mask.astype(np.float32) / 255.0
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0:
        return mask
    return cv2.filter2D(mask, -1, kernel / kernel_sum)


def tapered_segment_polygon(
    start: np.ndarray,
    end: np.ndarray,
    start_width: float,
    end_width: float,
) -> np.ndarray:
    direction = end - start
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    return np.array(
        [
            start - normal * start_width / 2.0,
            end - normal * end_width / 2.0,
            end + normal * end_width / 2.0,
            start + normal * start_width / 2.0,
        ],
        dtype=np.float32,
    )


def quadratic_curve(
    start: np.ndarray,
    control: np.ndarray,
    end: np.ndarray,
    points: int = 18,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, points, dtype=np.float32)[:, None]
    return (1.0 - t) ** 2 * start + 2.0 * (1.0 - t) * t * control + t**2 * end


def polyline_normals(points: np.ndarray) -> np.ndarray:
    tangents = np.empty_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangents[1:-1] = points[2:] - points[:-2]
    lengths = np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-6)
    tangents = tangents / lengths
    return np.column_stack([-tangents[:, 1], tangents[:, 0]]).astype(np.float32)


def ribbon_polygon(
    centerline: np.ndarray,
    start_width: float,
    end_width: float,
) -> np.ndarray:
    widths = np.linspace(start_width, end_width, len(centerline), dtype=np.float32)[:, None]
    return profiled_ribbon_polygon(centerline, widths)


def profiled_ribbon_polygon(
    centerline: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    widths = np.asarray(widths, dtype=np.float32).reshape(-1, 1)
    if len(widths) != len(centerline):
        raise ValueError("width profile must have one value per centerline point")
    normals = polyline_normals(centerline)
    side_a = centerline - normals * widths / 2.0
    side_b = centerline + normals * widths / 2.0
    return np.vstack([side_a, side_b[::-1]]).astype(np.float32)


def distal_pad_width_profile(
    point_count: int,
    root_width: float,
    jaw_tip_width: float,
    pad_width: float,
    pad_length_fraction: float,
) -> np.ndarray:
    if point_count < 2:
        raise ValueError("point_count must be at least 2")
    t = np.linspace(0.0, 1.0, point_count, dtype=np.float32)
    widths = root_width + (jaw_tip_width - root_width) * t
    pad_start = float(np.clip(1.0 - pad_length_fraction, 0.45, 0.92))
    pad_t = np.clip((t - pad_start) / max(1e-6, 1.0 - pad_start), 0.0, 1.0)
    pad_profile = pad_width * (1.0 - 0.42 * pad_t)
    blend = np.clip(pad_t * 3.0, 0.0, 1.0)
    return (widths * (1.0 - blend) + pad_profile * blend).astype(np.float32)


def draw_soft_tapered_segment(
    image: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    start_width: float,
    end_width: float,
    color: tuple[int, int, int],
    alpha: float,
    blur: int = 0,
    motion_blur: int = 0,
) -> None:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    polygon = tapered_segment_polygon(start, end, start_width, end_width)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], color=255, lineType=cv2.LINE_AA)
    if motion_blur > 1:
        mask = motion_blur_mask(mask, end - start, motion_blur)
    if blur > 0:
        blur = blur + 1 if blur % 2 == 0 else blur
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    blend_mask(image, mask, color, alpha)


def draw_metal_ribbon(
    image: np.ndarray,
    centerline: np.ndarray,
    start_width: float,
    end_width: float,
    rng: np.random.Generator,
    *,
    brightness: float = 1.0,
    highlight_side: float = -0.22,
    width_profile: Optional[np.ndarray] = None,
) -> np.ndarray:
    widths = (
        np.linspace(start_width, end_width, len(centerline), dtype=np.float32)
        if width_profile is None
        else np.asarray(width_profile, dtype=np.float32).reshape(-1)
    )
    if len(widths) != len(centerline):
        raise ValueError("width profile must have one value per centerline point")
    polygon = profiled_ribbon_polygon(centerline, widths)
    normals = polyline_normals(centerline)
    widths_column = widths[:, None]

    def material_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(int(np.clip(channel * brightness, 0, 255)) for channel in color)

    draw_soft_polygon(image, polygon, material_color((35, 36, 39)), alpha=0.92, blur=5)
    inner_line = centerline + normals * widths_column * 0.04
    inner_polygon = profiled_ribbon_polygon(inner_line, widths * 0.75)
    draw_soft_polygon(image, inner_polygon, material_color((104, 101, 96)), alpha=0.90, blur=2)

    draw_soft_line(
        image,
        centerline + normals * widths_column * highlight_side,
        material_color((205, 199, 186)),
        thickness=max(1, round(min(start_width, end_width) * 0.13)),
        alpha=0.48,
        blur=3,
    )
    draw_soft_line(
        image,
        centerline - normals * widths_column * 0.38,
        material_color((31, 31, 34)),
        thickness=max(1, round(min(start_width, end_width) * 0.13)),
        alpha=0.54,
        blur=3,
    )
    if rng.random() < 0.75:
        scratch_start = int(rng.integers(2, max(3, len(centerline) // 2)))
        scratch_end = min(len(centerline), scratch_start + int(rng.integers(3, 7)))
        draw_soft_line(
            image,
            centerline[scratch_start:scratch_end]
            + normals[scratch_start:scratch_end]
            * widths_column[scratch_start:scratch_end]
            * rng.uniform(-0.12, 0.12),
            material_color((224, 213, 193)),
            thickness=1,
            alpha=0.18,
            blur=1,
        )
    return polygon


def composite_realistic_shadow(
    image: np.ndarray,
    polygons: list[np.ndarray],
    rng: np.random.Generator,
    *,
    opacity: float,
    softness: float,
) -> None:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(
        mask,
        [np.round(polygon).astype(np.int32) for polygon in polygons],
        color=255,
        lineType=cv2.LINE_AA,
    )
    sigma = max(0.0, softness)
    if sigma <= 1e-6:
        umbra = mask
        penumbra = mask
    else:
        umbra = cv2.GaussianBlur(mask, (0, 0), max(0.15, sigma * 0.34))
        penumbra = cv2.GaussianBlur(mask, (0, 0), sigma)
    combined = np.clip(
        umbra.astype(np.float32) * 0.64 + penumbra.astype(np.float32) * 0.46,
        0,
        255,
    )

    low_frequency = rng.normal(
        1.0,
        0.035,
        size=(max(2, image.shape[0] // 80), max(2, image.shape[1] // 80)),
    ).astype(np.float32)
    low_frequency = cv2.resize(
        low_frequency,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    alpha = np.clip(combined / 255.0 * opacity * low_frequency, 0.0, 0.72)[:, :, None]
    shadow_tint = np.array((27, 34, 43), dtype=np.float32)
    image[:] = np.clip(
        image.astype(np.float32) * (1.0 - alpha) + shadow_tint * alpha,
        0,
        255,
    ).astype(np.uint8)


def draw_metal_segment(
    image: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    start_width: float,
    end_width: float,
    rng: np.random.Generator,
    highlight_side: float = -0.22,
) -> None:
    direction = end - start
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    dark_edge = (35, 35, 38)
    base = (99, 94, 88)
    highlight = (194, 186, 171)

    draw_soft_tapered_segment(image, start, end, start_width, end_width, dark_edge, 0.88, blur=7)
    draw_soft_tapered_segment(
        image,
        start + normal * start_width * 0.03,
        end + normal * end_width * 0.03,
        start_width * 0.78,
        end_width * 0.78,
        base,
        0.92,
        blur=5,
    )
    draw_soft_line(
        image,
        np.vstack([start + normal * start_width * 0.42, end + normal * end_width * 0.42]),
        (21, 20, 21),
        thickness=max(1, round(min(start_width, end_width) * 0.12)),
        alpha=0.58,
        blur=5,
    )
    draw_soft_line(
        image,
        np.vstack([start - normal * start_width * 0.42, end - normal * end_width * 0.42]),
        (38, 34, 33),
        thickness=max(1, round(min(start_width, end_width) * 0.10)),
        alpha=0.42,
        blur=5,
    )
    draw_soft_line(
        image,
        np.vstack(
            [
                start + normal * start_width * highlight_side + direction * start_width * 0.35,
                end + normal * end_width * highlight_side - direction * end_width * 0.35,
            ]
        ),
        highlight,
        thickness=max(1, round(min(start_width, end_width) * 0.12)),
        alpha=0.46,
        blur=5,
    )

    for _ in range(3):
        t0 = rng.uniform(0.18, 0.82)
        t1 = min(0.96, t0 + rng.uniform(0.08, 0.20))
        p0 = start * (1.0 - t0) + end * t0
        p1 = start * (1.0 - t1) + end * t1
        offset = normal * rng.uniform(-0.18, 0.18) * min(start_width, end_width)
        draw_soft_line(
            image,
            np.vstack([p0 + offset, p1 + offset]),
            (103, 95, 88),
            thickness=1,
            alpha=0.16,
            blur=3,
        )


def noisy_retina_background(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = width * rng.uniform(0.46, 0.53)
    cy = height * rng.uniform(0.46, 0.53)
    radius = min(width, height) * rng.uniform(0.42, 0.47)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius

    base = np.zeros((height, width, 3), dtype=np.float32)
    base[:, :, 2] = rng.uniform(132, 176) - 13 * rr + 20 * (xx / width)
    base[:, :, 1] = rng.uniform(50, 78) - 7 * rr + 10 * (yy / height)
    base[:, :, 0] = rng.uniform(19, 35) - 2 * rr + 4 * (xx / width)

    noise = rng.normal(0, 5.5, size=(height, width, 1)).astype(np.float32)
    low_freq = rng.normal(0, 10, size=(max(2, height // 64), max(2, width // 64), 1)).astype(np.float32)
    low_freq = cv2.resize(low_freq, (width, height), interpolation=cv2.INTER_CUBIC)
    if low_freq.ndim == 2:
        low_freq = low_freq[:, :, None]
    warm_mottle = rng.normal(0, 10, size=(max(2, height // 34), max(2, width // 34), 1)).astype(np.float32)
    warm_mottle = cv2.resize(warm_mottle, (width, height), interpolation=cv2.INTER_CUBIC)
    if warm_mottle.ndim == 2:
        warm_mottle = warm_mottle[:, :, None]
    base[:, :, 1:3] += warm_mottle * np.array([[[0.35, 0.78]]], dtype=np.float32)
    fine_texture = rng.normal(0, 2.2, size=(height, width, 1)).astype(np.float32)
    image = np.clip(base + noise + low_freq + fine_texture, 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (0, 0), 0.35)

    draw_retina_details(image, (cx, cy, radius), rng)
    sharpened = cv2.addWeighted(image, 1.16, cv2.GaussianBlur(image, (0, 0), 1.6), -0.16, 0)
    image[:] = np.clip(sharpened, 0, 255).astype(np.uint8)
    apply_scope_border(image, (cx, cy, radius), rng)
    return image


def draw_retina_details(
    image: np.ndarray,
    roi: tuple[float, float, float],
    rng: np.random.Generator,
) -> None:
    height, width = image.shape[:2]
    cx, cy, radius = roi
    disc = np.array(
        [
            cx - radius * rng.uniform(0.50, 0.62),
            cy - radius * rng.uniform(0.10, 0.25),
        ],
        dtype=np.float32,
    )
    disc_poly = ellipse_polygon(
        disc,
        radius * rng.uniform(0.085, 0.13),
        radius * rng.uniform(0.10, 0.15),
        rng.uniform(-0.45, 0.45),
        points=28,
    )
    draw_soft_polygon(image, disc_poly, (28, 118, 185), alpha=rng.uniform(0.42, 0.58), blur=13)
    draw_soft_polygon(
        image,
        ellipse_polygon(disc + np.array([radius * 0.025, radius * 0.02], dtype=np.float32), radius * 0.045, radius * 0.055, points=20),
        (51, 153, 211),
        alpha=0.24,
        blur=7,
    )

    for _ in range(rng.integers(14, 22)):
        start_angle = rng.uniform(-2.2, 2.0)
        length = radius * rng.uniform(0.45, 1.05)
        thickness = int(rng.integers(1, 4))
        pts = wavy_vessel(disc, start_angle, length, rng)
        draw_soft_line(image, pts, (15, 21, 74), thickness=thickness, alpha=rng.uniform(0.34, 0.58), blur=1)
        draw_soft_line(image, pts, (11, 14, 45), thickness=max(1, thickness - 1), alpha=0.25, blur=0)
        if rng.random() < 0.9:
            branch_start = pts[rng.integers(max(2, len(pts) // 4), len(pts) - 1)]
            branch = wavy_vessel(
                branch_start,
                start_angle + rng.choice([-1, 1]) * rng.uniform(0.35, 0.8),
                length * rng.uniform(0.28, 0.52),
                rng,
            )
            draw_soft_line(image, branch, (14, 22, 65), thickness=max(1, thickness - 1), alpha=0.30, blur=1)

    for _ in range(rng.integers(8, 15)):
        angle = rng.uniform(0, math.tau)
        dist = radius * math.sqrt(rng.uniform(0.03, 0.92))
        center = np.array([cx + math.cos(angle) * dist, cy + math.sin(angle) * dist], dtype=np.float32)
        if not (0 <= center[0] < width and 0 <= center[1] < height):
            continue
        spot = ellipse_polygon(
            center,
            rng.uniform(4, 18),
            rng.uniform(3, 15),
            rng.uniform(0, math.tau),
            points=12,
        )
        color = [(21, 39, 68), (27, 54, 91), (15, 25, 47)][int(rng.integers(0, 3))]
        draw_soft_polygon(image, spot, color, alpha=rng.uniform(0.05, 0.13), blur=7)

    for _ in range(rng.integers(10, 18)):
        angle = rng.uniform(-0.35, 0.65)
        direction = unit(angle)
        start = np.array(
            [rng.uniform(-width * 0.05, width * 1.05), rng.uniform(0, height)],
            dtype=np.float32,
        )
        end = start + direction * rng.uniform(width * 0.18, width * 0.42)
        color = [(28, 45, 83), (23, 36, 70), (34, 58, 90)][int(rng.integers(0, 3))]
        draw_soft_line(
            image,
            np.vstack([start, end]),
            color,
            thickness=int(rng.integers(1, 3)),
            alpha=rng.uniform(0.08, 0.17),
            blur=5,
        )


def wavy_vessel(
    start: np.ndarray,
    angle: float,
    length: float,
    rng: np.random.Generator,
    segments: int = 16,
) -> np.ndarray:
    direction = unit(angle)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    points: list[np.ndarray] = []
    drift = rng.uniform(-0.45, 0.45)
    for i in range(segments):
        t = i / (segments - 1)
        curve = math.sin(t * math.pi * rng.uniform(0.8, 2.0) + drift) * length * rng.uniform(0.015, 0.035)
        points.append(start + direction * length * t + normal * curve)
    return np.array(points, dtype=np.float32)


def apply_scope_border(
    image: np.ndarray,
    roi: tuple[float, float, float],
    rng: np.random.Generator,
) -> None:
    height, width = image.shape[:2]
    cx, cy, radius = roi
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    edge_vignette = np.clip((dist - radius * 0.84) / (radius * 0.22), 0, 1)
    edge_vignette = cv2.GaussianBlur(edge_vignette, (0, 0), radius * 0.022)
    warm_edge = np.array((42, 44, 76), dtype=np.float32)
    mix = edge_vignette[:, :, None] * 0.16
    image[:] = np.clip(image.astype(np.float32) * (1.0 - mix) + warm_edge * mix, 0, 255)

    rim_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(rim_mask, (int(cx), int(cy)), int(radius * 1.08), 255, thickness=int(radius * 0.026), lineType=cv2.LINE_AA)
    rim_mask = cv2.GaussianBlur(rim_mask, (0, 0), radius * 0.035)
    blend_mask(image, rim_mask, (58, 48, 73), alpha=0.07)

    for _ in range(rng.integers(0, 2)):
        angle = rng.uniform(0, math.tau)
        center = np.array(
            [cx + math.cos(angle) * radius * rng.uniform(1.02, 1.2), cy + math.sin(angle) * radius * rng.uniform(1.02, 1.2)],
            dtype=np.float32,
        )
        glint = ellipse_polygon(center, radius * 0.09, radius * 0.018, angle + math.pi / 2, points=18)
        draw_soft_polygon(image, glint, (59, 84, 119), alpha=0.08, blur=25)


def circular_alpha_mask(width: int, height: int, feather: float | None = None) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    radius = max(1.0, min(width, height) / 2.0 - 1.0)
    feather_width = feather if feather is not None else max(1.0, min(width, height) * 0.012)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip((radius - dist) / feather_width, 0.0, 1.0)
    return np.round(alpha * 255).astype(np.uint8)


def circular_png_image(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    alpha = circular_alpha_mask(width, height)
    mix = alpha.astype(np.float32)[:, :, None] / 255.0
    return np.clip(image.astype(np.float32) * mix, 0, 255).astype(np.uint8)


def pose_inside_circular_view(
    pose: Pose,
    width: int,
    height: int,
    margin: float = 0.0,
) -> bool:
    center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32)
    radius = min(width, height) / 2.0 - 1.0 - margin
    if radius <= 0:
        return False
    polygons = [*pose.tip_polygons, *pose.shadow_polygons]
    return all(
        np.all(np.linalg.norm(polygon - center, axis=1) <= radius)
        for polygon in polygons
    )


def circular_object_safety_margin(
    width: int,
    height: int,
    maximum_shadow_blur: float,
) -> float:
    view_diameter = float(min(width, height))
    return max(
        12.0,
        view_diameter * 0.07,
        maximum_shadow_blur * 2.0 + view_diameter * 0.015,
    )


def load_background(background: Path, width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    source = cv2.imread(str(background), cv2.IMREAD_UNCHANGED)
    if source is None:
        raise RuntimeError(f"failed to read background image: {background}")
    source_alpha: Optional[np.ndarray] = None
    if source.ndim == 2:
        image = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    elif source.shape[2] == 4:
        source_alpha = source[:, :, 3].astype(np.float32) / 255.0
        image = np.clip(
            source[:, :, :3].astype(np.float32) * source_alpha[:, :, None],
            0,
            255,
        ).astype(np.uint8)
    else:
        image = source[:, :, :3]
    if image is None:
        raise RuntimeError(f"failed to read background image: {background}")

    src_h, src_w = image.shape[:2]
    scale = max(width / src_w, height / src_h)
    resized = cv2.resize(image, (math.ceil(src_w * scale), math.ceil(src_h * scale)), interpolation=cv2.INTER_AREA)
    resized_alpha = None
    if source_alpha is not None:
        resized_alpha = cv2.resize(
            source_alpha,
            (resized.shape[1], resized.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    max_x = resized.shape[1] - width
    max_y = resized.shape[0] - height
    x0 = int(rng.integers(0, max(1, max_x + 1)))
    y0 = int(rng.integers(0, max(1, max_y + 1)))
    crop = resized[y0 : y0 + height, x0 : x0 + width].copy()
    crop = cv2.GaussianBlur(crop, (0, 0), rng.uniform(0.25, 1.4))
    gain = rng.uniform(0.90, 1.10)
    bias = rng.uniform(-7, 7)
    crop = np.clip(crop.astype(np.float32) * gain + bias, 0, 255)
    if resized_alpha is not None:
        crop_alpha = resized_alpha[y0 : y0 + height, x0 : x0 + width]
        crop *= crop_alpha[:, :, None]
    return crop.astype(np.uint8)


def select_background(
    backgrounds: tuple[Path, ...],
    rng: np.random.Generator,
) -> Optional[Path]:
    if not backgrounds:
        return None
    if len(backgrounds) == 1:
        return backgrounds[0]
    return backgrounds[int(rng.integers(0, len(backgrounds)))]


def rotate_background(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    if abs(angle_degrees) < 1e-6:
        return image

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    radians = math.radians(angle_degrees)
    cos_a = abs(math.cos(radians))
    sin_a = abs(math.sin(radians))
    cover_scale = max(
        (width * cos_a + height * sin_a) / width,
        (width * sin_a + height * cos_a) / height,
    )
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, cover_scale)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def rotate_image_and_pose(
    image: np.ndarray,
    pose: Pose,
    angle_degrees: float,
    *,
    fit_to_frame: bool = True,
) -> tuple[np.ndarray, Pose]:
    if abs(angle_degrees) < 1e-6:
        return image, pose

    height, width = image.shape[:2]
    radians = math.radians(angle_degrees)
    cos_a = abs(math.cos(radians))
    sin_a = abs(math.sin(radians))
    rotated_width = width * cos_a + height * sin_a
    rotated_height = width * sin_a + height * cos_a
    fit_scale = (
        min(width / max(rotated_width, 1e-6), height / max(rotated_height, 1e-6))
        if fit_to_frame
        else 1.0
    )
    matrix = cv2.getRotationMatrix2D(
        ((width - 1) / 2.0, (height - 1) / 2.0),
        angle_degrees,
        fit_scale,
    )
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    def transform_polygon(polygon: np.ndarray) -> np.ndarray:
        homogeneous = np.column_stack(
            [polygon.astype(np.float32), np.ones(len(polygon), dtype=np.float32)]
        )
        return (homogeneous @ matrix.T).astype(np.float32)

    variation = pose.variation
    if variation is not None:
        variation = replace(variation, image_rotation_degrees=angle_degrees)
    transformed_pose = Pose(
        tip_polygons=[transform_polygon(polygon) for polygon in pose.tip_polygons],
        shadow_polygons=[transform_polygon(polygon) for polygon in pose.shadow_polygons],
        variation=variation,
    )
    return rotated, transformed_pose


def select_image_rotation(
    rotations: tuple[float, ...],
    rng: np.random.Generator,
) -> float:
    if not rotations:
        return 0.0
    if len(rotations) == 1:
        return float(rotations[0])
    return float(rotations[int(rng.integers(0, len(rotations)))])


def sample_expressive_roll(rng: np.random.Generator, max_degrees: float) -> float:
    if max_degrees <= 0:
        return 0.0
    if rng.random() < 0.58:
        magnitude = rng.uniform(max_degrees * 0.38, max_degrees)
        return float(magnitude * rng.choice([-1.0, 1.0]))
    return float(rng.uniform(-max_degrees, max_degrees))


def sample_roll_pair(
    rng: np.random.Generator,
    forceps_axis_roll: float,
    shadow_axis_roll: float,
) -> tuple[float, float]:
    if forceps_axis_roll <= 0 and shadow_axis_roll <= 0:
        return 0.0, 0.0

    mode = int(rng.integers(0, 4))
    forceps_neutral = min(12.0, forceps_axis_roll * 0.10)
    shadow_neutral = min(12.0, shadow_axis_roll * 0.10)
    if mode == 0:
        return (
            sample_expressive_roll(rng, forceps_axis_roll),
            float(rng.uniform(-shadow_neutral, shadow_neutral)),
        )
    if mode == 1:
        return (
            float(rng.uniform(-forceps_neutral, forceps_neutral)),
            sample_expressive_roll(rng, shadow_axis_roll),
        )
    if mode == 2:
        return (
            sample_expressive_roll(rng, forceps_axis_roll),
            sample_expressive_roll(rng, shadow_axis_roll),
        )

    forceps_roll = sample_expressive_roll(rng, forceps_axis_roll)
    correlated_shadow = forceps_roll + rng.normal(0.0, max(4.0, shadow_axis_roll * 0.12))
    return forceps_roll, float(np.clip(correlated_shadow, -shadow_axis_roll, shadow_axis_roll))


def render_forceps(
    image: np.ndarray,
    rng: np.random.Generator,
    axis_roll: float,
    shadow_axis_roll: float = 180.0,
    shadow_scale_range: tuple[float, float] = (0.90, 1.90),
    tip_scale_range: tuple[float, float] = (0.85, 1.85),
    shadow_opacity_range: tuple[float, float] = (0.30, 0.55),
    shadow_blur_range: tuple[float, float] = (3.0, 18.0),
) -> Pose:
    height, width = image.shape[:2]
    image_scale = width / 820.0
    is_large_forceps = rng.random() < 0.38
    is_far_shadow = rng.random() < 0.42
    forceps_roll_degrees, shadow_roll_degrees = sample_roll_pair(
        rng,
        axis_roll,
        shadow_axis_roll,
    )
    forceps_roll = math.radians(forceps_roll_degrees)
    shadow_roll = math.radians(shadow_roll_degrees)
    shadow_scale = float(rng.uniform(*shadow_scale_range))
    tip_scale = float(rng.uniform(*tip_scale_range))
    if rng.random() < 0.34:
        shadow_scale = max(shadow_scale, shadow_scale_range[0] + 0.68 * (shadow_scale_range[1] - shadow_scale_range[0]))
    if rng.random() < 0.34:
        tip_scale = max(tip_scale, tip_scale_range[0] + 0.68 * (tip_scale_range[1] - tip_scale_range[0]))

    forceps_roll_projection = 0.14 + 0.86 * abs(math.cos(forceps_roll))
    shadow_roll_projection = 0.08 + 0.92 * abs(math.cos(shadow_roll))
    for _ in range(500):
        tip_center = np.array(
            [
                rng.uniform(width * 0.30, width * 0.72),
                rng.uniform(height * 0.28, height * 0.70),
            ],
            dtype=np.float32,
        )
        direction_angle = rng.uniform(math.radians(184), math.radians(242))
        direction = unit(direction_angle)
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        if is_large_forceps:
            jaw_len = rng.uniform(width * 0.155, width * 0.245)
            jaw_open = rng.uniform(width * 0.070, width * 0.135)
            shaft_thickness = rng.uniform(width * 0.068, width * 0.092)
        else:
            jaw_len = rng.uniform(width * 0.115, width * 0.190)
            jaw_open = rng.uniform(width * 0.048, width * 0.103)
            shaft_thickness = rng.uniform(width * 0.046, width * 0.068)
        shaft_thickness = max(12.0, shaft_thickness)
        projected_open = jaw_open * forceps_roll_projection
        forceps_depth_shift = math.sin(forceps_roll) * jaw_open * 0.26
        base = tip_center - direction * jaw_len
        tip_a = (
            base
            + direction * (jaw_len + forceps_depth_shift / 2.0)
            + normal * projected_open / 2.0
        )
        tip_b = (
            base
            + direction * (jaw_len - forceps_depth_shift / 2.0)
            - normal * projected_open / 2.0
        )
        shadow_skew = math.radians(rng.uniform(-14.0, 14.0))
        shadow_direction = unit(direction_angle + shadow_skew)
        shadow_normal = np.array([-shadow_direction[1], shadow_direction[0]], dtype=np.float32)
        if is_far_shadow:
            offset_length = rng.uniform(width * 0.18, width * 0.42)
        else:
            offset_length = rng.uniform(width * 0.075, width * 0.235)
        shadow_offset_direction = unit(rng.uniform(math.radians(-30.0), math.radians(22.0)))
        shadow_offset = shadow_offset_direction * offset_length
        shadow_base = base + shadow_offset
        shadow_jaw_len = jaw_len * shadow_scale * rng.uniform(0.95, 1.06)
        shadow_open = jaw_open * shadow_scale * shadow_roll_projection
        shadow_depth_shift = math.sin(shadow_roll) * jaw_open * shadow_scale * 0.28
        shadow_center = shadow_base + shadow_direction * shadow_jaw_len
        shadow_a = (
            shadow_center
            + shadow_direction * (shadow_depth_shift / 2.0)
            + shadow_normal * shadow_open / 2.0
        )
        shadow_b = (
            shadow_center
            - shadow_direction * (shadow_depth_shift / 2.0)
            - shadow_normal * shadow_open / 2.0
        )
        margin = max(10.0, shaft_thickness * 0.36)
        all_centers = np.vstack([tip_a, tip_b, base, shadow_a, shadow_b, shadow_base])
        entry = base - direction * (max(width, height) * 0.72)
        shadow_points = np.vstack([shadow_a, shadow_b, shadow_base])
        forceps_segments = [(entry, base), (base, tip_a), (base, tip_b)]
        min_shadow_clearance = max(shaft_thickness * 0.58, jaw_open * 0.22, width * 0.018)
        in_frame = np.all((all_centers[:, 0] > margin) & (all_centers[:, 0] < width - margin) & (all_centers[:, 1] > margin) & (all_centers[:, 1] < height - margin))
        shadow_visible = shadow_clears_forceps(
            shadow_points,
            forceps_segments,
            min_shadow_clearance,
        )
        if in_frame and shadow_visible:
            break
    else:
        raise RuntimeError("could not sample a valid forceps pose")

    jaw_root_offset = normal * (shaft_thickness * (0.19 + 0.16 * forceps_roll_projection))
    jaw_root_a = base + jaw_root_offset
    jaw_root_b = base - jaw_root_offset
    shaft_visible_length = float(np.linalg.norm(base - entry))
    shadow_entry = shadow_base - shadow_direction * shaft_visible_length * shadow_scale
    shadow_root_offset = shadow_normal * shaft_thickness * shadow_scale * (0.18 + 0.16 * shadow_roll_projection)
    shadow_root_a = shadow_base + shadow_root_offset
    shadow_root_b = shadow_base - shadow_root_offset

    tip_length = shaft_thickness * rng.uniform(0.30, 0.46) * tip_scale
    tip_width = shaft_thickness * rng.uniform(0.115, 0.18) * tip_scale
    tip_width = max(2.4 * image_scale, tip_width)
    jaw_width_root = rng.uniform(shaft_thickness * 0.19, shaft_thickness * 0.27)
    jaw_width_tip = max(tip_width * 0.72, shaft_thickness * 0.075)
    bow = normal * jaw_open * rng.uniform(0.05, 0.13)
    jaw_curve_a = quadratic_curve(
        jaw_root_a,
        (jaw_root_a + tip_a) / 2.0 + bow,
        tip_a,
    )
    jaw_curve_b = quadratic_curve(
        jaw_root_b,
        (jaw_root_b + tip_b) / 2.0 - bow,
        tip_b,
    )

    pad_length_fraction = float(np.clip(tip_length / max(jaw_len, 1e-6), 0.10, 0.38))
    jaw_width_profile = distal_pad_width_profile(
        len(jaw_curve_a),
        jaw_width_root,
        jaw_width_tip,
        tip_width,
        pad_length_fraction,
    )
    tip_polys = [
        keypoint_polygon(tip_a),
        keypoint_polygon(tip_b),
    ]
    tip_polys = sorted(tip_polys, key=lambda p: float(np.mean(p[:, 0])))
    tip_polys.append(keypoint_polygon(base))
    shadow_tip_length = tip_length * shadow_scale * rng.uniform(1.05, 1.28)
    shadow_tip_width = tip_width * shadow_scale * rng.uniform(1.04, 1.24)
    shadow_polys = [
        ellipse_polygon(shadow_a, shadow_tip_length * 0.42, shadow_tip_width * 0.58, math.atan2(shadow_direction[1], shadow_direction[0]), points=14),
        ellipse_polygon(shadow_b, shadow_tip_length * 0.42, shadow_tip_width * 0.58, math.atan2(shadow_direction[1], shadow_direction[0]), points=14),
    ]
    shadow_polys = sorted(shadow_polys, key=lambda p: float(np.mean(p[:, 0])))
    shadow_polys.append(keypoint_polygon(shadow_base))

    shadow_bow = shadow_normal * jaw_open * shadow_scale * rng.uniform(0.04, 0.12)
    shadow_curve_a = quadratic_curve(
        shadow_root_a,
        (shadow_root_a + shadow_a) / 2.0 + shadow_bow,
        shadow_a,
    )
    shadow_curve_b = quadratic_curve(
        shadow_root_b,
        (shadow_root_b + shadow_b) / 2.0 - shadow_bow,
        shadow_b,
    )
    shadow_shaft_start_width = shaft_thickness * shadow_scale * rng.uniform(1.02, 1.24)
    shadow_shaft_end_width = shaft_thickness * shadow_scale * rng.uniform(0.62, 0.84)
    shadow_jaw_root_width = jaw_width_root * shadow_scale * rng.uniform(1.00, 1.20)
    shadow_jaw_tip_width = max(shadow_tip_width * 0.68, jaw_width_tip * shadow_scale)
    shadow_pad_length_fraction = float(
        np.clip(shadow_tip_length / max(shadow_jaw_len, 1e-6), 0.10, 0.40)
    )
    shadow_width_profile = distal_pad_width_profile(
        len(shadow_curve_a),
        shadow_jaw_root_width,
        shadow_jaw_tip_width,
        shadow_tip_width,
        shadow_pad_length_fraction,
    )
    shadow_shapes = [
        tapered_segment_polygon(
            shadow_entry,
            shadow_base,
            shadow_shaft_start_width,
            shadow_shaft_end_width,
        ),
        profiled_ribbon_polygon(shadow_curve_a, shadow_width_profile),
        profiled_ribbon_polygon(shadow_curve_b, shadow_width_profile),
    ]
    blur_min, blur_max = shadow_blur_range
    blur_position = (
        rng.beta(3.2, 2.0)
        if is_far_shadow
        else rng.beta(2.0, 3.2)
    )
    shadow_softness = float(blur_min + (blur_max - blur_min) * blur_position)
    opacity_min, opacity_max = shadow_opacity_range
    opacity_position = (
        rng.beta(2.0, 3.2)
        if is_far_shadow
        else rng.beta(3.2, 2.0)
    )
    shadow_opacity = float(opacity_min + (opacity_max - opacity_min) * opacity_position)
    composite_realistic_shadow(
        image,
        shadow_shapes,
        rng,
        opacity=shadow_opacity,
        softness=shadow_softness,
    )

    shaft_start_width = shaft_thickness * rng.uniform(1.10, 1.26)
    shaft_end_width = shaft_thickness * rng.uniform(0.72, 0.92)
    draw_metal_segment(
        image,
        entry,
        base,
        shaft_start_width,
        shaft_end_width,
        rng,
        highlight_side=-0.10 - math.sin(forceps_roll) * 0.24,
    )

    collar = oriented_box(
        base - direction * shaft_thickness * 0.05,
        direction,
        shaft_thickness * 0.62,
        shaft_thickness * (0.70 + 0.20 * forceps_roll_projection),
    )
    draw_soft_polygon(image, collar, (51, 49, 48), alpha=0.82, blur=4)
    draw_soft_polygon(
        image,
        oriented_box(base + direction * shaft_thickness * 0.08, direction, shaft_thickness * 0.42, shaft_thickness * 0.48),
        (145, 136, 123),
        alpha=0.39,
        blur=3,
    )

    depth = math.sin(forceps_roll)
    brightness_a = 1.0 + depth * 0.22
    brightness_b = 1.0 - depth * 0.22
    draw_metal_ribbon(
        image,
        jaw_curve_a,
        jaw_width_root,
        jaw_width_tip,
        rng,
        brightness=brightness_a,
        highlight_side=0.16,
        width_profile=jaw_width_profile,
    )
    draw_metal_ribbon(
        image,
        jaw_curve_b,
        jaw_width_root,
        jaw_width_tip,
        rng,
        brightness=brightness_b,
        highlight_side=-0.16,
        width_profile=jaw_width_profile,
    )

    pad_start_index = int(round((1.0 - pad_length_fraction) * (len(jaw_curve_a) - 1)))
    for index, tip_curve in enumerate((jaw_curve_a[pad_start_index:], jaw_curve_b[pad_start_index:])):
        side = 1.0 if index == 0 else -1.0
        normals = polyline_normals(tip_curve)
        tip_widths = jaw_width_profile[pad_start_index:, None]
        draw_soft_line(
            image,
            tip_curve + normals * tip_widths * 0.16 * side,
            (222, 214, 198),
            thickness=max(1, round(float(np.min(tip_widths)) * 0.12)),
            alpha=0.48,
            blur=2,
        )
        groove_start = tip_curve[max(1, len(tip_curve) // 3)]
        groove_end = tip_curve[-2]
        draw_soft_line(
            image,
            np.vstack([groove_start, groove_end]),
            (34, 35, 37),
            thickness=max(1, round(tip_width * 0.10)),
            alpha=0.40,
            blur=1,
        )

    return Pose(
        tip_polygons=tip_polys,
        shadow_polygons=shadow_polys,
        variation=RenderVariation(
            forceps_roll_degrees=forceps_roll_degrees,
            shadow_roll_degrees=shadow_roll_degrees,
            shadow_scale=shadow_scale,
            tip_scale=tip_scale,
            shadow_softness=shadow_softness,
            shadow_opacity=shadow_opacity,
        ),
    )


def format_yolo_value(value: float) -> str:
    return f"{value:.9f}".rstrip("0").rstrip(".")


def normalized_center(polygon: np.ndarray, width: int, height: int) -> tuple[float, float]:
    center = np.mean(polygon, axis=0)
    return (
        float(np.clip(center[0] / width, 0.0, 1.0)),
        float(np.clip(center[1] / height, 0.0, 1.0)),
    )


def object_bbox(
    keypoint_polygons: list[np.ndarray],
    width: int,
    height: int,
    padding: float = 0.02,
) -> list[float]:
    all_points = np.vstack(keypoint_polygons)
    x1 = max(0.0, float(np.min(all_points[:, 0])) / width - padding)
    y1 = max(0.0, float(np.min(all_points[:, 1])) / height - padding)
    x2 = min(1.0, float(np.max(all_points[:, 0])) / width + padding)
    y2 = min(1.0, float(np.max(all_points[:, 1])) / height + padding)
    return [
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        x2 - x1,
        y2 - y1,
    ]


def pose_label_line(
    class_id: int,
    keypoint_polygons: list[np.ndarray],
    width: int,
    height: int,
    visibility: int = 2,
) -> str:
    bbox = object_bbox(keypoint_polygons, width, height)
    values: list[float | int] = [class_id, *bbox]
    for polygon in keypoint_polygons:
        x, y = normalized_center(polygon, width, height)
        values.extend([x, y, visibility])

    return " ".join(format_yolo_value(float(value)) for value in values)


def pose_label_lines(pose: Pose, width: int, height: int, visibility: int = 2) -> list[str]:
    forceps_polygons = [pose.tip_polygons[0], pose.tip_polygons[1], pose.tip_polygons[2]]
    shadow_polygons = [pose.shadow_polygons[0], pose.shadow_polygons[1], pose.shadow_polygons[2]]
    lines = [
        pose_label_line(FORCEPS_CLASS_ID, forceps_polygons, width, height, visibility),
        pose_label_line(SHADOW_CLASS_ID, shadow_polygons, width, height, visibility),
    ]
    for line in lines:
        column_count = len(line.split())
        if column_count != EXPECTED_POSE_LABEL_COLUMNS:
            raise RuntimeError(
                "invalid YOLO pose label column count: "
                f"expected {EXPECTED_POSE_LABEL_COLUMNS}, got {column_count}"
            )
    return lines


def normalized_box_xyxy(
    keypoint_polygons: list[np.ndarray],
    width: int,
    height: int,
    padding: float = 0.02,
) -> tuple[int, int, int, int]:
    cx, cy, box_width, box_height = object_bbox(keypoint_polygons, width, height, padding)
    x1 = int(round((cx - box_width / 2.0) * width))
    y1 = int(round((cy - box_height / 2.0) * height))
    x2 = int(round((cx + box_width / 2.0) * width))
    y2 = int(round((cy + box_height / 2.0) * height))
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width - 1, x2)),
        max(0, min(height - 1, y2)),
    )


def render_preview(image: np.ndarray, pose: Pose) -> np.ndarray:
    preview = image.copy()
    height, width = preview.shape[:2]
    objects = [
        (
            FORCEPS_CLASS_ID,
            "forceps",
            FORCEPS_KEYPOINT_NAMES,
            [pose.tip_polygons[0], pose.tip_polygons[1], pose.tip_polygons[2]],
            CLASS_COLORS[0],
        ),
        (
            SHADOW_CLASS_ID,
            "shadow",
            SHADOW_KEYPOINT_NAMES,
            [pose.shadow_polygons[0], pose.shadow_polygons[1], pose.shadow_polygons[2]],
            CLASS_COLORS[2],
        ),
    ]
    for _class_id, class_name, keypoint_names, polygons, color in objects:
        x1, y1, x2, y2 = normalized_box_xyxy(polygons, width, height)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
        cv2.putText(
            preview,
            class_name,
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        for keypoint_name, polygon in zip(keypoint_names, polygons, strict=True):
            center = np.round(np.mean(polygon, axis=0)).astype(int)
            point = (int(center[0]), int(center[1]))
            cv2.circle(preview, point, radius=6, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(preview, point, radius=4, color=color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(
                preview,
                keypoint_name,
                (point[0] + 7, point[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    if pose.variation is not None:
        variation = pose.variation
        description = (
            f"roll forceps={variation.forceps_roll_degrees:+.0f}deg "
            f"shadow={variation.shadow_roll_degrees:+.0f}deg  "
            f"shadow={variation.shadow_scale:.2f}x/{variation.shadow_opacity:.2f}alpha "
            f"blur={variation.shadow_softness:.1f}px tips={variation.tip_scale:.2f}x "
            f"image={variation.image_rotation_degrees:+.0f}deg"
        )
        cv2.rectangle(preview, (6, height - 27), (min(width - 6, 640), height - 5), (12, 12, 12), -1)
        cv2.putText(
            preview,
            description,
            (12, height - 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (238, 238, 238),
            1,
            cv2.LINE_AA,
        )
    return preview


def split_for_index(index: int, count: int, val_fraction: float) -> str:
    if val_fraction <= 0:
        return "train"
    if val_fraction >= 1:
        return "val"
    val_count = round(count * val_fraction)
    return "val" if index >= count - val_count else "train"


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_generation_progress(
    completed: int,
    total: int,
    generated: dict[str, int],
    start_time: float,
    *,
    final: bool = False,
) -> None:
    elapsed = time.monotonic() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta = (total - completed) / rate if rate > 0 else math.inf
    percent = 100.0 * completed / total if total else 100.0
    message = (
        f"Generating synthetic images: {completed}/{total} "
        f"({percent:5.1f}%) train={generated['train']} val={generated['val']} "
        f"{rate:4.1f} img/s ETA {format_duration(eta)}"
    )
    if sys.stderr.isatty():
        print(f"\r{message}", end="\n" if final else "", file=sys.stderr, flush=True)
    else:
        print(message, file=sys.stderr, flush=True)


def resolve_worker_count(requested_workers: int, count: int) -> int:
    if requested_workers < 0:
        raise SystemExit("--workers must be non-negative")
    if requested_workers == 0:
        cpu_count = os.cpu_count() or 1
        return max(1, min(count, cpu_count - 1 if cpu_count > 1 else 1))
    return max(1, min(count, requested_workers))


def build_image_seeds(seed: Optional[int], count: int) -> list[int]:
    seed_sequence = np.random.SeedSequence(seed)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in seed_sequence.spawn(count)]


def generate_one_image(task: GenerationTask) -> str:
    cv2.setNumThreads(1)
    rng = np.random.default_rng(task.seed)
    split = split_for_index(task.index, task.count, task.val_fraction)
    name = f"{task.prefix}_{task.start_index + task.index:06d}"
    available_backgrounds = task.backgrounds
    if not available_backgrounds and task.background is not None:
        available_backgrounds = (task.background,)
    selected_background = select_background(available_backgrounds, rng)
    if selected_background is not None:
        image = load_background(selected_background, task.width, task.height, rng)
    else:
        image = noisy_retina_background(task.width, task.height, rng)
    if task.background_rotation > 0:
        image = rotate_background(image, rng.uniform(-task.background_rotation, task.background_rotation))

    clean_background = image
    placement_margin = circular_object_safety_margin(
        task.width,
        task.height,
        task.shadow_blur_max,
    )
    for _ in range(120):
        rendered_image = clean_background.copy()
        pose = render_forceps(
            rendered_image,
            rng,
            task.axis_roll,
            task.shadow_axis_roll,
            (task.shadow_scale_min, task.shadow_scale_max),
            (task.tip_scale_min, task.tip_scale_max),
            (task.shadow_opacity_min, task.shadow_opacity_max),
            (task.shadow_blur_min, task.shadow_blur_max),
        )
        if not task.circular_mask or pose_inside_circular_view(
            pose,
            task.width,
            task.height,
            margin=placement_margin,
        ):
            image = rendered_image
            break
    else:
        raise RuntimeError("could not place forceps and shadow inside the circular view")

    image_rotation = select_image_rotation(task.image_rotations, rng)
    if abs(image_rotation) > 1e-6:
        image, pose = rotate_image_and_pose(
            image,
            pose,
            image_rotation,
            fit_to_frame=not task.circular_mask,
        )
    image_path = task.out_dir / "images" / split / f"{name}.png"
    label_path = task.out_dir / "labels" / split / f"{name}.txt"
    output_image = circular_png_image(image) if task.circular_mask else image
    if not cv2.imwrite(str(image_path), output_image):
        raise RuntimeError(f"failed to write image: {image_path}")
    label_path.write_text("\n".join(pose_label_lines(pose, task.width, task.height)) + "\n")

    if task.index < task.preview:
        preview = render_preview(image, pose)
        preview_path = task.preview_dir / f"{name}.png"
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"failed to write preview: {preview_path}")

    return split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic retinal forceps pose data.")
    parser.add_argument("--count", type=int, default=100, help="Number of images to generate.")
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="Dataset root containing images/ and labels/.")
    parser.add_argument("--width", type=int, default=820, help="Output image width.")
    parser.add_argument("--height", type=int, default=920, help="Output image height.")
    parser.add_argument(
        "--background",
        type=Path,
        nargs="+",
        action="extend",
        metavar="PATH",
        help=(
            "One or more clean background images to sample uniformly. "
            "The option may be repeated; omit it to generate procedural backgrounds."
        ),
    )
    parser.add_argument(
        "--background-rotation",
        type=float,
        default=180.0,
        metavar="DEGREES",
        help="Randomly rotate each retina/background by +/- this many degrees before drawing forceps. Use 0 to disable.",
    )
    parser.add_argument(
        "--image-rotations",
        type=float,
        nargs="+",
        default=(90.0, 180.0, 270.0),
        metavar="DEGREES",
        help=(
            "Discrete completed-image rotations to sample from. "
            "Defaults to 90 180 270; exposed canvas is filled black."
        ),
    )
    parser.add_argument(
        "--axis-roll",
        type=float,
        default=180.0,
        metavar="DEGREES",
        help="Randomly roll the forceps around its shaft axis by +/- this many degrees. Use 0 to disable.",
    )
    parser.add_argument(
        "--shadow-axis-roll",
        type=float,
        default=180.0,
        metavar="DEGREES",
        help="Independently roll the projected shadow around its shaft axis by +/- this many degrees. Use 0 to disable.",
    )
    parser.add_argument(
        "--shadow-scale",
        type=float,
        nargs=2,
        default=(0.90, 1.90),
        metavar=("MIN", "MAX"),
        help="Random projected shadow scale range. Values above 1 create enlarged shadows.",
    )
    parser.add_argument(
        "--shadow-opacity",
        "--shadow-visibility",
        dest="shadow_opacity",
        type=float,
        nargs=2,
        default=(0.30, 0.55),
        metavar=("MIN", "MAX"),
        help=(
            "Shadow visibility bounds from 0 (transparent) to 1 (strongest). "
            "Every sampled shadow stays within this range."
        ),
    )
    parser.add_argument(
        "--shadow-blur",
        "--shadow-softness",
        dest="shadow_blur",
        type=float,
        nargs=2,
        default=(3.0, 18.0),
        metavar=("MIN", "MAX"),
        help=(
            "Shadow Gaussian-blur bounds in output pixels. "
            "Use 0 for a hard-edged shadow or equal values for constant blur."
        ),
    )
    parser.add_argument(
        "--tip-scale",
        type=float,
        nargs=2,
        default=(0.85, 1.85),
        metavar=("MIN", "MAX"),
        help="Random forceps gripping-tip scale range.",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducible generation.")
    parser.add_argument("--prefix", default="synthetic", help="Filename prefix.")
    parser.add_argument("--start-index", type=int, default=0, help="First numeric image index.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Fraction of generated images written to val/.")
    parser.add_argument("--preview", type=int, default=0, help="Also render N label-overlay preview images.")
    parser.add_argument("--preview-dir", type=Path, default=Path("runs/synthetic_preview"), help="Directory for previews.")
    parser.add_argument(
        "--image-ext",
        choices=["png"],
        default="png",
        help="Deprecated compatibility option; synthetic dataset images are always PNG.",
    )
    view_group = parser.add_mutually_exclusive_group()
    view_group.add_argument(
        "--circular-mask",
        dest="circular_mask",
        action="store_true",
        help=(
            "Use the default circular microscope view with solid black corners. "
            "Both labeled objects are constrained to remain inside the circle."
        ),
    )
    view_group.add_argument(
        "--rectangular-view",
        dest="circular_mask",
        action="store_false",
        help="Disable the circular microscope mask and use the full rectangular frame.",
    )
    parser.set_defaults(circular_mask=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel worker processes. Use 0 for auto, 1 for serial generation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if not 0 <= args.val_fraction <= 1:
        raise SystemExit("--val-fraction must be between 0 and 1")
    if args.background_rotation < 0:
        raise SystemExit("--background-rotation must be non-negative")
    if not args.image_rotations:
        raise SystemExit("--image-rotations requires at least one angle")
    if not all(math.isfinite(angle) for angle in args.image_rotations):
        raise SystemExit("--image-rotations values must be finite")
    if args.axis_roll < 0:
        raise SystemExit("--axis-roll must be non-negative")
    if args.shadow_axis_roll < 0:
        raise SystemExit("--shadow-axis-roll must be non-negative")
    for flag, values in (("--shadow-scale", args.shadow_scale), ("--tip-scale", args.tip_scale)):
        if values[0] <= 0 or values[1] <= 0:
            raise SystemExit(f"{flag} values must be positive")
        if values[0] > values[1]:
            raise SystemExit(f"{flag} MIN must not exceed MAX")
    if not 0 <= args.shadow_opacity[0] <= 1 or not 0 <= args.shadow_opacity[1] <= 1:
        raise SystemExit("--shadow-opacity values must be between 0 and 1")
    if args.shadow_opacity[0] > args.shadow_opacity[1]:
        raise SystemExit("--shadow-opacity MIN must not exceed MAX")
    if args.shadow_blur[0] < 0 or args.shadow_blur[1] < 0:
        raise SystemExit("--shadow-blur values must be non-negative")
    if args.shadow_blur[0] > args.shadow_blur[1]:
        raise SystemExit("--shadow-blur MIN must not exceed MAX")
    backgrounds = tuple(args.background or ())
    for background in backgrounds:
        if not background.is_file():
            raise SystemExit(f"background image does not exist or is not a file: {background}")
    worker_count = resolve_worker_count(args.workers, args.count)

    image_dirs = {split: args.out_dir / "images" / split for split in ("train", "val")}
    label_dirs = {split: args.out_dir / "labels" / split for split in ("train", "val")}
    for path in [*image_dirs.values(), *label_dirs.values()]:
        path.mkdir(parents=True, exist_ok=True)
    if args.preview:
        args.preview_dir.mkdir(parents=True, exist_ok=True)

    image_seeds = build_image_seeds(args.seed, args.count)
    tasks = [
        GenerationTask(
            index=i,
            seed=image_seeds[i],
            count=args.count,
            out_dir=args.out_dir,
            width=args.width,
            height=args.height,
            background=None,
            background_rotation=args.background_rotation,
            axis_roll=args.axis_roll,
            prefix=args.prefix,
            start_index=args.start_index,
            val_fraction=args.val_fraction,
            preview=args.preview,
            preview_dir=args.preview_dir,
            shadow_axis_roll=args.shadow_axis_roll,
            shadow_scale_min=args.shadow_scale[0],
            shadow_scale_max=args.shadow_scale[1],
            tip_scale_min=args.tip_scale[0],
            tip_scale_max=args.tip_scale[1],
            backgrounds=backgrounds,
            shadow_opacity_min=args.shadow_opacity[0],
            shadow_opacity_max=args.shadow_opacity[1],
            shadow_blur_min=args.shadow_blur[0],
            shadow_blur_max=args.shadow_blur[1],
            circular_mask=args.circular_mask,
            image_rotations=tuple(float(angle % 360.0) for angle in args.image_rotations),
        )
        for i in range(args.count)
    ]

    generated = {"train": 0, "val": 0}
    start_time = time.monotonic()
    progress_interval = max(1, min(100, args.count // 100))
    print(f"Using {worker_count} synthetic generation worker(s)", file=sys.stderr, flush=True)
    print_generation_progress(0, args.count, generated, start_time)

    completed = 0
    if worker_count == 1:
        for task in tasks:
            split = generate_one_image(task)
            generated[split] += 1
            completed += 1
            if completed == args.count or completed % progress_interval == 0:
                print_generation_progress(
                    completed,
                    args.count,
                    generated,
                    start_time,
                    final=completed == args.count,
                )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(generate_one_image, task) for task in tasks]
            for future in as_completed(futures):
                split = future.result()
                generated[split] += 1
                completed += 1
                if completed == args.count or completed % progress_interval == 0:
                    print_generation_progress(
                        completed,
                        args.count,
                        generated,
                        start_time,
                        final=completed == args.count,
                    )

    if completed < args.count:
        print_generation_progress(
            completed,
            args.count,
            generated,
            start_time,
            final=True,
        )

    print(
        "Generated "
        f"{generated['train']} train and {generated['val']} val synthetic images under {args.out_dir}"
    )
    if args.preview:
        print(f"Wrote {min(args.preview, args.count)} previews under {args.preview_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
