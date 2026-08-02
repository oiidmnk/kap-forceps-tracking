from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.mask_retinal_video import (
    Circle,
    detect_retinal_circle,
    make_circular_mask,
    track_retinal_circle,
    validate_circle,
    validate_video_extensions,
)


def synthetic_retinal_frame(
    width: int = 640,
    height: int = 600,
    center: tuple[int, int] = (314, 306),
    radius: int = 142,
) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(frame, (width // 2, height // 2), 285, (20, 35, 55), -1)
    cv2.circle(frame, center, radius + 24, (8, 12, 17), -1)
    cv2.circle(frame, center, radius, (28, 78, 170), -1)
    cv2.line(
        frame,
        (center[0] - radius, center[1]),
        (center[0] + radius, center[1] + 35),
        (20, 25, 32),
        7,
    )
    return frame


def test_detect_retinal_circle_finds_inner_aperture() -> None:
    expected = Circle(314, 306, 142)
    frames = [synthetic_retinal_frame() for _ in range(3)]

    detected = detect_retinal_circle(frames)

    assert detected.center_x == pytest.approx(expected.center_x, abs=5)
    assert detected.center_y == pytest.approx(expected.center_y, abs=5)
    assert detected.radius == pytest.approx(expected.radius, abs=5)


def test_hard_mask_blacks_out_pixels_outside_circle() -> None:
    mask = make_circular_mask(11, 11, Circle(5, 5, 3))

    assert mask[5, 5] == 255
    assert mask[5, 8] == 255
    assert mask[5, 9] == 0
    assert mask[0, 0] == 0
    assert set(np.unique(mask)) == {0, 255}


def test_feathered_mask_transitions_inside_edge() -> None:
    mask = make_circular_mask(15, 15, Circle(7, 7, 5), feather=2)

    assert mask[7, 7] == 255
    assert mask[7, 10] == 255
    assert 0 < mask[7, 11] < 255
    assert mask[7, 12] == 0


def test_tracking_follows_a_moving_aperture() -> None:
    previous = Circle(314, 306, 142)
    frame = synthetic_retinal_frame(center=(326, 312), radius=146)

    tracked, accepted = track_retinal_circle(
        frame,
        previous,
        min_radius_fraction=0.16,
        max_radius_fraction=0.36,
        margin=0,
        smoothing=1,
        max_center_shift=30,
        max_radius_change=15,
    )

    assert accepted
    assert tracked.center_x == pytest.approx(326, abs=5)
    assert tracked.center_y == pytest.approx(312, abs=5)
    assert tracked.radius == pytest.approx(146, abs=5)


def test_tracking_reuses_previous_circle_when_detection_fails() -> None:
    previous = Circle(314, 306, 142)
    blank_frame = np.zeros((600, 640, 3), dtype=np.uint8)

    tracked, accepted = track_retinal_circle(
        blank_frame,
        previous,
        min_radius_fraction=0.16,
        max_radius_fraction=0.36,
        margin=0,
        smoothing=0.45,
        max_center_shift=30,
        max_radius_change=15,
    )

    assert not accepted
    assert tracked == previous


def test_validate_circle_rejects_circle_outside_frame() -> None:
    with pytest.raises(ValueError, match="fit completely"):
        validate_circle(Circle(2, 5, 4), width=10, height=10)


@pytest.mark.parametrize("filename", ["input.mp4", "input.MP4", "input.mov", "input.MOV"])
def test_validate_video_extensions_accepts_mp4_and_mov(filename: str) -> None:
    validate_video_extensions(Path(filename), Path("output.mp4"))


def test_validate_video_extensions_rejects_other_input_formats() -> None:
    with pytest.raises(ValueError, match="supported input"):
        validate_video_extensions(Path("input.avi"), Path("output.mp4"))
