from pathlib import Path

import cv2
import numpy as np

from scripts.video_realism import (
    RealismProfile,
    apply_realism_degradation,
    build_video_realism_assets,
    estimate_realism_profile,
    recover_video_backgrounds,
)


def test_recovers_background_behind_moving_dark_foreground(tmp_path: Path) -> None:
    frames = []
    for x in (4, 18, 32, 46, 60, 74, 88, 102):
        frame = np.full((64, 128, 3), (35, 105, 185), dtype=np.uint8)
        cv2.rectangle(frame, (x, 20), (min(127, x + 18), 42), (15, 20, 25), -1)
        frames.append(frame)

    paths = recover_video_backgrounds(frames, tmp_path, count=2, seed=4)
    recovered = cv2.imread(str(paths[0]))

    assert len(paths) == 2
    assert recovered.shape == frames[0].shape
    assert np.mean(recovered[20:43, 20:108, 2]) > 150


def test_realism_profile_round_trips_and_degrades_image(tmp_path: Path) -> None:
    frames = [
        np.full((48, 64, 3), (30 + index, 80 + index, 150 + index), dtype=np.uint8)
        for index in range(4)
    ]
    profile = estimate_realism_profile(frames)
    path = tmp_path / "profile.json"
    profile.write(path)

    loaded = RealismProfile.read(path)
    source = np.full((48, 64, 3), (110, 110, 110), dtype=np.uint8)
    degraded = apply_realism_degradation(source, loaded, np.random.default_rng(9))

    assert loaded == profile
    assert degraded.shape == source.shape
    assert degraded.dtype == np.uint8
    assert float(np.mean(degraded[:, :, 2])) > float(np.mean(degraded[:, :, 0]))


def test_builds_realism_assets_from_unlabeled_video(tmp_path: Path) -> None:
    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (80, 60),
    )
    assert writer.isOpened()
    for index in range(10):
        frame = np.full((60, 80, 3), (35, 90, 170), dtype=np.uint8)
        cv2.circle(frame, (8 + index * 6, 30), 5, (10, 15, 20), -1)
        writer.write(frame)
    writer.release()

    backgrounds, profile = build_video_realism_assets(
        video,
        tmp_path / "assets",
        width=64,
        height=64,
        sample_count=8,
        background_count=2,
        seed=2,
    )

    assert len(backgrounds) == 2
    assert all(path.is_file() for path in backgrounds)
    assert (tmp_path / "assets" / "realism_profile.json").is_file()
    assert profile.laplacian_variance > 0
