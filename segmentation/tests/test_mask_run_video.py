from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.mask_run_video import cut, detect_circle, mask_video


def test_detect_circle_fits_largest_bright_disc() -> None:
    frame = np.zeros((160, 200, 3), dtype=np.uint8)
    cv2.circle(frame, (104, 76), 52, (130, 170, 210), -1, lineType=cv2.LINE_AA)

    circle = detect_circle(frame, 12)

    assert circle is not None
    assert circle[0] == pytest.approx(104, abs=2)
    assert circle[1] == pytest.approx(76, abs=2)
    assert circle[2] == pytest.approx(52, abs=3)


def test_cut_black_pads_outside_source() -> None:
    frame = np.full((20, 20, 3), 180, dtype=np.uint8)

    output = cut(frame, 2, 2, 16)

    assert output.shape == (16, 16, 3)
    assert np.all(output[0, 0] == 0)
    assert np.all(output[-1, -1] == 180)


def test_mask_video_tracks_one_disc_and_writes_run_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "scope.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (128, 128))
    assert writer.isOpened()
    for offset in range(6):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.circle(frame, (62 + offset, 64), 42, (90, 145, 190), -1, lineType=cv2.LINE_AA)
        writer.write(frame)
    writer.release()

    metadata = mask_video(source, tmp_path / "output", smooth=3, erode=1, crf=28)

    output = tmp_path / "output" / "masked.mp4"
    capture = cv2.VideoCapture(str(output))
    ok, first = capture.read()
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert ok
    assert first.shape[0] == first.shape[1]
    assert frame_count == 6
    assert metadata["frames"] == 6
    assert metadata["detections"] == 6
    assert (tmp_path / "output" / "preview.png").is_file()
    assert (tmp_path / "output" / "mask.json").is_file()
