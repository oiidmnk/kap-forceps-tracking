import cv2
import numpy as np
from pathlib import Path

from scripts.detect_classical_roi import detect_classical_roi, process_image, process_video


def _intersects(box, expected) -> bool:
    return not (
        box[2] < expected[0]
        or box[0] > expected[2]
        or box[3] < expected[1]
        or box[1] > expected[3]
    )


def test_classical_roi_finds_border_connected_forceps_and_shadow() -> None:
    image = np.full((360, 608, 3), (25, 70, 180), dtype=np.uint8)
    cv2.line(image, (20, 40), (580, 100), (18, 45, 125), 3, cv2.LINE_AA)
    shadow = np.array([[285, 115], [607, 180], [607, 260], [310, 185]], dtype=np.int32)
    cv2.fillPoly(image, [shadow], (18, 45, 115), lineType=cv2.LINE_AA)
    forceps = np.array([[75, 205], [105, 190], [315, 359], [240, 359]], dtype=np.int32)
    cv2.fillPoly(image, [forceps], (45, 55, 85), lineType=cv2.LINE_AA)

    result = detect_classical_roi(image)

    assert result.forceps_box is not None
    assert result.shadow_box is not None
    assert result.roi_box is not None
    assert _intersects(result.forceps_box, (75, 190, 315, 359))
    assert _intersects(result.shadow_box, (285, 115, 607, 260))
    assert result.roi_box[0] <= result.forceps_box[0]
    assert result.roi_box[1] <= result.shadow_box[1]
    assert result.roi_box[2] >= result.shadow_box[2]
    assert result.roi_box[3] >= result.forceps_box[3]


def test_classical_roi_rejects_empty_image() -> None:
    try:
        detect_classical_roi(np.empty((0, 0, 3), dtype=np.uint8))
    except ValueError as error:
        assert "empty image" in str(error)
    else:
        raise AssertionError("empty image should be rejected")


def _example_frame(offset: int = 0) -> np.ndarray:
    image = np.full((180, 304, 3), (25, 70, 180), dtype=np.uint8)
    shadow = np.array(
        [[142 + offset, 58], [303, 90], [303, 130], [155 + offset, 93]], dtype=np.int32
    )
    cv2.fillPoly(image, [shadow], (18, 45, 115), lineType=cv2.LINE_AA)
    forceps = np.array(
        [[38 + offset, 103], [53 + offset, 95], [158 + offset, 179], [120 + offset, 179]],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [forceps], (45, 55, 85), lineType=cv2.LINE_AA)
    return image


def test_process_image_writes_annotated_media_and_summary(tmp_path: Path) -> None:
    source = tmp_path / "frame.png"
    cv2.imwrite(str(source), _example_frame())

    output, summary = process_image(source, tmp_path / "output")

    assert output.is_file()
    assert (tmp_path / "output" / "detections.json").is_file()
    assert (tmp_path / "output" / "summary.json").is_file()
    assert summary["frames"] == 1
    assert summary["media_type"] == "image"


def test_process_video_writes_every_frame_and_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (304, 180)
    )
    assert writer.isOpened()
    for offset in (0, 2, 4):
        writer.write(_example_frame(offset))
    writer.release()

    output, summary = process_video(
        source,
        tmp_path / "output",
        browser_compatible=False,
    )

    assert output.is_file()
    assert summary["frames"] == 3
    lines = (tmp_path / "output" / "detections.jsonl").read_text().splitlines()
    assert len(lines) == 3
    capture = cv2.VideoCapture(str(output))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    capture.release()
