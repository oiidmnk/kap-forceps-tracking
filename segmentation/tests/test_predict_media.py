from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

import scripts.predict_media as predict_media


def test_predict_video_writes_every_rendered_frame(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(source),
        cv2.VideoWriter_fourcc(*"MJPG"),
        12,
        (64, 48),
    )
    assert writer.isOpened()
    for value in (30, 90, 150):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(
        predict_media,
        "predict_frame",
        lambda _model, frame, _args, _preset: frame,
    )
    monkeypatch.setattr(predict_media.shutil, "which", lambda _name: None)
    args = Namespace(conf=0.25, imgsz=64, device="cpu", max_det=10, scene_filter=True)

    output = predict_media.predict_video(None, source, tmp_path / "output", args, None)

    capture = cv2.VideoCapture(str(output))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert output.suffix == ".mp4"
    assert frame_count == 3


def test_track_result_uses_persistent_ultralytics_tracking() -> None:
    expected = object()

    class FakeModel:
        def track(self, **kwargs):
            self.kwargs = kwargs
            return [expected]

    model = FakeModel()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    args = Namespace(
        conf=0.25,
        imgsz=64,
        device="cpu",
        max_det=10,
        tracker="botsort.yaml",
    )

    result = predict_media.track_result(model, frame, args, None)

    assert result is expected
    assert model.kwargs["source"] is frame
    assert model.kwargs["persist"] is True
    assert model.kwargs["tracker"] == "botsort.yaml"


def test_segmentation_roi_unions_boxes_adds_padding_and_clips() -> None:
    boxes = type("Boxes", (), {"xyxy": np.array([[5, 10, 25, 30], [35, 20, 55, 45]])})()
    result = type("Result", (), {"boxes": boxes})()

    roi = predict_media.segmentation_roi(result, (50, 60, 3), padding=0.2)

    assert roi is not None
    x1, y1, x2, y2 = roi
    assert x1 == 0
    assert y1 == 0
    assert x2 <= 60 and y2 <= 50
    assert x2 >= 55 and y2 >= 45
