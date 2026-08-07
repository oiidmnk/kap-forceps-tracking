from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results

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


def test_restore_pose_coordinates_clones_inference_tensors() -> None:
    crop = np.zeros((50, 60, 3), dtype=np.uint8)
    full_frame = np.zeros((100, 120, 3), dtype=np.uint8)
    with torch.inference_mode():
        boxes = torch.tensor([[1.0, 2.0, 30.0, 40.0, 0.9, 0.0]])
        keypoints = torch.tensor([[[2.0, 3.0, 0.9], [4.0, 5.0, 0.8], [6.0, 7.0, 0.7]]])
    assert boxes.is_inference() and keypoints.is_inference()
    result = Results(crop, path="frame.png", names={0: "forceps"}, boxes=boxes, keypoints=keypoints)

    restored = predict_media.restore_pose_coordinates(result, full_frame, (10, 20, 70, 70))

    assert restored.boxes.xyxy.tolist() == [[11.0, 22.0, 40.0, 60.0]]
    assert restored.keypoints.xy.tolist() == [[[12.0, 23.0], [14.0, 25.0], [16.0, 27.0]]]
    assert not restored.boxes.data.is_inference()
    assert not restored.keypoints.data.is_inference()
