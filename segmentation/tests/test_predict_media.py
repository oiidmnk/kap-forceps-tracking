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
