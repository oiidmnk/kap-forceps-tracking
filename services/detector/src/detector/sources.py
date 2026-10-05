"""Frame sources: video file (looped, paced to real time), network stream, or camera index."""
from __future__ import annotations

import time
from typing import Iterator

import cv2
import numpy as np


def open_capture(source: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)


def frames(source: str, loop: bool = True, realtime: bool = True) -> Iterator[np.ndarray]:
    """Yield frames forever (or until a non-looping file ends). Live inputs reconnect on failure."""
    is_file = not (source.isdigit() or "://" in source)
    while True:
        cap = open_capture(source)
        if not cap.isOpened():
            time.sleep(1.0)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        t0, n = time.monotonic(), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if is_file and realtime:                       # pace files like a camera would
                delay = t0 + n / fps - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            n += 1
            yield frame
        cap.release()
        if is_file and not loop:
            return
        time.sleep(0.2)
