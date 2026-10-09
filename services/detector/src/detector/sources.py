"""Frame sources: video file (looped, paced to real time), network stream, or camera index."""
from __future__ import annotations

import os
import time
from typing import Iterator

import cv2
import numpy as np


def open_capture(source: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)


def is_file_source(source: str) -> bool:
    return not (source.isdigit() or "://" in source)


def frames(source: str, loop: bool = True, realtime: bool = True, should_stop=None, on_error=None) -> Iterator[np.ndarray]:
    """Yield frames (forever for live inputs / looped files). Never blocks past `should_stop()`:
    an unopenable source is retried once a second *and* abandoned as soon as the config changes."""
    stop = should_stop or (lambda: False)
    is_file = is_file_source(source)
    while not stop():
        cap = open_capture(source)
        if not cap.isOpened():
            cap.release()
            if on_error:
                on_error(f"cannot open source '{source}'" + (" (file not found in the detector)" if is_file and not os.path.exists(source) else ""))
            for _ in range(10):                      # 1 s, but responsive to config changes
                if stop():
                    return
                time.sleep(0.1)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        t0, n = time.monotonic(), 0
        try:
            while not stop():
                ok, frame = cap.read()
                if not ok:
                    break
                if is_file and realtime:                   # pace files like a camera would
                    delay = t0 + n / fps - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                n += 1
                yield frame
        finally:
            cap.release()           # also runs when the consumer closes the generator (source switch)
        if is_file and not loop:
            return
        time.sleep(0.2)
