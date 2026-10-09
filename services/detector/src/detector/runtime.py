"""Runtime configuration (source + eye circle) that can be changed while the service is running."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def parse_circle(value) -> tuple[float, float, float] | None:
    if value in (None, "", {}):
        return None
    cx, cy, r = float(value["cx"]), float(value["cy"]), float(value["r"])
    if r < 20:
        raise ValueError("circle radius must be >= 20 px")
    return cx, cy, r


class RuntimeConfig:
    """Thread-safe {source, circle}; every change bumps `version` so the live loop can react."""

    def __init__(self, source: str = "", state_path: str | None = None):
        self._lock = threading.Lock()
        self.version = 0
        self.reset_count = 0          # bumped by restart(): hard reset of background + source
        self.source = source
        self.circle: tuple[float, float, float] | None = None      # in raw source-frame pixels
        self.path = Path(state_path) if state_path else None
        self._load()

    def _load(self) -> None:
        if not (self.path and self.path.is_file()):
            return
        try:
            data = json.loads(self.path.read_text())
            self.source = data.get("source", self.source) or self.source
            self.circle = parse_circle(data.get("circle"))
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.as_dict()))
        except OSError:
            pass          # state dir not writable: keep working in memory

    def update(self, source=..., circle=...) -> None:
        with self._lock:
            if source is not ...:
                self.source = str(source or "")
            if circle is not ...:
                self.circle = parse_circle(circle)
            self.version += 1
            self._save()

    def restart(self) -> None:
        """Hard reset: reopen the source from the start and discard everything learned (background)."""
        with self._lock:
            self.reset_count += 1
            self.version += 1

    def snapshot(self):
        with self._lock:
            return self.version, self.source, self.circle

    def as_dict(self) -> dict:
        c = self.circle
        return {"source": self.source, "circle": None if c is None else {"cx": c[0], "cy": c[1], "r": c[2]}}


def list_video_files(directory: str | None = None) -> list[str]:
    d = Path(directory or os.environ.get("VIDEO_DIR", "/data/video"))
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)


def scan_cameras(skip: set[int], limit: int = 4) -> list[dict]:
    """Probe local capture devices (only useful when the detector runs natively, not in Docker)."""
    import cv2
    found = []
    for i in range(limit):
        if i in skip:
            continue
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, f = cap.read()
            if ok:
                found.append({"index": i, "width": f.shape[1], "height": f.shape[0]})
        cap.release()
    return found
