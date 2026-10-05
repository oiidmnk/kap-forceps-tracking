"""Live worker: source -> detector -> annotated MJPEG + orchestrator push."""
from __future__ import annotations

import threading
import time

import cv2
import httpx

from .config import Settings
from .pipeline import Detection, Detector, annotate
from .sources import frames


class Smoother:
    """EMA on the pushed points; jaw points that vanish simply stop updating."""

    def __init__(self, w: float):
        self.w, self.last = w, None

    def __call__(self, pts: dict) -> dict:
        if self.last is None or self.w >= 1:
            self.last = pts
        else:
            self.last = {k: [self.w * v[0] + (1 - self.w) * self.last[k][0],
                             self.w * v[1] + (1 - self.w) * self.last[k][1]] for k, v in pts.items()}
        return self.last


class LiveWorker(threading.Thread):
    def __init__(self, detector: Detector, cfg: Settings):
        super().__init__(daemon=True, name="live")
        self.det, self.cfg = detector, cfg
        self.jpeg: bytes | None = None
        self.detection: Detection | None = None
        self.stats = {"fps": 0.0, "frames": 0, "pushed": 0, "push_error": None, "source": cfg.source}
        self.cond = threading.Condition()
        self._stop = threading.Event()
        self._smooth = Smoother(cfg.smoothing)
        self._last_push = 0.0

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        cfg = self.cfg
        client = httpx.Client(timeout=2.0) if cfg.orchestrator_url else None
        t_fps, n_fps, last_boot = time.monotonic(), 0, 0.0
        for frame in frames(cfg.source, cfg.source_loop, cfg.source_realtime):
            if self._stop.is_set():
                break
            work, _ = self.det._prepare(frame)
            if self.det.bg.state != "static":              # learn the empty retina from the live feed
                now = time.monotonic()
                if now - last_boot >= cfg.bootstrap_seconds / 30:
                    last_boot = now
                    self.det.bg.add_sample(work)
            d = self.det.detect(frame, adapt=True)
            ok, buf = cv2.imencode(".jpg", annotate(frame, d), [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
            with self.cond:
                self.detection, self.jpeg = d, buf.tobytes() if ok else None
                self.stats["frames"] += 1
                self.cond.notify_all()
            n_fps += 1
            if time.monotonic() - t_fps >= 1.0:
                self.stats["fps"] = round(n_fps / (time.monotonic() - t_fps), 1)
                t_fps, n_fps = time.monotonic(), 0
            pts = d.orchestrator_points()
            if client and pts and time.monotonic() - self._last_push >= 1.0 / cfg.push_hz:
                self._last_push = time.monotonic()
                try:
                    r = client.post(f"{cfg.orchestrator_url.rstrip('/')}/api/manual-points", json=self._smooth(pts))
                    r.raise_for_status()
                    self.stats["pushed"] += 1
                    self.stats["push_error"] = None
                except Exception as exc:  # noqa: BLE001 - keep streaming when the orchestrator is down
                    self.stats["push_error"] = str(exc)[:200]

    def mjpeg(self):
        last = -1
        while not self._stop.is_set():
            with self.cond:
                self.cond.wait_for(lambda: self.stats["frames"] != last, timeout=2.0)
                jpg, last = self.jpeg, self.stats["frames"]
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg) + jpg + b"\r\n"
