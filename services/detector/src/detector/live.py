"""Live worker: source -> eye-circle normalisation -> detector -> annotated MJPEG + orchestrator push."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

from .background import Background
from .config import Settings
from .pipeline import Detection, Detector, annotate
from .runtime import RuntimeConfig
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
    def __init__(self, detector: Detector, cfg: Settings, runtime: RuntimeConfig):
        super().__init__(daemon=True, name="live")
        self.det, self.cfg, self.rt = detector, cfg, runtime
        self.jpeg: bytes | None = None
        self.raw: np.ndarray | None = None          # last un-normalised frame (for the circle editor)
        self.detection: Detection | None = None
        self.stats = {"fps": 0.0, "frames": 0, "pushed": 0, "push_error": None, "detect_error": None,
                      "source_error": None}
        self.cond = threading.Condition()
        self._stop = threading.Event()
        self._smooth = Smoother(cfg.smoothing)
        self._last_push = 0.0
        self._applied_version = -1
        self._applied_resets = 0

    def stop(self) -> None:
        self._stop.set()

    # -- configuration changes -------------------------------------------------
    def _apply(self, version: int, source: str, circle) -> None:
        first = self._applied_version < 0
        hard = self.rt.reset_count != self._applied_resets
        self._applied_version, self._applied_resets = version, self.rt.reset_count
        self.det.roi = circle
        if hard:
            self.det.bg.reset()                        # discard everything learned; relearn from the stream
        elif not first:
            self._reset_background(source, circle)
        self._smooth.last = None
        self.stats.update(detect_error=None, source_error=None)
        with self.cond:
            self.jpeg, self.detection = None, None

    def _reset_background(self, source: str, circle) -> None:
        """Geometry changed -> the old background no longer fits; relearn (or reload the shipped one)."""
        self.det.bg.reset()
        path = self.cfg.background_path
        if circle is None and source == self.cfg.source and path and Path(path).is_file():
            self.det.bg = Background(path, self.cfg.work_size)

    # -- main loop ------------------------------------------------------------
    def run(self) -> None:
        client = httpx.Client(timeout=2.0) if self.cfg.orchestrator_url else None
        while not self._stop.is_set():
            version, source, circle = self.rt.snapshot()
            if not source:
                time.sleep(0.3)
                continue
            self._apply(version, source, circle)
            gen = frames(source, self.cfg.source_loop, self.cfg.source_realtime,
                         should_stop=lambda: self._stop.is_set() or self.rt.version != version,
                         on_error=lambda msg: self.stats.update(source_error=msg))
            try:
                self._consume(gen, version, client)
            finally:
                gen.close()

    def _consume(self, gen, version: int, client) -> None:
        cfg = self.cfg
        t_fps, n_fps, last_boot = time.monotonic(), 0, 0.0
        for frame in gen:
            if self._stop.is_set() or self.rt.version != version:
                return
            self.raw = frame
            self.stats["source_error"] = None
            prepared = self.det._prepare(frame)
            work = prepared[0]
            if self.det.bg.state != "static":              # learn the empty retina from the live feed
                now = time.monotonic()
                if now - last_boot >= cfg.bootstrap_seconds / 30:
                    last_boot = now
                    self.det.bg.add_sample(work)
            try:
                d = self.det.detect(frame, adapt=True, prepared=prepared)
            except Exception as exc:  # noqa: BLE001 - one bad frame must not kill the live loop
                self.stats["detect_error"] = repr(exc)[:200]
                continue
            ok, buf = cv2.imencode(".jpg", annotate(self.det.view, d), [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
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

    def raw_jpeg(self, max_side: int = 1280) -> tuple[bytes, tuple[int, int]] | None:
        frame = self.raw
        if frame is None:
            return None
        h, w = frame.shape[:2]
        k = min(1.0, max_side / max(h, w))
        img = frame if k == 1.0 else cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return (buf.tobytes(), (w, h)) if ok else None

    def mjpeg(self):
        last = -1
        while not self._stop.is_set():
            with self.cond:
                self.cond.wait_for(lambda: self.stats["frames"] != last, timeout=2.0)
                jpg, last = self.jpeg, self.stats["frames"]
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg) + jpg + b"\r\n"
