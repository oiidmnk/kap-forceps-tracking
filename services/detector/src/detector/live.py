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
from .contact import RodContact
from .pipeline import Detection, Detector, annotate
from .runtime import RuntimeConfig
from .sources import frames, rotate_frame


class Smoother:
    """EMA on the pushed points. Points may come and go (forceps and rod are optional): a new one starts unsmoothed,
    one that vanishes is held for `hold` pushes (bridges single missed frames) and then dropped. Values that are not
    points (the rod_contact flag) pass through unchanged."""

    def __init__(self, w: float, hold: int = 3):
        self.w, self.hold, self.last, self.missing = w, hold, None, {}

    # The rod's ends are unordered: keep each (end, its shadow) pair on the side it was on in the last push.
    SWAP = (("rod_a_px", "rod_a_shadow_px"), ("rod_b_px", "rod_b_shadow_px"))

    def __call__(self, pts: dict) -> dict:
        flags = {k: v for k, v in pts.items() if not isinstance(v, (list, tuple))}
        pts = {k: v for k, v in pts.items() if k not in flags}
        last = self.last or {}
        pts = self._align(pts, last)
        out = {}
        for k, v in pts.items():
            p = last.get(k)
            out[k] = v if p is None or self.w >= 1 else [self.w * v[0] + (1 - self.w) * p[0],
                                                         self.w * v[1] + (1 - self.w) * p[1]]
            self.missing.pop(k, None)
        for k, p in last.items():
            if k not in pts and self.missing.get(k, 0) < self.hold:
                self.missing[k] = self.missing.get(k, 0) + 1
                out[k] = p
        self.missing = {k: n for k, n in self.missing.items() if k in out}
        self.last = out
        return {**out, **flags}

    def _align(self, pts: dict, last: dict) -> dict:
        (a, a_s), (b, b_s) = self.SWAP
        if not all(k in pts and k in last for k in (a, b)):
            return pts
        d = lambda p, q: (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
        if d(pts[a], last[b]) + d(pts[b], last[a]) < d(pts[a], last[a]) + d(pts[b], last[b]):
            pts = {**pts, a: pts[b], b: pts[a], a_s: pts[b_s], b_s: pts[a_s]}
        return pts


ROD_KEYS = ("rod_a_px", "rod_b_px", "rod_a_shadow_px", "rod_b_shadow_px")


class RodHold:
    """The rod floats still unless something pushes it: while it is not found (hidden by the forceps, its shadow merged
    with the forceps shadow, a weak frame) keep its last position instead of letting it vanish. If the forceps touched it
    when it was lost, it is being carried (and is often hidden by the forceps then): it moves along with the tips for up
    to `follow` frames, then stays where it is. Frames without anything found are still not pushed."""

    def __init__(self, follow: int = 30):
        self.follow = follow
        self.reset()

    def reset(self) -> None:
        self.last: dict | None = None
        self.tips: tuple[float, float] | None = None
        self.lost = 0

    @staticmethod
    def _tips(pts: dict):
        if "left_tip_px" not in pts or "right_tip_px" not in pts:
            return None
        (lx, ly), (rx, ry) = pts["left_tip_px"], pts["right_tip_px"]
        return (lx + rx) / 2, (ly + ry) / 2

    def __call__(self, pts: dict | None) -> dict | None:
        if not pts:
            return pts
        tips = self._tips(pts)
        if all(k in pts for k in ROD_KEYS):
            self.last = {k: pts[k] for k in ROD_KEYS} | {"rod_contact": bool(pts.get("rod_contact"))}
            self.tips, self.lost = tips, 0
            return pts
        if self.last is None:
            return pts
        self.lost += 1
        if self.last["rod_contact"] and tips is not None and self.tips is not None and self.lost <= self.follow:
            dx, dy = tips[0] - self.tips[0], tips[1] - self.tips[1]
            self.last = {k: [self.last[k][0] + dx, self.last[k][1] + dy] for k in ROD_KEYS} | {"rod_contact": True}
            self.tips = tips
        elif self.lost > self.follow or tips is None:
            self.last = {**self.last, "rod_contact": False}
        return {**pts, **self.last}


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
        self._contact = RodContact()
        self._rod_hold = RodHold()
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
        self._contact.reset()
        self._rod_hold.reset()
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
            if cfg.rotate:
                frame = rotate_frame(frame, cfg.rotate)     # circle editor, autocircle and preview see the turned image
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
            contact = self._contact(d)                     # every frame: the rod's motion is measured per frame
            if d.rod is not None:
                d.rod.contact = contact
            ok, buf = cv2.imencode(".jpg", annotate(self.det.view, d), [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
            with self.cond:
                self.detection, self.jpeg = d, buf.tobytes() if ok else None
                self.stats["frames"] += 1
                self.cond.notify_all()
            n_fps += 1
            if time.monotonic() - t_fps >= 1.0:
                self.stats["fps"] = round(n_fps / (time.monotonic() - t_fps), 1)
                t_fps, n_fps = time.monotonic(), 0
            pts = self._rod_hold(d.orchestrator_points())      # every frame: a carried rod follows the tips
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
