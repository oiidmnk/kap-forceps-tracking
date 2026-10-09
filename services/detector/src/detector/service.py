"""HTTP API. `/segment` is contract-compatible with the YOLO segmentation service the orchestrator expects."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from pydantic import BaseModel

from .background import find_eye_disc
from .config import Settings
from .live import LiveWorker
from .pipeline import Detector, annotate
from .runtime import RuntimeConfig, list_video_files, scan_cameras

CLASSES = {"tip_left": ("forceps", "left_tip"), "tip_right": ("forceps", "right_tip"),
           "shadow_left": ("shadow", "left_tip"), "shadow_right": ("shadow", "right_tip")}


class CircleIn(BaseModel):
    cx: float
    cy: float
    r: float


class ConfigIn(BaseModel):
    source: str | None = None
    circle: CircleIn | None = None
    clear_circle: bool = False


def _decode(payload: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "uploaded file is not a readable image")
    return img


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or Settings()
    detector = Detector(cfg)
    runtime = RuntimeConfig(cfg.source, cfg.state_path)
    detector.roi = runtime.circle

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.live = LiveWorker(detector, cfg, runtime)
        app.state.live.start()
        yield
        if app.state.live:
            app.state.live.stop()

    app = FastAPI(title="Forceps Detector", version="1.0.0", lifespan=lifespan)
    app.state.detector, app.state.runtime, app.state.live = detector, runtime, None

    @app.get("/health")
    def health() -> dict:
        live = app.state.live
        return {"status": "ok", "weights_available": True, "background": detector.bg.state,
                "live": bool(live and runtime.source), "live_frames": live.stats["frames"] if live else 0}

    @app.post("/detect")
    async def detect(image: UploadFile = File(...)) -> dict:
        return detector.detect(_decode(await image.read())).as_dict()

    @app.post("/detect/annotated")
    async def detect_annotated(image: UploadFile = File(...)) -> Response:
        img = _decode(await image.read())
        ok, buf = cv2.imencode(".jpg", annotate(img, detector.detect(img)))
        return Response(buf.tobytes(), media_type="image/jpeg")

    @app.post("/segment")
    async def segment(image: UploadFile = File(...)) -> dict:
        """Orchestrator-compatible: one point-instance per keypoint (tip_left/tip_right/shadow_left/shadow_right)."""
        d = detector.detect(_decode(await image.read()))
        inst = []
        for cid, (name, (tool, key)) in enumerate(CLASSES.items()):
            p = getattr(getattr(d, tool), key)
            if p is not None:
                x, y = p
                inst.append({"class_id": cid, "class_name": name, "confidence": 1.0,
                             "box": {"xyxy": [x - 1, y - 1, x + 1, y + 1]}, "segments": []})
        return {"instances": inst, "junctions": {"forceps": d.forceps.junction, "shadow": d.shadow.junction},
                "detection": d.as_dict()}

    @app.post("/background")
    async def set_background(image: UploadFile = File(...)) -> dict:
        detector.bg.set_image(detector._prepare(_decode(await image.read()))[0])
        return {"background": detector.bg.state}

    @app.post("/background/reset")
    def reset_background() -> dict:
        detector.bg.reset()
        return {"background": detector.bg.state}

    def _need_live() -> LiveWorker:
        if not app.state.live:
            raise HTTPException(503, "live worker not running")
        return app.state.live

    def _config() -> dict:
        live = app.state.live
        raw = live.raw if live else None
        return {**runtime.as_dict(), "normalized_size": cfg.work_size,
                "frame_size": None if raw is None else [raw.shape[1], raw.shape[0]],
                "background": detector.bg.state}

    @app.get("/config")
    def get_config() -> dict:
        return _config()

    @app.post("/config")
    def set_config(body: ConfigIn) -> dict:
        """Change source and/or eye circle (raw-frame px). The pipeline input becomes the circle content, 1:1."""
        kw = {}
        if body.source is not None:
            kw["source"] = body.source
        if body.clear_circle:
            kw["circle"] = None
        elif body.circle is not None:
            kw["circle"] = body.circle.model_dump()
        try:
            runtime.update(**kw)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        detector.roi = runtime.circle
        return _config()

    @app.post("/reset")
    def reset() -> dict:
        """Clean restart: reopen the source from the beginning, forget the learned background, relearn it."""
        runtime.restart()
        return _config()

    @app.get("/sources")
    def sources(scan: bool = False) -> dict:
        cur = runtime.source
        out = {"current": cur, "files": list_video_files(),
               "cameras": [], "note": "cameras are only visible when the detector runs natively on the host"}
        if scan:
            skip = {int(cur)} if cur.isdigit() else set()
            out["cameras"] = scan_cameras(skip)
            if cur.isdigit():
                out["cameras"].append({"index": int(cur), "in_use": True})
        return out

    @app.get("/preview.jpg")
    def preview() -> Response:
        got = _need_live().raw_jpeg()
        if got is None:
            raise HTTPException(404, "no frame yet")
        jpg, (w, h) = got
        return Response(jpg, media_type="image/jpeg",
                        headers={"X-Frame-Width": str(w), "X-Frame-Height": str(h), "Cache-Control": "no-store",
                                 "Access-Control-Expose-Headers": "X-Frame-Width, X-Frame-Height"})

    @app.get("/background.jpg")
    def background_image() -> Response:
        if detector.bg.bgr is None:
            raise HTTPException(404, "no background yet")
        ok, buf = cv2.imencode(".jpg", np.clip(detector.bg.bgr, 0, 255).astype(np.uint8))
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/autocircle")
    def autocircle() -> dict:
        raw = _need_live().raw
        if raw is None:
            raise HTTPException(404, "no frame yet")
        disc = find_eye_disc(raw)
        if disc is None:
            raise HTTPException(404, "no eye circle found")
        return {"cx": disc[0], "cy": disc[1], "r": disc[2]}

    @app.get("/state")
    def state() -> dict:
        live = _need_live()
        return {"detection": live.detection.as_dict() if live.detection else None, **live.stats,
                "background": detector.bg.state, **{k: v for k, v in _config().items() if k != "background"}}

    @app.get("/live.mjpg")
    def live_mjpg() -> StreamingResponse:
        return StreamingResponse(_need_live().mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (Path(__file__).parent / "viewer.html").read_text()

    return app


app = create_app()
