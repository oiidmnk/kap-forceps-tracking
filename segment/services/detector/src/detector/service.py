"""HTTP API. `/segment` is contract-compatible with the YOLO segmentation service the orchestrator expects."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from .config import Settings
from .live import LiveWorker
from .pipeline import Detector, annotate

CLASSES = {"tip_left": ("forceps", "left_tip"), "tip_right": ("forceps", "right_tip"),
           "shadow_left": ("shadow", "left_tip"), "shadow_right": ("shadow", "right_tip")}


def _decode(payload: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "uploaded file is not a readable image")
    return img


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or Settings()
    detector = Detector(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if cfg.source:
            app.state.live = LiveWorker(detector, cfg)
            app.state.live.start()
        yield
        if app.state.live:
            app.state.live.stop()

    app = FastAPI(title="Forceps Detector", version="1.0.0", lifespan=lifespan)
    app.state.detector, app.state.live = detector, None

    @app.get("/health")
    def health() -> dict:
        live = app.state.live
        return {"status": "ok", "weights_available": True, "background": detector.bg.state,
                "live": bool(live), "live_frames": live.stats["frames"] if live else 0}

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
            raise HTTPException(404, "no live SOURCE configured")
        return app.state.live

    @app.get("/state")
    def state() -> dict:
        live = _need_live()
        return {"detection": live.detection.as_dict() if live.detection else None, **live.stats,
                "background": detector.bg.state}

    @app.get("/live.mjpg")
    def live_mjpg() -> StreamingResponse:
        return StreamingResponse(_need_live().mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (Path(__file__).parent / "viewer.html").read_text()

    return app


app = create_app()
