"""Source selection + eye-circle editor: a thin, persistent proxy to the detector service.

The detector turns the chosen source into a normalised 1:1 square (circle content, black outside). Saving a circle
therefore also resets the eye calibration to that square (centre = size/2, radius = size/2).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import File, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse


class DetectorStore:
    """Persists which detector the UI controls (e.g. docker `detector` vs. a native one on the host)."""

    def __init__(self, path: Path, default_url: str) -> None:
        self.path, self.default_url = path, default_url

    def get(self) -> str:
        try:
            return json.loads(self.path.read_text())["url"]
        except (OSError, ValueError, KeyError):
            return self.default_url

    def set(self, url: str) -> str:
        url = url.strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError("detector URL must start with http:// or https://")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"url": url}))
        except OSError:
            pass
        return url


DEFAULT_PATH = Path(os.getenv("ORCHESTRATOR_DETECTOR_PATH", "/state/detector.json"))


def add_source_routes(service, apply_calibration, store: DetectorStore, transport=None) -> None:
    service.state.segmentation_url = store.get()
    async def call(method: str, path: str, **kw) -> httpx.Response:
        url = f"{store.get()}{path}"
        try:
            async with httpx.AsyncClient(transport=transport, timeout=kw.pop("timeout", 10.0)) as client:
                return await client.request(method, url, **kw)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"detector not reachable at {store.get()}: {exc}") from exc

    def check(resp: httpx.Response) -> Any:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise HTTPException(resp.status_code if resp.status_code < 500 else 502, f"detector: {detail}")
        return resp.json()

    @service.get("/api/source")
    async def get_source() -> dict[str, Any]:
        out: dict[str, Any] = {"detector_url": store.get(), "default_detector_url": store.default_url}
        try:
            out["config"] = check(await call("GET", "/config"))
            out["sources"] = check(await call("GET", "/sources"))
        except HTTPException as exc:
            out["error"] = exc.detail
        return out

    @service.get("/api/source/scan")
    async def scan_sources() -> dict[str, Any]:
        return check(await call("GET", "/sources", params={"scan": "true"}, timeout=30.0))

    @service.put("/api/source/detector-url")
    async def put_detector_url(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            url = store.set(str(payload.get("url", "")))
            service.state.segmentation_url = url      # /api/status and /api/process follow the chosen detector
            return {"detector_url": url}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @service.get("/api/source/autocircle")
    async def autocircle() -> dict[str, Any]:
        return check(await call("GET", "/autocircle"))

    @service.post("/api/source/config")
    async def set_source_config(payload: dict[str, Any]) -> dict[str, Any]:
        """Forward {source?, circle?, clear_circle?} to the detector; a new circle re-bases the eye calibration."""
        config = check(await call("POST", "/config", json=payload))
        calibration = None
        if payload.get("circle"):
            size = float(config["normalized_size"])
            current = service.state.calibration_store.load()
            current.update(eye_center_px=[size / 2, size / 2], eye_radius_px=size / 2)
            calibration = service.state.calibration_store.save(current)
            try:
                await apply_calibration()
            except HTTPException:
                pass            # stream has no seed yet; the calibration is saved and applies with the next points
        return {"config": config, "calibration": calibration}

    @service.post("/api/source/reset")
    async def reset_detector() -> dict[str, Any]:
        """Restart the video from the start with a clean detector state (background knowledge discarded)."""
        return {"config": check(await call("POST", "/reset"))}

    @service.post("/api/source/upload")
    async def upload_source(file: UploadFile = File(...)) -> dict[str, Any]:
        try:
            if Path(file.filename or "").suffix.lower() != ".mp4":
                raise HTTPException(400, "upload an MP4 video")
            size = file.file.seek(0, 2)
            await file.seek(0)
            if size == 0:
                raise HTTPException(400, "uploaded video is empty")
            if size > 512 * 1024 * 1024:
                raise HTTPException(413, "video must be 512 MiB or smaller")
            config = check(await call("POST", "/source/upload", timeout=1200.0,
                                      files={"file": (Path(file.filename.replace("\\", "/")).name, file.file, "video/mp4")}))
            return {"config": config}
        finally:
            await file.close()

    @service.get("/api/source/preview.jpg")
    async def preview() -> Response:
        resp = await call("GET", "/preview.jpg")
        if resp.status_code >= 400:
            raise HTTPException(resp.status_code if resp.status_code < 500 else 502, "no frame from detector yet")
        headers = {k: v for k, v in resp.headers.items() if k.lower() in ("x-frame-width", "x-frame-height")}
        headers["Cache-Control"] = "no-store"
        return Response(resp.content, media_type="image/jpeg", headers=headers)

    @service.get("/api/source/live.mjpg")
    async def live() -> StreamingResponse:
        url = f"{store.get()}/live.mjpg"

        async def gen():
            try:
                async with httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(10.0, read=None)) as client:
                    async with client.stream("GET", url) as resp:
                        async for chunk in resp.aiter_raw():
                            yield chunk
            except httpx.HTTPError:
                return

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
