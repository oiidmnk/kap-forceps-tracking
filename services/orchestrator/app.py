"""Operator-facing orchestration API and Web UI."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

CALIBRATION_KEYS = [
    "light_rot_up",
    "light_rot_clock",
    "light_depth_mm",
    "forceps_rot_up",
    "forceps_rot_clock",
    "eye_center_px",
    "eye_radius_px",
    "eye_radius_mm",
    "jaw_length_mm",
]

REQUIRED_CLASS_TO_INPUT = {
    "tip_left": "left_tip_px",
    "tip_right": "right_tip_px",
    "shadow_left": "left_shadow_px",
    "shadow_right": "right_shadow_px",
}

REQUIRED_CLASS_ID_TO_INPUT = {
    0: "left_tip_px",
    1: "right_tip_px",
    2: "left_shadow_px",
    3: "right_shadow_px",
}

DEFAULT_SEGMENTATION_URL = "http://segmentation:8000"
DEFAULT_STREAM_URL = "http://stream:8765"
DEFAULT_CALIBRATION_PATH = Path(os.getenv("ORCHESTRATOR_CALIBRATION_PATH", "/data/calibration.json"))
DEFAULT_CALIBRATION_SOURCE = Path(
    os.getenv("ORCHESTRATOR_DEFAULT_CALIBRATION", "/app/default_calibration.json")
)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES = Jinja2Templates(directory=BASE_DIR / "templates")


class OrchestratorError(ValueError):
    """Raised for validation errors that should be shown to the operator."""


class CalibrationStore:
    def __init__(self, path: Path, default_source: Path | None = None) -> None:
        self.path = path
        self.default_source = default_source

    def load(self) -> dict[str, Any]:
        if self.path.is_file():
            return validate_calibration(read_json_object(self.path))
        if self.default_source is not None and self.default_source.is_file():
            calibration = validate_calibration(read_json_object(self.default_source))
            self.save(calibration)
            return calibration
        raise OrchestratorError(
            f"calibration file not found: {self.path}; default source not found: {self.default_source}"
        )

    def save(self, calibration: dict[str, Any]) -> dict[str, Any]:
        normalized = validate_calibration(calibration)
        atomic_write_json(self.path, normalized)
        return normalized


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise OrchestratorError(f"{path} must contain a JSON object")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_name = temp_file.name
            json.dump(payload, temp_file, indent=2)
            temp_file.write("\n")
        os.replace(temp_name, path)
    except Exception:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
        raise


def _as_vec2(value: Any, key: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise OrchestratorError(f"{key} must be [x, y]")
    return [float(value[0]), float(value[1])]


def validate_calibration(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise OrchestratorError("calibration must be a JSON object")
    missing = [key for key in CALIBRATION_KEYS if key not in payload]
    if missing:
        raise OrchestratorError("calibration missing keys: " + ", ".join(missing))

    calibration = {
        "light_rot_up": float(payload["light_rot_up"]),
        "light_rot_clock": float(payload["light_rot_clock"]),
        "light_depth_mm": float(payload["light_depth_mm"]),
        "forceps_rot_up": float(payload["forceps_rot_up"]),
        "forceps_rot_clock": float(payload["forceps_rot_clock"]),
        "eye_center_px": _as_vec2(payload["eye_center_px"], "eye_center_px"),
        "eye_radius_px": float(payload["eye_radius_px"]),
        "eye_radius_mm": float(payload["eye_radius_mm"]),
        "jaw_length_mm": float(payload["jaw_length_mm"]),
    }
    if calibration["eye_radius_px"] <= 0:
        raise OrchestratorError("eye_radius_px must be greater than 0")
    if calibration["eye_radius_mm"] <= 0:
        raise OrchestratorError("eye_radius_mm must be greater than 0")
    if calibration["jaw_length_mm"] <= 0:
        raise OrchestratorError("jaw_length_mm must be greater than 0")
    return calibration


def polygon_centroid(points: list[list[float]]) -> list[float]:
    if not points:
        raise OrchestratorError("polygon has no points")
    if len(points) < 3:
        array = np.asarray(points, dtype=float)
        return [float(array[:, 0].mean()), float(array[:, 1].mean())]

    area_twice = 0.0
    cx_acc = 0.0
    cy_acc = 0.0
    for current, nxt in zip(points, points[1:] + points[:1], strict=True):
        x0, y0 = float(current[0]), float(current[1])
        x1, y1 = float(nxt[0]), float(nxt[1])
        cross = x0 * y1 - x1 * y0
        area_twice += cross
        cx_acc += (x0 + x1) * cross
        cy_acc += (y0 + y1) * cross

    if abs(area_twice) < 1e-9:
        array = np.asarray(points, dtype=float)
        return [float(array[:, 0].mean()), float(array[:, 1].mean())]
    return [cx_acc / (3.0 * area_twice), cy_acc / (3.0 * area_twice)]


def point_from_instance(instance: dict[str, Any]) -> list[float]:
    segments = instance.get("segments") or []
    for segment in segments:
        points = segment.get("points") or []
        if points:
            return polygon_centroid([[float(x), float(y)] for x, y in points])

    box = instance.get("box") or {}
    xyxy = box.get("xyxy")
    if isinstance(xyxy, (list, tuple)) and len(xyxy) == 4:
        x1, y1, x2, y2 = [float(value) for value in xyxy]
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
    raise OrchestratorError(
        f"instance {instance.get('class_name', '<unknown>')} has neither segment points nor box"
    )


def validate_manual_points(payload: Any) -> dict[str, list[float]]:
    """Validate hand-clicked points meant to stand in for segmentation output.

    Same four keys ``select_preprocessor_points`` would have produced from a
    trained model, so callers (the stream merge) can't tell the difference.
    """
    if not isinstance(payload, dict):
        raise OrchestratorError("manual points payload must be a JSON object")
    required = list(REQUIRED_CLASS_TO_INPUT.values())
    missing = [key for key in required if key not in payload]
    if missing:
        raise OrchestratorError("manual points missing keys: " + ", ".join(missing))
    return {key: _as_vec2(payload[key], key) for key in required}


def select_preprocessor_points(segmentation: dict[str, Any]) -> dict[str, list[float]]:
    instances = segmentation.get("instances")
    if not isinstance(instances, list):
        raise OrchestratorError("segmentation response missing instances list")

    selected: dict[str, dict[str, Any]] = {}
    for instance in instances:
        if not isinstance(instance, dict):
            continue
        input_key = preprocessor_input_key(instance)
        if input_key is None:
            continue
        current = selected.get(input_key)
        confidence = instance.get("confidence")
        current_confidence = current.get("confidence") if current else None
        if current is None or _confidence_value(confidence) > _confidence_value(current_confidence):
            selected[input_key] = instance

    missing = [input_key for input_key in REQUIRED_CLASS_TO_INPUT.values() if input_key not in selected]
    if missing:
        raise OrchestratorError(missing_classes_message(missing, instances))

    return {
        input_key: point_from_instance(selected[input_key])
        for input_key in REQUIRED_CLASS_TO_INPUT.values()
    }


def preprocessor_input_key(instance: dict[str, Any]) -> str | None:
    class_name = str(instance.get("class_name", ""))
    if class_name in REQUIRED_CLASS_TO_INPUT:
        return REQUIRED_CLASS_TO_INPUT[class_name]

    try:
        class_id = int(instance.get("class_id"))
    except (TypeError, ValueError):
        return None
    return REQUIRED_CLASS_ID_TO_INPUT.get(class_id)


def missing_classes_message(missing_input_keys: list[str], instances: list[Any]) -> str:
    missing_names = [
        class_name
        for class_name, input_key in REQUIRED_CLASS_TO_INPUT.items()
        if input_key in missing_input_keys
    ]
    detected = detection_summary(instances)
    if not detected:
        return (
            "segmentation returned no detections for required classes: "
            + ", ".join(missing_names)
            + ". Check that the uploaded frame contains the forceps/shadows, "
            "or lower SEGMENTATION_CONF if the model is too strict."
        )

    return (
        "segmentation missing required classes: "
        + ", ".join(missing_names)
        + ". Detected classes were: "
        + "; ".join(detected)
        + ". Expected class names tip_left, tip_right, shadow_left, shadow_right "
        "or class IDs 0, 1, 2, 3."
    )


def detection_summary(instances: list[Any]) -> list[str]:
    summary = []
    for instance in instances:
        if not isinstance(instance, dict):
            continue
        class_id = instance.get("class_id")
        class_name = instance.get("class_name")
        confidence = instance.get("confidence")
        label = f"{class_name} (id={class_id}, conf={confidence})"
        if label not in summary:
            summary.append(label)
    return summary


def _confidence_value(value: Any) -> float:
    return float(value) if value is not None else -1.0


def decode_upload_frame(payload: bytes, filename: str | None, content_type: str | None, frame_time_s: float) -> bytes:
    if is_video_upload(filename, content_type):
        image = decode_video_frame(payload, frame_time_s)
    else:
        image = decode_image(payload)

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OrchestratorError("failed to encode frame for segmentation")
    return encoded.tobytes()


def is_video_upload(filename: str | None, content_type: str | None) -> bool:
    if content_type and content_type.startswith("video/"):
        return True
    suffix = Path(filename or "").suffix.lower()
    return suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def decode_image(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise OrchestratorError("uploaded file is not a readable image")
    return image


def decode_video_frame(payload: bytes, frame_time_s: float) -> np.ndarray:
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".upload-video", delete=False) as temp_file:
            temp_name = temp_file.name
            temp_file.write(payload)
        capture = cv2.VideoCapture(temp_name)
        if not capture.isOpened():
            raise OrchestratorError("uploaded file is not a readable video")
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, frame_time_s) * 1000.0)
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            raise OrchestratorError(f"could not read video frame at {frame_time_s:.3f}s")
        return frame
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


async def post_segmentation(
    segmentation_url: str,
    frame_png: bytes,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    async with httpx.AsyncClient(transport=transport, timeout=60.0) as client:
        response = await client.post(
            f"{segmentation_url.rstrip('/')}/segment",
            files={"image": ("frame.png", frame_png, "image/png")},
        )
    if response.status_code >= 400:
        detail = _response_detail(response)
        if "segmentation weights not found" in detail:
            detail = (
                f"{detail}. Put trained weights at "
                "segmentation/runs/segment/forceps/weights/best.pt, or set "
                "SEGMENTATION_WEIGHTS to another path that is mounted inside "
                "the segmentation container."
            )
        raise HTTPException(
            status_code=502,
            detail=f"segmentation service failed: {detail}",
        )
    return response.json()


async def put_stream_inputs(
    stream_url: str,
    payload: dict[str, Any],
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    async with httpx.AsyncClient(transport=transport, timeout=30.0) as client:
        response = await client.put(f"{stream_url.rstrip('/')}/inputs", json=payload)
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"stream service failed: {_response_detail(response)}")
    return response.json()


async def get_stream_inputs(
    stream_url: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    async with httpx.AsyncClient(transport=transport, timeout=30.0) as client:
        response = await client.get(f"{stream_url.rstrip('/')}/inputs")
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"stream service failed: {_response_detail(response)}")
    return response.json()


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text
    if isinstance(payload, dict) and payload.get("detail") is not None:
        return str(payload["detail"])
    return str(payload)


async def get_service_health(
    service_url: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(transport=transport, timeout=5.0) as client:
            response = await client.get(f"{service_url.rstrip('/')}/health")
        payload = response.json() if response.content else {}
        return {
            "available": response.status_code < 400,
            "status_code": response.status_code,
            "payload": payload,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - status endpoint should report dependency state
        return {
            "available": False,
            "status_code": None,
            "payload": None,
            "error": str(exc),
        }


def create_app(
    calibration_path: Path = DEFAULT_CALIBRATION_PATH,
    default_calibration_source: Path | None = DEFAULT_CALIBRATION_SOURCE,
    segmentation_url: str = os.getenv("SEGMENTATION_URL", DEFAULT_SEGMENTATION_URL),
    stream_url: str = os.getenv("STREAM_URL", DEFAULT_STREAM_URL),
    segmentation_transport: httpx.AsyncBaseTransport | None = None,
    stream_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    service = FastAPI(title="Forceps Orchestrator", version="0.1.0")
    service.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    service.state.calibration_store = CalibrationStore(calibration_path, default_calibration_source)
    service.state.segmentation_url = segmentation_url
    service.state.stream_url = stream_url
    service.state.segmentation_transport = segmentation_transport
    service.state.stream_transport = stream_transport

    @service.get("/")
    async def index(request: Request):
        return TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {
                "title": "Forceps Orchestrator",
            },
        )

    @service.get("/api/status")
    async def get_status() -> dict[str, Any]:
        segmentation = await get_service_health(
            service.state.segmentation_url,
            service.state.segmentation_transport,
        )
        stream = await get_service_health(
            service.state.stream_url,
            service.state.stream_transport,
        )
        weights_available = bool(
            isinstance(segmentation.get("payload"), dict)
            and segmentation["payload"].get("weights_available")
        )
        return {
            "segmentation": segmentation,
            "stream": stream,
            "ready": segmentation["available"] and stream["available"] and weights_available,
            "message": None
            if weights_available
            else (
                "Segmentation weights are not available. Put trained weights at "
                "segmentation/runs/segment/forceps/weights/best.pt or set "
                "SEGMENTATION_WEIGHTS to another mounted container path."
            ),
        }

    @service.get("/api/calibration")
    async def get_calibration() -> dict[str, Any]:
        try:
            return service.state.calibration_store.load()
        except OrchestratorError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @service.put("/api/calibration")
    async def put_calibration(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return service.state.calibration_store.save(payload)
        except OrchestratorError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @service.post("/api/apply-calibration")
    async def apply_calibration() -> dict[str, Any]:
        """Push the saved calibration onto the live stream.

        The stream's ``PUT /inputs`` requires a full input set, so we fetch the
        current inputs (which hold the last tip/shadow detections) and overlay
        the calibration on top. This lets a calibration change (e.g. clicking a
        new trocar) take effect immediately without re-uploading a frame.
        """
        try:
            calibration = service.state.calibration_store.load()
            current = await get_stream_inputs(
                service.state.stream_url,
                service.state.stream_transport,
            )
            current_inputs = current.get("inputs") if isinstance(current, dict) else None
            if not isinstance(current_inputs, dict):
                raise OrchestratorError(
                    "stream has no current inputs to update; process a frame first "
                    "so the tip/shadow points are seeded"
                )
            stream_payload = {**current_inputs, **calibration}
            stream_result = await put_stream_inputs(
                service.state.stream_url,
                stream_payload,
                service.state.stream_transport,
            )
        except HTTPException:
            raise
        except OrchestratorError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "calibration": calibration,
            "stream_payload": stream_payload,
            "stream_result": stream_result,
        }

    @service.post("/api/manual-points")
    async def post_manual_points(payload: dict[str, Any]) -> dict[str, Any]:
        """Push hand-clicked tip/shadow points onto the stream, bypassing segmentation.

        For testing the viz before the YOLO model is trained: takes the four
        points a trained model would have produced (in the same native camera
        pixel space as the live-calibration trocars), merges them with the saved
        calibration exactly like ``/api/process`` merges segmentation output, and
        pushes the result straight to the stream.
        """
        try:
            points = validate_manual_points(payload)
            calibration = service.state.calibration_store.load()
            stream_payload = {**calibration, **points}
            stream_result = await put_stream_inputs(
                service.state.stream_url,
                stream_payload,
                service.state.stream_transport,
            )
        except HTTPException:
            raise
        except OrchestratorError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "calibration": calibration,
            "predicted_points": points,
            "stream_payload": stream_payload,
            "stream_result": stream_result,
        }

    @service.post("/api/process")
    async def process_upload(
        file: UploadFile = File(...),
        frame_time_s: float = Form(0.0),
    ) -> dict[str, Any]:
        try:
            upload = await file.read()
            frame_png = decode_upload_frame(upload, file.filename, file.content_type, frame_time_s)
            segmentation = await post_segmentation(
                service.state.segmentation_url,
                frame_png,
                service.state.segmentation_transport,
            )
            predicted_points = select_preprocessor_points(segmentation)
            calibration = service.state.calibration_store.load()
            stream_payload = {**calibration, **predicted_points}
            stream_result = await put_stream_inputs(
                service.state.stream_url,
                stream_payload,
                service.state.stream_transport,
            )
        except HTTPException:
            raise
        except OrchestratorError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "filename": file.filename,
            "frame_time_s": frame_time_s,
            "calibration": calibration,
            "predicted_points": predicted_points,
            "segmentation": segmentation,
            "stream_payload": stream_payload,
            "stream_result": stream_result,
        }

    return service


app = create_app()
