#!/usr/bin/env python3
"""HTTP API for single-image YOLO segmentation inference."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from ultralytics import YOLO

from scripts.predict import DEFAULT_WEIGHTS, serialize_segmentation_result
from scripts.preprocessing import (
    DEFAULT_PREPROCESS_CONFIG,
    CropTransform,
    apply_preprocessing,
    load_preprocess_preset,
)


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


WEIGHTS = _env_path("SEGMENTATION_WEIGHTS", DEFAULT_WEIGHTS)
CONFIDENCE = _env_float("SEGMENTATION_CONF", 0.25)
IMAGE_SIZE = _env_int("SEGMENTATION_IMGSZ", 1024)
DEVICE = os.getenv("SEGMENTATION_DEVICE") or None
PREPROCESS_CONFIG = _env_path("SEGMENTATION_PREPROCESS_CONFIG", DEFAULT_PREPROCESS_CONFIG)
PREPROCESS_PRESET = os.getenv("SEGMENTATION_PREPROCESS_PRESET") or None

app = FastAPI(title="Forceps Segmentation API", version="0.1.0")


@lru_cache(maxsize=1)
def model() -> YOLO:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"segmentation weights not found: {WEIGHTS}")
    return YOLO(str(WEIGHTS))


@lru_cache(maxsize=1)
def preprocess_preset() -> dict[str, Any] | None:
    if not PREPROCESS_PRESET:
        return None
    return load_preprocess_preset(PREPROCESS_PRESET, PREPROCESS_CONFIG)


def decode_image(payload: bytes) -> np.ndarray:
    array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("uploaded file is not a readable image")
    return image


def predict_image(image: np.ndarray) -> dict[str, Any]:
    source_height, source_width = image.shape[:2]
    transform = CropTransform(source_width, source_height, 0, 0, source_width, source_height)
    model_input = image
    preset = preprocess_preset()
    if preset is not None:
        preprocessed = apply_preprocessing(image, preset)
        model_input = preprocessed.image
        transform = preprocessed.transform

    results = model().predict(
        source=model_input,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        save=False,
        verbose=False,
    )
    if not results:
        raise RuntimeError("model returned no prediction results")

    payload = serialize_segmentation_result(results[0], transform)
    payload["model"] = {
        "weights": str(WEIGHTS),
        "confidence": CONFIDENCE,
        "imgsz": IMAGE_SIZE,
        "device": DEVICE,
        "preprocess_preset": PREPROCESS_PRESET,
    }
    return payload


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "weights": str(WEIGHTS),
        "weights_available": WEIGHTS.exists(),
        "preprocess_preset": PREPROCESS_PRESET,
    }


@app.post("/segment")
async def segment(image: Annotated[UploadFile, File(...)]) -> dict[str, Any]:
    try:
        decoded = decode_image(await image.read())
        payload = predict_image(decoded)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - translate model failures into API errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    payload["filename"] = image.filename
    return payload


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
