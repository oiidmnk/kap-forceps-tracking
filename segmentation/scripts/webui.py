#!/usr/bin/env python3
"""Local web UI for launching and inspecting segmentation project runs."""

from __future__ import annotations

import json
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
import yaml

from scripts.common import REPO_ROOT


WEBUI_ROOT = Path(__file__).resolve().parent.parent / "webui"
RUNS_ROOT = Path(os.getenv("SEGMENTATION_WEBUI_RUNS", REPO_ROOT / "runs" / "webui"))
UPLOADS_ROOT = RUNS_ROOT / "_uploads"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024
RUN_ID_PATTERN = re.compile(r"^[a-z0-9-]+$")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
MEDIA_SUFFIXES = IMAGE_SUFFIXES | {".bmp", ".tif", ".tiff"} | VIDEO_SUFFIXES
PREVIEW_NAMES = {
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "labels.jpg",
    "labels_correlogram.jpg",
    "train_batch0.jpg",
    "val_batch0_pred.jpg",
    "preview.png",
}
ACTIVE_RUN_STATUSES = {"queued", "running", "cancelling"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NumberRange(BaseModel):
    minimum: float
    maximum: float

    @model_validator(mode="after")
    def ordered(self) -> "NumberRange":
        if self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        return self


class TrainingParameters(BaseModel):
    model: str = Field(default="yolo11n-pose.pt", min_length=1, max_length=500)
    starting_model_run_id: str | None = Field(default=None, pattern=r"^[a-z0-9-]+$")
    config: str = Field(default="configs/forceps_pose.yaml", min_length=1, max_length=500)
    epochs: int = Field(default=100, ge=1, le=10000)
    imgsz: int = Field(default=1024, ge=64, le=8192)
    batch: int = Field(default=8, ge=-1, le=4096)
    device: str | None = Field(default=None, max_length=100)
    patience: int = Field(default=20, ge=0, le=10000)
    preprocess_preset: str | None = Field(default=None, max_length=100)
    rebuild_preprocessed: bool = False
    dataset_run_id: str | None = Field(default=None, pattern=r"^[a-z0-9-]+$")

    @field_validator(
        "device", "preprocess_preset", "dataset_run_id", "starting_model_run_id", mode="before"
    )
    @classmethod
    def empty_to_none(cls, value: Any) -> Any:
        return None if value == "" else value


class SegmentationTrainingParameters(TrainingParameters):
    model: str = Field(default="yolo11n-seg.pt", min_length=1, max_length=500)
    config: str = Field(default="configs/forceps_object_seg.yaml", min_length=1, max_length=500)


class SyntheticParameters(BaseModel):
    label_format: Literal["pose", "segment"] = "pose"
    count: int = Field(default=500, ge=1, le=1_000_000)
    width: int = Field(default=820, ge=128, le=8192)
    height: int = Field(default=920, ge=128, le=8192)
    preview: int = Field(default=12, ge=0, le=1000)
    workers: int = Field(default=0, ge=0, le=256)
    seed: int | None = None
    val_fraction: float = Field(default=0.15, ge=0, le=1)
    prefix: str = Field(default="synthetic", pattern=r"^[A-Za-z0-9_-]+$", max_length=80)
    backgrounds: list[str] = Field(default_factory=list, max_length=100)
    realism_video: str | None = Field(default=None, max_length=2000)
    realism_video_run_id: str | None = Field(default=None, pattern=r"^[a-z0-9-]+$")
    video_backgrounds: int = Field(default=6, ge=1, le=32)
    video_samples: int = Field(default=64, ge=2, le=500)
    video_degradation: bool = True
    background_rotation: float = Field(default=180, ge=0, le=360)
    image_rotations: list[float] = Field(default_factory=lambda: [90, 180, 270], min_length=1, max_length=16)
    axis_roll: float = Field(default=180, ge=0, le=360)
    shadow_axis_roll: float = Field(default=180, ge=0, le=360)
    shadow_scale: NumberRange = Field(default_factory=lambda: NumberRange(minimum=0.9, maximum=1.9))
    shadow_opacity: NumberRange = Field(default_factory=lambda: NumberRange(minimum=0.3, maximum=0.55))
    shadow_blur: NumberRange = Field(default_factory=lambda: NumberRange(minimum=3, maximum=18))
    tip_scale: NumberRange = Field(default_factory=lambda: NumberRange(minimum=0.85, maximum=1.85))
    forceps_blur: NumberRange = Field(default_factory=lambda: NumberRange(minimum=0, maximum=2.5))
    forceps_contrast: NumberRange = Field(default_factory=lambda: NumberRange(minimum=0.25, maximum=0.80))
    shadow_correlation: float = Field(default=0.75, ge=0, le=1)
    circular_mask: bool = True

    @field_validator("image_rotations")
    @classmethod
    def finite_rotations(cls, values: list[float]) -> list[float]:
        if any(not -3600 <= value <= 3600 for value in values):
            raise ValueError("image rotations must be between -3600 and 3600 degrees")
        return values

    @field_validator("backgrounds")
    @classmethod
    def safe_background_arguments(cls, values: list[str]) -> list[str]:
        if any(not value or value.startswith("-") for value in values):
            raise ValueError("background paths must be non-empty paths")
        return values

    @field_validator("realism_video", "realism_video_run_id", mode="before")
    @classmethod
    def empty_realism_video_to_none(cls, value: Any) -> Any:
        return None if value == "" else value

    @model_validator(mode="after")
    def valid_render_ranges(self) -> "SyntheticParameters":
        if self.realism_video and self.realism_video_run_id:
            raise ValueError("choose either a realism video source or a masked video run")
        for name in ("shadow_scale", "tip_scale"):
            value = getattr(self, name)
            if value.minimum <= 0:
                raise ValueError(f"{name.replace('_', ' ')} values must be positive")
        for name in ("shadow_opacity", "forceps_contrast"):
            value = getattr(self, name)
            if value.minimum < 0 or value.maximum > 1:
                raise ValueError(f"{name.replace('_', ' ')} values must be between 0 and 1")
        for name in ("shadow_blur", "forceps_blur"):
            if getattr(self, name).minimum < 0:
                raise ValueError(f"{name.replace('_', ' ')} values must be non-negative")
        return self


class DatasetSplitParameters(BaseModel):
    source_run_id: str = Field(pattern=r"^[a-z0-9-]+$")
    train_ratio: float = Field(default=0.85, gt=0, le=1)
    seed: int = 42


class PredictionParameters(BaseModel):
    model_run_id: str = Field(pattern=r"^[a-z0-9-]+$")
    segmentation_model_run_id: str | None = Field(default=None, pattern=r"^[a-z0-9-]+$")
    source: str = Field(default="", max_length=2000)
    masked_video_run_id: str | None = Field(default=None, pattern=r"^[a-z0-9-]+$")
    confidence: float = Field(default=0.25, gt=0, le=1)
    max_detections: int = Field(default=300, ge=1, le=10000)
    imgsz: int = Field(default=1024, ge=64, le=8192)
    device: str | None = Field(default=None, max_length=100)
    preprocess_preset: str | None = Field(default=None, max_length=100)
    scene_filter: bool = True
    temporal_filter: bool = True
    segmentation_confidence: float = Field(default=0.25, gt=0, le=1)
    roi_padding: float = Field(default=0.25, ge=0, le=2)

    @field_validator("device", "preprocess_preset", "segmentation_model_run_id", mode="before")
    @classmethod
    def prediction_empty_to_none(cls, value: Any) -> Any:
        return None if value == "" else value

    @model_validator(mode="after")
    def prediction_has_media(self) -> "PredictionParameters":
        if not self.source and not self.masked_video_run_id:
            raise ValueError("choose a masked video run, upload media, or provide a source path")
        return self


class VideoMaskParameters(BaseModel):
    source: str = Field(min_length=1, max_length=2000)
    circle: Literal["outer", "inner"] = "outer"
    inner_threshold: int = Field(default=0, ge=0, le=255)
    probe: int = Field(default=24, ge=4, le=500)
    radius: int = Field(default=0, ge=0, le=10000)
    track: bool = True
    smooth: int = Field(default=9, ge=0, le=1001)
    erode: int = Field(default=2, ge=0, le=1000)
    threshold: int = Field(default=12, ge=0, le=255)
    size: int = Field(default=0, ge=0, le=8192)
    crf: int = Field(default=18, ge=0, le=51)


class ClassicalRoiParameters(BaseModel):
    source: str = Field(min_length=1, max_length=2000)
    forceps_max_saturation: int = Field(default=190, ge=0, le=255)
    forceps_max_value: int = Field(default=165, ge=0, le=255)
    canny_low: int = Field(default=25, ge=0, le=255)
    canny_high: int = Field(default=70, ge=1, le=255)
    temporal_smoothing: bool = True
    temporal_alpha: float = Field(default=0.65, ge=0, le=1)
    temporal_max_gap: int = Field(default=5, ge=0, le=1000)

    @model_validator(mode="after")
    def ordered_canny_thresholds(self) -> "ClassicalRoiParameters":
        if self.canny_low >= self.canny_high:
            raise ValueError("Canny low threshold must be smaller than the high threshold")
        return self


class CreateRunRequest(BaseModel):
    kind: Literal[
        "training", "segmentation_training", "synthetic", "dataset_split", "prediction", "video_mask", "classical_roi"
    ]
    name: str = Field(default="", max_length=100)
    parameters: dict[str, Any] = Field(default_factory=dict)


def _range_args(flag: str, value: NumberRange) -> list[str]:
    return [flag, str(value.minimum), str(value.maximum)]


def build_command(
    kind: str,
    parameters: dict[str, Any],
    run_dir: Path,
    source_dataset_root: Path | None = None,
    source_model_weights: Path | None = None,
    segmentation_model_weights: Path | None = None,
    source_media: Path | None = None,
    starting_model_weights: Path | None = None,
    realism_video_source: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    if kind in {"training", "segmentation_training"}:
        parameter_model = TrainingParameters if kind == "training" else SegmentationTrainingParameters
        parsed = parameter_model.model_validate(parameters)
        params = parsed.model_dump()
        command = [
            sys.executable,
            "scripts/train.py",
            "--model",
            str(starting_model_weights) if starting_model_weights else parsed.model,
            "--config",
            str(run_dir / "input_dataset.yaml") if parsed.dataset_run_id else parsed.config,
            "--epochs",
            str(parsed.epochs),
            "--imgsz",
            str(parsed.imgsz),
            "--batch",
            str(parsed.batch),
            "--patience",
            str(parsed.patience),
            "--project",
            str(run_dir / "artifacts"),
            "--name",
            "segmentation_training" if kind == "segmentation_training" else "training",
        ]
        if parsed.device:
            command.extend(["--device", parsed.device])
        if parsed.preprocess_preset:
            command.extend(["--preprocess-preset", parsed.preprocess_preset])
        if parsed.rebuild_preprocessed:
            command.append("--rebuild-preprocessed")
        return command, params

    if kind == "prediction":
        parsed = PredictionParameters.model_validate(parameters)
        if source_model_weights is None or source_media is None:
            raise ValueError("A completed model run and readable media source are required")
        command = [
            sys.executable,
            "scripts/predict_media.py",
            "--weights",
            str(source_model_weights),
            "--source",
            str(source_media),
            "--output-dir",
            str(run_dir / "artifacts" / "predictions"),
            "--conf",
            str(parsed.confidence),
            "--max-det",
            str(parsed.max_detections),
            "--imgsz",
            str(parsed.imgsz),
        ]
        if parsed.device:
            command.extend(["--device", parsed.device])
        if parsed.preprocess_preset:
            command.extend(["--preprocess-preset", parsed.preprocess_preset])
        if not parsed.scene_filter:
            command.append("--no-scene-filter")
        if not parsed.temporal_filter:
            command.append("--no-temporal-filter")
        if segmentation_model_weights is not None:
            command.extend(
                [
                    "--segmentation-weights",
                    str(segmentation_model_weights),
                    "--segmentation-conf",
                    str(parsed.segmentation_confidence),
                    "--roi-padding",
                    str(parsed.roi_padding),
                ]
            )
        return command, parsed.model_dump()

    if kind == "video_mask":
        parsed = VideoMaskParameters.model_validate(parameters)
        if source_media is None or source_media.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError("A readable video source is required")
        command = [
            sys.executable, "scripts/mask_run_video.py", "--source", str(source_media),
            "--output-dir", str(run_dir / "artifacts" / "masking"),
            "--circle", parsed.circle, "--inner-thresh", str(parsed.inner_threshold),
            "--probe", str(parsed.probe), "--radius", str(parsed.radius),
            "--smooth", str(parsed.smooth), "--erode", str(parsed.erode),
            "--thresh", str(parsed.threshold), "--size", str(parsed.size),
            "--crf", str(parsed.crf),
        ]
        if not parsed.track:
            command.append("--no-track")
        return command, parsed.model_dump()

    if kind == "classical_roi":
        parsed = ClassicalRoiParameters.model_validate(parameters)
        if source_media is None:
            raise ValueError("A readable image or video source is required")
        command = [
            sys.executable,
            "scripts/detect_classical_roi.py",
            "--source",
            str(source_media),
            "--output-dir",
            str(run_dir / "artifacts" / "classical_roi"),
            "--forceps-max-saturation",
            str(parsed.forceps_max_saturation),
            "--forceps-max-value",
            str(parsed.forceps_max_value),
            "--canny-low",
            str(parsed.canny_low),
            "--canny-high",
            str(parsed.canny_high),
            "--temporal-alpha",
            str(parsed.temporal_alpha),
            "--temporal-max-gap",
            str(parsed.temporal_max_gap),
        ]
        if not parsed.temporal_smoothing:
            command.append("--no-temporal-smoothing")
        return command, parsed.model_dump()

    if kind == "dataset_split":
        parsed = DatasetSplitParameters.model_validate(parameters)
        if source_dataset_root is None:
            raise ValueError("A completed source dataset run is required")
        return (
            [
                sys.executable,
                "scripts/split_run_dataset.py",
                "--source-root",
                str(source_dataset_root),
                "--out-dir",
                str(run_dir / "artifacts" / "dataset"),
                "--train-ratio",
                str(parsed.train_ratio),
                "--seed",
                str(parsed.seed),
            ],
            parsed.model_dump(),
        )

    parsed = SyntheticParameters.model_validate(parameters)
    params = parsed.model_dump()
    command = [
        sys.executable,
        "scripts/generate_synthetic_dataset.py",
        "--count",
        str(parsed.count),
        "--label-format",
        parsed.label_format,
        "--out-dir",
        str(run_dir / "artifacts" / "dataset"),
        "--width",
        str(parsed.width),
        "--height",
        str(parsed.height),
        "--preview",
        str(min(parsed.preview, parsed.count)),
        "--preview-dir",
        str(run_dir / "artifacts" / "previews"),
        "--workers",
        str(parsed.workers),
        "--val-fraction",
        str(parsed.val_fraction),
        "--prefix",
        parsed.prefix,
        "--background-rotation",
        str(parsed.background_rotation),
        "--image-rotations",
        *[str(value) for value in parsed.image_rotations],
        "--axis-roll",
        str(parsed.axis_roll),
        "--shadow-axis-roll",
        str(parsed.shadow_axis_roll),
        *_range_args("--shadow-scale", parsed.shadow_scale),
        *_range_args("--shadow-opacity", parsed.shadow_opacity),
        *_range_args("--shadow-blur", parsed.shadow_blur),
        *_range_args("--tip-scale", parsed.tip_scale),
        *_range_args("--forceps-blur", parsed.forceps_blur),
        *_range_args("--forceps-contrast", parsed.forceps_contrast),
        "--shadow-correlation",
        str(parsed.shadow_correlation),
        "--circular-mask" if parsed.circular_mask else "--rectangular-view",
    ]
    if parsed.seed is not None:
        command.extend(["--seed", str(parsed.seed)])
    if parsed.backgrounds:
        command.extend(["--background", *parsed.backgrounds])
    realism_video = realism_video_source or parsed.realism_video
    if realism_video:
        command.extend(
            [
                "--realism-video",
                str(realism_video),
                "--video-backgrounds",
                str(parsed.video_backgrounds),
                "--video-samples",
                str(parsed.video_samples),
            ]
        )
    if not parsed.video_degradation:
        command.append("--no-video-degradation")
    return command, params


class RunManager:
    def __init__(self, root: Path = RUNS_ROOT) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self.lock = threading.RLock()
        self._recover_interrupted_runs()

    def _run_dir(self, run_id: str) -> Path:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise KeyError(run_id)
        return self.root / run_id

    def _metadata_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "run.json"

    def _load(self, run_id: str) -> dict[str, Any]:
        path = self._metadata_path(run_id)
        if not path.is_file():
            raise KeyError(run_id)
        return json.loads(path.read_text())

    def _save(self, metadata: dict[str, Any]) -> None:
        path = self._metadata_path(metadata["id"])
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(path)

    def _recover_interrupted_runs(self) -> None:
        for path in self.root.glob("*/run.json"):
            try:
                metadata = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("status") in {"queued", "running", "cancelling"}:
                metadata["status"] = "interrupted"
                metadata["finished_at"] = utc_now()
                metadata["message"] = "The web UI stopped before this run finished."
                self._save(metadata)

    def create(self, request: CreateRunRequest) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        run_id = f"{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        run_dir = self._run_dir(run_id)
        source_run_id = (
            request.parameters.get("dataset_run_id")
            if request.kind in {"training", "segmentation_training"}
            else request.parameters.get("source_run_id")
            if request.kind == "dataset_split"
            else None
        )
        source_metadata = None
        source_dataset_root = None
        source_model_weights = None
        segmentation_model_weights = None
        source_media = None
        starting_model_weights = None
        if source_run_id:
            source_metadata, source_dataset_root = self._resolve_dataset_run(str(source_run_id))
        model_run_id = request.parameters.get("model_run_id") if request.kind == "prediction" else None
        segmentation_model_run_id = (
            request.parameters.get("segmentation_model_run_id")
            if request.kind == "prediction"
            else None
        )
        starting_model_run_id = (
            request.parameters.get("starting_model_run_id")
            if request.kind in {"training", "segmentation_training"}
            else None
        )
        masked_video_run_id = (
            request.parameters.get("masked_video_run_id") if request.kind == "prediction" else None
        )
        realism_video_run_id = (
            request.parameters.get("realism_video_run_id") if request.kind == "synthetic" else None
        )
        realism_video_source = None
        if model_run_id:
            source_model_weights = self._resolve_model_run(str(model_run_id), "training")
        if segmentation_model_run_id:
            segmentation_model_weights = self._resolve_model_run(
                str(segmentation_model_run_id), "segmentation_training"
            )
        if starting_model_run_id:
            starting_model_weights = self._resolve_model_run(str(starting_model_run_id), request.kind)
        if masked_video_run_id:
            source_media = self._resolve_masked_video_run(str(masked_video_run_id))
        elif request.kind in {"prediction", "video_mask", "classical_roi"} and request.parameters.get("source"):
            source_media = self._resolve_media(str(request.parameters["source"]))
        if realism_video_run_id:
            realism_video_source = self._resolve_masked_video_run(str(realism_video_run_id))
        elif request.kind == "synthetic" and request.parameters.get("realism_video"):
            realism_video_source = self._resolve_video(str(request.parameters["realism_video"]))
        command, parameters = build_command(
            request.kind,
            request.parameters,
            run_dir,
            source_dataset_root=source_dataset_root,
            source_model_weights=source_model_weights,
            segmentation_model_weights=segmentation_model_weights,
            source_media=source_media,
            starting_model_weights=starting_model_weights,
            realism_video_source=realism_video_source,
        )
        run_dir.mkdir(parents=True)
        (run_dir / "artifacts").mkdir()
        if request.kind in {"training", "segmentation_training"} and source_dataset_root is not None:
            expected_format = "pose" if request.kind == "training" else "segment"
            actual_format = source_metadata.get("dataset_format", "pose")
            if actual_format != expected_format:
                raise ValueError(
                    f"{request.kind.replace('_', ' ').title()} requires a {expected_format} dataset run"
                )
            self._write_dataset_config(
                run_dir / "input_dataset.yaml",
                source_dataset_root,
                actual_format,
            )
        default_names = {
            "training": "Model training",
            "segmentation_training": "Object segmentation training",
            "synthetic": "Synthetic dataset",
            "dataset_split": "Dataset split",
            "prediction": "Media prediction",
            "video_mask": "Masked video",
            "classical_roi": "Classical ROI",
        }
        metadata = {
            "id": run_id,
            "kind": request.kind,
            "name": request.name.strip() or default_names[request.kind],
            "status": "queued",
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "message": "Waiting to start",
            "parameters": parameters,
            "command": command,
            "input_runs": [
                run
                for run in (
                    source_run_id,
                    starting_model_run_id,
                    model_run_id,
                    segmentation_model_run_id,
                    masked_video_run_id,
                    realism_video_run_id,
                )
                if run
            ],
            "dataset_format": (
                parameters.get("label_format", "pose")
                if request.kind == "synthetic"
                else source_metadata.get("dataset_format", "pose")
                if request.kind == "dataset_split" and source_metadata
                else None
            ),
        }
        self._save(metadata)
        thread = threading.Thread(target=self._execute, args=(run_id,), daemon=True)
        thread.start()
        return self.get(run_id)

    def _resolve_model_run(self, run_id: str, expected_kind: str = "training") -> Path:
        try:
            metadata = self._load(run_id)
        except KeyError as exc:
            raise ValueError(f"Model run not found: {run_id}") from exc
        if metadata.get("kind") != expected_kind:
            task = "segmentation" if expected_kind == "segmentation_training" else "pose"
            raise ValueError(f"Selected run does not produce a trained {task} model")
        if metadata.get("status") != "completed":
            raise ValueError("Training run must be completed before its model can be used")
        artifacts = self._run_dir(run_id) / "artifacts"
        weights = next(artifacts.glob("**/weights/best.pt"), None)
        if weights is None:
            weights = next(artifacts.glob("**/weights/last.pt"), None)
        if weights is None or not weights.is_file():
            raise ValueError("Selected training run has no usable weights")
        return weights.resolve()

    def _resolve_masked_video_run(self, run_id: str) -> Path:
        try:
            metadata = self._load(run_id)
        except KeyError as exc:
            raise ValueError(f"Masked video run not found: {run_id}") from exc
        if metadata.get("kind") != "video_mask":
            raise ValueError("Selected run does not produce a masked video")
        if metadata.get("status") != "completed":
            raise ValueError("Video masking must finish before its output can be used")
        video = self._run_dir(run_id) / "artifacts" / "masking" / "masked.mp4"
        if not video.is_file():
            raise ValueError("Selected masking run has no usable video")
        return video.resolve()

    @staticmethod
    def _resolve_media(value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"Media source does not exist: {value}")
        if path.suffix.lower() not in MEDIA_SUFFIXES:
            raise ValueError(f"Unsupported media type: {path.suffix or 'none'}")
        return path

    @staticmethod
    def _resolve_video(value: str) -> Path:
        path = RunManager._resolve_media(value)
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported realism video type: {path.suffix or 'none'}")
        return path

    def _resolve_dataset_run(self, run_id: str) -> tuple[dict[str, Any], Path]:
        try:
            metadata = self._load(run_id)
        except KeyError as exc:
            raise ValueError(f"Dataset run not found: {run_id}") from exc
        if metadata.get("kind") not in {"synthetic", "dataset_split"}:
            raise ValueError("Selected run does not produce a dataset")
        if metadata.get("status") != "completed":
            raise ValueError("Dataset run must be completed before it can be used")
        root = self._run_dir(run_id) / "artifacts" / "dataset"
        if not (root / "images").is_dir() or not (root / "labels").is_dir():
            raise ValueError("Selected run has no usable dataset artifacts")
        return metadata, root.resolve()

    @staticmethod
    def _write_dataset_config(path: Path, dataset_root: Path, dataset_format: str) -> None:
        if dataset_format not in {"pose", "segment"}:
            raise ValueError(f"Unsupported run dataset format: {dataset_format}")
        payload = {
            "path": str(dataset_root),
            "train": "images/train",
            "val": "images/val",
            "nc": 2,
            "names": {0: "forceps", 1: "shadow"},
        }
        if dataset_format == "pose":
            payload.update({"kpt_shape": [3, 3], "flip_idx": [1, 0, 2]})
        path.write_text(yaml.safe_dump(payload, sort_keys=False))

    def _execute(self, run_id: str) -> None:
        with self.lock:
            metadata = self._load(run_id)
            if metadata["status"] == "cancelling":
                metadata.update(status="cancelled", finished_at=utc_now(), message="Cancelled before starting")
                self._save(metadata)
                return
            metadata.update(status="running", started_at=utc_now(), message="Run in progress")
            self._save(metadata)
        log_path = self._run_dir(run_id) / "run.log"
        try:
            with log_path.open("a", buffering=1) as log:
                log.write(f"$ {' '.join(metadata['command'])}\n\n")
                process = subprocess.Popen(
                    metadata["command"],
                    cwd=REPO_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                with self.lock:
                    self.processes[run_id] = process
                exit_code = process.wait()
            with self.lock:
                current = self._load(run_id)
                was_cancelled = current["status"] == "cancelling"
                current.update(
                    status="cancelled" if was_cancelled else ("completed" if exit_code == 0 else "failed"),
                    exit_code=exit_code,
                    finished_at=utc_now(),
                    message=(
                        "Cancelled by operator"
                        if was_cancelled
                        else ("Run completed successfully" if exit_code == 0 else f"Process exited with code {exit_code}")
                    ),
                )
                self._save(current)
        except Exception as exc:  # noqa: BLE001 - persist process startup failures
            with self.lock:
                current = self._load(run_id)
                current.update(status="failed", finished_at=utc_now(), message=str(exc))
                self._save(current)
            with log_path.open("a") as log:
                log.write(f"\nWeb UI error: {exc}\n")
        finally:
            with self.lock:
                self.processes.pop(run_id, None)

    def cancel(self, run_id: str) -> dict[str, Any]:
        with self.lock:
            metadata = self._load(run_id)
            if metadata["status"] not in {"queued", "running"}:
                raise ValueError(f"Run is already {metadata['status']}")
            metadata.update(status="cancelling", message="Stopping process…")
            self._save(metadata)
            process = self.processes.get(run_id)
            if process and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        return self.get(run_id)

    def delete(self, run_id: str) -> None:
        with self.lock:
            metadata = self._load(run_id)
            if metadata["status"] in ACTIVE_RUN_STATUSES:
                raise ValueError("Stop the run before deleting it")
            shutil.rmtree(self._run_dir(run_id))

    def list(self) -> list[dict[str, Any]]:
        runs = []
        for path in self.root.glob("*/run.json"):
            try:
                metadata = json.loads(path.read_text())
                metadata["progress"] = self._progress(metadata)
                runs.append(metadata)
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(runs, key=lambda run: run["created_at"], reverse=True)

    def get(self, run_id: str) -> dict[str, Any]:
        return self._enrich(self._load(run_id), include_log=True)

    def _enrich(self, metadata: dict[str, Any], *, include_log: bool) -> dict[str, Any]:
        result = dict(metadata)
        log = None
        if include_log:
            log = self._read_log_tail(metadata["id"], 200_000)
        result["progress"] = self._progress(metadata, log=log)
        result["summary"] = self._summary(metadata)
        result["artifacts"] = self._artifacts(metadata["id"])
        if include_log:
            result["log"] = log or ""
        return result

    def _read_log_tail(self, run_id: str, max_bytes: int) -> str:
        log_path = self._run_dir(run_id) / "run.log"
        try:
            with log_path.open("rb") as log_file:
                log_file.seek(0, os.SEEK_END)
                size = log_file.tell()
                log_file.seek(max(0, size - max_bytes))
                return log_file.read().decode(errors="replace")
        except OSError:
            return ""

    def _progress(self, metadata: dict[str, Any], *, log: str | None = None) -> float | None:
        if metadata["status"] == "completed":
            return 100.0
        if log is None:
            log = self._read_log_tail(metadata["id"], 100_000)
        else:
            log = log[-100_000:]
        if not log:
            return 0.0 if metadata["status"] in {"queued", "running"} else None
        if metadata["kind"] in {"synthetic", "dataset_split"}:
            matches = re.findall(r"Generating synthetic images: (\d+)/(\d+)", log)
            if matches:
                done, total = map(int, matches[-1])
                return round(done / total * 100, 1) if total else 100.0
        elif metadata["kind"] == "prediction":
            matches = re.findall(r"Predicting video: (\d+)/(\d+) frames", log)
            if matches:
                done, total = map(int, matches[-1])
                return round(done / total * 100, 1) if total else 0.0
        elif metadata["kind"] == "video_mask":
            scan_matches = re.findall(r"\[mask\] scan (\d+)/(\d+)", log)
            frame_matches = re.findall(r"\[mask\] frame (\d+)/(\d+)", log)
            if frame_matches:
                done, total = map(int, frame_matches[-1])
                return round(50 + done / total * 50, 1) if total else 50.0
            if scan_matches:
                done, total = map(int, scan_matches[-1])
                return round(done / total * 50, 1) if total else 0.0
        elif metadata["kind"] == "classical_roi":
            matches = re.findall(r"Classical ROI video: (\d+)/(\d+) frames", log)
            if matches:
                done, total = map(int, matches[-1])
                return round(done / total * 100, 1) if total else 0.0
        else:
            epochs = int(metadata["parameters"]["epochs"])
            matches = re.findall(r"(?:^|\s)(\d+)\s*/\s*" + str(epochs) + r"(?:\s|$)", log, re.MULTILINE)
            if matches:
                return round(min(int(matches[-1]), epochs) / epochs * 100, 1)
        return 0.0 if metadata["status"] in {"queued", "running"} else None

    def _summary(self, metadata: dict[str, Any]) -> dict[str, Any]:
        artifacts = self._run_dir(metadata["id"]) / "artifacts"
        if metadata["kind"] in {"synthetic", "dataset_split"}:
            return {
                "train_images": len(list((artifacts / "dataset" / "images" / "train").glob("*.png"))),
                "val_images": len(list((artifacts / "dataset" / "images" / "val").glob("*.png"))),
                "labels": len(list((artifacts / "dataset" / "labels").glob("**/*.txt"))),
                "previews": len(list((artifacts / "previews").glob("*.png"))),
            }
        if metadata["kind"] == "prediction":
            outputs = list((artifacts / "predictions").glob("*"))
            media = [path for path in outputs if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES]
            return {
                "outputs": len(media),
                "media_type": "video" if any(path.suffix.lower() in VIDEO_SUFFIXES for path in media) else "image",
            }
        if metadata["kind"] == "video_mask":
            mask_root = artifacts / "masking"
            metadata_path = mask_root / "mask.json"
            mask_data = {}
            if metadata_path.is_file():
                try:
                    mask_data = json.loads(metadata_path.read_text())
                except (OSError, json.JSONDecodeError):
                    pass
            return {
                "frames": mask_data.get("frames", 0),
                "radius": mask_data.get("radius", "—"),
                "detections": mask_data.get("detections", 0),
                "misses": mask_data.get("detection_misses", 0),
            }
        if metadata["kind"] == "classical_roi":
            summary_path = artifacts / "classical_roi" / "summary.json"
            if summary_path.is_file():
                try:
                    return json.loads(summary_path.read_text())
                except (OSError, json.JSONDecodeError):
                    pass
            return {}
        results_csv = next(artifacts.glob("**/results.csv"), None)
        summary: dict[str, Any] = {}
        if results_csv:
            try:
                lines = [line for line in results_csv.read_text().splitlines() if line.strip()]
                if len(lines) > 1:
                    headers = [item.strip() for item in lines[0].split(",")]
                    values = [item.strip() for item in lines[-1].split(",")]
                    for header, value in zip(headers, values, strict=False):
                        if header == "epoch" or "mAP50" in header or header.endswith("loss"):
                            try:
                                summary[header] = round(float(value), 4)
                            except ValueError:
                                pass
            except OSError:
                pass
        return summary

    def _artifacts(self, run_id: str) -> list[dict[str, Any]]:
        root = self._run_dir(run_id) / "artifacts"
        if not root.exists():
            return []
        artifacts = []
        for directory, subdirectories, filenames in os.walk(root):
            subdirectories[:] = [name for name in subdirectories if name != "dataset"]
            for filename in filenames:
                path = Path(directory) / filename
                relative = path.relative_to(root)
                is_image = path.suffix.lower() in IMAGE_SUFFIXES
                is_video = path.suffix.lower() in VIDEO_SUFFIXES
                if is_image and not (
                    "previews" in relative.parts
                    or "predictions" in relative.parts
                    or "classical_roi" in relative.parts
                    or path.name in PREVIEW_NAMES
                    or path.name.startswith(("val_batch", "train_batch"))
                ):
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                artifacts.append(
                    {
                        "path": relative.as_posix(),
                        "name": path.name,
                        "size": size,
                        "type": "image" if is_image else "video" if is_video else path.suffix.lower().lstrip(".") or "file",
                        "url": f"/api/runs/{run_id}/artifacts/{relative.as_posix()}",
                    }
                )
        artifacts.sort(key=lambda item: (item["type"] not in {"image", "video"}, item["path"]))
        return artifacts[:250]

    def artifact(self, run_id: str, artifact_path: str) -> Path:
        root = (self._run_dir(run_id) / "artifacts").resolve()
        path = (root / artifact_path).resolve()
        if root not in path.parents or not path.is_file():
            raise KeyError(artifact_path)
        return path


manager = RunManager()
app = FastAPI(title="Segmentation Run Studio", version="0.1.0")
app.mount("/assets", StaticFiles(directory=WEBUI_ROOT), name="webui-assets")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEBUI_ROOT / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "project": str(REPO_ROOT), "runs_root": str(manager.root)}


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    return manager.list()


@app.post("/api/uploads", status_code=201)
async def upload_media(media: UploadFile = File(...)) -> dict[str, Any]:
    original_name = Path(media.filename or "media").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in MEDIA_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {suffix or 'none'}")
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(original_name).stem).strip("-") or "media"
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    destination = UPLOADS_ROOT / f"{uuid.uuid4().hex[:12]}-{safe_stem}{suffix}"
    size = 0
    try:
        with destination.open("wb") as output:
            while chunk := await media.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Upload exceeds the 8 GB limit")
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await media.close()
    return {"source": str(destination), "filename": original_name, "size": size}


@app.post("/api/runs", status_code=201)
def create_run(request: CreateRunRequest) -> dict[str, Any]:
    try:
        return manager.create(request)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    try:
        return manager.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc


@app.delete("/api/runs/{run_id}", status_code=204)
def delete_run(run_id: str) -> None:
    try:
        manager.delete(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict[str, Any]:
    try:
        return manager.cancel(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
def get_artifact(run_id: str, artifact_path: str) -> FileResponse:
    try:
        return FileResponse(manager.artifact(run_id, artifact_path))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found") from exc


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8010)


if __name__ == "__main__":
    main()
