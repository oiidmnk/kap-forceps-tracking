import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import scripts.webui as webui
from scripts.webui import CreateRunRequest, RunManager, build_command
from scripts.split_run_dataset import split_run_dataset


def test_training_command_is_scoped_to_run_artifacts(tmp_path: Path) -> None:
    command, parameters = build_command(
        "training",
        {
            "model": "yolo11n-pose.pt",
            "config": "configs/forceps_pose.yaml",
            "epochs": 5,
            "batch": 2,
            "device": "cpu",
        },
        tmp_path,
    )

    assert command[1:2] == ["scripts/train.py"]
    assert command[command.index("--project") + 1] == str(tmp_path / "artifacts")
    assert command[command.index("--name") + 1] == "training"
    assert parameters["epochs"] == 5
    assert parameters["device"] == "cpu"


def test_training_command_uses_weights_from_starting_model_run(tmp_path: Path) -> None:
    weights = tmp_path / "parent" / "weights" / "best.pt"
    command, parameters = build_command(
        "training",
        {
            "model": "yolo11n-pose.pt",
            "starting_model_run_id": "20260101-120000-abcdef",
            "config": "configs/forceps_pose.yaml",
        },
        tmp_path / "child",
        starting_model_weights=weights,
    )

    assert command[command.index("--model") + 1] == str(weights)
    assert parameters["starting_model_run_id"] == "20260101-120000-abcdef"


def test_synthetic_command_maps_ranges_and_output_paths(tmp_path: Path) -> None:
    command, parameters = build_command(
        "synthetic",
        {
            "count": 7,
            "preview": 3,
            "seed": 42,
            "image_rotations": [0, 90],
            "shadow_opacity": {"minimum": 0.4, "maximum": 0.7},
        },
        tmp_path,
    )

    assert command[1:2] == ["scripts/generate_synthetic_dataset.py"]
    assert command[command.index("--out-dir") + 1] == str(tmp_path / "artifacts" / "dataset")
    assert command[command.index("--preview-dir") + 1] == str(tmp_path / "artifacts" / "previews")
    assert command[command.index("--seed") + 1] == "42"
    opacity_index = command.index("--shadow-opacity")
    assert command[opacity_index + 1 : opacity_index + 3] == ["0.4", "0.7"]
    assert "--forceps-opacity" not in command
    assert "forceps_opacity" not in parameters
    assert parameters["count"] == 7


def test_dataset_split_command_references_source_run_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source" / "artifacts" / "dataset"
    command, parameters = build_command(
        "dataset_split",
        {"source_run_id": "20260101-120000-abcdef", "train_ratio": 0.75, "seed": 9},
        tmp_path / "child",
        source_dataset_root=source,
    )

    assert command[1] == "scripts/split_run_dataset.py"
    assert command[command.index("--source-root") + 1] == str(source)
    assert command[command.index("--out-dir") + 1] == str(tmp_path / "child" / "artifacts" / "dataset")
    assert parameters == {
        "source_run_id": "20260101-120000-abcdef",
        "train_ratio": 0.75,
        "seed": 9,
    }


def test_split_run_dataset_repartitions_pairs_without_changing_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for split in ("train", "val"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
    for index in range(10):
        split = "train" if index < 6 else "val"
        (source / "images" / split / f"sample_{index}.png").write_bytes(b"image")
        (source / "labels" / split / f"sample_{index}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    train_count, val_count = split_run_dataset(source, tmp_path / "output", 0.7, 42)

    assert (train_count, val_count) == (7, 3)
    assert len(list((tmp_path / "output" / "images" / "train").glob("*.png"))) == 7
    assert len(list((tmp_path / "output" / "labels" / "val").glob("*.txt"))) == 3
    assert len(list((source / "images").glob("**/*.png"))) == 10


def test_training_dataset_config_points_to_completed_dataset_run(tmp_path: Path) -> None:
    manager = RunManager(tmp_path / "runs")
    source_id = "20260101-120000-abcdef"
    source_dir = tmp_path / "runs" / source_id
    for relative in ("artifacts/dataset/images/train", "artifacts/dataset/images/val", "artifacts/dataset/labels/train", "artifacts/dataset/labels/val"):
        (source_dir / relative).mkdir(parents=True, exist_ok=True)
    (source_dir / "run.json").write_text(json.dumps({
        "id": source_id,
        "kind": "synthetic",
        "status": "completed",
        "dataset_format": "pose",
    }))

    metadata, dataset_root = manager._resolve_dataset_run(source_id)
    config = tmp_path / "input_dataset.yaml"
    manager._write_dataset_config(config, dataset_root, metadata["dataset_format"])

    contents = config.read_text()
    assert f"path: {dataset_root}" in contents
    assert "train: images/train" in contents
    assert "kpt_shape:" in contents


def test_training_run_records_dataset_lineage_and_uses_generated_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RunManager(tmp_path / "runs")
    source_id = "20260101-120000-abcdef"
    source_dir = tmp_path / "runs" / source_id
    for relative in ("artifacts/dataset/images/train", "artifacts/dataset/images/val", "artifacts/dataset/labels/train", "artifacts/dataset/labels/val"):
        (source_dir / relative).mkdir(parents=True, exist_ok=True)
    (source_dir / "run.json").write_text(json.dumps({
        "id": source_id,
        "kind": "synthetic",
        "status": "completed",
        "dataset_format": "pose",
    }))
    monkeypatch.setattr("scripts.webui.threading.Thread.start", lambda _self: None)

    run = manager.create(CreateRunRequest(
        kind="training",
        parameters={"dataset_run_id": source_id, "model": "yolo11n-pose.pt", "epochs": 2},
    ))

    assert run["input_runs"] == [source_id]
    config_path = tmp_path / "runs" / run["id"] / "input_dataset.yaml"
    assert config_path.is_file()
    assert run["command"][run["command"].index("--config") + 1] == str(config_path)


def test_training_run_uses_parent_model_and_records_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RunManager(tmp_path / "runs")
    parent_id = "20260101-120000-abcdef"
    parent_dir = tmp_path / "runs" / parent_id
    weights = parent_dir / "artifacts" / "training" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    (parent_dir / "run.json").write_text(json.dumps({
        "id": parent_id,
        "kind": "training",
        "status": "completed",
    }))
    monkeypatch.setattr("scripts.webui.threading.Thread.start", lambda _self: None)

    run = manager.create(CreateRunRequest(
        kind="training",
        parameters={
            "starting_model_run_id": parent_id,
            "model": "yolo11n-pose.pt",
            "config": "configs/forceps_pose.yaml",
            "epochs": 2,
        },
    ))

    assert run["input_runs"] == [parent_id]
    assert run["command"][run["command"].index("--model") + 1] == str(weights.resolve())


def test_prediction_command_uses_model_run_weights_and_media(tmp_path: Path) -> None:
    weights = tmp_path / "training" / "weights" / "best.pt"
    source = tmp_path / "frame.png"
    command, parameters = build_command(
        "prediction",
        {
            "model_run_id": "20260101-120000-abcdef",
            "source": str(source),
            "confidence": 0.6,
            "device": "cpu",
            "scene_filter": False,
            "temporal_filter": False,
        },
        tmp_path / "prediction",
        source_model_weights=weights,
        source_media=source,
    )

    assert command[1] == "scripts/predict_media.py"
    assert command[command.index("--weights") + 1] == str(weights)
    assert command[command.index("--source") + 1] == str(source)
    assert command[command.index("--conf") + 1] == "0.6"
    assert "--no-scene-filter" in command
    assert "--no-temporal-filter" in command
    assert parameters["model_run_id"] == "20260101-120000-abcdef"


def test_model_run_resolves_best_weights(tmp_path: Path) -> None:
    manager = RunManager(tmp_path / "runs")
    run_id = "20260101-120000-abcdef"
    run_dir = tmp_path / "runs" / run_id
    weights = run_dir / "artifacts" / "training" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    (run_dir / "run.json").write_text(json.dumps({
        "id": run_id, "kind": "training", "status": "completed",
    }))

    assert manager._resolve_model_run(run_id) == weights.resolve()


def test_media_upload_accepts_video_and_returns_staged_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(webui, "UPLOADS_ROOT", tmp_path / "uploads")
    client = TestClient(webui.app)

    response = client.post("/api/uploads", files={"media": ("scope.mov", b"video-data", "video/quicktime")})

    assert response.status_code == 201
    payload = response.json()
    assert payload["filename"] == "scope.mov"
    assert Path(payload["source"]).read_bytes() == b"video-data"


def test_video_mask_command_uses_single_disc_reference_parameters(tmp_path: Path) -> None:
    source = tmp_path / "scope.mov"
    command, parameters = build_command(
        "video_mask",
        {
            "source": str(source), "circle": "inner", "inner_threshold": 45,
            "track": False, "smooth": 7, "erode": 3, "size": 768,
        },
        tmp_path / "run",
        source_media=source,
    )

    assert command[1] == "scripts/mask_run_video.py"
    assert command[command.index("--circle") + 1] == "inner"
    assert command[command.index("--inner-thresh") + 1] == "45"
    assert command[command.index("--output-dir") + 1] == str(tmp_path / "run" / "artifacts" / "masking")
    assert "--no-track" in command
    assert parameters["size"] == 768


def test_prediction_can_resolve_completed_mask_run(tmp_path: Path) -> None:
    manager = RunManager(tmp_path / "runs")
    run_id = "20260101-120000-abcdef"
    run_dir = tmp_path / "runs" / run_id
    video = run_dir / "artifacts" / "masking" / "masked.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    (run_dir / "run.json").write_text(json.dumps({
        "id": run_id, "kind": "video_mask", "status": "completed",
    }))

    assert manager._resolve_masked_video_run(run_id) == video.resolve()


def test_prediction_run_records_model_and_masked_video_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RunManager(tmp_path / "runs")
    model_id = "20260101-120000-abcdef"
    model_dir = tmp_path / "runs" / model_id
    weights = model_dir / "artifacts" / "training" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    (model_dir / "run.json").write_text(json.dumps({
        "id": model_id, "kind": "training", "status": "completed",
    }))
    mask_id = "20260102-120000-fedcba"
    mask_dir = tmp_path / "runs" / mask_id
    video = mask_dir / "artifacts" / "masking" / "masked.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    (mask_dir / "run.json").write_text(json.dumps({
        "id": mask_id, "kind": "video_mask", "status": "completed",
    }))
    monkeypatch.setattr("scripts.webui.threading.Thread.start", lambda _self: None)

    run = manager.create(CreateRunRequest(kind="prediction", parameters={
        "model_run_id": model_id, "masked_video_run_id": mask_id,
    }))

    assert run["input_runs"] == [model_id, mask_id]
    assert run["command"][run["command"].index("--weights") + 1] == str(weights.resolve())
    assert run["command"][run["command"].index("--source") + 1] == str(video.resolve())


def test_invalid_synthetic_range_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="minimum must not exceed maximum"):
        build_command(
            "synthetic",
            {"shadow_blur": {"minimum": 12, "maximum": 2}},
            tmp_path,
        )


def test_run_manager_recovers_active_metadata_as_interrupted(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260101-120000-abcdef"
    run_dir.mkdir()
    metadata = {
        "id": run_dir.name,
        "kind": "training",
        "name": "old run",
        "status": "running",
        "created_at": "2026-01-01T12:00:00+00:00",
        "started_at": "2026-01-01T12:00:01+00:00",
        "finished_at": None,
        "exit_code": None,
        "message": "Run in progress",
        "parameters": {"epochs": 10},
        "command": ["python", "scripts/train.py"],
    }
    (run_dir / "run.json").write_text(json.dumps(metadata))

    manager = RunManager(tmp_path)

    recovered = manager.get(run_dir.name)
    assert recovered["status"] == "interrupted"
    assert recovered["finished_at"] is not None


def test_list_runs_does_not_scan_summaries_or_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "20260101-120000-abcdef"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({
        "id": run_id,
        "kind": "synthetic",
        "name": "large dataset",
        "status": "completed",
        "created_at": "2026-01-01T12:00:00+00:00",
    }))
    manager = RunManager(tmp_path)
    monkeypatch.setattr(manager, "_summary", lambda _metadata: pytest.fail("summary scanned"))
    monkeypatch.setattr(manager, "_artifacts", lambda _run_id: pytest.fail("artifacts scanned"))

    runs = manager.list()

    assert runs[0]["id"] == run_id
    assert runs[0]["progress"] == 100.0
    assert "summary" not in runs[0]
    assert "artifacts" not in runs[0]


def test_log_tail_reads_only_requested_bytes(tmp_path: Path) -> None:
    run_id = "20260101-120000-abcdef"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "run.log").write_bytes(b"prefix-" + b"x" * 100 + b"-suffix")
    manager = RunManager(tmp_path)

    assert manager._read_log_tail(run_id, 10) == "xxx-suffix"


def test_delete_run_removes_metadata_logs_and_artifacts(tmp_path: Path) -> None:
    run_id = "20260101-120000-abcdef"
    run_dir = tmp_path / run_id
    artifact = run_dir / "artifacts" / "results.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("result")
    (run_dir / "run.log").write_text("finished")
    (run_dir / "run.json").write_text(json.dumps({
        "id": run_id,
        "status": "completed",
    }))
    manager = RunManager(tmp_path)

    manager.delete(run_id)

    assert not run_dir.exists()


def test_delete_run_endpoint_returns_no_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "20260101-120000-abcdef"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({
        "id": run_id,
        "status": "completed",
    }))
    monkeypatch.setattr(webui, "manager", RunManager(tmp_path))

    response = TestClient(webui.app).delete(f"/api/runs/{run_id}")

    assert response.status_code == 204
    assert response.content == b""
    assert not run_dir.exists()


@pytest.mark.parametrize("status", ["queued", "running", "cancelling"])
def test_delete_run_rejects_active_statuses(tmp_path: Path, status: str) -> None:
    run_id = "20260101-120000-abcdef"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({
        "id": run_id,
        "status": status,
    }))
    manager = RunManager(tmp_path)
    metadata = json.loads((run_dir / "run.json").read_text())
    metadata["status"] = status
    (run_dir / "run.json").write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="Stop the run before deleting it"):
        manager.delete(run_id)

    assert run_dir.is_dir()


def test_artifacts_cannot_escape_run_directory(tmp_path: Path) -> None:
    run_id = "20260101-120000-abcdef"
    run_dir = tmp_path / run_id
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "run.json").write_text("{}")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    manager = RunManager(tmp_path)

    with pytest.raises(KeyError):
        manager.artifact(run_id, "../../secret.txt")
