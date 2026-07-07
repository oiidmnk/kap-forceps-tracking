from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_CONFIG = REPO_ROOT / "configs" / "forceps_seg.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_data_config(config_path: Path | None = None) -> dict:
    path = config_path or DEFAULT_DATA_CONFIG
    with path.open() as f:
        return yaml.safe_load(f)


def data_root(config: dict, config_path: Path | None = None) -> Path:
    base = config_path.parent if config_path else DEFAULT_DATA_CONFIG.parent
    data_path = Path(config["path"])
    if data_path.is_absolute():
        return data_path
    return (base / data_path).resolve()


def split_dirs(data_dir: Path, config: dict, split: str) -> tuple[Path, Path]:
    images_dir = data_dir / config[split]
    labels_dir = data_dir / "labels" / split
    return images_dir, labels_dir


def list_images(images_dir: Path) -> dict[str, Path]:
    if not images_dir.is_dir():
        return {}
    return {
        p.stem: p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }


def list_labels(labels_dir: Path) -> dict[str, Path]:
    if not labels_dir.is_dir():
        return {}
    return {p.stem: p for p in labels_dir.iterdir() if p.is_file() and p.suffix == ".txt"}
