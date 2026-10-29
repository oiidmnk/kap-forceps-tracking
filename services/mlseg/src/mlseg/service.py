"""Same HTTP API / live worker / viewer as the classical detector, backed by the learned keypoint model."""
from __future__ import annotations

import os

from detector.config import Settings
from detector.service import create_app as create_detector_app

from .detector import MLDetector


def create_app(ckpt: str | None = None, cfg: Settings | None = None):
    cfg = cfg or Settings(background_path="")         # no reference background needed
    det = MLDetector(ckpt or os.environ.get("MLSEG_CKPT", "/weights/last.pt"), cfg,
                     device=os.environ.get("MLSEG_DEVICE") or None,
                     min_score=float(os.environ.get("MLSEG_MIN_SCORE") or 0.3),
                     shadow_min_score=float(os.environ.get("MLSEG_SHADOW_MIN_SCORE") or 0.1))
    app = create_detector_app(cfg, det)
    app.title = "Forceps Segmenter (learned)"
    return app


app = create_app() if os.environ.get("MLSEG_CKPT") else None
