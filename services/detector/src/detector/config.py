"""Runtime settings, all overridable through environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _f(name: str, default: float) -> float:
    return float(os.getenv(name) or default)


def _s(name: str, default: str = "") -> str:
    return os.getenv(name) or default


@dataclass(frozen=True)
class Settings:
    # --- detection ---
    work_size: int = 1080                 # frames are center-cropped/resized to this square
    disc_radius: float = 0.435            # analysis disc radius, fraction of work_size
    disc_eye: float = 0.94                # analysis disc as fraction of the detected eye radius
    mask_disc_eye: float = 0.98
    mask_disc_radius: float = 0.458       # mask disc radius, fraction of work_size
    shadow_ratio: float = field(default_factory=lambda: _f("SHADOW_RATIO", 0.88))
    fallback_rel: float = field(default_factory=lambda: _f("FALLBACK_REL", 0.90))
    forceps_angle: float = field(default_factory=lambda: _f("FORCEPS_ANGLE", 0.0))   # deg; 0 disables the chroma cue
    forceps_sat_max: int = 150
    forceps_val_max: int = 135
    background_path: str = field(default_factory=lambda: os.environ.get("BACKGROUND_PATH", "/data/background.png"))  # empty = learn from live feed
    state_path: str = field(default_factory=lambda: os.environ.get("DETECTOR_STATE", "/state/config.json"))
    # --- live source ---
    source: str = field(default_factory=lambda: _s("SOURCE"))          # file | rtsp/http url | camera index
    source_loop: bool = field(default_factory=lambda: _s("SOURCE_LOOP", "1") != "0")
    source_realtime: bool = field(default_factory=lambda: _s("SOURCE_REALTIME", "1") != "0")
    bootstrap_seconds: float = field(default_factory=lambda: _f("BOOTSTRAP_SECONDS", 8))
    jpeg_quality: int = field(default_factory=lambda: int(_f("JPEG_QUALITY", 80)))
    # --- orchestrator push ---
    orchestrator_url: str = field(default_factory=lambda: _s("ORCHESTRATOR_URL"))
    push_hz: float = field(default_factory=lambda: _f("PUSH_HZ", 15))
    smoothing: float = field(default_factory=lambda: _f("SMOOTHING", 0.5))   # EMA weight of the new sample, 1 = off
