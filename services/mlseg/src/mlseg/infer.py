"""Inference wrapper: BGR square frame(s) -> 4 keypoints in that frame's pixel coords + confidence."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .data import to_tensor
from .model import KEYS, KeypointNet, decode


class Segmenter:
    def __init__(self, ckpt: str, device: str | None = None, min_score: float = 0.3):
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        ck = torch.load(ckpt, map_location="cpu")
        a = ck["args"]
        self.size, self.min_score = a["size"], min_score
        self.model = KeypointNet(a["backbone"], pretrained=False, n_out=ck["model"]["head.1.weight"].shape[0])
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.amp = self.device == "cuda"

    @torch.no_grad()
    def __call__(self, frames: list[np.ndarray]):
        """-> pts [N,4,2] (px of the input frames), score [N,4]. Frames must be square."""
        S = frames[0].shape[0]
        x = torch.stack([to_tensor(cv2.resize(f, (self.size, self.size), interpolation=cv2.INTER_AREA)) for f in frames])
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.amp):
            lg = self.model(x.to(self.device))
        xy, sc = decode(lg)
        return xy.cpu().numpy() * 2 * S / self.size, sc.cpu().numpy()
