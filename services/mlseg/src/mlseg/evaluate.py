import numpy as np
import torch

from .data import Aug, to_tensor
from .model import KEYS, decode


@torch.no_grad()
def predict(model, frames, size=768, stride=2, device="cuda"):
    """-> pts [N,K,2] in original px, score [N,K]."""
    model.eval()
    aug = Aug(size, stride, train=False)
    x = torch.stack([to_tensor(aug(f[1], f[2], None)[0]) for f in frames]).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model(x)
    xy, sc = decode(lg)
    return xy.cpu().numpy() * stride * frames[0][1].shape[0] / size, sc.cpu().numpy()


def report(frames, pred, score=None):
    """Errors over keypoints with state==1 only (tips are real labels in 'real' frames; junctions are detector pseudo-labels)."""
    gt = np.stack([f[2] for f in frames]); st = np.stack([f[3] for f in frames]) == 1
    err = np.linalg.norm(pred - gt, axis=-1)
    s = {}
    tips = st[:, :4]
    s["mean_px"] = float(err[:, :4][tips].mean()); s["median_px"] = float(np.median(err[:, :4][tips]))
    for t in (5, 10, 20):
        s[f"pck@{t}"] = float((err[:, :4][tips] < t).mean())
    for i, k in enumerate(KEYS):
        if st[:, i].any():
            s[k] = float(err[:, i][st[:, i]].mean())
    if score is not None:     # absent keypoints should score low
        absent = np.stack([f[3] for f in frames]) == 0
        if absent.any():
            s["absent_max_score"] = float(score[absent].max())
    return s, err
