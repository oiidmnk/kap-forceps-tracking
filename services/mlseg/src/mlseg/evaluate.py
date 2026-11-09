import numpy as np
import torch

from .data import Aug, to_tensor
from .model import ROWS, decode_rows

ROD_PAIRS = ((6, 7), (8, 9))       # unordered end-point pairs: rod, rod shadow


@torch.no_grad()
def predict(model, frames, size=768, stride=2, device="cuda", batch=16):
    """-> pts [N,R,2] in original px, score [N,R] (one row per label row; rod rows are the channel's two peaks)."""
    model.eval()
    aug = Aug(size, stride, train=False)
    P, S = [], []
    for i in range(0, len(frames), batch):
        x = torch.stack([to_tensor(aug(f[1], f[2], None)[0]) for f in frames[i:i + batch]]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            lg = model(x)
        xy, sc = decode_rows(lg)
        P.append(xy.cpu().numpy() * stride * frames[0][1].shape[0] / size)
        S.append(sc.cpu().numpy())
    return np.concatenate(P), np.concatenate(S)


def match_pairs(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Reorder the unordered rod peaks of `pred` [N,R,2] to the closer assignment to `gt`."""
    pred = pred.copy()
    for a, b in ROD_PAIRS:
        if pred.shape[1] <= b:
            continue
        keep = np.linalg.norm(pred[:, a] - gt[:, a], axis=-1) + np.linalg.norm(pred[:, b] - gt[:, b], axis=-1)
        swap = np.linalg.norm(pred[:, a] - gt[:, b], axis=-1) + np.linalg.norm(pred[:, b] - gt[:, a], axis=-1)
        s = swap < keep
        pred[s, a], pred[s, b] = pred[s, b].copy(), pred[s, a].copy()
    return pred


def report(frames, pred, score=None):
    """Errors over keypoints with state==1 only (tips are real labels in 'real' frames; junctions are detector pseudo-labels)."""
    gt = np.stack([f[2] for f in frames]); st = np.stack([f[3] for f in frames]) == 1
    pred = match_pairs(pred, gt)
    R = min(pred.shape[1], gt.shape[1])
    err = np.linalg.norm(pred[:, :R] - gt[:, :R], axis=-1)
    s = {}
    tips = st[:, :4]
    if tips.any():
        s["mean_px"] = float(err[:, :4][tips].mean()); s["median_px"] = float(np.median(err[:, :4][tips]))
        for t in (5, 10, 20):
            s[f"pck@{t}"] = float((err[:, :4][tips] < t).mean())
    for i, k in enumerate(ROWS[:R]):
        if st[:, i].any():
            s[k] = float(err[:, i][st[:, i]].mean())
    if score is not None:     # absent keypoints should score low
        absent = np.stack([f[3] for f in frames])[:, :R] == 0
        if absent.any():
            s["absent_max_score"] = float(score[:, :R][absent].max())
    return s, err
