import torch


def focal(logits, gt, mask=None, a=2.0, b=4.0):
    """CornerNet penalty-reduced focal loss on gaussian heatmaps; mask [B,K] drops ignored keypoint channels."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = gt.ge(0.999).float()
    neg = 1 - pos
    m = 1.0 if mask is None else mask[:, :, None, None]
    lp = torch.log(p) * (1 - p) ** a * pos * m
    ln = torch.log(1 - p) * p ** a * (1 - gt) ** b * neg * m
    n = pos.sum().clamp_min(1)
    return -(lp.sum() + ln.sum()) / n
