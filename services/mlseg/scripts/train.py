"""Train the keypoint net.  python scripts/train.py --data ../../data --out runs/r34 [--val-every 5]"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mlseg.data import Aug, PointDataset, load_frames
from mlseg.evaluate import predict, report
from mlseg.losses import focal
from mlseg.model import KeypointNet
from mlseg.relight import DiscRelight
from mlseg.shadowreloc import ShadowBank
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--backbone", default="resnet34")
ap.add_argument("--size", type=int, default=768)
ap.add_argument("--sigma", type=float, default=3.0)
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--repeat", type=int, default=8)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=4e-4)
ap.add_argument("--val-every", type=int, default=5, help="every n-th frame is validation; 0 = train on everything")
ap.add_argument("--val-offset", type=int, default=0)
ap.add_argument("--fold", type=int, default=-1, help="block CV: fold index (overrides --val-every)")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--pseudo", type=int, default=0, help="add pseudo-labelled video frames")
ap.add_argument("--real-weight", type=int, default=6, help="repeat factor of real frames vs pseudo")
ap.add_argument("--relight", default="", help="empty-retina background image -> disc relight augmentation")
ap.add_argument("--reloc", type=float, default=0.0, help="prob. of the shadow relocation augmentation (needs --relight bg)")
ap.add_argument("--reloc-bg", nargs="*", default=[], help="further empty-retina backgrounds (other videos) for shadow relocation")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

torch.manual_seed(a.seed)
os.makedirs(a.out, exist_ok=True)
real = load_frames(a.data, "real")
pseudo = load_frames(a.data, "pseudo") if a.pseudo else []
frames = real
idx = np.arange(len(frames))
val_m = (idx % a.val_every == a.val_offset) if a.val_every else np.zeros(len(idx), bool)
if a.fold >= 0:                       # contiguous block = the harder, honest split (neighbouring frames look alike)
    b = np.array_split(idx, a.folds)[a.fold]
    val_m = np.isin(idx, b)
train = [f for f, v in zip(frames, val_m) if not v]
train = train * a.real_weight + pseudo
val = [f for f, v in zip(frames, val_m) if v]
print(f"train {len(train)} (real {len(train) - len(pseudo)} incl. repeats, pseudo {len(pseudo)}) val {len(val)}", flush=True)

dl = DataLoader(PointDataset(train, Aug(a.size, 2, a.sigma, True, DiscRelight(cv2.imread(a.relight)) if a.relight else None, 0.5,
                                       ShadowBank(train, [cv2.imread(b) for b in [a.relight] + a.reloc_bg]) if a.reloc > 0 else None, a.reloc), max(1, a.repeat // a.real_weight), a.seed), batch_size=a.bs, shuffle=True,
                num_workers=8, drop_last=True, persistent_workers=True)
model = KeypointNet(a.backbone).cuda().to(memory_format=torch.channels_last)
enc = list(model.enc.parameters())
ids = {id(p) for p in enc}
opt = torch.optim.AdamW([{"params": enc, "lr": a.lr / 4}, {"params": [p for p in model.parameters() if id(p) not in ids], "lr": a.lr}],
                        weight_decay=1e-4)
total = a.epochs * len(dl)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr / 4, a.lr], total_steps=total, pct_start=0.1)
log, best = [], 1e9
t0 = time.time()
for ep in range(a.epochs):
    model.train()
    tl = 0
    for x, hm, _, mk in dl:
        x, hm = x.cuda(non_blocking=True).to(memory_format=torch.channels_last), hm.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x)
        loss = focal(lg, hm, mk.cuda())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        tl += loss.item()
    row = {"ep": ep, "loss": tl / len(dl), "t": round(time.time() - t0)}
    if val and (ep % 5 == 4 or ep == a.epochs - 1):
        pr, sc = predict(model, val, a.size)
        s, _ = report(val, pr, sc)
        row.update({"val_" + k: v for k, v in s.items()})
    log.append(row)
    print(json.dumps(row), flush=True)
torch.save({"model": model.state_dict(), "args": vars(a)}, os.path.join(a.out, "last.pt"))
json.dump(log, open(os.path.join(a.out, "log.json"), "w"))
