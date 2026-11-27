"""Run old YOLO pose checkpoints on the labelled simulator frames.

For every model and frame it stores the most confident forceps and shadow
detection separately: {"tips": {"conf", "kpts": [left, right]}, "shadows": ...}.
Handles both label layouts of the old projects:
  - one class, 4 keypoints (tip_left, tip_right, shadow_left, shadow_right)
  - two classes (0 forceps, 1 shadow), 2 or 3 keypoints each (the first two are the tips)

    python run_yolo.py <annotated dir> <out.json> name=weights.pt [...]
"""
import glob
import json
import os
import sys

from ultralytics import YOLO

ann_dir, out = sys.argv[1], sys.argv[2]
models = dict(a.split("=", 1) for a in sys.argv[3:])
frames = sorted(glob.glob(os.path.join(ann_dir, "*.png")))
res = {}
for name, weights in models.items():
    m = YOLO(weights)
    per = {}
    for f in frames:
        r = m.predict(f, imgsz=1024, conf=0.001, verbose=False)[0]
        entry = {"tips": None, "shadows": None}
        if r.keypoints is not None and len(r.boxes):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            xy = r.keypoints.xy.cpu().numpy()
            if xy.shape[1] == 4:
                i = int(conf.argmax())
                entry["tips"] = {"conf": float(conf[i]), "kpts": xy[i, :2].tolist()}
                entry["shadows"] = {"conf": float(conf[i]), "kpts": xy[i, 2:4].tolist()}
            else:
                for c, key in ((0, "tips"), (1, "shadows")):
                    idx = [j for j in range(len(cls)) if cls[j] == c]
                    if idx:
                        j = max(idx, key=lambda j: conf[j])
                        entry[key] = {"conf": float(conf[j]), "kpts": xy[j, :2].tolist()}
        per[os.path.basename(f)] = entry
    res[name] = {"weights": weights, "kpt_shape": list(m.model.yaml.get("kpt_shape", [])), "frames": per}
    print(name, res[name]["kpt_shape"], flush=True)
json.dump(res, open(out, "w"))
