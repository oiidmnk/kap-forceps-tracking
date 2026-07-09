# kap-forceps-tracking

Forceps tracking workspace with:

- `preprocessing/`: websocket preprocessing service for tool positions.
- `orchestrator/`: operator Web UI/API that uploads a frame, calls segmentation,
  and updates the stream inputs.
- `microforceps-viz/`: React/Three.js visualization frontend.
- `segmentation/`: YOLO instance segmentation pipeline for forceps tips and shadows.

## Segmentation

The segmentation tooling lives as a self-contained Python project under
`segmentation/`.

```bash
cd segmentation
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/check_labels.py
python scripts/train.py
```

Place training data under:

```text
segmentation/data/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

See `segmentation/README.md` for the full workflow, preprocessing presets,
prediction, benchmarking, and remote GPU training.

## Docker services

The compose stack includes:

- `stream`: websocket preprocessing feed on `ws://localhost:8765/ws`, with
  preprocessor input API on `http://localhost:8765/inputs`.
- `orchestrator`: upload/calibration UI on `http://localhost:8090`.
- `viz`: React/Three.js dashboard on `http://localhost:8080`.
- `segmentation`: HTTP segmentation inference API on `http://localhost:8000`.

Send an image to the segmentation service:

```bash
docker compose up segmentation
curl -F image=@segmentation/data/images/val/frame_001.png http://localhost:8000/segment
```

Run the operator workflow:

```bash
docker compose up stream segmentation orchestrator viz
```

Open `http://localhost:8090` to upload an image or video frame and edit the
static calibration values. Successful uploads update the live stream consumed by
the dashboard at `http://localhost:8080`.

The segmentation service needs trained weights mounted in the container. By
default it expects:

```text
segmentation/runs/segment/forceps/weights/best.pt
```

If your weights live somewhere else, copy or symlink them there, or start Compose
with `SEGMENTATION_WEIGHTS` set to a path that exists inside the segmentation
container.
