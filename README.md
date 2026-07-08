# kap-forceps-tracking

Forceps tracking workspace with:

- `preprocessing/`: websocket preprocessing service for tool positions.
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

- `stream`: websocket preprocessing feed on `ws://localhost:8765`.
- `viz`: React/Three.js dashboard on `http://localhost:8080`.
- `segmentation`: HTTP segmentation inference API on `http://localhost:8000`.

Send an image to the segmentation service:

```bash
docker compose up segmentation
curl -F image=@segmentation/data/images/val/frame_001.png http://localhost:8000/segment
```
