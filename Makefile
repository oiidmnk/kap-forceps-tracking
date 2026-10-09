PY := .venv/bin/python

.PHONY: setup test eval up down logs live-cam live-url render background
setup:            ## local dev env (python>=3.10)
	$(or $(PYTHON),python3) -m venv .venv && .venv/bin/pip install -q -U pip && .venv/bin/pip install -q -e "services/detector[dev]"
test:
	.venv/bin/pip install -q aiohttp websockets jinja2
	.venv/bin/pytest -q services/detector
	PYTHONPATH=services .venv/bin/pytest -q services/orchestrator/tests
	PYTHONPATH=services/stream .venv/bin/pytest -q services/stream/tests
eval:             ## accuracy vs. the LabelMe annotations in data/annotated
	$(PY) scripts/evaluate.py
up:               ## full stack: detector -> orchestrator -> stream -> viz
	docker compose up -d --build
	@echo "detector http://localhost:8000  orchestrator http://localhost:8090  viz http://localhost:8080"
down:
	docker compose down
logs:
	docker compose logs -f --tail 50
live-cam:         ## local webcam/capture card (index 0) -> stack running in docker; learns the background
	VIDEO_DIR="$(CURDIR)/data/video" DETECTOR_STATE=.detector_state.json SOURCE=$${CAM:-0} BACKGROUND_PATH= ORCHESTRATOR_URL=http://localhost:8090 .venv/bin/uvicorn detector.service:app --port 8001
live-url:         ## URL=rtsp://... or http://... MJPEG
	VIDEO_DIR="$(CURDIR)/data/video" DETECTOR_STATE=.detector_state.json SOURCE=$(URL) BACKGROUND_PATH= ORCHESTRATOR_URL=http://localhost:8090 .venv/bin/uvicorn detector.service:app --port 8001
render:           ## overlay video + csv of the sample video -> artifacts/
	mkdir -p artifacts && $(PY) scripts/render_video.py data/video/forceps_lev1.mp4 artifacts/result
background:       ## rebuild data/background.png (median of a tool-free-ish video)
	$(PY) scripts/make_background.py data/video/forceps_lev1.mp4 data/background.png
