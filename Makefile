PY := .venv/bin/python

.PHONY: setup test eval up down logs live-cam live-url render background ml-live ml-cam cams ml-down classic
CAM ?= 0
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
ml-live:          ## learned segmenter, native (uses the Apple GPU via MPS) on :8002; SOURCE=file|url|cam idx, PUSH=1 to feed the orchestrator
	test -d .venv-ml || (/opt/homebrew/bin/python3.14 -m venv .venv-ml && .venv-ml/bin/pip install -q torch torchvision timm opencv-python-headless numpy fastapi httpx "uvicorn[standard]" python-multipart)
	VIDEO_DIR="$(CURDIR)/data/video" PYTHONPATH=services/mlseg/src:services/detector/src MLSEG_CKPT=services/mlseg/weights/last.pt DETECTOR_STATE=.segmenter_state.json \
	SOURCE=$${SOURCE:-data/video/forceps_lev1.mp4} BACKGROUND_PATH= ORCHESTRATOR_URL=$${PUSH:+http://localhost:8090} \
	.venv-ml/bin/uvicorn mlseg.service:app --port 8002
cams:             ## list capture devices (index + resolution) - the ML venv is created by ml-live/ml-cam
	test -d .venv-ml || $(MAKE) .venv-ml
	.venv-ml/bin/python scripts/list_cams.py
.venv-ml:
	/opt/homebrew/bin/python3.14 -m venv .venv-ml && .venv-ml/bin/pip install -q torch torchvision timm opencv-python-headless numpy fastapi httpx "uvicorn[standard]" python-multipart
ml-cam:           ## ONE COMMAND: capture card (CAM=<index>, default 0) -> learned segmenter (Apple GPU) -> orchestrator -> viz.  KEEP=1 keeps the saved eye circle
	@docker info >/dev/null 2>&1 || { echo "Docker/OrbStack is not running - start it first (orb start)"; exit 1; }
	docker compose stop detector
	docker compose up -d --no-deps orchestrator stream viz
	$(if $(KEEP),,rm -f .segmenter_state.json)
	@until curl -sf localhost:8090/api/status >/dev/null; do sleep 1; done
	@curl -s -X PUT localhost:8090/api/source/detector-url -H 'content-type: application/json' -d '{"url":"http://host.docker.internal:8002"}' >/dev/null && echo "orchestrator -> segmenter on :8002"
	@echo "live view http://localhost:8002   calibration + eye circle http://localhost:8090   viz http://localhost:8080   (Ctrl+C stops the segmenter)"
	SOURCE=$(CAM) PUSH=1 $(MAKE) ml-live
ml-down:          ## stop segmenter + dockerised orchestrator/stream/viz
	-lsof -ti tcp:8002 | xargs kill
	docker compose stop orchestrator stream viz
classic:          ## back to the classical detector (docker) as the stack's detector
	-lsof -ti tcp:8002 | xargs kill
	docker compose up -d detector orchestrator stream viz
	@until curl -sf localhost:8090/api/status >/dev/null; do sleep 1; done
	curl -s -X PUT localhost:8090/api/source/detector-url -H 'content-type: application/json' -d '{"url":"http://detector:8000"}'
render:           ## overlay video + csv of the sample video -> artifacts/
	mkdir -p artifacts && $(PY) scripts/render_video.py data/video/forceps_lev1.mp4 artifacts/result
background:       ## rebuild data/background.png (median of a tool-free-ish video)
	$(PY) scripts/make_background.py data/video/forceps_lev1.mp4 data/background.png
