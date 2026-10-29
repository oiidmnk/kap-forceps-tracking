# build from the repo root:  docker build -f services/mlseg/Dockerfile .
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY services/detector/requirements.txt detector-requirements.txt
RUN pip install --no-cache-dir -r detector-requirements.txt timm
COPY services/detector/src /app/detector-src
COPY services/mlseg/src /app/src
ENV PYTHONPATH=/app/src:/app/detector-src MLSEG_CKPT=/weights/last.pt
EXPOSE 8000
CMD ["uvicorn", "mlseg.service:app", "--host", "0.0.0.0", "--port", "8000"]
