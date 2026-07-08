FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY scripts ./scripts

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install --no-cache-dir -e .

ENV YOLO_CONFIG_DIR=/tmp

EXPOSE 8000

CMD ["uvicorn", "scripts.service:app", "--host", "0.0.0.0", "--port", "8000"]
