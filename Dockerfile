# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /app

COPY requirements.txt .
# CPU-only torch wheel first (the default wheel bundles CUDA and is ~2.5GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY frontend ./frontend
COPY data/fixtures ./data/fixtures
COPY data/eval ./data/eval
COPY pytest.ini ./
COPY tests ./tests

# model weights cache + ingested data live in named volumes (see compose)
VOLUME ["/app/data/cache", "/app/data/vectorstore", "/home/user/.cache/huggingface"]

EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
