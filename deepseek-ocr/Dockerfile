# ─────────────────────────────────────────────────────────────────────────────
# deepseek-ocr — GPU-Accelerated OCR Model Service
# ─────────────────────────────────────────────────────────────────────────────
# Base: CPU-only by default. For GPU support, switch to nvidia/cuda base image
# and uncomment the deploy.resources section in docker-compose.yml.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

HEALTHCHECK --interval=15s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9093/health || exit 1

EXPOSE 9093

CMD ["python", "main.py"]
