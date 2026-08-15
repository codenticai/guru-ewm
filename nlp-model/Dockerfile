# ─────────────────────────────────────────────────────────────────────────────
# nlp-model — NanoLM English NLP Model (model #2)
# ─────────────────────────────────────────────────────────────────────────────
# Python/FastAPI service. CPU-only. Consumes hllset-next (:9090) for lattice
# ingestion and ipfs (:5001) for document ingestion + snapshots.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py english_cards.py .

HEALTHCHECK --interval=15s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:9095/health || exit 1

EXPOSE 9095

CMD ["python", "main.py"]
