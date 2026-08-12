# ─────────────────────────────────────────────────────────────────────────────
# medical-diagnostic — Clinical Diagnosis Service (EWM backend)
# ─────────────────────────────────────────────────────────────────────────────
# A separate EWM backend that interprets OCR'd clinical text and/or medical
# images to produce a structured diagnostic report. Designed with a pluggable
# engine so real medical models (clinical LLM / medical VLM) can be dropped in.
#
# NOTE: The default engine is a SIMULATION for architecture validation.
# It is NOT a medical device and must not be used for real diagnosis.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ecg_cards.py .

HEALTHCHECK --interval=15s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:9094/health || exit 1

EXPOSE 9094

CMD ["python", "main.py"]
