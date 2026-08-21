# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project

Guru-EWM is a self-hosted, CPU-first platform combining NLP question answering, OCR document analysis, and clinical-report interpretation behind a single chat UI. It uses an HLLSet (HyperLogLog-set) lattice for content-addressed knowledge retrieval and IPFS for durable storage. No external LLM and no GPU are required.

## Architecture (Docker services)

| Service | Port | Tech | Purpose |
|---|---|---|---|
| `ewm-ui` | 8080 | NiceGUI | Chat UI (NLP / OCR / Diagnose modes) |
| `ewm-gateway` | 8001 | FastAPI | Routing, aggregated health, service catalog, IPFS proxy |
| `nlp-model` | 9095 | FastAPI | NanoLM English Q&A + document ingestion |
| `medical-diagnostic` | 9094 | FastAPI | Text lattice matching, knee-MRI fingerprint, BiomedCLIP zero-shot |
| `deepseek-ocr` | 9093 | FastAPI + Tesseract | OCR text extraction |
| `ipfs` | 5001 / 8081 | Kubo | Content-addressed storage |

Optional (sources live outside this repo; enable via `docker-compose.optional.yml`):

- `hllset-next` (9090, Rust) — HLLSet algebra API (ingest + inclusion)
- `hllset-cortex` (9092, Flask) — HLLSet semantic compressor

## Repository layout

- `ewm-gateway/` — central API gateway; forwards to the services above.
- `nlp-model/` — retrieval-based English Q&A; `english_cards.py` is the seed corpus.
- `medical-diagnostic/` — clinical card corpora (`*_cards.py`), `vision.py` (BiomedCLIP), `knee_mri.py` (synthetic fingerprint classifier), `knee_cnn.py` (CPU CNN).
- `deepseek-ocr/` — Tesseract CPU OCR; `deepseek_model.py` is the optional DeepSeek-OCR GPU path; `model/` is vendored upstream source.
- `docker-compose.yml` — core services; `docker-compose.optional.yml` — optional HLLSet services.
- `.env.example` — all configuration (copy to `.env`).

`invoice-extractor/`, `docs/`, `tests/`, `scripts/`, and `output/` are developer-local (gitignored — not part of the published repo).

## Build & run

```bash
cp .env.example .env
docker compose build
docker compose up -d
docker compose logs -f
```

- UI: http://localhost:8080
- Gateway: http://localhost:8001 (see `/health` and `/services`)

## Conventions & constraints

- CPU-only by default — no GPU required; OCR uses Tesseract.
- Deterministic retrieval: no embeddings/vector DB and no external LLM calls.
- The DeepSeek-OCR model is an optional GPU-only path (`DEEPSEEK_OCR_GPU_ENABLED`); the default engine is Tesseract.
- The HLLSet services are optional — retrieval and diagnosis run locally in Python, and durability comes from IPFS snapshots.
- Keep configuration in `.env` (template: `.env.example`); don't commit `.env`.
