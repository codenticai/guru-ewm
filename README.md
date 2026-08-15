# Guru-EWM — Emerging World Models Platform

A self-hosted, CPU-first platform that combines NLP question answering, OCR document analysis, and clinical-report interpretation behind a single chat UI. It uses an HLLSet (HyperLogLog-set) lattice for content-addressed knowledge retrieval and IPFS for durable storage — no external LLM, no GPU required.

> ⚠️ **Not a medical device.** The diagnostic features are a research demo. They do not replace clinician review.

## Highlights

- **NanoLM NLP** — deterministic retrieval-based Q&A over a 55k+ card knowledge corpus (keyword index + IDF-weighted cosine + union-of-occurrences replies with pagination).
- **Clinical text reports** — ECG / X-ray / CT / knee-MRI / lab reports matched against curated finding cards with numeric reference ranges.
- **OCR** — Tesseract-based document/image text extraction (ECG band crop and full-page modes).
- **Vision** — BiomedCLIP zero-shot image classification and a synthetic knee-MRI fingerprint classifier (CPU-only).
- **IPFS** — content-addressed storage for the knowledge snapshot and ingested documents.
- **HLLSet lattice** — ingestion and inclusion queries via `hllset-next` (Rust).

## Architecture

```
                ┌──────────────────────────────┐
   browser ───▶ │  ewm-ui (NiceGUI :8080)      │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │  ewm-gateway (FastAPI :8000) │
                └──┬──────┬──────┬──────┬──────┘
                   │      │      │      │
        ┌──────────▼┐ ┌───▼────┐ ┌▼──────────┐ ┌▼─────────────────┐
        │ nlp-model │ │ deep-  │ │ medical-  │ │ hllset-next      │
        │ (:9095)   │ │ seek   │ │ diagnostic│ │ (Rust :9090)     │
        │           │ │ OCR    │ │ (:9094)   │ │                  │
        │           │ │ (:9093)│ │           │ │                  │
        └─────┬─────┘ └───┬────┘ └─────┬─────┘ └────────┬─────────┘
              │           │            │                │
              └───────────┴─────┬──────┴────────────────┘
                                ▼
                  ┌──────────────────────────┐
                  │ hllset-cortex (Flask :9092)│
                  └───────────┬──────────────┘
                              ▼
                  ┌──────────────────────────┐
                  │ ipfs (kubo :5001)         │
                  └──────────────────────────┘
```

## Services

| Service | Port | Technology | Purpose |
|---|---|---|---|
| `ewm-ui` | 8080 | NiceGUI | Chat UI (NLP / OCR / Diagnose modes) |
| `ewm-gateway` | 8000 | FastAPI | Routing, service catalog, IPFS proxy |
| `nlp-model` | 9095 | FastAPI | NanoLM English Q&A + document ingestion |
| `medical-diagnostic` | 9094 | FastAPI | Text lattice + knee-MRI + BiomedCLIP zero-shot |
| `deepseek-ocr` | 9093 | FastAPI + Tesseract | OCR text extraction |
| `hllset-next` | 9090 | Rust (axum) | HLLSet algebra API (ingest + inclusion) |
| `hllset-cortex` | 9092 | Flask | HLLSet semantic compressor |
| `ipfs` | 5001 / 8081 | Kubo | Content-addressed storage |

## Quick start

### Prerequisites

- Docker Engine + Docker Compose plugin (Linux) or Docker Desktop (Windows/macOS).
- The `hllset-next` and `hllset-cortex` sources live outside this repository. Copy them into `./hllset-next` and `./hllset-cortex` (the compose defaults), or point the build contexts at your local copies via `.env`:

```bash
cp .env.example .env
# edit .env:
#   HLLSET_NEXT_CONTEXT=/path/to/hllset-next
#   HLLSET_CORTEX_CONTEXT=/path/to/hllset_cortex
```

### Build & run

```bash
docker compose build
docker compose up -d
docker compose logs -f
```

Open <http://localhost:8080>. The API gateway is at <http://localhost:8000>.

### Optional: UI resource badge

By default the header does **not** mount the Docker socket (security). To show live CPU/RAM usage, set in `.env`:

```bash
DOCKER_SOCK_MOUNT=/var/run/docker.sock   # Linux
# Docker Desktop Windows: DOCKER_SOCK_MOUNT=//var/run/docker.sock
```

then `docker compose up -d ewm-ui`.

## Configuration

All settings live in `.env` (see `.env.example`): host ports, internal service URLs, IPFS profile, and optional GPU passthrough for OCR.

## Documentation

- [Architecture & implementation plan](docs/ARCHITECTURE.md)
- [NanoLM specification](docs/NANOLM_SPECIFICATION.md)
- [NanoLM NLP model specification](docs/NANOLM_NLP_MODEL_SPECIFICATION.md)

## Testing

```bash
pip install -r tests/requirements-test.txt
pytest tests/ -v                       # full suite (requires a running stack)
pytest tests/test_nlp_quality.py -v    # NLP reply quality (needs stack up)
pytest tests/test_keyword_query.py -v  # keyword non-fallback (needs stack up)
pytest tests/test_ui_gui.py --base-url http://localhost:8080
```

## Data & model sources

| Asset | Source | License note |
|---|---|---|
| NLP corpus (trivia/capitals/currencies/inventors) | OpenTriviaQA + Wikidata | Wikidata = CC0. **Verify OpenTriviaQA redistribution terms before publishing the bundled corpus.** |
| `nlp_keywords.csv` | derived from the NLP corpus | follows the corpus license |
| `knee_cnn.pt` | trained on synthetic MRI data in this repo | project license |
| BiomedCLIP weights | downloaded at runtime from Hugging Face | subject to the model's HF license |

## Disclaimer

Clinical/diagnostic outputs are for research and demonstration only. They are not a substitute for professional medical advice, diagnosis, or treatment.

## License

[MIT](LICENSE) © 2026 Guru-EWM contributors.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md). Report vulnerabilities privately per [SECURITY.md](SECURITY.md).
