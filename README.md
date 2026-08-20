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
                │  ewm-gateway (FastAPI :8001) │
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
| `ewm-gateway` | 8001 | FastAPI | Routing, service catalog, IPFS proxy |
| `nlp-model` | 9095 | FastAPI | NanoLM English Q&A + document ingestion |
| `medical-diagnostic` | 9094 | FastAPI | Text lattice + knee-MRI + BiomedCLIP zero-shot |
| `deepseek-ocr` | 9093 | FastAPI + Tesseract | OCR text extraction |
| `hllset-next` | 9090 | Rust (axum) | HLLSet algebra API (ingest + inclusion) |
| `hllset-cortex` | 9092 | Flask | HLLSet semantic compressor |
| `ipfs` | 5001 / 8081 | Kubo | Content-addressed storage |

## Roadmap

The long-term direction is the **EWM (Emerging World Models) Rust workspace** — a self-modifying, content-addressed agent system built on `hllset-next`.

### Phase 0 — Foundation

Get EWM compiling against `hllset-next` and define the core types.

- Create the EWM Rust workspace and wire `Cargo.toml` path dependencies to `hllset-next` crates.
- Define `EmergenceTracker`, `OntologyView`, and `GateManager` in `ewm-core`.
- Define the `BootstrapFamily` trait with an `NgramBootstrap` implementation.
- Define `ControllerConfig` (the six-knob control surface) and the `OpponentProcessSignals` / `ControllerAction` enums.

*Target: `cargo build` passes with unit tests for `EmergenceTracker`.*

### Phase 1 — The Controller

Wire the integration layer — the controller is the heart of EWM.

- Implement `CrossLayerMatrix` — pairwise R-links across temporal layers.
- Implement `NoetherController::step()` — the full decision loop.
- Add temperature-controlled explore/exploit scheduling.
- Track rank flux over a sliding window (ΔR, Δ²R) with Fisher-guided divergence analysis.
- Compute the BSS opponent-process triad.
- Test that the controller stabilizes under repeated identical input and detects phase transitions on novel input.

### Phase 2 — Agent Network

Implement `[UM]-Net` in Rust and validate it against the Python notebook results.

- Implement the `Agent` trait and an `AgentDAG` with wave execution.
- Merge outputs with CRDTs at confluence nodes and resolve cycles via temporal separation.
- Add the fire threshold `θ_fire`.
- Test two-agent pipelines, three-agent confluence, cycle loops, and controller-steered agent graphs.

### Phase 3 — Memory & Lifecycle

Add the holographic time-lens and system reproduction.

- Implement `HolographicMemory` with a term-frequency stack and a `reconstruct(commit_t)` time-lens.
- Implement `reproduce()` to spawn a child system and `check_health()` for rank-bubble detection.
- Test round-trip reconstruction fidelity and parent → child knowledge transfer.

### Phase 4 — Self-Ingestion

Let the codebase observe itself.

- Implement the `ingest_commit()` pipeline and auto-generated `llms.txt` from doc comments.
- Add folder views (`v:<sha1>` = union of directory contents) and the `hot_files()` rank-flux query.
- Test that EWM ingests its own source history and surfaces the highest-flux files.

### Phase 5 — CLI & Evaluation

Ship a runnable binary and evaluation notebooks.

- Build the `ewm` CLI (`ewm run`, `ewm commit`, `ewm query`).
- Produce evaluation notebooks for controller stabilization, agent-graph cycle memory, holographic reconstruction, self-ingestion, and a CAAL-LLM-style driving-rule demo.

## Quick start

### Prerequisites

- Docker Engine + Docker Compose plugin (Linux) or Docker Desktop (Windows/macOS).

```bash
cp .env.example .env   # adjust ports/options if needed
```

### Build & run

```bash
docker compose build
docker compose up -d
docker compose logs -f
```

Open <http://localhost:8080>. The API gateway is at <http://localhost:8001>.

### Optional: UI resource badge

By default the header does **not** mount the Docker socket (security). To show live CPU/RAM usage, set in `.env`:

```bash
DOCKER_SOCK_MOUNT=/var/run/docker.sock   # Linux
# Docker Desktop Windows: DOCKER_SOCK_MOUNT=//var/run/docker.sock
```

then `docker compose up -d ewm-ui`.

### Optional: HLLSet services (`hllset-next` / `hllset-cortex`)

The application runs fully without them — NLP retrieval and clinical matching are computed locally in Python, OCR uses Tesseract, and durability comes from IPFS snapshots. They are only needed if you want the HLLSet lattice (content-addressed ingestion).

To enable them (requires the sources, which are not in this repository):

```bash
# 1. Put the two projects next to this repo, or point .env at your copies:
#    HLLSET_NEXT_CONTEXT=/path/to/hllset-next
#    HLLSET_CORTEX_CONTEXT=/path/to/hllset-cortex
# 2. Run with the override file:
docker compose -f docker-compose.yml -f docker-compose.optional.yml up -d --build
```

> **Note on `hllset-cortex` vs DeepSeek-OCR:** `hllset-cortex` is this project's own HLLSet semantic-compressor service — it is **not** part of the [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) repository. That repo is the upstream OCR *model*. This project's `deepseek-ocr` service currently uses Tesseract (CPU-only), so the DeepSeek-OCR model is not required either.

## Configuration

All settings live in `.env` (see `.env.example`): host ports, internal service URLs, IPFS profile, and optional GPU passthrough for OCR.

## Documentation

- [Deployment guide — Ubuntu + Docker](DEPLOYMENT.md)
- [Changelog](CHANGELOG.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Data & model sources

| Asset | Source | License note |
|---|---|---|
| NLP corpus (trivia/capitals/currencies/inventors) | OpenTriviaQA + Wikidata | Wikidata = CC0. **Verify OpenTriviaQA redistribution terms before publishing the bundled corpus.** |
| `nlp_keywords.csv` | derived from the NLP corpus | follows the corpus license |
| `knee_cnn.pt` | trained on synthetic MRI data in this repo | project license |
| BiomedCLIP weights | downloaded at runtime from Hugging Face | subject to the model's HF license |

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full third-party attribution.

## Disclaimer

Clinical/diagnostic outputs are for research and demonstration only. They are not a substitute for professional medical advice, diagnosis, or treatment.

## License

[MIT](LICENSE) © 2026 Codentic AI.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md). Report vulnerabilities privately per [SECURITY.md](SECURITY.md).

## Support

For questions, issues, or support requests, contact:

- **Deependra Kumar**
- Phone: [9845555760](tel:+919845555760)
- Email: [deependar@codenticai.com](mailto:deependar@codenticai.com)
