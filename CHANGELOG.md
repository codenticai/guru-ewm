# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-21

First open-source release.

### Added

- `nlp-model` — NanoLM English Q&A: IDF-weighted cosine retrieval, single-keyword rescue, union-of-occurrences replies with 15-per-page pagination, typo correction, clarification prompts, pronoun/context resolution, negation and elaboration handling, meta-question handling, persistent sessions, and bulk corpus ingestion.
- `medical-diagnostic` — ECG / X-ray / CT / knee-MRI / lab text-report matching with numeric reference ranges, negation + domain filtering, and head-term scoring; BiomedCLIP zero-shot image classification; synthetic knee-MRI fingerprint classifier (CPU-only) plus an optional CPU CNN (`/classify/knee/deep`).
- `deepseek-ocr` — Tesseract CPU OCR (full-page and ECG band modes), PDF text extraction, plain-text pass-through, and an optional DeepSeek-OCR GPU model path (`deepseek_model.py` + vendored model source).
- `invoice-extractor` — dedicated PDF/image → template-populated invoice JSON with a browser UI and optional model-backed OCR.
- `ewm-gateway` — central routing, aggregated health, service catalog, and IPFS proxy.
- `ewm-ui` — NiceGUI chat with NLP / OCR / Diagnose modes, sample reports, light/dark themes, logo, and an optional live CPU/RAM header badge.
- IPFS (Kubo) content-addressed snapshots with local JSON backup and startup restore.
- Community + CI: README, DEPLOYMENT, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, THIRD_PARTY_NOTICES, MIT license, and a compose-validating CI workflow.



## Development history (pre-release)

- **2026-08-12** — Initial NanoLM medical-diagnostic: ECG knowledge-card lattice, HLLSet ingestion, IPFS snapshots.
- **2026-08-13** — X-ray + lab report support; scoring hardening; second model `nlp-model` with IDF-cosine retrieval, typo correction, clarification, and context tracking.
- **2026-08-14** — Knee-MRI classifier (Python port), vision encoder merged into `medical-diagnostic`, dead-subsystem cleanup, CT support, and the UI redesign + 3-mode dropdown.
- **2026-08-15** — Numeric-keyword cleanup, single-keyword non-fallback sweep, union-of-occurrences + pagination, and negation fixes.
- **2026-08-16** — Open-source packaging: portable compose file, MIT license, community docs, CI, and the UI resource badge.
- **2026-08-17** — IPFS local backup + startup restore.
- **2026-08-19** — Invoice extraction, segregated into the `invoice-extractor` service with a browser UI.
- **2026-08-20** — DeepSeek-OCR model integration (CPU fallback preserved), vendored DeepSeek-OCR source, and full test-suite validation.
- **2026-08-21** — Open-source finalization: third-party notices, code owner, support contact, and legacy code removal.
