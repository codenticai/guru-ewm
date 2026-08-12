# Guru-EWM — Emerging World Models Platform

> Unified entry point and orchestration platform for all Emerging World Model (EWM) backends.

## Overview

Guru-EWM provides a single gateway for managing, routing, and interacting with
Emerging World Models — backend AI/ML services exposed as containerized microservices
with a generic UI and IPFS-based content-addressed storage.

## Architecture

```
ewm-ui (NiceGUI :8080) → ewm-gateway (FastAPI :8000) → ┬─ hllset-cortex (Flask :9092)
                                                        ├─ deepseek-ocr (GPU :9093)
                                                        └─ ipfs (ipfrs-core :5001)
```

## Components

| Component | Port | Technology | Description |
|-----------|------|------------|-------------|
| **ewm-ui** | 8080 | NiceGUI | Generic UI for all EWM backends |
| **ewm-gateway** | 8000 | FastAPI | API gateway, routing, service discovery |
| **hllset-cortex** | 9092 | Flask + hllset_py (Rust) | HLLSet semantic compressor for OCR |
| **deepseek-ocr** | 9093 | FastAPI + DeepSeek-OCR | GPU-accelerated OCR inference |
| **ipfs** | 5001 | ipfrs-core + sled | Content-addressed storage |

## Documentation

- [Architecture & Implementation Plan](docs/ARCHITECTURE.md)

## Status

🔄 **Phase 1: Architecture & Design** — Documenting existing systems and planning containerized deployment.

### ✅ hllset-cortex Pipeline Validated (2026-08-12)

| Test | Result |
|------|--------|
| Real ds-OCR BPE tokenizer → 128K vocab | ✅ |
| IICA-compliant HLLSet with real token IDs | ✅ |
| gate_TF vocabulary filtering | ✅ |
| TF-ranked materialization (set) | ✅ |
| De Bruijn ordered reconstruction | ✅ |
| **Roundtrip: 100% word retention** | ✅ |
| OCR inference on RTX 3060 | ✅ |

> **Key insight:** hllset-cortex is encoding-agnostic — MurmurHash3 operates
> identically on simulated (`enc10253`) or real (`tid671`) token IDs. De Bruijn
> reconstruction achieves zero order loss.

## Quick Start

### Prerequisites

```bash
# 1. Build hllset_py (Rust → Python bindings)
cd D:\innovation\DeepSeek-OCR\hllset_cortex\crates\hllset_py
maturin develop --release

# 2. Install hllset-cortex
cd D:\innovation\DeepSeek-OCR\hllset_cortex
pip install -e .
```

### Run Validation Tests

```bash
cd d:\innovation\guru-ewm
pip install -r tests/requirements-test.txt
pytest tests/ -v
```

### Start Services (Coming in Phase 2)

```bash
docker-compose up -d
```
