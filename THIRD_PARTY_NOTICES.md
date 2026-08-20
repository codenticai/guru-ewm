# Third-Party Notices

Guru-EWM bundles, vendors, or downloads the following third-party components.
This file records their sources and licenses so attribution travels with the code.

## Vendored source code

### DeepSeek-OCR (`deepseek-ocr/model/`)

- Source: https://github.com/deepseek-ai/DeepSeek-OCR
- Purpose: optional DeepSeek-OCR model integration (GPU-only inference path)
- License: see the upstream repository's `LICENSE` file (verify before publishing)
- Note: vendored for offline builds. It is **not required** to run the app —
  the default CPU OCR engine is Tesseract. No model weights are included here.

## Downloaded at runtime (not bundled)

### BiomedCLIP

- Model: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- Source: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- License: see the Hugging Face model card.

### DeepSeek-OCR model weights

- Downloaded from Hugging Face only when GPU inference is enabled
  (`DEEPSEEK_OCR_GPU_ENABLED=true`). See the model card for its license.

## Container / system dependencies

### Tesseract OCR

- License: Apache License 2.0 — https://github.com/tesseract-ocr/tesseract

### IPFS (Kubo)

- License: MIT / Apache-2.0 — https://github.com/ipfs/kubo

## Data

### NLP corpus

- OpenTriviaQA + Wikidata (see the "Data & model sources" section of the README).
- Wikidata is CC0. OpenTriviaQA redistribution terms must be verified before
  publishing the bundled corpus.

---

Python package dependencies are declared in each service's `requirements.txt`
and carry their own licenses.
