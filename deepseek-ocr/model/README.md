# Vendored DeepSeek-OCR source

This directory contains source code vendored from the upstream
[deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
repository (the "Visual Causal Flow" OCR model).

- `DeepSeek-OCR2-hf/` — Hugging Face `transformers` integration
- `DeepSeek-OCR2-vllm/` — vLLM integration

It is used only for the optional GPU inference path (`deepseek_model.py`).
The default CPU OCR engine is Tesseract. No model weights are included here.

## License

The upstream project is distributed under its own license — see the upstream
repository's `LICENSE` file. This vendored copy is provided for build
convenience and remains subject to the upstream license.
