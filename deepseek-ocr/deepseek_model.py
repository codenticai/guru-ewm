"""
DeepSeek-OCR model wrapper — optional GPU path.

The real DeepSeek-OCR model (deepseek-ai/DeepSeek-OCR) is a multi-billion
parameter vision-language model that requires a CUDA GPU and multi-GB weights
(flash-attention + bfloat16). The guru-ewm deepseek-ocr service runs on CPU by
default (Tesseract). This module lazy-loads the real model ONLY when a GPU and
the weights are available, so the service still boots and serves OCR on
CPU-only hosts.

Env:
  DEEPSEEK_OCR_GPU_ENABLED — "1"/"true"/"yes"/"on" force-on,
                             "0"/"false"/"no"/"off" force-off,
                             "auto" (default) = on iff a CUDA GPU is present.
  DEEPSEEK_OCR_MODEL_PATH  — HF repo id or local path to the weights
                             (default "deepseek-ai/DeepSeek-OCR").
  DEEPSEEK_OCR_PROMPT      — optional prompt override (see DEFAULT_PROMPT).
  DEEPSEEK_OCR_ENGINE      — "transformers" (default), "vllm", or "auto".
  DEEPSEEK_OCR_BASE_SIZE   — base size (default 1024).
  DEEPSEEK_OCR_IMAGE_SIZE  — tile size (default 768).
  DEEPSEEK_OCR_CROP_MODE   — "0"/"false" to disable cropping (default on).
"""

import glob
import os
import sys
import tempfile

DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

# "Gundam" mode (recommended for documents): base_size=1024, image_size=768,
# crop_mode=True — matches DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/config.py.
BASE_SIZE = int(os.environ.get("DEEPSEEK_OCR_BASE_SIZE", "1024"))
IMAGE_SIZE = int(os.environ.get("DEEPSEEK_OCR_IMAGE_SIZE", "768"))
CROP_MODE = os.environ.get("DEEPSEEK_OCR_CROP_MODE", "1").strip().lower() not in ("0", "false", "no", "off")
ENGINE = os.environ.get("DEEPSEEK_OCR_ENGINE", "transformers").strip().lower()

_model = None
_load_error = None


def gpu_available() -> bool:
    """True when torch sees a CUDA GPU. Lazy — torch is absent on CPU images."""
    try:
        import torch  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def enabled() -> bool:
    """Whether the GPU model path is enabled (env override or auto-detect)."""
    setting = os.environ.get("DEEPSEEK_OCR_GPU_ENABLED", "auto").strip().lower()
    if setting in ("1", "true", "yes", "on"):
        return True
    if setting in ("0", "false", "no", "off"):
        return False
    return gpu_available()


def is_available() -> bool:
    """Whether the real DeepSeek-OCR model can be used for this request."""
    return enabled() and gpu_available()


def load_model():
    """Load the model once (lazy). Returns an engine handle dict or None."""
    global _model, _load_error
    if _model is not None:
        return _model
    if _load_error is not None:
        return None
    if not enabled():
        _load_error = "DeepSeek-OCR GPU path disabled (DEEPSEEK_OCR_GPU_ENABLED)"
        return None
    if not gpu_available():
        _load_error = "no CUDA GPU available"
        return None

    engine = ENGINE
    if engine == "auto":
        engine = "transformers"

    if engine == "transformers":
        handle = _load_transformers()
    elif engine == "vllm":
        handle = _load_vllm()
    else:
        _load_error = f"unknown DEEPSEEK_OCR_ENGINE={ENGINE!r} (use transformers or vllm)"
        return None

    if handle is None:
        return None  # _load_error already set
    _model = {"engine": engine, **handle}
    return _model


def _load_transformers():
    """Load via HuggingFace transformers (flash-attn → sdpa → eager)."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as e:  # pragma: no cover — torch/transformers absent on CPU image
        _load_error = f"missing torch/transformers for DeepSeek-OCR: {e}"
        return None

    model_path = os.environ.get("DEEPSEEK_OCR_MODEL_PATH", "deepseek-ai/DeepSeek-OCR")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        _load_error = f"failed to load DeepSeek-OCR tokenizer from {model_path}: {e}"
        return None

    # flash-attn is fastest but optional — fall back to sdpa / eager.
    model = None
    for attn in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = AutoModel.from_pretrained(
                model_path,
                _attn_implementation=attn,
                trust_remote_code=True,
                use_safetensors=True,
            )
            break
        except Exception:
            continue
    if model is None:
        _load_error = f"failed to load DeepSeek-OCR model from {model_path}"
        return None

    model = model.eval().cuda().to(torch.bfloat16)
    return {"model": model, "tokenizer": tokenizer}


def _load_vllm():
    """Load via vLLM using the bundled DeepSeek-OCR model registration."""
    try:
        from vllm import LLM, SamplingParams
        from vllm.model_executor.models.registry import ModelRegistry
    except Exception as e:
        _load_error = f"missing vllm for DeepSeek-OCR: {e}"
        return None

    # DeepSeek-OCR has no built-in vLLM support — it ships its own model
    # class and logits processor. Load them from the bundled source so the
    # model architecture can be registered before vLLM builds the engine.
    vllm_dir = os.path.join(os.path.dirname(__file__), "model", "DeepSeek-OCR2-vllm")
    if vllm_dir not in sys.path:
        sys.path.insert(0, vllm_dir)
    try:
        from deepseek_ocr2 import DeepseekOCR2ForCausalLM
        from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
        ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)
    except Exception as e:
        _load_error = f"failed to import DeepSeek-OCR vLLM model code: {e}"
        return None

    model_path = os.environ.get("DEEPSEEK_OCR_MODEL_PATH", "deepseek-ai/DeepSeek-OCR")
    kwargs = {
        "model": model_path,
        "hf_overrides": {"architectures": ["DeepseekOCR2ForCausalLM"]},
        "trust_remote_code": True,
        "max_model_len": 8192,
        "enforce_eager": False,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "disable_mm_preprocessor_cache": True,
    }
    try:
        llm = LLM(**kwargs)
    except Exception as e:
        _load_error = f"failed to load DeepSeek-OCR via vllm from {model_path}: {e}"
        return None

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=[
            NoRepeatNGramLogitsProcessor(
                ngram_size=20, window_size=90, whitelist_token_ids={128821, 128822}
            )
        ],
        skip_special_tokens=False,
    )
    return {"model": llm, "sampling": sampling}


def _infer_text(model, tokenizer, image_path: str, prompt: str, out_dir: str) -> str:
    """Call model.infer(...) and extract the OCR text from its result."""
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=out_dir,
        base_size=BASE_SIZE,
        image_size=IMAGE_SIZE,
        crop_mode=CROP_MODE,
        save_results=False,
    )

    # Prefer an explicit string return.
    if isinstance(result, str) and result.strip():
        return result.strip()

    # Some versions return a dict with the text under a known key.
    if isinstance(result, dict):
        for key in ("text", "result", "output", "response", "predict"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    # Fall back to any markdown/text file the model wrote to out_dir.
    for pattern in ("*.md", "*.txt"):
        for path in sorted(glob.glob(os.path.join(out_dir, pattern))):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read().strip()
                if content:
                    return content
            except OSError:
                continue
    return ""


def _infer_vllm(llm, sampling, image_path: str, prompt: str) -> str:
    """Run vLLM generation and return the OCR text."""
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    model_input = [{"prompt": prompt, "multi_modal_data": {"image": image}}]
    outputs = llm.generate(model_input, sampling)
    if outputs and outputs[0].outputs:
        return outputs[0].outputs[0].text.strip()
    return ""


def infer_image(raw: bytes, prompt: str = None):
    """Run the DeepSeek-OCR model on an image.

    Returns (text, mode) on success — or (None, reason) when unavailable or
    when the model produced no text (caller should fall back to Tesseract).
    """
    if not is_available():
        return None, "unavailable"
    loaded = load_model()
    if loaded is None:
        return None, _load_error or "model unavailable"
    prompt = prompt or os.environ.get("DEEPSEEK_OCR_PROMPT") or DEFAULT_PROMPT

    with tempfile.TemporaryDirectory() as tmp:
        image_path = os.path.join(tmp, "input.png")
        with open(image_path, "wb") as fh:
            fh.write(raw)
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        try:
            if loaded["engine"] == "vllm":
                text = _infer_vllm(loaded["model"], loaded["sampling"], image_path, prompt)
            else:
                text = _infer_text(loaded["model"], loaded["tokenizer"], image_path, prompt, out_dir)
        except Exception as e:
            return None, f"inference failed: {e}"
        if text:
            return text, "deepseek-ocr"
        return None, "model produced no text"
