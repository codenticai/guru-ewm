"""
deepseek-ocr — OCR Model Service.

Runs the DeepSeek-OCR model (deepseek-ai/DeepSeek-OCR) when a CUDA GPU and the
model weights are available; otherwise falls back to Tesseract OCR on CPU.

Endpoints:
  GET  /health       — Service health + GPU/model status
  POST /ocr/upload   — Upload a PDF/image → extract text
"""

import os
import io
import logging
from datetime import datetime

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

import deepseek_model

# ── Config ──────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("deepseek-ocr")

# ── GPU Detection ───────────────────────────────────────────────────
GPU_AVAILABLE = False
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass

# ── PDF Extraction ──────────────────────────────────────────────────
try:
    import pypdf  # noqa: F401

    PDF_AVAILABLE = True
except ImportError:  # pragma: no cover
    PDF_AVAILABLE = False

# ── Image OCR (CPU, no GPU required) ────────────────────────────────
try:
    import pytesseract  # noqa: F401
    from PIL import Image, ImageOps  # noqa: F401

    IMAGE_OCR_AVAILABLE = True
except ImportError:  # pragma: no cover
    IMAGE_OCR_AVAILABLE = False

# File types treated as plain text rather than pixel images.
_TEXT_EXTENSIONS = (".txt", ".md", ".csv", ".json", ".log", ".text", ".rtf")

app = FastAPI(
    title="DeepSeek-OCR Service",
    description="GPU-accelerated OCR inference (DeepSeek-OCR model)",
    version="0.1.0",
)

# ═══════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    gpu_info = {"available": GPU_AVAILABLE}
    if GPU_AVAILABLE:
        gpu_info["device"] = torch.cuda.get_device_name(0)
        gpu_info["memory"] = f"{torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB"

    return {
        "status": "ok",
        "service": "deepseek-ocr",
        "gpu": gpu_info,
        "pdf_support": PDF_AVAILABLE,
        "image_ocr": "tesseract-cpu" if IMAGE_OCR_AVAILABLE else "unavailable",
        "model": "deepseek-ocr" if deepseek_model.is_available() else "tesseract-cpu",
        "deepseek_model": {
            "enabled": deepseek_model.enabled(),
            "gpu_available": deepseek_model.gpu_available(),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════
# OCR Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.post("/ocr/upload")
async def ocr_upload(file: UploadFile = File(...), mode: str = "full", prompt: str = ""):
    """Upload a PDF or image → extract text → OCR pipeline.

    - PDF: text extracted natively via pypdf (works on CPU).
    - Image (PNG/JPEG): OCR via Tesseract — CPU-only, no GPU needed.
    - mode="full" (default): OCR the whole image → all contents.
    - mode="ecg": ECG printout crop (header + footer bands).
    - prompt: optional model prompt override (used only on the GPU model path).
    """
    raw = await file.read()
    name = (file.filename or "").lower()
    extracted = ""
    source = ""
    used_model = False

    if name.endswith(".pdf"):
        if not PDF_AVAILABLE:
            return JSONResponse({"error": "PDF support not installed (pypdf)"}, status_code=503)
        try:
            reader = pypdf.PdfReader(io.BytesIO(raw))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            extracted = "\n".join(pages).strip()
            source = "pdf"
        except Exception as e:
            return JSONResponse({"error": f"PDF extraction failed: {e}"}, status_code=422)
    elif name.endswith(_TEXT_EXTENSIONS):
        # Plain-text report files (txt/md/csv/json) — read the text directly
        # instead of treating them as unreadable "images".
        try:
            extracted = raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            extracted = raw.decode("latin-1", errors="ignore").strip()
        source = "text"
    elif name.endswith((".dcm", ".dicom")):
        return JSONResponse(
            {"error": "DICOM images can't be read as text or pixels here — "
                      "paste the report text instead."},
            status_code=415,
        )
    else:
        # OCR any non-PDF file as an image. Try the real DeepSeek-OCR model
        # first (requires a CUDA GPU + weights); fall back to Tesseract CPU
        # OCR when it's unavailable or produces no text. Pillow opens
        # PNG/JPEG/TIFF/BMP as well as WebP/JFIF/GIF, so accept them too.
        if deepseek_model.is_available():
            model_text, _ = deepseek_model.infer_image(raw, prompt=prompt or None)
            if model_text:
                extracted = model_text
                source = "image"
                used_model = True
        if not used_model:
            if not IMAGE_OCR_AVAILABLE:
                return JSONResponse(
                    {"error": "Image OCR unavailable (tesseract/pytesseract not installed)"},
                    status_code=503,
                )
            try:
                extracted = _ocr_image_ecg(raw) if mode == "ecg" else _ocr_image_full(raw)
                source = "image"
            except Exception as e:
                return JSONResponse(
                    {"error": f"Unsupported or unreadable image: {e}. Use .pdf, .png, .jpg, .jpeg, .tif, .webp, .jfif."},
                    status_code=415,
                )

    if not extracted:
        return {
            "filename": file.filename,
            "size_bytes": len(raw),
            "text": "",
            "source": source,
            "mode": "cpu-fallback",
            "notice": "No text could be extracted from this document.",
        }

    return {
        "filename": file.filename,
        "size_bytes": len(raw),
        "source": source,
        "page_text": extracted,
        "text": extracted,
        "mode": "deepseek-ocr" if used_model else ("ecg" if mode == "ecg" else "full"),
        "engine": "deepseek-ocr" if used_model else "tesseract-cpu",
    }


def _upscale(im, target_width=4000):
    if im.width < target_width:
        scale = target_width / im.width
        return im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    return im


def _ocr_image_full(raw: bytes) -> str:
    """OCR the entire image — produce all text contents (no cropping).

    Tries several page-segmentation modes and keeps the longest result so
    full-page documents aren't truncated to header/footer bands."""
    image = ImageOps.autocontrast(_upscale(Image.open(io.BytesIO(raw)).convert("L")))
    best = ""
    for psm in ("3", "4", "6", "11"):
        try:
            t = pytesseract.image_to_string(image, config=f"--psm {psm}").strip()
        except Exception:
            continue
        if len(t) > len(best):
            best = t
    return best


def _ocr_image_ecg(raw: bytes) -> str:
    """ECG printout OCR: crop to the top header band and bottom technical
    line (the middle is waveform noise), then merge. Falls back to full-image
    OCR when the bands yield nothing."""
    image = Image.open(io.BytesIO(raw)).convert("L")
    w, h = image.size

    def score(t: str) -> int:
        tl = t.lower()
        hits = sum(1 for k in _ECG_KEYWORDS if k in tl)
        return hits * 10 + sum(c.isdigit() for c in tl)

    top = ImageOps.autocontrast(_upscale(image.crop((0, 0, w, int(h * 0.45)))))
    bottom = ImageOps.autocontrast(_upscale(image.crop((0, int(h * 0.90), w, h))))

    # Try several page-segmentation modes on the header band, keep the best.
    best_top = ""
    for psm in ("3", "4", "6", "11"):
        try:
            t = pytesseract.image_to_string(top, config=f"--psm {psm}").strip()
        except Exception:
            continue
        if score(t) > score(best_top):
            best_top = t

    bottom_text = ""
    try:
        bottom_text = pytesseract.image_to_string(bottom, config="--psm 6").strip()
    except Exception:
        pass

    merged = "\n".join(x for x in (best_top, bottom_text) if x).strip()
    if merged:
        return merged
    return _ocr_image_full(raw)


_ECG_KEYWORDS = [
    "vent", "rate", "bpm", "pr", "interval", "ms", "qrs", "qt", "qtc",
    "axes", "deg", "sinus", "rhythm", "normal", "ecg", "unconfirmed",
    "diagnosis", "male", "female", "years", "name", "id",
]


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 9093))
    logger.info(f"deepseek-ocr starting on port {port} (GPU: {GPU_AVAILABLE})")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=LOG_LEVEL.lower())
