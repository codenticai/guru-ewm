"""
deepseek-ocr — GPU-Accelerated OCR Model Service.

Wraps the DeepSeek-OCR model as a microservice.
Communicates with hllset-cortex for semantic compression.

Endpoints:
  GET  /health       — Service health + GPU status
  POST /ocr          — Run OCR on provided text/encoding IDs
  POST /ocr/upload   — Upload a PDF/image → extract text → OCR pipeline
  POST /ocr/encode   — Text → encoding IDs (simulated when no GPU)
  POST /ocr/decode   — Encoding IDs → text
"""

import os
import io
import logging
from datetime import datetime

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
import httpx

# ── Config ──────────────────────────────────────────────────────────
HLLSET_CORTEX_URL = os.environ.get("HLLSET_CORTEX_URL", "http://hllset-cortex:9092")
IPFS_API_URL = os.environ.get("IPFS_API_URL", "http://ipfs:5001")
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

_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    return _client


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
        "hllset_cortex_url": HLLSET_CORTEX_URL,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════
# OCR Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.post("/ocr")
async def ocr(request: Request):
    """Run OCR pipeline on input text/encoding IDs.

    When GPU is available, runs the full DeepSeek-OCR model.
    In CPU mode, acts as a pass-through to hllset-cortex.
    """
    body = await request.json()
    text = body.get("text", body.get("encoding_ids", ""))

    if not text:
        return JSONResponse({"error": "Missing 'text' or 'encoding_ids'"}, status_code=400)

    if GPU_AVAILABLE:
        # Full DeepSeek-OCR pipeline (GPU required)
        return await _ocr_gpu(text)
    else:
        # CPU fallback: forward to hllset-cortex for semantic processing
        return await _ocr_cpu_fallback(text)


@app.post("/ocr/upload")
async def ocr_upload(file: UploadFile = File(...), mode: str = "full"):
    """Upload a PDF or image → extract text → OCR pipeline.

    - PDF: text extracted natively via pypdf (works on CPU).
    - Image (PNG/JPEG): OCR via Tesseract — CPU-only, no GPU needed.
    - mode="full" (default): OCR the whole image → all contents.
    - mode="ecg": ECG printout crop (header + footer bands).
    """
    raw = await file.read()
    name = (file.filename or "").lower()
    extracted = ""
    source = ""

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
        # OCR any non-PDF file as an image. Pillow opens PNG/JPEG/TIFF/BMP as
        # well as WebP/JFIF/GIF, so accept them too instead of 415-ing.
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

    # Run the OCR pipeline on the extracted text
    result = await _ocr_cpu_fallback(extracted) if not GPU_AVAILABLE else await _ocr_gpu(extracted)
    return {
        "filename": file.filename,
        "size_bytes": len(raw),
        "source": source,
        "page_text": extracted,
        **result,
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


async def _ocr_image_gpu(raw: bytes, filename: str) -> dict:
    """GPU path: run DeepSeek-OCR on an image (requires model + weights)."""
    # Placeholder — real implementation loads the DeepSeek-OCR model and
    # calls model.infer(tokenizer, prompt=..., image_file=...).
    return {
        "filename": filename,
        "size_bytes": len(raw),
        "source": "image",
        "mode": "gpu",
        "notice": "Image OCR on GPU requires the DeepSeek-OCR model weights to be mounted.",
    }


async def _ocr_gpu(text: str) -> dict:
    """GPU-accelerated OCR with DeepSeek-OCR model."""
    # This would load the actual DeepSeek-OCR model and tokenizer
    # For now, returns the encoding IDs with metadata
    encoding_ids = [f"tid{ord(c):04d}" for c in text if c.isalpha()][:100]

    # Forward to hllset-cortex for semantic compression
    client = await get_client()
    try:
        r = await client.post(
            f"{HLLSET_CORTEX_URL}/process",
            json={"text": " ".join(encoding_ids), "format": "basic"},
        )
        hllset_result = r.json() if r.status_code == 200 else {}
    except Exception as e:
        hllset_result = {"error": str(e)}

    return {
        "text": text,
        "encoding_ids": encoding_ids,
        "encoding_count": len(encoding_ids),
        "hllset": hllset_result,
        "mode": "gpu",
    }


async def _ocr_cpu_fallback(text: str) -> dict:
    """CPU fallback: simulate encoding IDs and forward to hllset-cortex."""
    # Generate simulated encoding IDs from text
    words = text.split()
    encoding_ids = [f"enc{abs(hash(w)) % 90000 + 10000}" for w in words]

    # Forward to hllset-cortex
    client = await get_client()
    try:
        r = await client.post(
            f"{HLLSET_CORTEX_URL}/process",
            json={"text": " ".join(encoding_ids), "format": "basic"},
        )
        hllset_result = r.json() if r.status_code == 200 else {}
    except Exception as e:
        hllset_result = {"error": str(e)}

    return {
        "text": text,
        "encoding_ids": encoding_ids,
        "encoding_count": len(encoding_ids),
        "hllset": hllset_result,
        "mode": "cpu-fallback",
    }


@app.post("/ocr/encode")
async def ocr_encode(request: Request):
    """Text → encoding IDs."""
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return JSONResponse({"error": "Missing 'text'"}, status_code=400)

    words = text.split()
    encoding_ids = [f"enc{abs(hash(w)) % 90000 + 10000}" for w in words]
    return {"encoding_ids": encoding_ids, "count": len(encoding_ids)}


@app.post("/ocr/decode")
async def ocr_decode(request: Request):
    """Encoding IDs → text via hllset-cortex materialization."""
    body = await request.json()
    encoding_ids = body.get("encoding_ids", [])

    if not encoding_ids:
        return JSONResponse({"error": "Missing 'encoding_ids'"}, status_code=400)

    stream = " ".join(encoding_ids)
    client = await get_client()
    try:
        r = await client.post(
            f"{HLLSET_CORTEX_URL}/process",
            json={"text": stream, "format": "basic"},
        )
        return r.json() if r.status_code == 200 else {"error": "hllset-cortex error"}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 9093))
    logger.info(f"deepseek-ocr starting on port {port} (GPU: {GPU_AVAILABLE})")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=LOG_LEVEL.lower())
