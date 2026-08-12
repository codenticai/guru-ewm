"""
ewm-gateway — FastAPI Central Gateway for Emerging World Models.

Routes requests to EWM backend services:
  - hllset-cortex  (Flask :9092) — HLLSet semantic compressor
  - deepseek-ocr   (FastAPI :9093) — GPU-accelerated OCR
  - ipfs           (Kubo :5001) — Content-addressed storage

Endpoints:
  GET  /                  — API root with service catalog
  GET  /health            — Aggregated health check
  GET  /services          — Registered service list
  POST /ocr/process       — Forward to hllset-cortex
  POST /ocr/full          — Full pipeline (OCR → HLLSet → decode)
  POST /ipfs/upload       — Upload to IPFS
  GET  /ipfs/{cid}        — Retrieve from IPFS
"""

import os
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
import httpx

# ── Config ──────────────────────────────────────────────────────────
HLLSET_CORTEX_URL = os.environ.get("HLLSET_CORTEX_URL", "http://hllset-cortex:9092")
DEEPSEEK_OCR_URL = os.environ.get("DEEPSEEK_OCR_URL", "http://deepseek-ocr:9093")
MEDICAL_DIAGNOSTIC_URL = os.environ.get("MEDICAL_DIAGNOSTIC_URL", "http://medical-diagnostic:9094")
IPFS_API_URL = os.environ.get("IPFS_API_URL", "http://ipfs:5001")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("ewm-gateway")

app = FastAPI(
    title="Guru-EWM Gateway",
    description="Central API gateway for Emerging World Models",
    version="0.1.0",
)

# ── HTTP Client (shared) ────────────────────────────────────────────
_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    return _client


# ═══════════════════════════════════════════════════════════════════════
# Health & Discovery
# ═══════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "name": "Guru-EWM Gateway",
        "version": "0.1.0",
        "services": {
            "hllset-cortex": HLLSET_CORTEX_URL,
            "deepseek-ocr": DEEPSEEK_OCR_URL,
            "medical-diagnostic": MEDICAL_DIAGNOSTIC_URL,
            "ipfs": IPFS_API_URL,
        },
    }


@app.get("/health")
async def health():
    client = await get_client()
    services_status = {}

    # Check hllset-cortex
    try:
        r = await client.get(f"{HLLSET_CORTEX_URL}/health")
        services_status["hllset-cortex"] = r.json() if r.status_code == 200 else {"status": "unhealthy"}
    except Exception as e:
        services_status["hllset-cortex"] = {"status": "unreachable", "error": str(e)}

    # Check deepseek-ocr
    try:
        r = await client.get(f"{DEEPSEEK_OCR_URL}/health")
        services_status["deepseek-ocr"] = r.json() if r.status_code == 200 else {"status": "unhealthy"}
    except Exception:
        services_status["deepseek-ocr"] = {"status": "unreachable"}

    # Check medical-diagnostic
    try:
        r = await client.get(f"{MEDICAL_DIAGNOSTIC_URL}/health")
        services_status["medical-diagnostic"] = r.json() if r.status_code == 200 else {"status": "unhealthy"}
    except Exception:
        services_status["medical-diagnostic"] = {"status": "unreachable"}

    # Check IPFS
    try:
        r = await client.post(f"{IPFS_API_URL}/api/v0/id")
        services_status["ipfs"] = {"status": "ok"} if r.status_code == 200 else {"status": "unhealthy"}
    except Exception:
        services_status["ipfs"] = {"status": "unreachable"}

    all_healthy = all(
        s.get("status") in ("ok", "healthy") or s.get("hllset_available", True)
        for s in services_status.values()
    )

    return {
        "status": "ok" if all_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "services": services_status,
    }


@app.get("/services")
async def list_services():
    return {
        "hllset-cortex": {
            "url": HLLSET_CORTEX_URL,
            "type": "semantic-compressor",
            "endpoints": ["/health", "/process", "/process/debruijn", "/gate"],
        },
        "deepseek-ocr": {
            "url": DEEPSEEK_OCR_URL,
            "type": "ocr-inference",
            "endpoints": ["/health", "/ocr", "/ocr/batch"],
        },
        "medical-diagnostic": {
            "url": MEDICAL_DIAGNOSTIC_URL,
            "type": "clinical-diagnosis",
            "endpoints": ["/health", "/diagnose", "/diagnose/text", "/diagnose/image", "/engines"],
        },
        "ipfs": {
            "url": IPFS_API_URL,
            "type": "storage",
            "endpoints": ["/api/v0/add", "/api/v0/cat", "/api/v0/id"],
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# OCR Pipeline
# ═══════════════════════════════════════════════════════════════════════

@app.post("/ocr/process")
async def ocr_process(request: Request):
    """Forward to hllset-cortex for HLLSet semantic compression."""
    body = await request.json()
    client = await get_client()
    try:
        r = await client.post(
            f"{HLLSET_CORTEX_URL}/process",
            json=body,
        )
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        raise HTTPException(503, "hllset-cortex service unreachable")


@app.post("/ocr/process/debruijn")
async def ocr_process_debruijn(request: Request):
    """Forward to hllset-cortex for De Bruijn ordered reconstruction."""
    body = await request.json()
    client = await get_client()
    try:
        r = await client.post(
            f"{HLLSET_CORTEX_URL}/process",
            json={**body, "format": "debruijn"},
        )
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        raise HTTPException(503, "hllset-cortex service unreachable")


@app.post("/ocr/full")
async def ocr_full_pipeline(request: Request):
    """Full OCR pipeline: image → OCR → HLLSet → decode."""
    body = await request.json()
    client = await get_client()

    # Step 1: OCR inference
    try:
        ocr_r = await client.post(f"{DEEPSEEK_OCR_URL}/ocr", json=body)
        if ocr_r.status_code != 200:
            return JSONResponse(content={"error": "OCR inference failed"}, status_code=502)
        ocr_result = ocr_r.json()
    except httpx.ConnectError:
        raise HTTPException(503, "deepseek-ocr service unreachable")

    # Step 2: HLLSet semantic compression
    try:
        hllset_r = await client.post(
            f"{HLLSET_CORTEX_URL}/process",
            json={"text": ocr_result.get("text", ""), "format": "basic"},
        )
        hllset_result = hllset_r.json() if hllset_r.status_code == 200 else {}
    except httpx.ConnectError:
        hllset_result = {"error": "hllset-cortex unreachable"}

    return {
        "ocr": ocr_result,
        "hllset": hllset_result,
    }


# ═══════════════════════════════════════════════════════════════════════
# Diagnostic Pipeline
# ═══════════════════════════════════════════════════════════════════════

@app.post("/diagnose")
async def diagnose(request: Request):
    """Forward to medical-diagnostic service."""
    body = await request.json()
    client = await get_client()
    try:
        r = await client.post(f"{MEDICAL_DIAGNOSTIC_URL}/diagnose", json=body)
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        raise HTTPException(503, "medical-diagnostic service unreachable")


@app.post("/diagnose/text")
async def diagnose_text(request: Request):
    """Text-based diagnosis: forward OCR'd text to medical-diagnostic."""
    body = await request.json()
    client = await get_client()
    try:
        r = await client.post(f"{MEDICAL_DIAGNOSTIC_URL}/diagnose/text", json=body)
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        raise HTTPException(503, "medical-diagnostic service unreachable")


@app.post("/diagnose/full")
async def diagnose_full_pipeline(request: Request):
    """Full pipeline: OCR → HLLSet → text → diagnosis → IPFS."""
    body = await request.json()
    client = await get_client()

    # Step 1: OCR inference
    try:
        ocr_r = await client.post(f"{DEEPSEEK_OCR_URL}/ocr", json=body)
        if ocr_r.status_code != 200:
            return JSONResponse(content={"error": "OCR inference failed"}, status_code=502)
        ocr_result = ocr_r.json()
    except httpx.ConnectError:
        raise HTTPException(503, "deepseek-ocr service unreachable")

    ocr_text = ocr_result.get("text", "")

    # Step 2: Diagnosis over the OCR'd text
    try:
        diag_r = await client.post(
            f"{MEDICAL_DIAGNOSTIC_URL}/diagnose/text",
            json={"text": ocr_text},
        )
        diag_result = diag_r.json() if diag_r.status_code == 200 else {"error": "diagnosis failed"}
    except httpx.ConnectError:
        diag_result = {"error": "medical-diagnostic unreachable"}

    return {
        "ocr": ocr_result,
        "diagnosis": diag_result,
    }


# ═══════════════════════════════════════════════════════════════════════
# Medical Document Analysis (OCR → Diagnosis → Final Report)
# ═══════════════════════════════════════════════════════════════════════

@app.post("/analyze/ecg")
async def analyze_ecg(
    file: UploadFile = File(...),
    modality: str = Form("ECG"),
    instruction: str = Form(""),
):
    """End-to-end medical analysis:

        document/image → deepseek-ocr (text extraction)
          → medical-diagnostic (diagnosis)
            → final analysis report

    Accepts a multipart file upload (PDF or image) plus an optional
    free-text instruction describing how to process the file.
    """
    client = await get_client()

    # Step 1: OCR — extract text from the uploaded document
    try:
        ocr_files = {"file": (file.filename, await file.read(), file.content_type or "application/octet-stream")}
        ocr_r = await client.post(f"{DEEPSEEK_OCR_URL}/ocr/upload", files=ocr_files)
        if ocr_r.status_code != 200:
            return JSONResponse(
                content={"error": "OCR failed", "detail": ocr_r.text[:500]},
                status_code=502,
            )
        ocr_result = ocr_r.json()
    except httpx.ConnectError:
        raise HTTPException(503, "deepseek-ocr service unreachable")

    extracted_text = ocr_result.get("text") or ocr_result.get("page_text") or ""

    # Step 2: Diagnosis — build a final structured report from the text
    try:
        diag_r = await client.post(
            f"{MEDICAL_DIAGNOSTIC_URL}/analyze/hllset",
            json={"text": extracted_text, "modality": modality, "instruction": instruction},
        )
        report = diag_r.json() if diag_r.status_code == 200 else {"error": diag_r.text[:500]}
    except httpx.ConnectError:
        raise HTTPException(503, "medical-diagnostic service unreachable")

    return {
        "filename": file.filename,
        "modality": modality,
        "instruction": instruction,
        "ocr": {
            "source": ocr_result.get("source"),
            "extracted_text": extracted_text,
            "mode": ocr_result.get("mode"),
            "notice": ocr_result.get("notice"),
        },
        "report": report,
    }


@app.post("/analyze/document")
async def analyze_document(
    file: UploadFile = File(...),
    modality: str = Form("DOC"),
    instruction: str = Form(""),
):
    """Generic medical document analysis (same pipeline as /analyze/ecg)."""
    return await analyze_ecg(file, modality, instruction)


# ═══════════════════════════════════════════════════════════════════════
# IPFS Proxy
# ═══════════════════════════════════════════════════════════════════════

@app.post("/ipfs/upload")
async def ipfs_upload(request: Request):
    """Upload content to IPFS."""
    client = await get_client()
    try:
        form = await request.form()
        files = {}
        for key, value in form.items():
            if hasattr(value, "file"):
                files[key] = (value.filename, await value.read())
        r = await client.post(f"{IPFS_API_URL}/api/v0/add", files=files)
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        raise HTTPException(503, "IPFS service unreachable")


@app.get("/ipfs/{cid:path}")
async def ipfs_get(cid: str):
    """Retrieve content from IPFS by CID."""
    client = await get_client()
    try:
        r = await client.post(f"{IPFS_API_URL}/api/v0/cat", params={"arg": cid})
        return JSONResponse(content={"cid": cid, "data": r.text}, status_code=r.status_code)
    except httpx.ConnectError:
        raise HTTPException(503, "IPFS service unreachable")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"ewm-gateway starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=LOG_LEVEL.lower())
