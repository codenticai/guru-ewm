"""
ewm-gateway — FastAPI Central Gateway for Emerging World Models.

Routes requests to EWM backend services:
  - deepseek-ocr          (FastAPI :9093) — OCR text extraction
  - medical-diagnostic    (FastAPI :9094) — clinical diagnosis
  - nlp-model             (FastAPI :9095) — English NLP
  - ipfs                  (Kubo :5001) — Content-addressed storage

Endpoints:
  GET  /                  — API root with service catalog
  GET  /health            — Aggregated health check
  GET  /services          — Registered service list
  POST /ocr/extract       — Extract text from an uploaded file
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
NLP_MODEL_URL = os.environ.get("NLP_MODEL_URL", "http://nlp-model:9095")
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
        _client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
    return _client


def _infer_modality(instruction: str) -> str:
    """Best-effort modality hint from a free-text instruction."""
    s = (instruction or "").lower()
    if any(k in s for k in ("knee", "mri", "meniscus", "acl", "ligament")):
        return "MR"
    if " ct" in s or "ct scan" in s or "computed tomography" in s:
        return "CT"
    if any(k in s for k in ("x-ray", "xray", "chest", "radiograph")):
        return "DX"
    return "DX"


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
            "nlp-model": NLP_MODEL_URL,
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

    # Check nlp-model
    try:
        r = await client.get(f"{NLP_MODEL_URL}/health")
        services_status["nlp-model"] = r.json() if r.status_code == 200 else {"status": "unhealthy"}
    except Exception:
        services_status["nlp-model"] = {"status": "unreachable"}

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
            "endpoints": ["/health", "/ocr/upload"],
        },
        "medical-diagnostic": {
            "url": MEDICAL_DIAGNOSTIC_URL,
            "type": "clinical-diagnosis",
            "endpoints": ["/health", "/analyze/hllset", "/analyze/document", "/analyze/ecg", "/classify", "/classify/knee"],
        },
        "nlp-model": {
            "url": NLP_MODEL_URL,
            "type": "english-nlp",
            "endpoints": ["/health", "/chat", "/nlp/query", "/nlp/ingest", "/nlp/ingest/document"],
        },
        "ipfs": {
            "url": IPFS_API_URL,
            "type": "storage",
            "endpoints": ["/api/v0/add", "/api/v0/cat", "/api/v0/id"],
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# OCR
# ═══════════════════════════════════════════════════════════════════════

@app.post("/ocr/extract")
async def ocr_extract(file: UploadFile = File(...)):
    """Extract text from an uploaded file (image/PDF) via deepseek-ocr."""
    client = await get_client()
    try:
        files = {"file": (file.filename, await file.read(), file.content_type or "application/octet-stream")}
        r = await client.post(f"{DEEPSEEK_OCR_URL}/ocr/upload?mode=full", files=files)
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        raise HTTPException(503, "deepseek-ocr service unreachable")


# ═══════════════════════════════════════════════════════════════════════
# Diagnostic Pipeline
# ═══════════════════════════════════════════════════════════════════════

@app.post("/analyze/text")
async def analyze_text(request: Request):
    """Text query → NanoLM diagnostic lattice report (medical-diagnostic /analyze/hllset)."""
    body = await request.json()
    client = await get_client()
    try:
        r = await client.post(f"{MEDICAL_DIAGNOSTIC_URL}/analyze/hllset", json=body)
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        raise HTTPException(503, "medical-diagnostic service unreachable")


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
        # ECG printouts use the band-cropped OCR; everything else reads the
        # full page so no content is missed.
        ocr_mode = "ecg" if (modality or "ECG").upper() == "ECG" else "full"
        ocr_r = await client.post(f"{DEEPSEEK_OCR_URL}/ocr/upload?mode={ocr_mode}", files=ocr_files)
        if ocr_r.status_code != 200:
            try:
                detail = ocr_r.json().get("error", ocr_r.text[:300])
            except Exception:
                detail = ocr_r.text[:300]
            return JSONResponse(
                content={"error": f"OCR failed: {detail}"},
                status_code=502,
            )
        ocr_result = ocr_r.json()
    except httpx.ConnectError:
        raise HTTPException(503, "deepseek-ocr service unreachable")

    extracted_text = ocr_result.get("text") or ocr_result.get("page_text") or ""

    # Step 2: Diagnosis — build a final structured report from the text
    if not extracted_text.strip():
        return {
            "filename": file.filename,
            "modality": modality,
            "instruction": instruction,
            "ocr": {
                "source": ocr_result.get("source"),
                "extracted_text": "",
                "mode": ocr_result.get("mode"),
                "notice": ocr_result.get("notice"),
            },
            "report": {
                "report_type": "nanolm",
                "engine": "nanolm",
                "modality": modality,
                "findings": [],
                "assessment": (
                    "No text could be extracted from this file. If it's a scan "
                    "image (X-ray/CT/MRI), attach it as an image instead, or "
                    "paste the report text directly."
                ),
                "recommendation": (
                    "NanoLM lattice-matched reference criteria (not a medical "
                    "device). A licensed clinician must confirm before any "
                    "clinical use."
                ),
            },
        }

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


@app.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    instruction: str = Form(""),
):
    """Medical image diagnosis: image → vision-encoder zero-shot classification."""
    client = await get_client()
    files = {"file": (file.filename, await file.read(), file.content_type or "application/octet-stream")}
    try:
        r = await client.post(f"{MEDICAL_DIAGNOSTIC_URL}/classify?top_k=10", files=files)
    except httpx.TimeoutException:
        return JSONResponse(
            content={"error": "Vision model is still loading (first run). Please retry in a minute."},
            status_code=504,
        )
    except httpx.HTTPError:
        raise HTTPException(503, "medical-diagnostic service unreachable")
    if r.status_code != 200:
        return JSONResponse(content={"error": f"vision classification failed: {r.text[:300]}"}, status_code=502)

    result = r.json()
    raw_matches = result.get("matches", [])
    modality = _infer_modality(instruction)
    # Zero-shot CLIP/BiomedCLIP scores are not calibrated; require a minimum
    # similarity so blank/irrelevant images don't surface spurious findings.
    MIN_SCORE = float(os.environ.get("VISION_MIN_SCORE", "0.22"))
    matches = [
        {"signal": m.get("signal"), "note": m.get("note"), "severity": m.get("severity"), "bss": m.get("score")}
        for m in raw_matches
        if (m.get("score") or 0.0) >= MIN_SCORE
    ]
    # "clear/normal" labels aren't findings — treat them as a normal-study signal.
    normals = [m for m in matches if m["severity"] == "normal"]
    findings = [m for m in matches if m["severity"] != "normal"]
    criticals = [f for f in findings if f["severity"] == "critical"]
    if criticals:
        assessment = f"{len(criticals)} critical finding(s) — urgent clinician review required."
    elif findings:
        assessment = f"{len(findings)} possible finding(s) — clinician review advised."
    elif normals:
        top_norm = f"{normals[0]['signal'].rsplit('.', 1)[-1]} {normals[0]['bss']:.2f}"
        assessment = f"No acute finding — closest match '{top_norm}' (low confidence)."
    else:
        top = raw_matches[0] if raw_matches else None
        if top and (top.get("score") or 0.0) >= 0.15:
            label = top["signal"].rsplit(".", 1)[-1]
            assessment = (
                f"Closest match: {label} (confidence {top['score']:.2f}, low). "
                "Zero-shot image classification is approximate — paste the "
                "report text for reliable analysis."
            )
        else:
            assessment = "No recognizable finding in this image."

    report = {
        "report_type": "nanolm-vision",
        "engine": "vision-encoder",
        "modality": modality,
        "findings": findings,
        "assessment": assessment,
        "recommendation": (
            "Vision-encoder zero-shot retrieval (not a medical device). "
            "A licensed clinician must confirm before any clinical use."
        ),
    }
    return {
        "filename": file.filename,
        "modality": modality,
        "instruction": instruction,
        "report": report,
    }


@app.post("/analyze/knee")
async def analyze_knee(
    file: UploadFile = File(...),
    instruction: str = Form(""),
):
    """Knee MRI diagnosis: lattice fingerprint classifier, with a BiomedCLIP
    zero-shot fallback for real scans that don't match the synthetic library.

    NOTE: the synthetic-trained CNN (knee_cnn.py, /classify/knee/deep) is NOT
    used here — softmax on out-of-distribution (real) scans is over-confident
    (it mislabels a real knee MRI as "patellar dislocation" at 1.0). The CNN
    is a CPU-only training/retraining pipeline for a real labeled dataset."""
    client = await get_client()
    files = {"file": (file.filename, await file.read(), file.content_type or "application/octet-stream")}
    try:
        r = await client.post(f"{MEDICAL_DIAGNOSTIC_URL}/classify/knee", files=files)
    except httpx.HTTPError:
        raise HTTPException(503, "medical-diagnostic service unreachable")
    if r.status_code != 200:
        return JSONResponse(content={"error": f"knee classification failed: {r.text[:300]}"}, status_code=502)

    result = r.json()
    detected = result.get("detected") or []
    top_label = result.get("top_label") or "Normal"
    top_conf = result.get("confidence") or 0.0

    if detected:
        findings = [
            {"signal": s["abnormality"], "severity": s["severity"],
             "bss": s["confidence"],
             "note": f"{s['label']} — confidence {s['confidence']:.2f}."}
            for s in detected[:3]
        ]
        assessment = f"Top finding: {top_label} (confidence {top_conf:.2f})."
        engine = "hllset-knee"
    else:
        # Fingerprint found nothing (real scans rarely match the synthetic
        # library). Fall back to BiomedCLIP zero-shot over the knee labels.
        engine = "vision-encoder"
        findings = []
        assessment = (
            "No knee abnormality matched the reference library. If this is a "
            "different study, paste the report text for reliable analysis."
        )
        try:
            r2 = await client.post(f"{MEDICAL_DIAGNOSTIC_URL}/classify?top_k=10", files=files)
            if r2.status_code == 200:
                raw_matches = r2.json().get("matches", [])
                knee_matches = [m for m in raw_matches if m["signal"].startswith("knee.")]
                MIN_SCORE = float(os.environ.get("VISION_MIN_SCORE", "0.22"))
                findings = [
                    {"signal": m["signal"], "severity": m["severity"], "bss": m["score"], "note": m["note"]}
                    for m in knee_matches
                    if (m.get("score") or 0.0) >= MIN_SCORE
                ][:3]
                if findings:
                    assessment = (
                        f"{len(findings)} possible knee finding(s) — low-confidence "
                        "zero-shot (the specific finding can't be reliably "
                        "distinguished from the image); paste the report text for "
                        "reliable analysis."
                    )
        except httpx.HTTPError:
            pass

    report = {
        "report_type": "nanolm-knee",
        "engine": engine,
        "modality": "MR",
        "findings": findings,
        "assessment": assessment,
        "recommendation": (
            "Lattice fingerprint classification (not a medical device). "
            "A licensed clinician must confirm before any clinical use."
        ),
    }
    return {
        "filename": file.filename,
        "modality": "MR",
        "instruction": instruction,
        "report": report,
    }


# ═══════════════════════════════════════════════════════════════════════
# NLP Chat (NanoLM English model)
# ═══════════════════════════════════════════════════════════════════════

@app.post("/chat")
async def chat(request: Request):
    """Forward a chat message to the NanoLM English NLP model."""
    body = await request.json()
    client = await get_client()
    try:
        r = await client.post(f"{NLP_MODEL_URL}/chat", json=body)
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        raise HTTPException(503, "nlp-model service unreachable")


@app.post("/session/new")
async def session_new():
    """Create a new chat session."""
    client = await get_client()
    try:
        r = await client.post(f"{NLP_MODEL_URL}/session/new")
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        raise HTTPException(503, "nlp-model service unreachable")


@app.get("/sessions")
async def list_sessions():
    """List chat sessions."""
    client = await get_client()
    try:
        r = await client.get(f"{NLP_MODEL_URL}/sessions")
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        raise HTTPException(503, "nlp-model service unreachable")


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get a single chat session's history."""
    client = await get_client()
    try:
        r = await client.get(f"{NLP_MODEL_URL}/session/{session_id}")
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        raise HTTPException(503, "nlp-model service unreachable")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    client = await get_client()
    try:
        r = await client.delete(f"{NLP_MODEL_URL}/session/{session_id}")
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        raise HTTPException(503, "nlp-model service unreachable")


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
    port = int(os.environ.get("PORT", 8001))
    logger.info(f"ewm-gateway starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=LOG_LEVEL.lower())
