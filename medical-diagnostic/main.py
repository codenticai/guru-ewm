"""
medical-diagnostic — Clinical Diagnosis Service (EWM backend).

A separate EWM container that interprets OCR'd clinical text, medical images,
X-ray/ECG, and DICOM files into a structured diagnostic report.

Pipeline in Guru-EWM:
    Document/Image/DICOM → deepseek-ocr (:9093) → text/markdown
      → medical-diagnostic (:9094) → structured diagnosis
        → ipfs (:5001) → content-addressed storage

Capabilities:
  - Text diagnosis        — clinical LLM over OCR'd text (pluggable)
  - Image diagnosis       — medical VLM over X-ray/ECG/medical images (pluggable)
  - DICOM processing      — pydicom metadata extraction + pixel→PNG conversion
  - Modality awareness    — CR/DX (X-ray), ECG, CT, MR, US, etc.

Design:
  - Pluggable engine: swap in a real clinical LLM / medical VLM without
    changing the HTTP API.
  - The default `SimulatedDiagnosticEngine` exists ONLY to validate the
    architecture end-to-end. It is NOT a medical device.

Endpoints:
  GET  /health             — service + engine status
  GET  /engines            — list available diagnostic engines
  GET  /modalities         — list supported DICOM modalities
  POST /diagnose           — combined: OCR text and/or image/DICOM reference
  POST /diagnose/text      — text-based diagnosis (from OCR'd clinical text)
  POST /diagnose/image     — image/DICOM diagnosis (X-ray/ECG/etc.)
  POST /dicom/upload       — upload a DICOM file → metadata + PNG (to IPFS)
  POST /dicom/extract      — extract metadata from a stored DICOM (by CID)
"""

import asyncio
import io
import json
import os
import logging
import re
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
import httpx

try:
    from ecg_cards import ECG_CARDS
except ImportError:  # pragma: no cover
    ECG_CARDS = []

# ── Config ──────────────────────────────────────────────────────────
IPFS_API_URL = os.environ.get("IPFS_API_URL", "http://ipfs:5001")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
# Set to a real clinical model path / engine name when available.
DIAGNOSTIC_ENGINE = os.environ.get("DIAGNOSTIC_ENGINE", "simulated")
# hllset-next lattice (knowledge-card ingestion + BSS inclusion query)
HLLSET_NEXT_URL = os.environ.get("HLLSET_NEXT_URL", "http://hllset-next:9090")
# Seed/backup: knowledge-corpus snapshot stored in IPFS for restore-on-restart
KB_SNAPSHOT_CID = os.environ.get("KB_SNAPSHOT_CID", "")
KB_SNAPSHOT_FILE = os.environ.get("KB_SNAPSHOT_FILE", "/app/data/kb_snapshot.json")

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("medical-diagnostic")

app = FastAPI(
    title="Medical Diagnostic Service",
    description="Clinical diagnosis over OCR'd text, medical images, X-ray/ECG and DICOM (EWM backend)",
    version="0.2.0",
)

_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    return _client


# ═══════════════════════════════════════════════════════════════════════
# DICOM Processing (pydicom)
# ═══════════════════════════════════════════════════════════════════════

try:
    import pydicom
    import numpy as np
    from PIL import Image as PILImage

    DICOM_AVAILABLE = True
except ImportError as e:  # pragma: no cover
    logger.warning(f"DICOM support unavailable: {e}")
    DICOM_AVAILABLE = False

# Optional biosignal / neuroimaging / interoperability support
try:
    import pyedflib  # noqa: F401

    EDF_AVAILABLE = True
except ImportError:  # pragma: no cover
    EDF_AVAILABLE = False

try:
    import nibabel  # noqa: F401

    NIFTI_AVAILABLE = True
except ImportError:  # pragma: no cover
    NIFTI_AVAILABLE = False

# DICOM modality → clinical domain mapping
MODALITY_MAP = {
    "CR": {"domain": "radiology", "label": "Computed Radiography (X-ray)"},
    "DX": {"domain": "radiology", "label": "Digital Radiography (X-ray)"},
    "ECG": {"domain": "cardiology", "label": "Electrocardiogram"},
    "CT": {"domain": "radiology", "label": "Computed Tomography"},
    "MR": {"domain": "radiology", "label": "Magnetic Resonance"},
    "US": {"domain": "radiology", "label": "Ultrasound"},
    "MG": {"domain": "radiology", "label": "Mammography"},
    "XA": {"domain": "radiology", "label": "X-Ray Angiography"},
    "PT": {"domain": "radiology", "label": "Positron Emission Tomography"},
    "NM": {"domain": "radiology", "label": "Nuclear Medicine"},
    "RF": {"domain": "radiology", "label": "Radiofluoroscopy"},
    "ES": {"domain": "gastroenterology", "label": "Endoscopy"},
    "OCT": {"domain": "ophthalmology", "label": "Optical Coherence Tomography"},
    "OP": {"domain": "ophthalmology", "label": "Ophthalmic Photography"},
    "SM": {"domain": "pathology", "label": "Slide Microscopy (whole-slide)"},
    "GM": {"domain": "pathology", "label": "General Microscopy"},
    "IO": {"domain": "dental", "label": "Intra-oral Radiography"},
    "IVUS": {"domain": "cardiology", "label": "Intravascular Ultrasound"},
    "EPS": {"domain": "cardiology", "label": "Cardiac Electrophysiology"},
    "HD": {"domain": "cardiology", "label": "Hemodynamic Waveform"},
    "SEG": {"domain": "radiology", "label": "Segmentation"},
    "SR": {"domain": "radiology", "label": "Structured Report"},
    "RTSTRUCT": {"domain": "oncology", "label": "Radiotherapy Structure Set"},
    "RTPLAN": {"domain": "oncology", "label": "Radiotherapy Plan"},
    "RTDOSE": {"domain": "oncology", "label": "Radiotherapy Dose"},
    "OT": {"domain": "other", "label": "Other"},
}


def _dicom_to_png(ds) -> Optional[bytes]:
    """Convert DICOM pixel data to an 8-bit PNG (for viewing / IPFS)."""
    if not hasattr(ds, "pixel_array"):
        return None
    try:
        arr = ds.pixel_array.astype(np.float32)
        arr = arr - float(arr.min())
        if float(arr.max()) > 0:
            arr = arr / float(arr.max()) * 255.0
        arr = arr.astype(np.uint8)
        img = PILImage.fromarray(arr).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Pixel→PNG conversion failed: {e}")
        return None


def _dicom_waveform_summary(ds) -> Optional[dict]:
    """Extract ECG waveform summary if the DICOM contains waveform data."""
    try:
        if not hasattr(ds, "waveform_array"):
            return None
        waves = ds.waveform_array(0)
        data = np.asarray(waves)
        return {
            "channels": int(data.shape[0]) if data.ndim > 1 else 1,
            "samples": int(data.shape[-1]),
            "waveform_min": float(np.min(data)),
            "waveform_max": float(np.max(data)),
            "waveform_mean": float(np.mean(data)),
        }
    except Exception:
        return None


def parse_dicom(dicom_bytes: bytes) -> dict:
    """Parse a DICOM file → metadata + PNG bytes (if pixel data present)."""
    if not DICOM_AVAILABLE:
        return {"error": "DICOM support not installed (pydicom)"}

    ds = pydicom.dcmread(io.BytesIO(dicom_bytes))

    def tag(name: str):
        """Safely extract a DICOM tag, returning None if missing/invalid."""
        try:
            v = getattr(ds, name, None)
            if v is None:
                return None
            if isinstance(v, pydicom.valuerep.PersonName):
                return str(v)
            if hasattr(v, "item"):  # avoid nested Sequence objects
                return None
            return str(v) if not isinstance(v, (int, float)) else v
        except Exception:
            return None

    modality = tag("Modality") or "OT"
    mod_info = MODALITY_MAP.get(modality, {"domain": "other", "label": f"Modality {modality}"})

    png_bytes = _dicom_to_png(ds)
    waveform = _dicom_waveform_summary(ds)

    metadata = {
        "modality": modality,
        "modality_label": mod_info["label"],
        "domain": mod_info["domain"],
        "patient_id": tag("PatientID"),
        "patient_name": tag("PatientName"),
        "patient_birth_date": tag("PatientBirthDate"),
        "patient_sex": tag("PatientSex"),
        "study_description": tag("StudyDescription"),
        "series_description": tag("SeriesDescription"),
        "body_part": tag("BodyPartExamined"),
        "study_date": tag("StudyDate"),
        "study_time": tag("StudyTime"),
        "institution_name": tag("InstitutionName"),
        "manufacturer": tag("Manufacturer"),
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
        "samples_per_pixel": getattr(ds, "SamplesPerPixel", None),
        "photometric_interpretation": tag("PhotometricInterpretation"),
        "has_pixel_data": hasattr(ds, "pixel_array"),
        "waveform": waveform,
    }

    return {
        "metadata": metadata,
        "png_bytes": png_bytes,
    }


# ═══════════════════════════════════════════════════════════════════════
# Biosignal / Neuroimaging / Interoperability Parsers
# ═══════════════════════════════════════════════════════════════════════

def parse_edf(data: bytes) -> dict:
    """Parse an EDF/EDF+ biosignal file (EEG, sleep study, ECG)."""
    if not EDF_AVAILABLE:
        return {"error": "EDF support not installed (pyEDFlib)"}
    import tempfile
    import os

    tmp = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
    tmp.write(data)
    tmp.close()
    reader = None
    try:
        reader = pyedflib.EdfReader(tmp.name)
        channels = []
        for i in range(reader.signals_in_file):
            try:
                channels.append({
                    "label": reader.getLabel(i),
                    "dimension": reader.getPhysicalDimension(i),
                    "sample_frequency": float(reader.getSampleFrequency(i)),
                })
            except Exception:
                channels.append({"label": f"ch{i}", "error": "unreadable"})

        signal = reader.readSignal(0)
        return {
            "type": "edf",
            "channels": reader.signals_in_file,
            "channel_info": channels,
            "signal_min": float(np.min(signal)),
            "signal_max": float(np.max(signal)),
            "signal_mean": float(np.mean(signal)),
            "samples": int(len(signal)),
        }
    except Exception as e:
        return {"error": f"EDF parse failed: {e}"}
    finally:
        try:
            if reader is not None:
                reader.close()
        except Exception:
            pass
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def parse_scp_ecg(data: bytes) -> dict:
    """Best-effort SCP-ECG binary format detection and header parse.

    SCP-ECG is a binary standard for ECG exchange. Full parsing is complex;
    here we detect the magic and report basic structure.
    """
    try:
        has_magic = b"SCPECG" in data[:64]
        # First two bytes are a CRC-16; the version is in the section-0 header
        # (pointer sections). Best-effort: report size + magic presence.
        return {
            "type": "scp_ecg",
            "detected": has_magic,
            "size_bytes": len(data),
            "note": (
                "SCP-ECG detected (best-effort). Full waveform decoding "
                "requires a dedicated SCP-ECG library."
            ),
        }
    except Exception as e:
        return {"error": f"SCP-ECG parse failed: {e}"}


def parse_fhir(data: dict) -> dict:
    """Parse a FHIR resource (JSON) into a structured clinical summary.

    Handles the most common resource types: Patient, Observation,
    DiagnosticReport, Condition, MedicationRequest.
    """
    rt = data.get("resourceType", "Unknown")
    result: dict = {"resource_type": rt}

    if "id" in data:
        result["id"] = data["id"]

    if rt == "Patient":
        names = data.get("name") or [{}]
        result["patient_name"] = names[0].get("text") if names else None
        result["gender"] = data.get("gender")
        result["birth_date"] = data.get("birthDate")

    elif rt == "Observation":
        code = data.get("code", {})
        result["observation"] = (
            code.get("text")
            or (code.get("coding") or [{}])[0].get("display")
            or (code.get("coding") or [{}])[0].get("code")
        )
        result["status"] = data.get("status")
        value = data.get("valueQuantity")
        if value:
            result["value"] = {"value": value.get("value"), "unit": value.get("unit")}
        else:
            result["value"] = data.get("valueString", data.get("valueCodeableConcept"))

    elif rt == "DiagnosticReport":
        result["status"] = data.get("status")
        result["conclusion"] = data.get("conclusion")
        code = data.get("code", {})
        result["report_type"] = code.get("text") or (code.get("coding") or [{}])[0].get("display")

    elif rt == "Condition":
        code = data.get("code", {})
        result["condition"] = (
            code.get("text")
            or (code.get("coding") or [{}])[0].get("display")
            or (code.get("coding") or [{}])[0].get("code")
        )
        cs = data.get("clinicalStatus", {}).get("coding") or [{}]
        result["clinical_status"] = cs[0].get("code") if cs else None

    elif rt == "MedicationRequest":
        med = data.get("medicationCodeableConcept", {})
        result["medication"] = med.get("text") or (med.get("coding") or [{}])[0].get("display")

    return result


def parse_hl7(message: str) -> dict:
    """Parse an HL7 v2 message (pipe/caret delimited) into segments."""
    segments: dict = {}
    try:
        for line in message.strip().replace("\n", "\r").split("\r"):
            line = line.strip()
            if not line:
                continue
            fields = line.split("|")
            seg_name = fields[0]
            segments[seg_name] = [f.split("^") for f in fields]

        msh = segments.get("MSH")
        message_type = None
        if msh and len(msh) > 9:
            parts = msh[9]
            message_type = parts[1] if len(parts) > 1 else None

        return {
            "type": "hl7_v2",
            "message_type": message_type,
            "segments": {k: v for k, v in segments.items()},
        }
    except Exception as e:
        return {"error": f"HL7 parse failed: {e}"}


def parse_nifti(data: bytes) -> dict:
    """Parse a NIfTI neuroimaging file (.nii / .nii.gz) → volume summary."""
    if not NIFTI_AVAILABLE:
        return {"error": "NIfTI support not installed (nibabel)"}
    import tempfile
    import os

    suffix = ".nii.gz" if data[:2] == b"\x1f\x8b" else ".nii"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()
    try:
        img = nibabel.load(tmp.name)
        arr = img.get_fdata()
        return {
            "type": "nifti",
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "voxel_min": float(np.min(arr)),
            "voxel_max": float(np.max(arr)),
            "voxel_mean": float(np.mean(arr)),
            "affine": img.affine.tolist() if hasattr(img.affine, "tolist") else None,
        }
    except Exception as e:
        return {"error": f"NIfTI parse failed: {e}"}
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
# Pluggable Diagnostic Engine
# ═══════════════════════════════════════════════════════════════════════

class DiagnosticEngine:
    """Base interface for a diagnostic engine.

    Subclass this to plug in a real clinical model:
      - TextDiagnosticEngine  → clinical LLM over OCR'd text
      - ImageDiagnosticEngine → medical VLM over X-ray/DICOM/ECG
    """

    name: str = "base"
    kind: str = "base"  # "text" | "image" | "multimodal"
    available: bool = False

    def diagnose_text(self, text: str, meta: Optional[dict] = None) -> dict:
        raise NotImplementedError

    def diagnose_image(self, image_ref: str, meta: Optional[dict] = None) -> dict:
        raise NotImplementedError


class SimulatedDiagnosticEngine(DiagnosticEngine):
    """Simulation engine for architecture validation ONLY.

    Produces a deterministic, structured diagnostic report from the input
    text using simple keyword heuristics. This is NOT a medical device —
    a real clinical model must replace it before any real-world use.
    """

    name = "simulated"
    kind = "multimodal"
    available = True

    # Curated heuristic markers — architecture validation only.
    _MARKERS = {
        "cardiology": [
            ("chest pain", "Possible cardiac origin — recommend ECG review"),
            ("shortness of breath", "Respiratory/cardiac differential"),
            ("palpitation", "Arrhythmia screening advised"),
            ("hypertension", "Blood pressure management review"),
            ("st elevation", "ST elevation — possible ischemia/infarction"),
            ("st depression", "ST depression — possible ischemia"),
            ("atrial fibrillation", "Atrial fibrillation detected"),
            ("bundle branch block", "Bundle branch block present"),
            ("tachycardia", "Tachycardia (elevated heart rate)"),
            ("bradycardia", "Bradycardia (reduced heart rate)"),
            ("qt prolongation", "Prolonged QT interval — TdP risk"),
            ("ventricular", "Ventricular abnormality noted"),
            ("sinus rhythm", "Normal sinus rhythm"),
            ("normal ecg", "Normal ECG — no acute abnormality"),
            ("ecg", "ECG trace analyzed"),
        ],
        "radiology": [
            ("fracture", "Possible fracture — confirm via imaging"),
            ("opacity", "Pulmonary opacity — correlate clinically"),
            ("lesion", "Lesion noted — further characterization advised"),
            ("x-ray", "X-ray referenced — interpretation out of scope"),
        ],
        "laboratory": [
            ("elevated", "Abnormal value flagged for review"),
            ("abnormal", "Abnormal result — correlate clinically"),
            ("hemoglobin", "Hematological parameter noted"),
        ],
    }

    def diagnose_text(self, text: str, meta: Optional[dict] = None) -> dict:
        findings: list[dict] = []
        lowered = text.lower()

        for domain, markers in self._MARKERS.items():
            for keyword, note in markers:
                if keyword in lowered:
                    findings.append({
                        "domain": domain,
                        "signal": keyword,
                        "note": note,
                    })

        # Extract structured ECG measurements from the report text
        measurements = self._extract_ecg_measurements(text)

        return {
            "engine": self.name,
            "input_type": "text",
            "status": "completed",
            "measurements": measurements,
            "findings": findings,
            "summary": (
                f"{len(findings)} heuristic finding(s) identified "
                "(simulated — not a medical device)"
            ),
            "recommendation": (
                "Simulated output. A licensed clinical model and clinician "
                "review are required before any real use."
            ),
        }

    @staticmethod
    def _extract_ecg_measurements(text: str) -> dict:
        """Extract numeric ECG parameters (rate, intervals, axes) via regex."""
        import re
        measures: dict = {}

        def grab(pattern, key):
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    measures[key] = float(m.group(1))
                except (ValueError, IndexError):
                    pass

        grab(r"vent(?:\.)?\s*rate\s*(\d+)", "ventricular_rate_bpm")
        grab(r"heart\s*rate\s*(\d+)", "heart_rate_bpm")
        grab(r"pr\s*interval\s*(\d+)", "pr_interval_ms")
        grab(r"qrs\s*duration\s*(\d+)", "qrs_duration_ms")
        grab(r"qtc?\s*interval\s*(\d+)", "qt_interval_ms")
        grab(r"qt/qtc\s*interval\s*(\d+)/(\d+)", "qtc_ms")

        # QT/QTc split value (e.g., 366/378)
        m = re.search(r"qt/?qtc?\s*interval\s*(\d+)\s*/\s*(\d+)", text, re.IGNORECASE)
        if m:
            try:
                measures["qt_ms"] = float(m.group(1))
                measures["qtc_ms"] = float(m.group(2))
            except ValueError:
                pass

        # P/QRS/T axes (e.g., 57/72/60)
        m = re.search(r"p/?qrs/?t\s*axes\s*([\d-]+)\s*/\s*([\d-]+)\s*/\s*([\d-]+)", text, re.IGNORECASE)
        if m:
            try:
                measures["p_axis_deg"] = float(m.group(1))
                measures["qrs_axis_deg"] = float(m.group(2))
                measures["t_axis_deg"] = float(m.group(3))
            except ValueError:
                pass

        return measures

    def diagnose_image(self, image_ref: str, meta: Optional[dict] = None) -> dict:
        """Modality-aware image/DICOM diagnosis (metadata heuristics only)."""
        meta = meta or {}
        modality = meta.get("modality", "OT")
        mod_info = MODALITY_MAP.get(modality, {"domain": "other", "label": f"Modality {modality}"})
        domain = mod_info["domain"]
        label = mod_info["label"]

        findings: list[dict] = [
            {"domain": domain, "signal": "modality",
             "note": f"{label} acquired — interpretation out of scope (simulated)"},
        ]

        if domain == "cardiology":
            if meta.get("waveform"):
                w = meta["waveform"]
                findings.append({
                    "domain": "cardiology",
                    "signal": "ecg_signal_stats",
                    "note": (
                        f"{w.get('channels', '?')} channel(s), "
                        f"{w.get('samples', '?')} samples, "
                        f"range [{w.get('waveform_min')}, {w.get('waveform_max')}]"
                    ),
                })
        elif domain == "radiology":
            if meta.get("body_part"):
                findings.append({
                    "domain": "radiology",
                    "signal": "body_part",
                    "note": f"Body part examined: {meta['body_part']}",
                })
        # Other domains (gastroenterology, ophthalmology, pathology, dental,
        # oncology) fall through with the generic modality finding only.

        return {
            "engine": self.name,
            "input_type": "image",
            "status": "completed",
            "modality": modality,
            "modality_label": label,
            "domain": domain,
            "findings": findings,
            "summary": (
                f"Modality-aware report for {label} (simulated — metadata only, "
                "not pixel interpretation)"
            ),
            "recommendation": "Plug in a medical vision model for pixel-level diagnosis.",
            "image_ref": image_ref,
        }


# Engine registry — add real engines here as they become available.
_ENGINES: dict[str, DiagnosticEngine] = {
    "simulated": SimulatedDiagnosticEngine(),
}

_ACTIVE_ENGINE = _ENGINES.get(DIAGNOSTIC_ENGINE, _ENGINES["simulated"])


# ═══════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "medical-diagnostic",
        "engine": {
            "name": _ACTIVE_ENGINE.name,
            "kind": _ACTIVE_ENGINE.kind,
            "available": _ACTIVE_ENGINE.available,
        },
        "modules": {
            "dicom": DICOM_AVAILABLE,
            "edf": EDF_AVAILABLE,
            "nifti": NIFTI_AVAILABLE,
            "fhir": True,   # native JSON parsing
            "hl7_v2": True,  # native segment parsing
            "scp_ecg": True,  # best-effort detection
        },
        "ipfs_api_url": IPFS_API_URL,
        "disclaimer": "Simulated engine — not a medical device",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/engines")
async def list_engines():
    return {
        "active": _ACTIVE_ENGINE.name,
        "available": [
            {"name": e.name, "kind": e.kind, "available": e.available}
            for e in _ENGINES.values()
        ],
    }


@app.get("/modalities")
async def list_modalities():
    return {
        "supported": [
            {"code": k, "domain": v["domain"], "label": v["label"]}
            for k, v in MODALITY_MAP.items()
        ],
    }


@app.post("/diagnose")
async def diagnose(request: Request):
    """Combined endpoint: accept OCR text and/or an image reference."""
    body = await request.json()
    text = body.get("text", "")
    image_ref = body.get("image_ref", body.get("image_cid", ""))
    meta = body.get("meta", {})

    if not text and not image_ref:
        return JSONResponse(
            {"error": "Provide 'text' (OCR output) and/or 'image_ref'"},
            status_code=400,
        )

    result: dict = {"engine": _ACTIVE_ENGINE.name, "timestamp": datetime.utcnow().isoformat()}

    if text:
        result["text_diagnosis"] = _ACTIVE_ENGINE.diagnose_text(text, meta)
    if image_ref:
        result["image_diagnosis"] = _ACTIVE_ENGINE.diagnose_image(image_ref, meta)

    # Persist result to IPFS (content-addressed) if configured
    cid = await _store_to_ipfs(result)
    if cid:
        result["stored_cid"] = cid

    return result


@app.post("/diagnose/text")
async def diagnose_text(request: Request):
    """Text-based diagnosis from OCR'd clinical text."""
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return JSONResponse({"error": "Missing 'text'"}, status_code=400)

    result = _ACTIVE_ENGINE.diagnose_text(text, body.get("meta", {}))
    result["timestamp"] = datetime.utcnow().isoformat()

    cid = await _store_to_ipfs(result)
    if cid:
        result["stored_cid"] = cid

    return result


@app.post("/analyze/report")
async def analyze_report(request: Request):
    """Produce a structured final analysis report from OCR'd clinical text.

    Accepts:
      - text:      OCR-extracted clinical text (from deepseek-ocr)
      - modality:  optional modality hint (e.g., "ECG", "DX")
      - patient:   optional patient context dict

    Returns a formal report with sections: patient, findings, assessment,
    recommendation, and metadata.
    """
    body = await request.json()
    text = body.get("text", body.get("page_text", ""))
    modality = body.get("modality", "DOC")
    patient = body.get("patient", {})

    if not text:
        return JSONResponse({"error": "Missing 'text'"}, status_code=400)

    diagnosis = _ACTIVE_ENGINE.diagnose_text(text, body.get("meta", {}))

    report = {
        "report_type": "medical-analysis",
        "modality": modality,
        "generated_at": datetime.utcnow().isoformat(),
        "engine": _ACTIVE_ENGINE.name,
        "patient": {
            "id": patient.get("id", patient.get("patient_id")),
            "name": patient.get("name", patient.get("patient_name")),
            "sex": patient.get("sex"),
            "age": patient.get("age"),
            "birth_date": patient.get("birth_date"),
        },
        "measurements": diagnosis.get("measurements", {}),
        "findings": diagnosis.get("findings", []),
        "assessment": diagnosis.get("summary", ""),
        "recommendation": diagnosis.get("recommendation", ""),
        "source_text_preview": text[:500],
    }

    cid = await _store_to_ipfs(report)
    if cid:
        report["stored_cid"] = cid

    return report


@app.post("/diagnose/image")
async def diagnose_image(request: Request):
    """Image/DICOM diagnosis (X-ray/ECG/etc.) — modality-aware."""
    body = await request.json()
    image_ref = body.get("image_ref", body.get("image_cid", ""))
    meta = body.get("meta", {})

    if not image_ref:
        return JSONResponse({"error": "Missing 'image_ref'"}, status_code=400)

    result = _ACTIVE_ENGINE.diagnose_image(image_ref, meta)
    result["timestamp"] = datetime.utcnow().isoformat()

    cid = await _store_to_ipfs(result)
    if cid:
        result["stored_cid"] = cid

    return result


# ═══════════════════════════════════════════════════════════════════════
# DICOM endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.post("/dicom/upload")
async def dicom_upload(file: UploadFile = File(...)):
    """Upload a DICOM file → extract metadata + convert pixels → PNG (to IPFS)."""
    if not DICOM_AVAILABLE:
        return JSONResponse({"error": "DICOM support not installed"}, status_code=503)

    raw = await file.read()
    parsed = parse_dicom(raw)

    if "error" in parsed:
        return JSONResponse(parsed, status_code=422)

    result = {
        "filename": file.filename,
        "size_bytes": len(raw),
        "metadata": parsed["metadata"],
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Store the original DICOM and the converted PNG to IPFS
    dicom_cid = await _store_bytes_to_ipfs(raw, f"{file.filename or 'study'}.dcm")
    if dicom_cid:
        result["dicom_cid"] = dicom_cid

    if parsed.get("png_bytes"):
        png_cid = await _store_bytes_to_ipfs(parsed["png_bytes"], f"{file.filename or 'study'}.png")
        if png_cid:
            result["png_cid"] = png_cid

    return result


@app.post("/dicom/extract")
async def dicom_extract(request: Request):
    """Extract metadata from a DICOM stored in IPFS (by CID)."""
    if not DICOM_AVAILABLE:
        return JSONResponse({"error": "DICOM support not installed"}, status_code=503)

    body = await request.json()
    cid = body.get("cid", "")
    if not cid:
        return JSONResponse({"error": "Missing 'cid'"}, status_code=400)

    dicom_bytes = await _fetch_bytes_from_ipfs(cid)
    if dicom_bytes is None:
        return JSONResponse({"error": f"DICOM not found for CID {cid}"}, status_code=404)

    parsed = parse_dicom(dicom_bytes)
    if "error" in parsed:
        return JSONResponse(parsed, status_code=422)

    result = {
        "cid": cid,
        "metadata": parsed["metadata"],
        "timestamp": datetime.utcnow().isoformat(),
    }
    if parsed.get("png_bytes"):
        png_cid = await _store_bytes_to_ipfs(parsed["png_bytes"], f"{cid}.png")
        if png_cid:
            result["png_cid"] = png_cid

    return result


# ═══════════════════════════════════════════════════════════════════════
# Biosignal endpoints (EDF / SCP-ECG / CSV)
# ═══════════════════════════════════════════════════════════════════════

@app.post("/biosignal/upload")
async def biosignal_upload(file: UploadFile = File(...)):
    """Upload a biosignal file (EDF/EDF+/SCP-ECG/CSV) → summary + IPFS."""
    raw = await file.read()
    name = (file.filename or "").lower()

    if name.endswith(".edf") or name.endswith(".edf+"):
        parsed = parse_edf(raw)
    elif name.endswith(".scp") or name.endswith(".ecg") or name.endswith(".dat"):
        parsed = parse_scp_ecg(raw)
    elif name.endswith(".csv"):
        parsed = _parse_csv_waveform(raw)
    else:
        return JSONResponse(
            {"error": "Unsupported biosignal format (use .edf, .scp, .ecg, .csv)"},
            status_code=415,
        )

    if "error" in parsed:
        return JSONResponse(parsed, status_code=422)

    cid = await _store_bytes_to_ipfs(raw, file.filename or "biosignal.bin")
    result = {"filename": file.filename, "size_bytes": len(raw), **parsed}
    if cid:
        result["cid"] = cid
    return result


def _parse_csv_waveform(raw: bytes) -> dict:
    """Parse a CSV waveform file (columnar samples)."""
    try:
        text = raw.decode("utf-8", errors="replace")
        rows = [r for r in text.splitlines() if r.strip()]
        header = rows[0].split(",")
        # Sample a few data rows for stats
        import statistics
        vals = []
        for row in rows[1:50]:
            parts = row.split(",")
            for p in parts[1:]:
                try:
                    vals.append(float(p))
                except ValueError:
                    pass
        return {
            "type": "csv",
            "columns": header,
            "rows": len(rows) - 1,
            "sample_min": min(vals) if vals else None,
            "sample_max": max(vals) if vals else None,
            "sample_mean": round(statistics.mean(vals), 4) if vals else None,
        }
    except Exception as e:
        return {"error": f"CSV parse failed: {e}"}


# ═══════════════════════════════════════════════════════════════════════
# Interoperability endpoints (FHIR / HL7)
# ═══════════════════════════════════════════════════════════════════════

@app.post("/fhir/ingest")
async def fhir_ingest(request: Request):
    """Ingest a FHIR resource (JSON) → structured clinical summary."""
    body = await request.json()
    if isinstance(body, list):
        resources = [parse_fhir(r) for r in body]
        result = {"resources": resources, "count": len(resources)}
    else:
        result = parse_fhir(body)

    cid = await _store_to_ipfs(result)
    if cid:
        result["stored_cid"] = cid
    return result


@app.post("/hl7/ingest")
async def hl7_ingest(request: Request):
    """Ingest an HL7 v2 message → parsed segments."""
    body = await request.json()
    message = body.get("message", body.get("text", ""))
    if not message:
        return JSONResponse({"error": "Missing 'message'"}, status_code=400)

    result = parse_hl7(message)
    if "error" in result:
        return JSONResponse(result, status_code=422)

    cid = await _store_to_ipfs(result)
    if cid:
        result["stored_cid"] = cid
    return result


# ═══════════════════════════════════════════════════════════════════════
# Neuroimaging endpoint (NIfTI)
# ═══════════════════════════════════════════════════════════════════════

@app.post("/nifti/upload")
async def nifti_upload(file: UploadFile = File(...)):
    """Upload a NIfTI neuroimaging file (.nii / .nii.gz) → volume summary."""
    if not NIFTI_AVAILABLE:
        return JSONResponse({"error": "NIfTI support not installed (nibabel)"}, status_code=503)

    raw = await file.read()
    parsed = parse_nifti(raw)
    if "error" in parsed:
        return JSONResponse(parsed, status_code=422)

    cid = await _store_bytes_to_ipfs(raw, file.filename or "volume.nii")
    result = {"filename": file.filename, "size_bytes": len(raw), **parsed}
    if cid:
        result["cid"] = cid
    return result


# ═══════════════════════════════════════════════════════════════════════
# HLLSet lattice — knowledge-card ingestion + BSS inclusion query
# ═══════════════════════════════════════════════════════════════════════

def hll_tokenize(text: str, ngram_min: int = 1, ngram_max: int = 2) -> list:
    """Replicate hllset-next's word-pattern tokenizer (lowercase + trim,
    unigrams + bigrams joined with NUL). Deterministic, no estimation."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    if not words:
        return []
    tokens = []
    for n in range(ngram_min, ngram_max + 1):
        for i in range(len(words) - n + 1):
            if n == 1:
                tokens.append(words[i])
            else:
                tokens.append("\x00".join(words[i:i + n]))
    return tokens


def _infer_modality(instruction: str) -> str:
    """Infer a modality hint from a free-text instruction (best effort)."""
    s = instruction.lower()
    if any(k in s for k in ("ecg", "ekg", "electrocardi")):
        return "ECG"
    if any(k in s for k in ("x-ray", "xray", "chest", "radiograph")):
        return "DX"
    if any(k in s for k in ("mri", "magnetic resonance")):
        return "MR"
    if any(k in s for k in ("ultrasound", "sonogram")):
        return "US"
    if " ct" in s or "computed tomography" in s or "ct scan" in s:
        return "CT"
    return ""


def classify_measurements(measurements: dict) -> list:
    """Derive interval findings from numeric measurements vs reference ranges.

    BSS inclusion over tokens cannot distinguish "short PR" (a finding) from
    "PR interval" (a measurement label) — so numeric values are classified
    deterministically against ECG_REFERENCE_RANGES.
    """
    findings = []

    rate = measurements.get("ventricular_rate_bpm")
    if rate is not None:
        if rate < 60:
            findings.append({"domain": "rate", "signal": "bradycardia", "severity": "abnormal",
                             "value": rate, "note": f"Heart rate {rate:.0f} bpm — bradycardia (<60)."})
        elif rate > 100:
            findings.append({"domain": "rate", "signal": "tachycardia", "severity": "abnormal",
                             "value": rate, "note": f"Heart rate {rate:.0f} bpm — tachycardia (>100)."})
        else:
            findings.append({"domain": "rate", "signal": "normal_rate", "severity": "normal",
                             "value": rate, "note": f"Heart rate {rate:.0f} bpm — within normal range."})

    pr = measurements.get("pr_interval_ms")
    if pr is not None:
        if pr > 200:
            findings.append({"domain": "interval", "signal": "prolonged_pr", "severity": "abnormal",
                             "value": pr, "note": f"PR {pr:.0f} ms — prolonged (>200)."})
        elif pr < 120:
            findings.append({"domain": "interval", "signal": "short_pr", "severity": "abnormal",
                             "value": pr, "note": f"PR {pr:.0f} ms — short (<120)."})
        else:
            findings.append({"domain": "interval", "signal": "normal_pr", "severity": "normal",
                             "value": pr, "note": f"PR {pr:.0f} ms — within normal range."})

    qrs = measurements.get("qrs_duration_ms")
    if qrs is not None:
        if qrs >= 120:
            findings.append({"domain": "interval", "signal": "wide_qrs", "severity": "abnormal",
                             "value": qrs, "note": f"QRS {qrs:.0f} ms — prolonged (>=120)."})
        else:
            findings.append({"domain": "interval", "signal": "normal_qrs", "severity": "normal",
                             "value": qrs, "note": f"QRS {qrs:.0f} ms — within normal range."})

    qtc = measurements.get("qtc_ms")
    if qtc is not None:
        if qtc > 450:
            findings.append({"domain": "interval", "signal": "prolonged_qtc", "severity": "critical",
                             "value": qtc, "note": f"QTc {qtc:.0f} ms — prolonged (>450)."})
        elif qtc < 350:
            findings.append({"domain": "interval", "signal": "short_qtc", "severity": "critical",
                             "value": qtc, "note": f"QTc {qtc:.0f} ms — short (<350)."})
        else:
            findings.append({"domain": "interval", "signal": "normal_qtc", "severity": "normal",
                             "value": qtc, "note": f"QTc {qtc:.0f} ms — within normal range."})

    return findings


class HllsetKnowledgeBase:
    """Ingest ECG knowledge cards into the hllset-next lattice and query
    incoming OCR'd clinical text by BSS (Bell State Similarity) inclusion.

    Each card text is tokenized → 4KB HLLSet → stored under a `c:ecg:<id>`
    key. A query tokenizes the report text and returns, for every card,
    BSS(query, card) = |query ∩ card| / |card| — the confidence that the
    report contains that diagnostic pattern.
    """

    def __init__(self, base_url: str, cards: list):
        self.base_url = base_url.rstrip("/")
        self.cards = cards
        self.registry: dict[str, dict] = {}  # card key → card metadata
        self.card_tokens: dict[str, set] = {}  # card key → token set (exact scoring)
        self.ingested = False

    def _key(self, card_id: str) -> str:
        return f"c:ecg:{card_id}"

    async def ingest(self) -> dict:
        """Ingest all cards into the lattice (idempotent — same key overwritten).

        The HLLSet lattice is populated via hllset-next for content-addressed
        storage. Token sets are also cached locally for exact, deterministic
        scoring (HLL cardinality is unstable for small knowledge cards).
        """
        client = await get_client()

        async def one(card: dict):
            key = self._key(card["id"])
            tokens = set(hll_tokenize(card["text"]))
            err = None
            try:
                r = await client.post(
                    f"{self.base_url}/api/v1/hllset/ingest",
                    json={"key": key, "text": card["text"]},
                )
                r.raise_for_status()
            except Exception as e:
                err = str(e)
            return key, card, tokens, err

        results = await asyncio.gather(*(one(c) for c in self.cards))
        ingested = 0
        errors = []
        for key, card, tokens, err in results:
            self.registry[key] = card
            self.card_tokens[key] = tokens
            if err:
                errors.append(f"{card['id']}: {err}")
            else:
                ingested += 1
        self.ingested = len(self.registry) == len(self.cards)
        return {"cards": len(self.cards), "ingested": ingested, "errors": errors}

    async def query(self, text: str, top_k: int = 12) -> dict:
        """Rank cards by exact token-set inclusion: |query ∩ card| / |card|.

        Deterministic — no HLL cardinality estimation, which is statistically
        unstable for small knowledge cards and produced false inclusions.
        """
        query_tokens = set(hll_tokenize(text))
        scored = []
        for key, card in self.registry.items():
            card_tokens = self.card_tokens.get(key)
            if not card_tokens:
                continue
            score = len(query_tokens & card_tokens) / len(card_tokens)
            scored.append({"key": key, "bss": score, "card": card})
        scored.sort(key=lambda m: m["bss"], reverse=True)
        return {"matches": scored[:top_k], "top_k": top_k}

    def status(self) -> dict:
        return {
            "base_url": self.base_url,
            "cards": len(self.cards),
            "ingested": len(self.registry),
            "ready": self.ingested,
        }


_kb: HllsetKnowledgeBase | None = None
_last_snapshot_cid: str = ""


def get_kb() -> HllsetKnowledgeBase:
    global _kb
    if _kb is None:
        _kb = HllsetKnowledgeBase(HLLSET_NEXT_URL, ECG_CARDS)
    return _kb


# ═══════════════════════════════════════════════════════════════════════
# Seed / backup — restore the knowledge lattice on restart
# ═══════════════════════════════════════════════════════════════════════

def _save_snapshot_file(cid: str) -> None:
    try:
        os.makedirs(os.path.dirname(KB_SNAPSHOT_FILE), exist_ok=True)
        with open(KB_SNAPSHOT_FILE, "w") as f:
            json.dump({
                "cid": cid,
                "count": len(get_kb().cards),
                "saved_at": datetime.utcnow().isoformat(),
            }, f)
    except Exception as e:  # pragma: no cover
        logger.warning(f"could not save snapshot file: {e}")


def _load_snapshot_file() -> str:
    try:
        with open(KB_SNAPSHOT_FILE) as f:
            return json.load(f).get("cid", "")
    except Exception:
        return KB_SNAPSHOT_CID


async def _snapshot_kb(kb: HllsetKnowledgeBase) -> str:
    """Back up the knowledge corpus to IPFS as a durable seed snapshot."""
    global _last_snapshot_cid
    payload = {
        "type": "knowledge-snapshot",
        "engine": "hllset-lattice",
        "count": len(kb.cards),
        "cards": kb.cards,
        "created_at": datetime.utcnow().isoformat(),
    }
    cid = await _store_to_ipfs(payload)
    if cid:
        _last_snapshot_cid = cid
        _save_snapshot_file(cid)
        logger.info(f"knowledge snapshot stored at IPFS CID {cid}")
    return cid or ""


@app.on_event("startup")
async def restore_knowledge_base():
    """Restore the knowledge lattice on startup.

    Re-ingests the seed corpus (idempotent — repopulates HLLSets and rebuilds
    the in-memory LUT), then writes a durable IPFS snapshot. If the built-in
    corpus is missing, restores cards from the last IPFS snapshot instead.
    """
    kb = get_kb()

    if not kb.cards:
        cid = _load_snapshot_file() or KB_SNAPSHOT_CID
        if cid:
            raw = await _fetch_bytes_from_ipfs(cid)
            if raw:
                try:
                    cards = json.loads(raw.decode("utf-8")).get("cards", [])
                    if cards:
                        kb.cards = cards
                        logger.info(f"restored {len(cards)} cards from snapshot {cid}")
                except Exception as e:
                    logger.warning(f"snapshot parse failed: {e}")
        if not kb.cards:
            logger.warning("no seed corpus and no snapshot — knowledge base empty")
            return

    for attempt in range(10):
        result = await kb.ingest()
        if result.get("ingested", 0) > 0:
            await _snapshot_kb(kb)
            logger.info(f"knowledge base seeded: {result['ingested']} cards")
            return
        logger.warning(
            f"seed ingest attempt {attempt + 1}: hllset-next not ready "
            f"({len(result.get('errors', []))} errors)"
        )
        await asyncio.sleep(3)

    logger.error("could not seed knowledge base after 10 attempts")


@app.post("/hllset/ingest")
async def hllset_ingest():
    """Ingest the ECG knowledge-card corpus into the hllset-next lattice."""
    return await get_kb().ingest()


@app.get("/hllset/status")
async def hllset_status():
    """Report lattice corpus status."""
    return {**get_kb().status(), "last_snapshot_cid": _last_snapshot_cid}


@app.post("/analyze/hllset")
async def analyze_hllset(request: Request):
    """Diagnose via HLLSet lattice: BSS inclusion over the knowledge corpus.

    Accepts:
      - text:           OCR-extracted clinical text
      - instruction:    optional free-text instruction (modality inferred)
      - modality:       optional modality hint (default ECG)
      - patient:        optional patient context dict
      - bss_threshold:  minimum inclusion confidence (default 0.50)
      - top_k:          max cards to rank (default 12)
    """
    body = await request.json()
    text = body.get("text", body.get("page_text", ""))
    modality = body.get("modality", "ECG")
    instruction = body.get("instruction", "")
    if instruction:
        inferred = _infer_modality(instruction)
        if inferred:
            modality = inferred
    patient = body.get("patient", {})
    threshold = float(body.get("bss_threshold", 0.50))
    top_k = int(body.get("top_k", 12))

    if not text:
        return JSONResponse({"error": "Missing 'text'"}, status_code=400)

    kb = get_kb()
    if not kb.ingested:
        await kb.ingest()

    try:
        result = await kb.query(text, top_k=top_k)
    except Exception as e:
        return JSONResponse(
            {"error": f"hllset-next query failed: {e}"},
            status_code=502,
        )

    matched = [
        m for m in result["matches"]
        if m.get("card") and (m.get("bss") or 0.0) >= threshold
    ]

    severity_rank = {"critical": 3, "abnormal": 2, "benign": 1, "normal": 0}
    matched.sort(key=lambda m: (severity_rank.get(m["card"]["severity"], 0), m["bss"]), reverse=True)

    measurements = SimulatedDiagnosticEngine._extract_ecg_measurements(text)
    measurement_findings = classify_measurements(measurements)

    criticals = [m for m in matched if m["card"]["severity"] == "critical"]
    criticals += [f for f in measurement_findings if f["severity"] == "critical"]
    abnormals = [m for m in matched if m["card"]["severity"] in ("abnormal", "benign")]
    abnormals += [f for f in measurement_findings if f["severity"] in ("abnormal", "benign")]

    if criticals:
        assessment = f"{len(criticals)} critical pattern(s) matched — urgent clinician review required."
    elif abnormals:
        assessment = f"{len(abnormals)} abnormal pattern(s) matched — clinician review advised."
    elif matched or measurement_findings:
        assessment = f"{len(matched)} lattice + {len(measurement_findings)} measurement finding(s), none critical."
    else:
        assessment = "No reference patterns matched at threshold."

    report = {
        "report_type": "medical-analysis",
        "engine": "hllset-lattice",
        "modality": modality,
        "instruction": instruction,
        "generated_at": datetime.utcnow().isoformat(),
        "patient": {
            "id": patient.get("id", patient.get("patient_id")),
            "name": patient.get("name", patient.get("patient_name")),
            "sex": patient.get("sex"),
            "age": patient.get("age"),
            "birth_date": patient.get("birth_date"),
        },
        "measurements": measurements,
        "measurement_findings": measurement_findings,
        "findings": [
            {
                "domain": m["card"]["domain"],
                "signal": m["card"]["id"],
                "pattern": m["card"]["text"],
                "note": m["card"]["interpretation"],
                "severity": m["card"]["severity"],
                "bss": round(m["bss"], 4),
            }
            for m in matched
        ],
        "assessment": assessment,
        "recommendation": (
            "Lattice-matched reference criteria (simulated — not a medical device). "
            "A licensed clinician must confirm before any clinical use."
        ),
        "query_stats": {
            "cards_ingested": len(kb.registry),
            "threshold": threshold,
            "matched": len(matched),
        },
        "source_text_preview": text[:500],
    }

    cid = await _store_to_ipfs(report)
    if cid:
        report["stored_cid"] = cid

    return report


# ═══════════════════════════════════════════════════════════════════════
# IPFS integration
# ═══════════════════════════════════════════════════════════════════════

async def _store_to_ipfs(payload: dict) -> Optional[str]:
    """Store a diagnostic result in IPFS, returning its CID."""
    return await _store_bytes_to_ipfs(_json_bytes(payload), "diagnosis.json")


async def _store_bytes_to_ipfs(data: bytes, filename: str) -> Optional[str]:
    """Store raw bytes in IPFS, returning its CID."""
    client = await get_client()
    try:
        files = {"file": (filename, data, "application/octet-stream")}
        r = await client.post(f"{IPFS_API_URL}/api/v0/add", files=files)
        if r.status_code == 200:
            return r.json().get("Hash")
    except Exception as e:
        logger.warning(f"IPFS store failed: {e}")
    return None


async def _fetch_bytes_from_ipfs(cid: str) -> Optional[bytes]:
    """Fetch raw bytes from IPFS by CID."""
    client = await get_client()
    try:
        r = await client.post(f"{IPFS_API_URL}/api/v0/cat", params={"arg": cid})
        if r.status_code == 200:
            return r.content
    except Exception as e:
        logger.warning(f"IPFS fetch failed: {e}")
    return None


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, default=str).encode("utf-8")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 9094))
    logger.info(
        f"medical-diagnostic starting on port {port} "
        f"(engine: {_ACTIVE_ENGINE.name}, dicom: {DICOM_AVAILABLE})"
    )
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=LOG_LEVEL.lower())
