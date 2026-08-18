"""
medical-diagnostic — Clinical Diagnosis Service (EWM backend).

Interprets OCR'd clinical text and medical images (X-ray, ECG, knee MRI)
into a structured diagnostic report.

Pipeline in Guru-EWM:
    Document/Image → deepseek-ocr (:9093) → text/markdown
      → medical-diagnostic (:9094) → structured diagnosis
        → ipfs (:5001) → content-addressed storage

Endpoints:
  GET  /health           — service status
  POST /analyze/hllset   — text → NanoLM lattice report (ECG / X-ray / lab)
  POST /analyze/document — document → OCR → NanoLM report
  POST /analyze/ecg      — ECG document → OCR → NanoLM report
  POST /classify         — image → BiomedCLIP zero-shot classification
  POST /classify/knee    — knee MRI → lattice fingerprint classifier
  GET  /hllset/status    — lattice corpus status
  POST /hllset/ingest    — re-ingest the knowledge-card corpus
"""

import asyncio
import io
import json
import os
import logging
import re
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import httpx

try:
    from ecg_cards import ECG_CARDS
except ImportError:  # pragma: no cover
    ECG_CARDS = []

try:
    from xray_cards import XRAY_CARDS
except ImportError:  # pragma: no cover
    XRAY_CARDS = []

try:
    from lab_cards import LAB_CARDS, LAB_REFERENCE_RANGES
except ImportError:  # pragma: no cover
    LAB_CARDS = []
    LAB_REFERENCE_RANGES = []

try:
    from knee_cards import KNEE_CARDS
except ImportError:  # pragma: no cover
    KNEE_CARDS = []

try:
    from ct_cards import CT_CARDS
except ImportError:  # pragma: no cover
    CT_CARDS = []

import vision
import knee_mri
import knee_cnn

# Modalities whose findings are typically single-phrase mentions ("cardiomegaly")
# rather than dense numeric+text reports — so they use a lower inclusion threshold.
RADIOLOGY_MODALITIES = {"CR", "DX", "XA", "RF", "MG", "PT", "NM", "MR", "CT"}

# ── Config ──────────────────────────────────────────────────────────
IPFS_API_URL = os.environ.get("IPFS_API_URL", "http://ipfs:5001")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
# hllset-next lattice (knowledge-card ingestion + BSS inclusion query)
HLLSET_NEXT_URL = os.environ.get("HLLSET_NEXT_URL", "http://hllset-next:9090")
# deepseek-ocr — CPU OCR layer (the "model" that feeds NanoLM)
DEEPSEEK_OCR_URL = os.environ.get("DEEPSEEK_OCR_URL", "http://deepseek-ocr:9093")
# Seed/backup: knowledge-corpus snapshot stored in IPFS for restore-on-restart
KB_SNAPSHOT_CID = os.environ.get("KB_SNAPSHOT_CID", "")
KB_SNAPSHOT_FILE = os.environ.get("KB_SNAPSHOT_FILE", "/app/data/kb_snapshot.json")
# Full self-contained copy of the ingested corpus (cards), used to restore at
# startup even when the IPFS node is empty or unreachable.
KB_LOCAL_BACKUP_FILE = os.environ.get("KB_LOCAL_BACKUP_FILE", "/app/data/kb_snapshot_cards.json")
# CPU-only CNN knee-MRI classifier weights (trained via scripts/train_knee_cnn.py)
KNEE_CNN_MODEL = os.environ.get("KNEE_CNN_MODEL", "/app/knee_cnn.pt")

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("medical-diagnostic")

app = FastAPI(
    title="Medical Diagnostic Service",
    description="Clinical diagnosis over OCR'd text and medical images (EWM backend)",
    version="0.2.0",
)

_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    return _client


# ═══════════════════════════════════════════════════════════════════════
# ECG measurement extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_ecg_measurements(text: str) -> dict:
    """Extract numeric ECG parameters (rate, intervals, axes) via regex."""
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


# ═══════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    kb = get_kb()

    ocr_status = {"url": DEEPSEEK_OCR_URL, "status": "unknown"}
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{DEEPSEEK_OCR_URL}/health")
            ocr_status["status"] = "ok" if r.status_code == 200 else "unhealthy"
    except Exception:
        ocr_status["status"] = "unreachable"

    return {
        "status": "ok",
        "service": "medical-diagnostic",
        "nanolm": {
            "engine": "nanolm",
            "available": True,
            "ready": kb.ingested,
        },
        "lattice": {**kb.status(), "last_snapshot_cid": _last_snapshot_cid},
        "ocr": ocr_status,
        "ipfs_api_url": IPFS_API_URL,
        "disclaimer": "NanoLM — lattice-based language model; not a medical device",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════
# HLLSet lattice — knowledge-card ingestion + BSS inclusion query
# ═══════════════════════════════════════════════════════════════════════

def _singular(word: str) -> str:
    """Light plural→singular normalization (not full stemming)."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"   # opacities → opacity, cavities → cavity
    if word.endswith("es") and len(word) > 3:
        c = word[-3]
        # "es" plural only for stems ending in s/x/z or ch/sh ("masses"→"mass",
        # "boxes"→"box", "branches"→"branch"). "nodules" ends in "e"+"s" and
        # falls through to the plain "s" rule instead.
        if c in "sxz" or (c == "h" and len(word) > 4 and word[-4] in "cs"):
            return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
        return word[:-1]         # nodules → nodule, effusions → effusion
    return word


def hll_tokenize(text: str, ngram_min: int = 1, ngram_max: int = 2) -> list:
    """Replicate hllset-next's word-pattern tokenizer (lowercase + trim,
    unigrams + bigrams joined with NUL). Deterministic, no estimation."""
    words = [_singular(w) for w in re.findall(r"[a-zA-Z0-9]+", text.lower())]
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
    # CT must be checked before "chest" so "Chest CT" isn't read as a chest X-ray.
    if " ct" in s or "ct scan" in s or "computed tomography" in s:
        return "CT"
    if any(k in s for k in ("mri", "magnetic resonance")):
        return "MR"
    if any(k in s for k in ("ultrasound", "sonogram")):
        return "US"
    if any(k in s for k in ("x-ray", "xray", "radiograph", "cxr")):
        return "DX"
    if "chest" in s:
        return "DX"
    return ""


# Modality → knowledge-card id prefix. When a modality maps here, findings
# are restricted to that domain so a CT report doesn't surface X-ray cards
# that share a word (e.g. both define "nodule").
_MODALITY_PREFIX = {
    "ECG": "ecg", "EKG": "ecg",
    "CT": "ct",
    "MR": "knee", "MRI": "knee",
    "DX": "xr", "CR": "xr", "XA": "xr", "RF": "xr",
    "MG": "xr", "PT": "xr", "NM": "xr",
    "LAB": "lab",
}


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


def _extract_lab_values(text: str) -> dict:
    """Extract numeric lab values from free text using analyte aliases."""
    lowered = text.lower()
    values: dict = {}
    for spec in LAB_REFERENCE_RANGES:
        for alias in spec["aliases"]:
            pat = r"\b" + re.escape(alias) + r"\b[\s:=]*([\d]+(?:\.[\d]+)?)"
            m = re.search(pat, lowered)
            if m:
                try:
                    val = float(m.group(1))
                except ValueError:
                    continue
                # Sanity bound — reject implausible numbers (e.g. an alias
                # matching an unrelated value like a patient ID).
                if not (0.01 <= val <= 5000):
                    continue
                values[spec["id"]] = {"value": val, "unit": spec["unit"]}
                break
    return values


def classify_lab_values(lab_values: dict) -> list:
    """Derive findings from extracted lab values vs LAB_REFERENCE_RANGES."""
    findings = []
    by_id = {spec["id"]: spec for spec in LAB_REFERENCE_RANGES}
    for analyte_id, entry in lab_values.items():
        spec = by_id.get(analyte_id)
        if not spec:
            continue
        val = entry["value"]
        unit = spec["unit"]
        low, high = spec.get("low"), spec.get("high")
        if low is None:
            ref = f"≤ {high}"
        elif high is None:
            ref = f"≥ {low}"
        else:
            ref = f"{low}–{high}"
        if low is not None and val < low and spec.get("low_finding"):
            f = spec["low_finding"]
            findings.append({"domain": "lab", "signal": f["signal"], "severity": f["severity"],
                             "value": val, "unit": unit,
                             "note": f"{f['note']} ({val} {unit}; ref {ref})."})
        elif high is not None and val > high and spec.get("high_finding"):
            f = spec["high_finding"]
            findings.append({"domain": "lab", "signal": f["signal"], "severity": f["severity"],
                             "value": val, "unit": unit,
                             "note": f"{f['note']} ({val} {unit}; ref {ref})."})
        else:
            findings.append({"domain": "lab", "signal": f"{analyte_id}_normal", "severity": "normal",
                             "value": val, "unit": unit,
                             "note": f"{analyte_id.replace('_', ' ')} {val} {unit} — within reference range."})
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
        # Namespace by card family: c:ecg:<id> for ECG, c:xr:<id> for X-ray.
        ns = card_id.split(".")[0]
        return f"c:{ns}:{card_id}"

    async def ingest(self) -> dict:
        """Ingest all cards into the lattice (idempotent — same key overwritten).

        The HLLSet lattice is populated via hllset-next for content-addressed
        storage. Token sets are also cached locally for exact, deterministic
        scoring (HLL cardinality is unstable for small knowledge cards).

        If hllset-next is unreachable the lattice POSTs are skipped and only
        the local token index is built — diagnosis still works.
        """
        client = await get_client()
        try:
            h = await client.get(f"{self.base_url}/api/v1/health")
            h.raise_for_status()
        except Exception as e:
            for card in self.cards:
                key = self._key(card["id"])
                self.registry[key] = card
                self.card_tokens[key] = set(hll_tokenize(card["text"]))
            self.ingested = len(self.registry) == len(self.cards)
            return {"cards": len(self.cards), "ingested": 0,
                    "errors": [f"hllset-next unreachable: {e}"]}

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
        query_words = {_singular(w) for w in re.findall(r"[a-zA-Z0-9]+", text.lower())}
        scored = []
        for key, card in self.registry.items():
            card_tokens = self.card_tokens.get(key)
            if not card_tokens:
                continue
            # Require the card's head term (primary finding noun) to actually
            # appear — otherwise "pulmonary nodules" fires "pulmonary
            # embolism" via the shared word "pulmonary".
            card_text = (card.get("text") or "").strip()
            head = _singular(card_text.split()[-1]) if card_text else ""
            if head and head not in query_words:
                continue
            score = len(query_tokens & card_tokens) / len(card_tokens)
            scored.append({"key": key, "bss": score, "card": card})
        scored.sort(key=lambda m: m["bss"], reverse=True)
        return {"matches": scored[:top_k], "top_k": top_k}

    def status(self) -> dict:
        return {
            "base_url": self.base_url,
            "cards": len(self.cards),
            "corpus_count": len(self.cards),
            "ingested": len(self.registry),
            "ready": self.ingested,
        }


_kb: HllsetKnowledgeBase | None = None
_last_snapshot_cid: str = ""


def get_kb() -> HllsetKnowledgeBase:
    global _kb
    if _kb is None:
        _kb = HllsetKnowledgeBase(HLLSET_NEXT_URL, ECG_CARDS + XRAY_CARDS + LAB_CARDS + KNEE_CARDS + CT_CARDS)
    return _kb


_knee_library = None


def get_knee_library():
    """Build (once) the synthetic knee-MRI reference library for the lattice
    fingerprint classifier — no deep model, no weights."""
    global _knee_library
    if _knee_library is None:
        logger.info("building knee MRI reference library (synthetic patterns) …")
        _knee_library = knee_mri.build_reference_library(size=256, images_per_class=8)
        logger.info("knee MRI reference library ready")
    return _knee_library


_knee_cnn_model = None


def get_knee_cnn_model():
    """Load (once) the CPU-only CNN knee-MRI classifier, or None if weights are
    absent (e.g. a dev image without a trained model)."""
    global _knee_cnn_model
    if _knee_cnn_model is None and os.path.exists(KNEE_CNN_MODEL):
        logger.info(f"loading knee CNN classifier from {KNEE_CNN_MODEL} …")
        _knee_cnn_model = knee_cnn.load_model(KNEE_CNN_MODEL)
        logger.info("knee CNN classifier ready")
    return _knee_cnn_model


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


def _save_local_backup(payload: dict) -> None:
    try:
        os.makedirs(os.path.dirname(KB_LOCAL_BACKUP_FILE), exist_ok=True)
        with open(KB_LOCAL_BACKUP_FILE, "w") as f:
            json.dump(payload, f, default=str)
    except Exception as e:  # pragma: no cover
        logger.warning(f"could not save local backup: {e}")


def _load_local_backup_cards() -> list:
    try:
        with open(KB_LOCAL_BACKUP_FILE) as f:
            return json.load(f).get("cards", [])
    except Exception:
        return []


async def _snapshot_kb(kb: HllsetKnowledgeBase) -> str:
    """Back up the knowledge corpus to IPFS + a local file as a durable seed."""
    global _last_snapshot_cid
    payload = {
        "type": "knowledge-snapshot",
        "engine": "hllset-lattice",
        "count": len(kb.cards),
        "cards": kb.cards,
        "created_at": datetime.utcnow().isoformat(),
    }
    _save_local_backup(payload)
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
            cards = _load_local_backup_cards()
            if cards:
                kb.cards = cards
                logger.info(f"restored {len(cards)} cards from local backup")
        if not kb.cards:
            logger.warning("no seed corpus and no snapshot — knowledge base empty")
            return

    result = await kb.ingest()
    await _snapshot_kb(kb)
    if result.get("ingested", 0) > 0:
        logger.info(f"knowledge base seeded: {result['ingested']} cards")
    else:
        logger.warning(
            "hllset-next unavailable — using local token index only "
            f"({len(result.get('errors', []))} errors)"
        )


@app.post("/hllset/ingest")
async def hllset_ingest():
    """Ingest the ECG knowledge-card corpus into the hllset-next lattice."""
    return await get_kb().ingest()


@app.get("/hllset/status")
async def hllset_status():
    """Report lattice corpus status."""
    return {**get_kb().status(), "last_snapshot_cid": _last_snapshot_cid}


NEGATION_WORDS = {"no", "not", "without", "absent", "negative", "denies"}


def _negated_terms(text: str) -> set:
    """Collect tokens inside a negation scope ("no pneumothorax", "without
    effusion") so negated findings don't match their cards."""
    words = [_singular(w) for w in re.findall(r"[a-zA-Z0-9]+", text.lower())]
    negated: set = set()
    for i, w in enumerate(words):
        if w not in NEGATION_WORDS:
            continue
        j = i + 1
        while j < len(words) and words[j] in {
            "evidence", "of", "any", "a", "an", "the", "significant", "acute", "definite",
        }:
            j += 1
        while j < len(words) and words[j] not in {
            ",", ";", "and", "or", "but", "with", "is", "are", "was", "were", "there",
        }:
            negated.add(words[j])
            j += 1
    return negated


async def _analyze_text(
    text: str,
    modality: str = "ECG",
    instruction: str = "",
    patient: Optional[dict] = None,
    threshold: Optional[float] = None,
    top_k: int = 12,
) -> dict:
    """NanoLM core pipeline: tokenize → lattice match → numeric classify → compose.

    Shared by /analyze/hllset (JSON text) and /analyze/document|ecg (OCR text).
    """
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text'")

    inferred = _infer_modality(instruction) or _infer_modality(text)
    if inferred:
        modality = inferred

    if threshold is None:
        # Radiology findings are usually a single phrase, so require less
        # token overlap than the dense ECG vocabulary.
        threshold = 0.20 if modality in RADIOLOGY_MODALITIES else 0.50

    # Bare instructions ("analyze this x-ray report…") contain no clinical
    # content; don't let generic words like "diagnosis" match a card.
    looks_instruction = (
        any(k in text.lower() for k in ("analyze", "analyse", "diagnos", "produce"))
        and len(re.findall(r"[a-zA-Z0-9]+", text)) < 30
    )
    if looks_instruction and threshold < 0.50:
        threshold = 0.50

    patient = patient or {}
    kb = get_kb()
    if not kb.ingested:
        await kb.ingest()

    try:
        result = await kb.query(text, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"hllset-next query failed: {e}")

    matched = [
        m for m in result["matches"]
        if m.get("card") and (m.get("bss") or 0.0) >= threshold
    ]

    # Suppress negated findings ("no pneumothorax" must not match the
    # "pneumothorax" card).
    negated = _negated_terms(text)
    if negated:
        matched = [
            m for m in matched
            if not (set(hll_tokenize(m["card"]["text"])) & negated)
        ]

    # Restrict to the modality's domain so a CT report doesn't surface
    # X-ray/ECG cards that happen to share a word ("nodule").
    prefix = _MODALITY_PREFIX.get(modality)
    if prefix:
        matched = [m for m in matched if m["card"]["id"].startswith(prefix + ".")]

    severity_rank = {"critical": 3, "abnormal": 2, "benign": 1, "normal": 0}
    matched.sort(key=lambda m: (severity_rank.get(m["card"]["severity"], 0), m["bss"]), reverse=True)

    measurements = extract_ecg_measurements(text)
    measurement_findings = classify_measurements(measurements)

    lab_values = _extract_lab_values(text)
    measurement_findings = measurement_findings + classify_lab_values(lab_values)

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
        if looks_instruction:
            assessment = (
                "This looks like an instruction rather than a report. "
                "Paste the clinical report text (or upload the report file) to analyze it."
            )
        else:
            assessment = "No reference patterns matched at threshold."

    return {
        "report_type": "nanolm",
        "engine": "nanolm",
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
        "lab_values": lab_values,
        "measurement_findings": measurement_findings,
        "findings": [
            {
                "signal": m["card"]["id"],
                "note": m["card"]["interpretation"],
                "severity": m["card"]["severity"],
                "bss": round(m["bss"], 4),
            }
            for m in matched
        ] + [
            {
                "signal": f["signal"],
                "note": f["note"],
                "severity": f["severity"],
                "bss": None,
            }
            for f in measurement_findings
        ],
        "assessment": assessment,
        "recommendation": (
            "NanoLM lattice-matched reference criteria (not a medical device). "
            "A licensed clinician must confirm before any clinical use."
        ),
        "query_stats": {
            "cards_ingested": len(kb.registry),
            "threshold": threshold,
            "matched": len(matched),
        },
        "source_text_preview": text[:500],
    }


@app.post("/analyze/hllset")
async def analyze_hllset(request: Request):
    """JSON text → NanoLM lattice report (spec §5 /analyze/hllset)."""
    body = await request.json()
    report = await _analyze_text(
        text=body.get("text", body.get("page_text", "")),
        modality=body.get("modality", "ECG"),
        instruction=body.get("instruction", ""),
        patient=body.get("patient", {}),
        threshold=float(body["bss_threshold"]) if body.get("bss_threshold") is not None else None,
        top_k=int(body.get("top_k", 12)),
    )
    cid = await _store_to_ipfs(report)
    if cid:
        report["stored_cid"] = cid
    return report


async def _ocr_extract(filename: str, raw: bytes, content_type: str) -> dict:
    """Send an uploaded document to the OCR service and return its JSON result."""
    client = await get_client()
    try:
        files = {"file": (filename, raw, content_type or "application/octet-stream")}
        r = await client.post(f"{DEEPSEEK_OCR_URL}/ocr/upload", files=files)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"deepseek-ocr service unreachable: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OCR failed: {r.text[:300]}")
    return r.json()


@app.post("/analyze/document")
async def analyze_document(
    file: UploadFile = File(...),
    instruction: str = Form(""),
    modality: str = Form("DOC"),
):
    """Multipart document → OCR → NanoLM report (spec §5 /analyze/document)."""
    raw = await file.read()
    ocr = await _ocr_extract(file.filename or "document", raw, file.content_type or "")
    text = ocr.get("page_text") or ocr.get("text") or ""
    if not text:
        return JSONResponse(
            {"error": "No text could be extracted from the document", "ocr": ocr},
            status_code=422,
        )

    report = await _analyze_text(text=text, modality=modality, instruction=instruction)
    cid = await _store_to_ipfs(report)
    if cid:
        report["stored_cid"] = cid

    return {
        "filename": file.filename,
        "modality": report.get("modality", modality),
        "instruction": instruction,
        "ocr": {
            "source": ocr.get("source"),
            "mode": ocr.get("mode"),
            "notice": ocr.get("notice"),
        },
        "report": report,
    }


@app.post("/analyze/ecg")
async def analyze_ecg(
    file: UploadFile = File(...),
    instruction: str = Form(""),
):
    """Multipart ECG → OCR → NanoLM report (ECG modality default, spec §5)."""
    return await analyze_document(file, instruction, "ECG")


# ═══════════════════════════════════════════════════════════════════════
# Vision classification / embedding (merged from the former vision-encoder)
# ═══════════════════════════════════════════════════════════════════════

@app.post("/classify")
async def classify_image_endpoint(file: UploadFile = File(...), top_k: int = 3):
    """Zero-shot image classification (BiomedCLIP) over xr.* finding labels."""
    raw = await file.read()
    try:
        matches = vision.classify_image(raw, top_k=top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"model": vision.MODEL_NAME, "matches": matches}


@app.post("/classify/knee")
async def classify_knee_endpoint(file: UploadFile = File(...), top_k: int = 3):
    """Knee MRI classification via lattice image fingerprinting (no deep model)."""
    import numpy as np
    from PIL import Image
    raw = await file.read()
    gray = np.asarray(Image.open(io.BytesIO(raw)).convert("L")).astype(np.uint8)
    scores = knee_mri.classify(gray, get_knee_library())
    top = scores[0] if scores else None
    detected = [s for s in scores if s["detected"]]
    return {
        "engine": "hllset-knee",
        "top_finding": top["abnormality"] if top else "normal",
        "top_label": top["label"] if top else "Normal",
        "confidence": top["confidence"] if top else 0.0,
        "scores": scores[:max(1, min(top_k, len(scores)))],
        "detected": detected[:max(1, min(top_k, len(detected)))],
    }


@app.post("/classify/knee/deep")
async def classify_knee_deep_endpoint(file: UploadFile = File(...), top_k: int = 6):
    """Knee MRI classification via the CPU-only CNN (knee_cnn.py).

    Returns 404 when no trained weights are present (the caller should then
    fall back to the fingerprint / zero-shot classifiers).
    """
    model = get_knee_cnn_model()
    if model is None:
        raise HTTPException(status_code=404, detail="knee CNN weights not available")
    raw = await file.read()
    matches = knee_cnn.predict(model, raw, top_k=top_k)
    return {
        "engine": "knee-cnn",
        "top_label": matches[0]["label"] if matches else "Normal",
        "confidence": matches[0]["confidence"] if matches else 0.0,
        "scores": matches,
    }


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
    logger.info(f"medical-diagnostic starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=LOG_LEVEL.lower())
