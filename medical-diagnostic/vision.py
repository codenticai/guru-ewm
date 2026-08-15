"""
vision.py — biomedical image classification, merged into medical-diagnostic.

Contains the functionality formerly in the standalone vision-encoder service:
  - BiomedCLIP zero-shot classification over xr.* finding labels (/classify)

The knee-MRI lattice-fingerprint classifier lives in knee_mri.py (/classify/knee).
"""

import io
import os
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger("medical-diagnostic.vision")

MODEL_NAME = os.environ.get("VISION_MODEL", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

_model = None
_preprocess = None
_tokenizer = None
_ref_embeddings: list = []

# Zero-shot reference findings (mirrors the xr.* finding cards).
REFERENCE_FINDINGS = [
    {"id": "xr.cardio.cardiomegaly", "label": "A chest X-ray showing cardiomegaly, an enlarged heart.",
     "severity": "abnormal", "note": "Cardiomegaly — enlarged cardiac silhouette."},
    {"id": "xr.cardio.pulmonary_edema", "label": "A chest X-ray showing pulmonary edema.",
     "severity": "critical", "note": "Pulmonary edema — perihilar and interstitial opacities."},
    {"id": "xr.cardio.pericardial_effusion", "label": "A chest X-ray showing pericardial effusion.",
     "severity": "abnormal", "note": "Pericardial effusion — globular cardiac silhouette."},
    {"id": "xr.mediastinal.widened", "label": "A chest X-ray showing a widened mediastinum.",
     "severity": "critical", "note": "Widened mediastinum — consider aortic injury or mass."},
    {"id": "xr.pleural.effusion", "label": "A chest X-ray showing pleural effusion.",
     "severity": "abnormal", "note": "Pleural effusion — blunted costophrenic angle."},
    {"id": "xr.pleural.pneumothorax", "label": "A chest X-ray showing pneumothorax, a collapsed lung.",
     "severity": "critical", "note": "Pneumothorax — visceral pleural line."},
    {"id": "xr.pleural.hemothorax", "label": "A chest X-ray showing hemothorax, blood in the pleural space.",
     "severity": "critical", "note": "Hemothorax — pleural blood, often post-traumatic."},
    {"id": "xr.parenchyma.consolidation", "label": "A chest X-ray showing lung consolidation.",
     "severity": "abnormal", "note": "Consolidation — airspace opacification."},
    {"id": "xr.parenchyma.pneumonia", "label": "A chest X-ray showing pneumonia.",
     "severity": "abnormal", "note": "Pneumonia — pulmonary infiltrate."},
    {"id": "xr.parenchyma.atelectasis", "label": "A chest X-ray showing atelectasis, lung collapse.",
     "severity": "abnormal", "note": "Atelectasis — volume loss with opacification."},
    {"id": "xr.parenchyma.nodule", "label": "A chest X-ray showing a pulmonary nodule.",
     "severity": "abnormal", "note": "Pulmonary nodule — follow-up imaging advised."},
    {"id": "xr.parenchyma.mass", "label": "A chest X-ray showing a lung mass.",
     "severity": "abnormal", "note": "Lung mass — irregular opacity; needs characterization."},
    {"id": "xr.parenchyma.cavity", "label": "A chest X-ray showing a cavitary lung lesion.",
     "severity": "abnormal", "note": "Cavitary lesion — consider infection, abscess or neoplasm."},
    {"id": "xr.airway.hyperinflation", "label": "A chest X-ray showing hyperinflated lungs.",
     "severity": "abnormal", "note": "Hyperinflation — flattened diaphragms; consider COPD."},
    {"id": "xr.airway.bronchiectasis", "label": "A chest X-ray showing bronchiectasis.",
     "severity": "abnormal", "note": "Bronchiectasis — dilated, thickened airways."},
    {"id": "xr.msk.rib_fracture", "label": "A chest X-ray showing a rib fracture.",
     "severity": "abnormal", "note": "Rib fracture — cortical disruption."},
    {"id": "xr.msk.clavicle_fracture", "label": "A chest X-ray showing a clavicle fracture.",
     "severity": "abnormal", "note": "Clavicle fracture — mid-shaft cortical break."},
    {"id": "xr.msk.femur_fracture", "label": "An X-ray of the thigh showing a femur fracture.",
     "severity": "abnormal", "note": "Femur fracture — fracture of the femoral shaft or distal femur."},
    {"id": "xr.msk.tibia_fracture", "label": "An X-ray of the lower leg showing a tibia fracture.",
     "severity": "abnormal", "note": "Tibia fracture — fracture of the tibial shaft or plateau."},
    {"id": "xr.msk.fibula_fracture", "label": "An X-ray of the lower leg showing a fibula fracture.",
     "severity": "abnormal", "note": "Fibula fracture — fracture of the fibular shaft."},
    {"id": "xr.msk.hip_fracture", "label": "An X-ray of the hip showing a hip fracture.",
     "severity": "abnormal", "note": "Hip fracture — proximal femur fracture."},
    {"id": "xr.msk.humerus_fracture", "label": "An X-ray of the upper arm showing a humerus fracture.",
     "severity": "abnormal", "note": "Humerus fracture — fracture of the humeral shaft or proximal humerus."},
    {"id": "xr.msk.forearm_fracture", "label": "An X-ray of the forearm showing a radius or ulna fracture.",
     "severity": "abnormal", "note": "Forearm fracture — radius or ulna fracture."},
    {"id": "xr.normal.clear", "label": "A normal chest X-ray with clear lungs and no acute disease.",
     "severity": "normal", "note": "No acute cardiopulmonary disease."},

    # ── Knee MRI findings (RSNA knee-MRI task labels) ───────────────────
    {"id": "knee.acl_tear", "label": "A knee MRI showing an ACL tear.",
     "severity": "abnormal", "note": "ACL tear — anterior cruciate ligament disruption."},
    {"id": "knee.pcl_tear", "label": "A knee MRI showing a PCL tear.",
     "severity": "abnormal", "note": "PCL tear — posterior cruciate ligament disruption."},
    {"id": "knee.mcl_tear", "label": "A knee MRI showing an MCL tear.",
     "severity": "abnormal", "note": "MCL tear — medial collateral ligament disruption."},
    {"id": "knee.lcl_tear", "label": "A knee MRI showing an LCL tear.",
     "severity": "abnormal", "note": "LCL tear — lateral collateral ligament disruption."},
    {"id": "knee.meniscal_tear", "label": "A knee MRI showing a meniscal tear.",
     "severity": "abnormal", "note": "Meniscal tear — meniscus disruption."},
    {"id": "knee.cartilage_loss", "label": "A knee MRI showing cartilage loss.",
     "severity": "abnormal", "note": "Cartilage loss — full-thickness chondral wear."},
    {"id": "knee.chondral_defect", "label": "A knee MRI showing a chondral defect.",
     "severity": "abnormal", "note": "Chondral defect — focal articular cartilage lesion."},
    {"id": "knee.osteoarthritis", "label": "A knee MRI showing osteoarthritis.",
     "severity": "abnormal", "note": "Osteoarthritis — joint space narrowing and osteophytes."},
    {"id": "knee.bone_marrow_edema", "label": "A knee MRI showing bone marrow edema.",
     "severity": "abnormal", "note": "Bone marrow edema — increased marrow signal."},
    {"id": "knee.tibial_plateau_fracture", "label": "A knee MRI showing a tibial plateau fracture.",
     "severity": "abnormal", "note": "Tibial plateau fracture — proximal tibia fracture."},
    {"id": "knee.patellar_dislocation", "label": "A knee MRI showing patellar dislocation.",
     "severity": "abnormal", "note": "Patellar dislocation — lateral patellar subluxation."},
    {"id": "knee.patellar_tendinopathy", "label": "A knee MRI showing patellar tendinopathy.",
     "severity": "abnormal", "note": "Patellar tendinopathy — proximal patellar tendon change."},

    # ── CT (computed tomography) findings ───────────────────────────────
    {"id": "ct.lung.ground_glass", "label": "A chest CT showing ground-glass opacity.",
     "severity": "abnormal", "note": "Ground-glass opacity — hazy increased attenuation."},
    {"id": "ct.lung.consolidation", "label": "A chest CT showing lung consolidation.",
     "severity": "abnormal", "note": "Consolidation — airspace opacification."},
    {"id": "ct.lung.nodule", "label": "A chest CT showing a pulmonary nodule.",
     "severity": "abnormal", "note": "Pulmonary nodule — focal opacity under 3 cm."},
    {"id": "ct.lung.mass", "label": "A chest CT showing a lung mass.",
     "severity": "abnormal", "note": "Lung mass — lesion over 3 cm."},
    {"id": "ct.lung.cavity", "label": "A chest CT showing a cavitary lung lesion.",
     "severity": "abnormal", "note": "Cavitary lesion — air-filled lesion within an opacity."},
    {"id": "ct.lung.atelectasis", "label": "A chest CT showing atelectasis.",
     "severity": "abnormal", "note": "Atelectasis — collapsed lung tissue with volume loss."},
    {"id": "ct.lung.emphysema", "label": "A chest CT showing emphysema.",
     "severity": "abnormal", "note": "Emphysema — parenchymal destruction with enlarged airspaces."},
    {"id": "ct.lung.bronchiectasis", "label": "A chest CT showing bronchiectasis.",
     "severity": "abnormal", "note": "Bronchiectasis — dilated, thick-walled airways."},
    {"id": "ct.lung.fibrosis", "label": "A chest CT showing pulmonary fibrosis.",
     "severity": "abnormal", "note": "Pulmonary fibrosis — interstitial thickening."},
    {"id": "ct.lung.pulmonary_edema", "label": "A chest CT showing pulmonary edema.",
     "severity": "abnormal", "note": "Pulmonary edema — interstitial/alveolar fluid."},
    {"id": "ct.lung.pulmonary_embolism", "label": "A chest CT showing a pulmonary embolism.",
     "severity": "critical", "note": "Pulmonary embolism — intraluminal filling defect."},
    {"id": "ct.pleural.pneumothorax", "label": "A chest CT showing pneumothorax.",
     "severity": "critical", "note": "Pneumothorax — air in the pleural space."},
    {"id": "ct.pleural.effusion", "label": "A chest CT showing pleural effusion.",
     "severity": "abnormal", "note": "Pleural effusion — fluid in the pleural space."},
    {"id": "ct.mediastinum.lymphadenopathy", "label": "A chest CT showing lymphadenopathy.",
     "severity": "abnormal", "note": "Lymphadenopathy — enlarged lymph nodes."},
    {"id": "ct.liver.lesion", "label": "An abdominal CT showing a liver lesion.",
     "severity": "abnormal", "note": "Liver lesion — focal hepatic abnormality."},
    {"id": "ct.normal.clear", "label": "A normal chest CT with clear lungs and no acute disease.",
     "severity": "normal", "note": "No acute cardiopulmonary disease."},
]


# ── Model (lazy) ──────────────────────────────────────────────────────
def _load_model() -> bool:
    global _model, _preprocess, _tokenizer
    if _model is not None:
        return True
    try:
        import open_clip
        logger.info(f"loading vision model {MODEL_NAME} …")
        _model, _, _preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
        _tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        _model.eval()
        logger.info("vision model loaded")
        return True
    except Exception as e:  # pragma: no cover
        logger.error(f"could not load vision model: {e}")
        return False


def _embed_image(raw: bytes) -> np.ndarray:
    if _model is None and not _load_model():
        raise RuntimeError("vision model unavailable — install open_clip_torch + transformers")
    import torch
    img = _preprocess(Image.open(io.BytesIO(raw)).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        feats = _model.encode_image(img)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy().astype("float32")


def _embed_text(text: str) -> np.ndarray:
    if _model is None and not _load_model():
        raise RuntimeError("vision model unavailable — install open_clip_torch + transformers")
    import torch
    tokens = _tokenizer([text])
    with torch.no_grad():
        feats = _model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy().astype("float32")


def model_loaded() -> bool:
    return _model is not None


def _get_reference_embeddings() -> list:
    global _ref_embeddings
    if not _ref_embeddings:
        for f in REFERENCE_FINDINGS:
            _ref_embeddings.append({
                "id": f["id"], "note": f["note"], "severity": f["severity"],
                "embedding": _embed_text(f["label"]),
            })
    return _ref_embeddings


def classify_image(raw: bytes, top_k: int = 3) -> list:
    """Zero-shot classification: image → nearest reference findings (cosine)."""
    img_vec = _embed_image(raw)
    scored = []
    for r in _get_reference_embeddings():
        s = float(np.dot(img_vec, r["embedding"]))
        scored.append({"signal": r["id"], "note": r["note"], "severity": r["severity"],
                       "score": round(s, 4)})
    scored.sort(key=lambda m: m["score"], reverse=True)
    return scored[:max(1, min(top_k, len(scored)))]
