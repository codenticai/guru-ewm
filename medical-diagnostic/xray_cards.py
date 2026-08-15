"""
Chest X-ray (CR/DX) knowledge-card corpus for the HLLSet lattice.

Mirrors ecg_cards.py: each card is one atomic radiologic finding. The `text`
field is tokenized into a 4KB HLLSet by hllset-next and the card is retrieved
by token-overlap inclusion — how confidently the OCR'd report CONTAINS the
card's pattern.

IMPORTANT: These are reference interpretation criteria for architecture
validation only. This is NOT a medical device and must not be used for real
clinical decisions.
"""

XRAY_CARDS = [
    # ── Cardiac silhouette / mediastinum ────────────────────────────────
    {
        "id": "xr.cardio.cardiomegaly",
        "text": "cardiomegaly",
        "domain": "cardiac",
        "severity": "abnormal",
        "interpretation": "Cardiomegaly — enlarged cardiac silhouette; correlate with chamber dilation, cardiomyopathy or pericardial effusion.",
    },
    {
        "id": "xr.cardio.enlarged_silhouette",
        "text": "enlarged cardiac silhouette",
        "domain": "cardiac",
        "severity": "abnormal",
        "interpretation": "Enlarged cardiac silhouette — suggest cardiomegaly or pericardial effusion.",
    },
    {
        "id": "xr.cardio.pulmonary_edema",
        "text": "pulmonary edema",
        "domain": "cardiac",
        "severity": "critical",
        "interpretation": "Pulmonary edema — perihilar and interstitial opacities with vascular congestion; consider cardiogenic or non-cardiogenic causes.",
    },
    {
        "id": "xr.cardio.pericardial_effusion",
        "text": "pericardial effusion",
        "domain": "cardiac",
        "severity": "abnormal",
        "interpretation": "Pericardial effusion — globular cardiac silhouette; confirm by echocardiography.",
    },
    {
        "id": "xr.mediastinal.widened",
        "text": "widened mediastinum",
        "domain": "mediastinum",
        "severity": "critical",
        "interpretation": "Widened mediastinum — consider aortic injury or mediastinal mass; urgent correlation advised.",
    },

    # ── Pleural space ───────────────────────────────────────────────────
    {
        "id": "xr.pleural.effusion",
        "text": "pleural effusion",
        "domain": "pleural",
        "severity": "abnormal",
        "interpretation": "Pleural effusion — blunted costophrenic angle with meniscus sign.",
    },
    {
        "id": "xr.pleural.pneumothorax",
        "text": "pneumothorax",
        "domain": "pleural",
        "severity": "critical",
        "interpretation": "Pneumothorax — visceral pleural line with absent peripheral lung markings. Tension physiology is an emergency.",
    },
    {
        "id": "xr.pleural.hemothorax",
        "text": "hemothorax",
        "domain": "pleural",
        "severity": "critical",
        "interpretation": "Hemothorax — pleural blood, often post-traumatic.",
    },

    # ── Lung parenchyma ─────────────────────────────────────────────────
    {
        "id": "xr.parenchyma.consolidation",
        "text": "consolidation",
        "domain": "parenchyma",
        "severity": "abnormal",
        "interpretation": "Consolidation — airspace opacification with air bronchograms; consider pneumonia or atelectasis.",
    },
    {
        "id": "xr.parenchyma.pneumonia",
        "text": "pneumonia",
        "domain": "parenchyma",
        "severity": "abnormal",
        "interpretation": "Pneumonia — pulmonary infiltrate; correlate clinically for infectious process.",
    },
    {
        "id": "xr.parenchyma.atelectasis",
        "text": "atelectasis",
        "domain": "parenchyma",
        "severity": "abnormal",
        "interpretation": "Atelectasis — volume loss with opacification; may be post-obstructive or post-operative.",
    },
    {
        "id": "xr.parenchyma.nodule",
        "text": "pulmonary nodule",
        "domain": "parenchyma",
        "severity": "abnormal",
        "interpretation": "Pulmonary nodule — follow-up imaging advised to assess stability.",
    },
    {
        "id": "xr.parenchyma.mass",
        "text": "lung mass",
        "domain": "parenchyma",
        "severity": "abnormal",
        "interpretation": "Lung mass — irregular opacity > 3 cm; requires further characterization (CT, biopsy).",
    },
    {
        "id": "xr.parenchyma.cavity",
        "text": "cavitary lesion",
        "domain": "parenchyma",
        "severity": "abnormal",
        "interpretation": "Cavitary lesion — air-fluid level or thick-walled cavity; consider infection, abscess or neoplasm.",
    },

    # ── Airways ─────────────────────────────────────────────────────────
    {
        "id": "xr.airway.hyperinflation",
        "text": "hyperinflation",
        "domain": "airway",
        "severity": "abnormal",
        "interpretation": "Hyperinflation — flattened diaphragms and increased retrosternal airspace; consider COPD/emphysema.",
    },
    {
        "id": "xr.airway.bronchiectasis",
        "text": "bronchiectasis",
        "domain": "airway",
        "severity": "abnormal",
        "interpretation": "Bronchiectasis — dilated, thickened airways; confirm by CT.",
    },

    # ── Musculoskeletal ─────────────────────────────────────────────────
    {
        "id": "xr.msk.rib_fracture",
        "text": "rib fracture",
        "domain": "msk",
        "severity": "abnormal",
        "interpretation": "Rib fracture — cortical disruption; assess for pneumothorax or flail segment.",
    },
    {
        "id": "xr.msk.clavicle_fracture",
        "text": "clavicle fracture",
        "domain": "msk",
        "severity": "abnormal",
        "interpretation": "Clavicle fracture — mid-shaft cortical break.",
    },
    {
        "id": "xr.msk.femur_fracture",
        "text": "femur fracture",
        "domain": "msk",
        "severity": "abnormal",
        "interpretation": "Femur fracture — fracture of the femoral shaft or distal femur.",
    },
    {
        "id": "xr.msk.tibia_fracture",
        "text": "tibia fracture",
        "domain": "msk",
        "severity": "abnormal",
        "interpretation": "Tibia fracture — fracture of the tibial shaft or plateau.",
    },
    {
        "id": "xr.msk.fibula_fracture",
        "text": "fibula fracture",
        "domain": "msk",
        "severity": "abnormal",
        "interpretation": "Fibula fracture — fracture of the fibular shaft.",
    },
    {
        "id": "xr.msk.hip_fracture",
        "text": "hip fracture",
        "domain": "msk",
        "severity": "abnormal",
        "interpretation": "Hip fracture — proximal femur fracture.",
    },
    {
        "id": "xr.msk.humerus_fracture",
        "text": "humerus fracture",
        "domain": "msk",
        "severity": "abnormal",
        "interpretation": "Humerus fracture — fracture of the humeral shaft or proximal humerus.",
    },
    {
        "id": "xr.msk.forearm_fracture",
        "text": "forearm fracture",
        "domain": "msk",
        "severity": "abnormal",
        "interpretation": "Forearm fracture — radius or ulna fracture.",
    },

    # ── Normal ──────────────────────────────────────────────────────────
    {
        "id": "xr.normal.clear",
        "text": "no acute cardiopulmonary disease",
        "domain": "normal",
        "severity": "normal",
        "interpretation": "No acute cardiopulmonary disease — clear lung fields and normal cardiac silhouette.",
    },
]
