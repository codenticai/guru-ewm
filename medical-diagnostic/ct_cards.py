"""
CT (computed tomography) knowledge-card corpus for the HLLSet lattice.

Mirrors ecg_cards.py / xray_cards.py: each card is one atomic CT finding. The
`text` field is tokenized by hllset-next and the card is retrieved by
token-overlap inclusion — how confidently the OCR'd report CONTAINS the
card's pattern.

IMPORTANT: These are reference interpretation criteria for architecture
validation only. This is NOT a medical device and must not be used for real
clinical decisions.
"""

CT_CARDS = [
    # ── Lung parenchyma ───────────────────────────────────────────────
    {
        "id": "ct.lung.ground_glass",
        "text": "ground glass",
        "domain": "lung",
        "severity": "abnormal",
        "interpretation": "Ground-glass opacity — hazy increased attenuation without obscuring vessels; consider infection, edema, or early interstitial disease.",
    },
    {
        "id": "ct.lung.consolidation",
        "text": "consolidation",
        "domain": "lung",
        "severity": "abnormal",
        "interpretation": "Consolidation — airspace opacification; consider pneumonia, atelectasis, or aspiration.",
    },
    {
        "id": "ct.lung.nodule",
        "text": "nodule",
        "domain": "lung",
        "severity": "abnormal",
        "interpretation": "Pulmonary nodule — focal opacity < 3 cm; characterize and follow up.",
    },
    {
        "id": "ct.lung.mass",
        "text": "mass",
        "domain": "lung",
        "severity": "abnormal",
        "interpretation": "Lung mass — lesion > 3 cm; needs tissue characterization.",
    },
    {
        "id": "ct.lung.cavity",
        "text": "cavitary",
        "domain": "lung",
        "severity": "abnormal",
        "interpretation": "Cavitary lesion — air-filled lesion within an opacity; consider infection or neoplasm.",
    },
    {
        "id": "ct.lung.atelectasis",
        "text": "atelectasis",
        "domain": "lung",
        "severity": "abnormal",
        "interpretation": "Atelectasis — collapsed lung tissue with volume loss.",
    },
    {
        "id": "ct.lung.emphysema",
        "text": "emphysema",
        "domain": "lung",
        "severity": "abnormal",
        "interpretation": "Emphysema — parenchymal destruction with enlarged airspaces.",
    },
    {
        "id": "ct.lung.bronchiectasis",
        "text": "bronchiectasis",
        "domain": "lung",
        "severity": "abnormal",
        "interpretation": "Bronchiectasis — dilated, thick-walled airways.",
    },
    {
        "id": "ct.lung.fibrosis",
        "text": "fibrosis",
        "domain": "lung",
        "severity": "abnormal",
        "interpretation": "Pulmonary fibrosis — interstitial thickening with architectural distortion.",
    },
    {
        "id": "ct.lung.pulmonary_edema",
        "text": "pulmonary edema",
        "domain": "lung",
        "severity": "abnormal",
        "interpretation": "Pulmonary edema — interstitial/alveolar fluid; cardiogenic vs non-cardiogenic.",
    },
    {
        "id": "ct.lung.pulmonary_embolism",
        "text": "pulmonary embolism",
        "domain": "vascular",
        "severity": "critical",
        "interpretation": "Pulmonary embolism — intraluminal filling defect in a pulmonary artery.",
    },

    # ── Pleura / mediastinum ──────────────────────────────────────────
    {
        "id": "ct.pleural.pneumothorax",
        "text": "pneumothorax",
        "domain": "pleura",
        "severity": "critical",
        "interpretation": "Pneumothorax — air in the pleural space with a visceral pleural line.",
    },
    {
        "id": "ct.pleural.effusion",
        "text": "pleural effusion",
        "domain": "pleura",
        "severity": "abnormal",
        "interpretation": "Pleural effusion — fluid in the pleural space.",
    },
    {
        "id": "ct.mediastinum.lymphadenopathy",
        "text": "lymphadenopathy",
        "domain": "mediastinum",
        "severity": "abnormal",
        "interpretation": "Lymphadenopathy — enlarged lymph nodes.",
    },

    # ── Abdomen / other ───────────────────────────────────────────────
    {
        "id": "ct.liver.lesion",
        "text": "liver lesion",
        "domain": "abdomen",
        "severity": "abnormal",
        "interpretation": "Liver lesion — focal hepatic abnormality; characterize.",
    },
    {
        "id": "ct.normal.clear",
        "text": "unremarkable",
        "domain": "normal",
        "severity": "normal",
        "interpretation": "Unremarkable — no acute abnormality.",
    },
]
