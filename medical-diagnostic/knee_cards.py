"""
Knee MRI (MR) knowledge-card corpus for the HLLSet lattice.

Mirrors ecg_cards.py / xray_cards.py: each card is one atomic knee finding.
The `text` field is tokenized by hllset-next and the card is retrieved by
token-overlap inclusion — how confidently the OCR'd radiology report
CONTAINS the card's pattern.

The twelve findings mirror the clinically important knee abnormalities of the
RSNA knee-MRI task (ligament tears, meniscal damage, cartilage loss, marrow
lesions, fracture) so the same lattice pipeline can analyze knee reports.

IMPORTANT: These are reference interpretation criteria for architecture
validation only. This is NOT a medical device and must not be used for real
clinical decisions.
"""

KNEE_CARDS = [
    # ── Cruciate / collateral ligaments ────────────────────────────────
    {
        "id": "knee.acl_tear",
        "text": "acl tear",
        "domain": "ligament",
        "severity": "abnormal",
        "interpretation": "ACL tear — anterior cruciate ligament disruption (partial or complete).",
    },
    {
        "id": "knee.pcl_tear",
        "text": "pcl tear",
        "domain": "ligament",
        "severity": "abnormal",
        "interpretation": "PCL tear — posterior cruciate ligament disruption.",
    },
    {
        "id": "knee.mcl_tear",
        "text": "mcl tear",
        "domain": "ligament",
        "severity": "abnormal",
        "interpretation": "MCL tear — medial collateral ligament sprain or disruption.",
    },
    {
        "id": "knee.lcl_tear",
        "text": "lcl tear",
        "domain": "ligament",
        "severity": "abnormal",
        "interpretation": "LCL tear — lateral collateral ligament disruption.",
    },

    # ── Meniscus / cartilage ───────────────────────────────────────────
    {
        "id": "knee.meniscal_tear",
        "text": "meniscal tear",
        "domain": "meniscus",
        "severity": "abnormal",
        "interpretation": "Meniscal tear — medial or lateral meniscus disruption; correlate location and pattern.",
    },
    {
        "id": "knee.cartilage_loss",
        "text": "cartilage loss",
        "domain": "cartilage",
        "severity": "abnormal",
        "interpretation": "Cartilage loss — chondral wear up to full-thickness loss of articular cartilage.",
    },
    {
        "id": "knee.chondral_defect",
        "text": "chondral defect",
        "domain": "cartilage",
        "severity": "abnormal",
        "interpretation": "Chondral defect — focal articular cartilage lesion.",
    },

    # ── Degenerative / marrow / bone ───────────────────────────────────
    {
        "id": "knee.osteoarthritis",
        "text": "osteoarthritis",
        "domain": "degenerative",
        "severity": "abnormal",
        "interpretation": "Osteoarthritis — joint space narrowing, osteophytes, and subchondral change.",
    },
    {
        "id": "knee.bone_marrow_edema",
        "text": "bone marrow edema",
        "domain": "marrow",
        "severity": "abnormal",
        "interpretation": "Bone marrow edema — increased marrow signal; consider contusion or marrow lesion.",
    },
    {
        "id": "knee.tibial_plateau_fracture",
        "text": "tibial plateau fracture",
        "domain": "fracture",
        "severity": "abnormal",
        "interpretation": "Tibial plateau fracture — proximal tibia fracture; assess for articular depression.",
    },

    # ── Patellofemoral ─────────────────────────────────────────────────
    {
        "id": "knee.patellar_dislocation",
        "text": "patellar dislocation",
        "domain": "patellofemoral",
        "severity": "abnormal",
        "interpretation": "Patellar dislocation — lateral patellar subluxation or dislocation.",
    },
    {
        "id": "knee.patellar_tendinopathy",
        "text": "patellar tendinopathy",
        "domain": "patellofemoral",
        "severity": "abnormal",
        "interpretation": "Patellar tendinopathy — proximal patellar tendon change (jumper's knee).",
    },
]
