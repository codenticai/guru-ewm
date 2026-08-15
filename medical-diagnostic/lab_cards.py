"""
Laboratory knowledge-card corpus + reference ranges for the HLLSet lattice.

Two complementary layers, mirroring the ECG design:

1. `LAB_CARDS` — named-finding text cards (e.g. "anemia", "hyperkalemia")
   matched by token-overlap inclusion against the OCR'd report text.

2. `LAB_REFERENCE_RANGES` — a table-driven numeric classifier: each analyte
   has aliases (for regex extraction), a reference interval, and the findings
   to emit when the value falls low / high / in-range. This generalizes the
   ECG reference-range idea to common chemistry/hematology panels.

IMPORTANT: Reference values are adult population approximations for
architecture validation only. This is NOT a medical device and must not be
used for real clinical decisions.
"""

LAB_CARDS = [
    # ── Complete blood count (CBC) ─────────────────────────────────────
    {
        "id": "lab.cbc.anemia",
        "text": "anemia low hemoglobin low hemoglobin",
        "domain": "hematology",
        "severity": "abnormal",
        "interpretation": "Anemia — low hemoglobin; evaluate cause (blood loss, hemolysis, underproduction).",
    },
    {
        "id": "lab.cbc.leukocytosis",
        "text": "leukocytosis elevated white blood cell elevated wbc",
        "domain": "hematology",
        "severity": "abnormal",
        "interpretation": "Leukocytosis — elevated white cell count; consider infection, inflammation, or steroid effect.",
    },
    {
        "id": "lab.cbc.leukopenia",
        "text": "leukopenia low white blood cell low wbc",
        "domain": "hematology",
        "severity": "abnormal",
        "interpretation": "Leukopenia — low white cell count; consider viral infection, marrow suppression, or drug effect.",
    },
    {
        "id": "lab.cbc.thrombocytopenia",
        "text": "thrombocytopenia low platelets low platelet count",
        "domain": "hematology",
        "severity": "abnormal",
        "interpretation": "Thrombocytopenia — low platelet count; bleeding risk if severe.",
    },
    {
        "id": "lab.cbc.thrombocytosis",
        "text": "thrombocytosis elevated platelets high platelet count",
        "domain": "hematology",
        "severity": "abnormal",
        "interpretation": "Thrombocytosis — elevated platelet count; reactive or myeloproliferative.",
    },

    # ── Basic metabolic panel (BMP) / electrolytes ─────────────────────
    {
        "id": "lab.bmp.hyperkalemia",
        "text": "hyperkalemia high potassium elevated potassium",
        "domain": "chemistry",
        "severity": "critical",
        "interpretation": "Hyperkalemia — high potassium; risk of arrhythmia; urgent evaluation.",
    },
    {
        "id": "lab.bmp.hypokalemia",
        "text": "hypokalemia low potassium",
        "domain": "chemistry",
        "severity": "abnormal",
        "interpretation": "Hypokalemia — low potassium; consider GI losses, diuretics, or renal wasting.",
    },
    {
        "id": "lab.bmp.hyponatremia",
        "text": "hyponatremia low sodium",
        "domain": "chemistry",
        "severity": "abnormal",
        "interpretation": "Hyponatremia — low sodium; assess volume status and osmolarity.",
    },
    {
        "id": "lab.bmp.hypernatremia",
        "text": "hypernatremia high sodium elevated sodium",
        "domain": "chemistry",
        "severity": "abnormal",
        "interpretation": "Hypernatremia — high sodium; usually reflects water deficit.",
    },
    {
        "id": "lab.bmp.hyperglycemia",
        "text": "hyperglycemia high glucose elevated blood sugar",
        "domain": "chemistry",
        "severity": "abnormal",
        "interpretation": "Hyperglycemia — elevated glucose; consider diabetes or stress response.",
    },
    {
        "id": "lab.bmp.hypoglycemia",
        "text": "hypoglycemia low glucose low blood sugar",
        "domain": "chemistry",
        "severity": "critical",
        "interpretation": "Hypoglycemia — low glucose; potentially dangerous; correct promptly.",
    },
    {
        "id": "lab.bmp.hypocalcemia",
        "text": "hypocalcemia low calcium",
        "domain": "chemistry",
        "severity": "abnormal",
        "interpretation": "Hypocalcemia — low calcium; consider tetany, QT prolongation, or parathyroid disease.",
    },

    # ── Renal / hepatic ─────────────────────────────────────────────────
    {
        "id": "lab.renal.elevated_creatinine",
        "text": "elevated creatinine high creatinine acute kidney injury",
        "domain": "renal",
        "severity": "abnormal",
        "interpretation": "Elevated creatinine — impaired renal function; assess for acute kidney injury.",
    },
    {
        "id": "lab.liver.transaminitis",
        "text": "elevated alt elevated ast transaminitis elevated liver enzymes",
        "domain": "hepatic",
        "severity": "abnormal",
        "interpretation": "Transaminitis — elevated liver enzymes; consider hepatocellular injury.",
    },
    {
        "id": "lab.liver.hyperbilirubinemia",
        "text": "hyperbilirubinemia elevated bilirubin jaundice",
        "domain": "hepatic",
        "severity": "abnormal",
        "interpretation": "Hyperbilirubinemia — elevated bilirubin; consider hemolysis or hepatobiliary disease.",
    },
    {
        "id": "lab.cardiac.elevated_troponin",
        "text": "elevated troponin high troponin positive troponin",
        "domain": "cardiac",
        "severity": "critical",
        "interpretation": "Elevated troponin — myocardial injury; urgent cardiology evaluation.",
    },
]


# Numeric reference intervals. `low`/`high` may be None for analytes where only
# one side is clinically meaningful. Each side maps to a finding to emit.
LAB_REFERENCE_RANGES = [
    {
        "id": "hemoglobin",
        "aliases": ["hemoglobin", "hgb"],
        "unit": "g/dL",
        "low": 13.5, "high": 17.5,
        "low_finding": {"signal": "lab.cbc.anemia", "severity": "abnormal",
                        "note": "Anemia — hemoglobin below reference range."},
        "high_finding": {"signal": "lab.cbc.polycythemia", "severity": "abnormal",
                         "note": "Polycythemia — hemoglobin above reference range."},
    },
    {
        "id": "white_blood_cells",
        "aliases": ["wbc", "white blood cell", "white blood count", "leukocyte"],
        "unit": "x10^9/L",
        "low": 4.0, "high": 11.0,
        "low_finding": {"signal": "lab.cbc.leukopenia", "severity": "abnormal",
                        "note": "Leukopenia — white cell count below reference range."},
        "high_finding": {"signal": "lab.cbc.leukocytosis", "severity": "abnormal",
                         "note": "Leukocytosis — white cell count above reference range."},
    },
    {
        "id": "platelets",
        "aliases": ["platelets", "platelet"],
        "unit": "x10^9/L",
        "low": 150, "high": 450,
        "low_finding": {"signal": "lab.cbc.thrombocytopenia", "severity": "abnormal",
                        "note": "Thrombocytopenia — platelet count below reference range."},
        "high_finding": {"signal": "lab.cbc.thrombocytosis", "severity": "abnormal",
                         "note": "Thrombocytosis — platelet count above reference range."},
    },
    {
        "id": "sodium",
        "aliases": ["sodium", "na"],
        "unit": "mmol/L",
        "low": 135, "high": 145,
        "low_finding": {"signal": "lab.bmp.hyponatremia", "severity": "abnormal",
                        "note": "Hyponatremia — sodium below reference range."},
        "high_finding": {"signal": "lab.bmp.hypernatremia", "severity": "abnormal",
                         "note": "Hypernatremia — sodium above reference range."},
    },
    {
        "id": "potassium",
        "aliases": ["potassium", "k"],
        "unit": "mmol/L",
        "low": 3.5, "high": 5.0,
        "low_finding": {"signal": "lab.bmp.hypokalemia", "severity": "abnormal",
                        "note": "Hypokalemia — potassium below reference range."},
        "high_finding": {"signal": "lab.bmp.hyperkalemia", "severity": "critical",
                         "note": "Hyperkalemia — potassium above reference range; arrhythmia risk."},
    },
    {
        "id": "chloride",
        "aliases": ["chloride"],
        "unit": "mmol/L",
        "low": 98, "high": 106,
        "low_finding": {"signal": "lab.bmp.hypochloremia", "severity": "abnormal",
                        "note": "Hypochloremia — chloride below reference range."},
        "high_finding": {"signal": "lab.bmp.hyperchloremia", "severity": "abnormal",
                         "note": "Hyperchloremia — chloride above reference range."},
    },
    {
        "id": "bicarbonate",
        "aliases": ["bicarbonate", "hco3", "co2"],
        "unit": "mmol/L",
        "low": 22, "high": 29,
        "low_finding": {"signal": "lab.chem.metabolic_acidosis", "severity": "abnormal",
                        "note": "Metabolic acidosis — bicarbonate below reference range."},
        "high_finding": {"signal": "lab.chem.metabolic_alkalosis", "severity": "abnormal",
                         "note": "Metabolic alkalosis — bicarbonate above reference range."},
    },
    {
        "id": "creatinine",
        "aliases": ["creatinine"],
        "unit": "mg/dL",
        "low": None, "high": 1.2,
        "low_finding": None,
        "high_finding": {"signal": "lab.renal.elevated_creatinine", "severity": "abnormal",
                         "note": "Elevated creatinine — impaired renal function."},
    },
    {
        "id": "bun",
        "aliases": ["bun", "blood urea nitrogen", "urea"],
        "unit": "mg/dL",
        "low": None, "high": 20,
        "low_finding": None,
        "high_finding": {"signal": "lab.renal.elevated_bun", "severity": "abnormal",
                         "note": "Elevated BUN — consider dehydration or renal dysfunction."},
    },
    {
        "id": "glucose",
        "aliases": ["glucose", "blood sugar"],
        "unit": "mg/dL",
        "low": 70, "high": 100,
        "low_finding": {"signal": "lab.bmp.hypoglycemia", "severity": "critical",
                        "note": "Hypoglycemia — glucose below reference range."},
        "high_finding": {"signal": "lab.bmp.hyperglycemia", "severity": "abnormal",
                         "note": "Hyperglycemia — glucose above reference range."},
    },
    {
        "id": "calcium",
        "aliases": ["calcium"],
        "unit": "mg/dL",
        "low": 8.5, "high": 10.2,
        "low_finding": {"signal": "lab.bmp.hypocalcemia", "severity": "abnormal",
                        "note": "Hypocalcemia — calcium below reference range."},
        "high_finding": {"signal": "lab.bmp.hypercalcemia", "severity": "abnormal",
                         "note": "Hypercalcemia — calcium above reference range."},
    },
    {
        "id": "magnesium",
        "aliases": ["magnesium"],
        "unit": "mg/dL",
        "low": 1.7, "high": 2.2,
        "low_finding": {"signal": "lab.chem.hypomagnesemia", "severity": "abnormal",
                        "note": "Hypomagnesemia — magnesium below reference range."},
        "high_finding": {"signal": "lab.chem.hypermagnesemia", "severity": "abnormal",
                         "note": "Hypermagnesemia — magnesium above reference range."},
    },
    {
        "id": "alt",
        "aliases": ["alt", "sgpt"],
        "unit": "U/L",
        "low": None, "high": 56,
        "low_finding": None,
        "high_finding": {"signal": "lab.liver.transaminitis", "severity": "abnormal",
                         "note": "Elevated ALT — hepatocellular injury."},
    },
    {
        "id": "ast",
        "aliases": ["ast", "sgot"],
        "unit": "U/L",
        "low": None, "high": 40,
        "low_finding": None,
        "high_finding": {"signal": "lab.liver.transaminitis", "severity": "abnormal",
                         "note": "Elevated AST — hepatocellular injury."},
    },
    {
        "id": "total_bilirubin",
        "aliases": ["bilirubin", "total bilirubin"],
        "unit": "mg/dL",
        "low": None, "high": 1.2,
        "low_finding": None,
        "high_finding": {"signal": "lab.liver.hyperbilirubinemia", "severity": "abnormal",
                         "note": "Hyperbilirubinemia — bilirubin above reference range."},
    },
    {
        "id": "albumin",
        "aliases": ["albumin"],
        "unit": "g/dL",
        "low": 3.5, "high": 5.0,
        "low_finding": {"signal": "lab.chem.hypoalbuminemia", "severity": "abnormal",
                        "note": "Hypoalbuminemia — albumin below reference range."},
        "high_finding": None,
    },
    {
        "id": "troponin",
        "aliases": ["troponin"],
        "unit": "ng/mL",
        "low": None, "high": 0.04,
        "low_finding": None,
        "high_finding": {"signal": "lab.cardiac.elevated_troponin", "severity": "critical",
                         "note": "Elevated troponin — myocardial injury."},
    },
]
