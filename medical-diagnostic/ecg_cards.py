"""
ECG knowledge-card corpus for the HLLSet lattice.

Each card is one atomic ECG interpretation criterion. The `text` field is
tokenized into a 4KB HLLSet by hllset-next (`--serve`) and the card is
retrieved by BSS inclusion — how confidently the OCR'd report CONTAINS the
card's pattern.

IMPORTANT: These are reference interpretation criteria for architecture
validation only. This is NOT a medical device and must not be used for real
clinical decisions.
"""

ECG_CARDS = [
    # ── Rhythm ──────────────────────────────────────────────────────────
    {
        "id": "ecg.rhythm.sinus",
        "text": "sinus rhythm normal sinus rhythm",
        "domain": "rhythm",
        "severity": "normal",
        "interpretation": "Normal sinus rhythm — regular P waves preceding each QRS complex.",
    },
    {
        "id": "ecg.rhythm.sinus_bradycardia",
        "text": "sinus bradycardia bradycardia slow heart rate",
        "domain": "rhythm",
        "severity": "abnormal",
        "interpretation": "Sinus bradycardia — heart rate below 60 bpm. May be physiologic in athletes.",
    },
    {
        "id": "ecg.rhythm.sinus_tachycardia",
        "text": "sinus tachycardia tachycardia fast heart rate",
        "domain": "rhythm",
        "severity": "abnormal",
        "interpretation": "Sinus tachycardia — heart rate above 100 bpm. Consider pain, fever, anemia, dehydration.",
    },
    {
        "id": "ecg.rhythm.sinus_arrhythmia",
        "text": "sinus arrhythmia respiratory sinus arrhythmia",
        "domain": "rhythm",
        "severity": "normal",
        "interpretation": "Sinus arrhythmia — physiologic rate variation with respiration, common in the young.",
    },
    {
        "id": "ecg.rhythm.afib",
        "text": "atrial fibrillation afib af irregularly irregular rhythm",
        "domain": "rhythm",
        "severity": "abnormal",
        "interpretation": "Atrial fibrillation — absence of P waves with irregularly irregular ventricular rhythm. Stroke risk assessment advised.",
    },
    {
        "id": "ecg.rhythm.aflutter",
        "text": "atrial flutter flutter sawtooth flutter waves",
        "domain": "rhythm",
        "severity": "abnormal",
        "interpretation": "Atrial flutter — sawtooth flutter waves, often with regular 2:1 or 3:1 conduction.",
    },
    {
        "id": "ecg.rhythm.svt",
        "text": "supraventricular tachycardia svt narrow complex tachycardia",
        "domain": "rhythm",
        "severity": "abnormal",
        "interpretation": "Supraventricular tachycardia — regular narrow-complex tachycardia, rate typically 150-250 bpm.",
    },
    {
        "id": "ecg.rhythm.pac",
        "text": "premature atrial contraction pac atrial premature beat",
        "domain": "rhythm",
        "severity": "benign",
        "interpretation": "Premature atrial contraction — early P wave with usually normal QRS. Generally benign.",
    },
    {
        "id": "ecg.rhythm.pvc",
        "text": "premature ventricular contraction pvc ventricular ectopic ventricular premature beat",
        "domain": "rhythm",
        "severity": "abnormal",
        "interpretation": "Premature ventricular contraction — wide, early QRS without preceding P wave. Frequent or multifocal PVCs warrant review.",
    },
    {
        "id": "ecg.rhythm.vtach",
        "text": "ventricular tachycardia vtach wide complex tachycardia",
        "domain": "rhythm",
        "severity": "critical",
        "interpretation": "Ventricular tachycardia — wide-complex tachycardia. Potentially life-threatening; urgent management required.",
    },
    {
        "id": "ecg.rhythm.vfib",
        "text": "ventricular fibrillation vfib vf cardiac arrest",
        "domain": "rhythm",
        "severity": "critical",
        "interpretation": "Ventricular fibrillation — chaotic rhythm with no organized QRS. Medical emergency: immediate defibrillation.",
    },
    {
        "id": "ecg.rhythm.av_block_1",
        "text": "first degree av block first degree heart block prolonged pr",
        "domain": "rhythm",
        "severity": "abnormal",
        "interpretation": "First-degree AV block — PR interval > 200 ms with every P wave conducted. Often benign.",
    },
    {
        "id": "ecg.rhythm.av_block_2_mobitz1",
        "text": "mobitz i mobitz type 1 wenckebach second degree av block type 1",
        "domain": "rhythm",
        "severity": "abnormal",
        "interpretation": "Second-degree AV block Mobitz I (Wenckebach) — progressive PR prolongation with dropped QRS. Usually benign.",
    },
    {
        "id": "ecg.rhythm.av_block_2_mobitz2",
        "text": "mobitz ii mobitz type 2 second degree av block type 2",
        "domain": "rhythm",
        "severity": "critical",
        "interpretation": "Second-degree AV block Mobitz II — intermittent non-conducted P waves with fixed PR. May progress to complete block.",
    },
    {
        "id": "ecg.rhythm.av_block_3",
        "text": "third degree av block complete heart block complete av block",
        "domain": "rhythm",
        "severity": "critical",
        "interpretation": "Third-degree (complete) AV block — AV dissociation with escape rhythm. Urgent pacing evaluation required.",
    },
    {
        "id": "ecg.rhythm.lbbb",
        "text": "left bundle branch block lbbb wide qrs left bundle",
        "domain": "rhythm",
        "severity": "abnormal",
        "interpretation": "Left bundle branch block — QRS >= 120 ms with broad R waves in lateral leads. New LBBB is a red flag.",
    },
    {
        "id": "ecg.rhythm.rbbb",
        "text": "right bundle branch block rbbb right bundle",
        "domain": "rhythm",
        "severity": "abnormal",
        "interpretation": "Right bundle branch block — QRS >= 120 ms with RSR' pattern in V1-V3.",
    },
    {
        "id": "ecg.rhythm.lafb",
        "text": "left anterior fascicular block lafb left anterior hemiblock",
        "domain": "rhythm",
        "severity": "abnormal",
        "interpretation": "Left anterior fascicular block — left axis deviation with qR in I/aVL and rS in II/III/aVF.",
    },
    {
        "id": "ecg.rhythm.lpfb",
        "text": "left posterior fascicular block lpfb left posterior hemiblock",
        "domain": "rhythm",
        "severity": "abnormal",
        "interpretation": "Left posterior fascicular block — right axis deviation with rS in I/aVL and qR in II/III/aVF.",
    },
    {
        "id": "ecg.rhythm.bifascicular",
        "text": "bifascicular block",
        "domain": "rhythm",
        "severity": "abnormal",
        "interpretation": "Bifascicular block — RBBB with left anterior or left posterior fascicular block. Monitor for progression.",
    },
    {
        "id": "ecg.rhythm.junctional",
        "text": "junctional rhythm junctional escape rhythm",
        "domain": "rhythm",
        "severity": "abnormal",
        "interpretation": "Junctional rhythm — narrow QRS without preceding P waves, rate 40-60 bpm.",
    },
    {
        "id": "ecg.rhythm.idioventricular",
        "text": "idioventricular rhythm ventricular escape rhythm",
        "domain": "rhythm",
        "severity": "critical",
        "interpretation": "Idioventricular rhythm — wide QRS escape at 20-40 bpm. Serious; urgent evaluation.",
    },
    {
        "id": "ecg.rhythm.paced",
        "text": "paced rhythm ventricular paced pacemaker rhythm",
        "domain": "rhythm",
        "severity": "normal",
        "interpretation": "Paced rhythm — ventricular pacing spikes preceding wide QRS. Consistent with pacemaker.",
    },
    {
        "id": "ecg.rhythm.asystole",
        "text": "asystole flatline no cardiac activity",
        "domain": "rhythm",
        "severity": "critical",
        "interpretation": "Asystole — no electrical activity. Medical emergency: CPR and ACLS protocol.",
    },
    {
        "id": "ecg.rhythm.wap",
        "text": "wandering atrial pacemaker wandering pacemaker",
        "domain": "rhythm",
        "severity": "benign",
        "interpretation": "Wandering atrial pacemaker — varying P wave morphology with normal rate. Benign.",
    },
    {
        "id": "ecg.rhythm.wpw",
        "text": "wolff parkinson white wpw delta wave preexcitation",
        "domain": "rhythm",
        "severity": "abnormal",
        "interpretation": "Wolff-Parkinson-White — short PR with delta wave. Accessory pathway; risk of SVT.",
    },
    {
        "id": "ecg.rhythm.brugada",
        "text": "brugada pattern brugada syndrome",
        "domain": "rhythm",
        "severity": "critical",
        "interpretation": "Brugada pattern — coved ST elevation in V1-V3 with RBBB pattern. Risk of ventricular arrhythmia.",
    },
    {
        "id": "ecg.rhythm.long_qt_syndrome",
        "text": "long qt syndrome lqts",
        "domain": "rhythm",
        "severity": "critical",
        "interpretation": "Long QT syndrome — prolonged QTc with risk of torsades de pointes. Avoid QT-prolonging drugs.",
    },
    {
        "id": "ecg.rhythm.short_qt_syndrome",
        "text": "short qt syndrome sqts",
        "domain": "rhythm",
        "severity": "critical",
        "interpretation": "Short QT syndrome — abnormally short QTc. Risk of arrhythmia; specialist review.",
    },
    {
        "id": "ecg.rhythm.torsades",
        "text": "torsades de pointes torsade polymorphic ventricular tachycardia",
        "domain": "rhythm",
        "severity": "critical",
        "interpretation": "Torsades de pointes — polymorphic VT with twisting QRS axis. Emergency; treat with magnesium and defibrillation if unstable.",
    },

    # ── Rate (named findings; numeric intervals are classified deterministically) ──
    # NOTE: single-word cards are deliberately excluded — a one-token card maps
    # to a ~1-bit HLLSet whose BSS is statistically meaningless (bit collisions
    # with dense queries). "bradycardia"/"tachycardia" are already covered by
    # the multi-word rhythm cards and by numeric rate classification.
    {
        "id": "ecg.rate.tachycardia",
        "text": "sinus tachycardia rapid heart rate",
        "domain": "rate",
        "severity": "abnormal",
        "interpretation": "Tachycardia — heart rate above 100 bpm.",
    },

    # ── ST / T changes ─────────────────────────────────────────────────
    {
        "id": "ecg.st.elevation",
        "text": "st elevation st segment elevation",
        "domain": "st",
        "severity": "critical",
        "interpretation": "ST elevation — possible acute ischemia/infarction or pericarditis. Urgent evaluation.",
    },
    {
        "id": "ecg.st.depression",
        "text": "st depression st segment depression",
        "domain": "st",
        "severity": "abnormal",
        "interpretation": "ST depression — possible ischemia or strain pattern.",
    },
    {
        "id": "ecg.st.stemi",
        "text": "st elevation myocardial infarction stemi acute myocardial infarction",
        "domain": "st",
        "severity": "critical",
        "interpretation": "ST-elevation myocardial infarction — emergency. Immediate reperfusion strategy required.",
    },
    {
        "id": "ecg.st.nstemi",
        "text": "non st elevation nstemi non st elevation myocardial infarction",
        "domain": "st",
        "severity": "critical",
        "interpretation": "Non-ST-elevation MI — ischemia without persistent ST elevation. Urgent risk stratification.",
    },
    {
        "id": "ecg.st.anterior_stemi",
        "text": "anterior st elevation anterior stemi v1 v4",
        "domain": "st",
        "severity": "critical",
        "interpretation": "Anterior STEMI — ST elevation in V1-V4. LAD territory; emergency.",
    },
    {
        "id": "ecg.st.inferior_stemi",
        "text": "inferior st elevation inferior stemi ii iii avf",
        "domain": "st",
        "severity": "critical",
        "interpretation": "Inferior STEMI — ST elevation in II, III, aVF. Consider right-sided leads.",
    },
    {
        "id": "ecg.st.lateral_stemi",
        "text": "lateral st elevation lateral stemi i avl v5 v6",
        "domain": "st",
        "severity": "critical",
        "interpretation": "Lateral STEMI — ST elevation in I, aVL, V5-V6. Circumflex territory.",
    },
    {
        "id": "ecg.t.inversion",
        "text": "t wave inversion inverted t waves",
        "domain": "t",
        "severity": "abnormal",
        "interpretation": "T wave inversion — may indicate ischemia, strain, or be a normal variant.",
    },
    {
        "id": "ecg.t.peaked",
        "text": "peaked t waves tall t waves hyperacute t waves",
        "domain": "t",
        "severity": "abnormal",
        "interpretation": "Peaked T waves — consider hyperkalemia or early ischemia (hyperacute).",
    },
    {
        "id": "ecg.t.flattened",
        "text": "flattened t waves flat t waves",
        "domain": "t",
        "severity": "abnormal",
        "interpretation": "Flattened T waves — nonspecific; may indicate ischemia or electrolyte imbalance.",
    },
    {
        "id": "ecg.t.biphasic",
        "text": "biphasic t waves",
        "domain": "t",
        "severity": "abnormal",
        "interpretation": "Biphasic T waves — may indicate ischemia, especially in anterior leads (Wellens).",
    },
    {
        "id": "ecg.t.wellens",
        "text": "wellens syndrome wellens pattern",
        "domain": "t",
        "severity": "critical",
        "interpretation": "Wellens syndrome — deeply inverted/biphasic T waves in V2-V3. Critical LAD stenosis; urgent angiography.",
    },
    {
        "id": "ecg.t.pericarditis",
        "text": "pericarditis diffuse st elevation pr depression",
        "domain": "t",
        "severity": "abnormal",
        "interpretation": "Pericarditis — diffuse ST elevation with PR depression. Inflammatory, not ischemic.",
    },

    # ── Axis / hypertrophy ─────────────────────────────────────────────
    {
        "id": "ecg.axis.lad",
        "text": "left axis deviation lad",
        "domain": "axis",
        "severity": "abnormal",
        "interpretation": "Left axis deviation — QRS axis more negative than -30°. Consider LVH or fascicular block.",
    },
    {
        "id": "ecg.axis.rad",
        "text": "right axis deviation rad",
        "domain": "axis",
        "severity": "abnormal",
        "interpretation": "Right axis deviation — QRS axis beyond +90°. Consider RVH or pulmonary disease.",
    },
    {
        "id": "ecg.axis.extreme",
        "text": "extreme axis deviation northwest axis indeterminate axis",
        "domain": "axis",
        "severity": "abnormal",
        "interpretation": "Extreme axis deviation — QRS axis between -90° and -180°. Often ventricular rhythm.",
    },
    {
        "id": "ecg.axis.normal_axis",
        "text": "normal axis normal qrs axis",
        "domain": "axis",
        "severity": "normal",
        "interpretation": "QRS axis within normal range.",
    },
    {
        "id": "ecg.hypertrophy.lvh",
        "text": "left ventricular hypertrophy lvh",
        "domain": "hypertrophy",
        "severity": "abnormal",
        "interpretation": "Left ventricular hypertrophy — voltage criteria with possible strain pattern.",
    },
    {
        "id": "ecg.hypertrophy.rvh",
        "text": "right ventricular hypertrophy rvh",
        "domain": "hypertrophy",
        "severity": "abnormal",
        "interpretation": "Right ventricular hypertrophy — right axis with tall R in V1.",
    },
    {
        "id": "ecg.hypertrophy.lae",
        "text": "left atrial enlargement lae left atrial abnormality",
        "domain": "hypertrophy",
        "severity": "abnormal",
        "interpretation": "Left atrial enlargement — broad notched P wave in II (P mitrale).",
    },
    {
        "id": "ecg.hypertrophy.rae",
        "text": "right atrial enlargement rae right atrial abnormality",
        "domain": "hypertrophy",
        "severity": "abnormal",
        "interpretation": "Right atrial enlargement — tall peaked P wave in II (P pulmonale).",
    },
    {
        "id": "ecg.hypertrophy.biatrial",
        "text": "biatrial enlargement biatrial abnormality",
        "domain": "hypertrophy",
        "severity": "abnormal",
        "interpretation": "Biatrial enlargement — combined left and right atrial abnormality.",
    },

    # ── Ischemia / infarction ──────────────────────────────────────────
    {
        "id": "ecg.mi.anterior",
        "text": "anterior myocardial infarction anterior mi",
        "domain": "infarction",
        "severity": "critical",
        "interpretation": "Anterior myocardial infarction — Q waves/ST changes in anterior leads.",
    },
    {
        "id": "ecg.mi.inferior",
        "text": "inferior myocardial infarction inferior mi",
        "domain": "infarction",
        "severity": "critical",
        "interpretation": "Inferior myocardial infarction — Q waves/ST changes in II, III, aVF.",
    },
    {
        "id": "ecg.mi.lateral",
        "text": "lateral myocardial infarction lateral mi",
        "domain": "infarction",
        "severity": "critical",
        "interpretation": "Lateral myocardial infarction — Q waves/ST changes in I, aVL, V5-V6.",
    },
    {
        "id": "ecg.mi.posterior",
        "text": "posterior myocardial infarction posterior mi",
        "domain": "infarction",
        "severity": "critical",
        "interpretation": "Posterior myocardial infarction — tall R waves with ST depression in V1-V3.",
    },
    {
        "id": "ecg.mi.septal",
        "text": "septal myocardial infarction septal mi",
        "domain": "infarction",
        "severity": "critical",
        "interpretation": "Septal myocardial infarction — Q waves in V1-V2.",
    },
    {
        "id": "ecg.mi.old",
        "text": "old myocardial infarction old mi prior infarction",
        "domain": "infarction",
        "severity": "abnormal",
        "interpretation": "Old myocardial infarction — pathologic Q waves without acute ST changes.",
    },
    {
        "id": "ecg.mi.pathologic_q",
        "text": "pathologic q waves significant q waves",
        "domain": "infarction",
        "severity": "abnormal",
        "interpretation": "Pathologic Q waves — consistent with prior myocardial infarction.",
    },
    {
        "id": "ecg.mi.ischemia",
        "text": "ischemia ischemic changes myocardial ischemia",
        "domain": "infarction",
        "severity": "abnormal",
        "interpretation": "Myocardial ischemia — ST/T changes suggestive of ischemia. Correlate with symptoms and troponin.",
    },
    {
        "id": "ecg.mi.injury",
        "text": "injury pattern acute injury",
        "domain": "infarction",
        "severity": "critical",
        "interpretation": "Acute injury pattern — ST elevation of active injury. Emergency.",
    },

    # ── Electrolyte / drug effects ─────────────────────────────────────
    {
        "id": "ecg.electrolyte.hyperkalemia",
        "text": "hyperkalemia high potassium peaked t waves",
        "domain": "electrolyte",
        "severity": "critical",
        "interpretation": "Hyperkalemia — peaked T waves, widened QRS. Emergency; check potassium.",
    },
    {
        "id": "ecg.electrolyte.hypokalemia",
        "text": "hypokalemia low potassium u wave",
        "domain": "electrolyte",
        "severity": "abnormal",
        "interpretation": "Hypokalemia — flattened T waves with prominent U waves.",
    },
    {
        "id": "ecg.electrolyte.hypercalcemia",
        "text": "hypercalcemia high calcium short qt",
        "domain": "electrolyte",
        "severity": "abnormal",
        "interpretation": "Hypercalcemia — shortened QT interval.",
    },
    {
        "id": "ecg.electrolyte.hypocalcemia",
        "text": "hypocalcemia low calcium prolonged qt",
        "domain": "electrolyte",
        "severity": "abnormal",
        "interpretation": "Hypocalcemia — prolonged QT interval.",
    },
    {
        "id": "ecg.electrolyte.digoxin",
        "text": "digoxin effect digitalis effect",
        "domain": "electrolyte",
        "severity": "abnormal",
        "interpretation": "Digoxin effect — scooped ST depression. Not toxicity by itself.",
    },
    {
        "id": "ecg.electrolyte.drug_qt",
        "text": "drug induced qt prolongation qt prolonging medication",
        "domain": "electrolyte",
        "severity": "abnormal",
        "interpretation": "QT prolongation possibly drug-induced — review medication list.",
    },

    # ── Normal / nonspecific / other ───────────────────────────────────
    {
        "id": "ecg.normal.ecg",
        "text": "normal ecg normal electrocardiogram",
        "domain": "normal",
        "severity": "normal",
        "interpretation": "Normal ECG — no acute abnormality.",
    },
    {
        "id": "ecg.normal.limits",
        "text": "within normal limits no abnormality",
        "domain": "normal",
        "severity": "normal",
        "interpretation": "ECG within normal limits.",
    },
    {
        "id": "ecg.normal.unconfirmed",
        "text": "unconfirmed diagnosis unconfirmed",
        "domain": "normal",
        "severity": "normal",
        "interpretation": "Unconfirmed diagnosis — findings not yet clinically validated.",
    },
    {
        "id": "ecg.normal.borderline",
        "text": "borderline ecg borderline",
        "domain": "normal",
        "severity": "normal",
        "interpretation": "Borderline ECG — minor nonspecific changes.",
    },
    {
        "id": "ecg.normal.abnormal",
        "text": "abnormal ecg abnormal electrocardiogram",
        "domain": "normal",
        "severity": "abnormal",
        "interpretation": "Abnormal ECG — requires clinician interpretation.",
    },
    {
        "id": "ecg.other.poor_r_wave",
        "text": "poor r wave progression",
        "domain": "other",
        "severity": "abnormal",
        "interpretation": "Poor R wave progression — may indicate anterior MI or lead placement.",
    },
    {
        "id": "ecg.other.early_repol",
        "text": "early repolarization early repolarization pattern",
        "domain": "other",
        "severity": "benign",
        "interpretation": "Early repolarization — benign J-point elevation, common in young patients.",
    },
    {
        "id": "ecg.other.low_voltage",
        "text": "low voltage qrs low voltage",
        "domain": "other",
        "severity": "abnormal",
        "interpretation": "Low QRS voltage — consider pericardial effusion, obesity, or emphysema.",
    },
    {
        "id": "ecg.other.lv_strain",
        "text": "strain pattern lv strain",
        "domain": "other",
        "severity": "abnormal",
        "interpretation": "Strain pattern — ST depression with T inversion in lateral leads, often with LVH.",
    },
]


# Numeric reference ranges for deterministic interval classification.
#
# Token inclusion (BSS) cannot distinguish "short PR" (a finding) from
# "PR interval" (a measurement label) — the words overlap. Numeric values
# are therefore classified against reference ranges instead of BSS.
ECG_REFERENCE_RANGES = {
    "ventricular_rate_bpm": {"low": 60, "high": 100, "unit": "bpm"},
    "pr_interval_ms": {"low": 120, "high": 200, "unit": "ms"},
    "qrs_duration_ms": {"low": 0, "high": 120, "unit": "ms"},
    "qtc_ms": {"low": 350, "high": 450, "unit": "ms"},
}


def get_card_by_id(card_id: str):
    for card in ECG_CARDS:
        if card["id"] == card_id:
            return card
    return None
