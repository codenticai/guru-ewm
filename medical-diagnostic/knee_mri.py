"""
knee_mri.py — HLLSet-style knee MRI abnormality classifier (Python port).

Faithful port of the old Rust hllset-NanoLM pipeline:

  src/ocr/hllset_image.rs     → fingerprint_image (patch → token → n-gram)
  src/ocr/hllset_training.rs  → generate_synthetic (per-abnormality patterns)
  src/ocr/hllset_classifier.rs → classify (directional BSS tau/rho)
  src/ocr/hllset_reference.rs → build_reference_library (union per class)

The image is tokenized into patch-feature tokens (mean / std-dev / edge
density), expanded into unigrams+bigrams+trigrams, then compared to a
reference library of synthetic MRI patterns using directional BSS:

    tau = |scan ∩ ref| / |ref|      (inclusion of ref in scan)
    rho = |scan \\ ref| / |ref|      (exclusion)
    confidence = 0.7 * tau + 0.3 * (1 - rho)
    detected   = tau >= 0.70 and rho < 0.30

Exact Python sets are used instead of the 32,768-bit HLLSet approximation —
same math, deterministic, more precise.

NOTE: This is a pattern-matching demonstration over synthetic reference
images, NOT a trained medical model and NOT a medical device.
"""

import numpy as np

# ── Abnormality types ────────────────────────────────────────────────
ABNORMALITIES = [
    "acl_tear",
    "meniscal_tear",
    "osteoarthritis",
    "patellar_dislocation",
    "bone_marrow_lesion",
    "chondral_defect",
]

LABELS = {
    "acl_tear": "ACL Tear",
    "meniscal_tear": "Meniscal Tear",
    "osteoarthritis": "Osteoarthritis",
    "patellar_dislocation": "Patellar Dislocation",
    "bone_marrow_lesion": "Bone Marrow Lesion",
    "chondral_defect": "Chondral Defect",
}

SEVERITY = {
    "acl_tear": "high",
    "meniscal_tear": "moderate",
    "osteoarthritis": "moderate",
    "patellar_dislocation": "high",
    "bone_marrow_lesion": "moderate",
    "chondral_defect": "moderate",
}

# ── Image fingerprint config (mirrors HllsetImageConfig::default) ─────
DEFAULT_CONFIG = {
    "patch_size": 16,
    "intensity_bins": 4,
    "std_bins": 4,
    "edge_bins": 4,
    "edge_threshold": 30,
    "ngram_min": 1,
    "ngram_max": 3,
}


def _quantize(value: float, lo: float, hi: float, bins: int) -> int:
    norm = (value - lo) / (hi - lo)
    norm = max(0.0, min(norm, 0.9999))
    return int(norm * bins)


def _expand_ngrams(tokens, cfg):
    result = set()
    n = len(tokens)
    for k in range(cfg["ngram_min"], cfg["ngram_max"] + 1):
        if k > n:
            break
        for i in range(0, n - k + 1):
            if k == 1:
                result.add(tokens[i])
            else:
                result.add("|".join(tokens[i:i + k]))
    return result


def fingerprint_image(gray, cfg=None):
    """Convert a grayscale uint8 image (2D ndarray) to a set of n-gram tokens."""
    cfg = cfg or DEFAULT_CONFIG
    h, w = gray.shape
    ps = cfg["patch_size"]

    if w < ps or h < ps:
        # Tiny image → single patch (mirrors tokenize_single_patch).
        mean = float(gray.mean())
        std = float(gray.std())
        m = _quantize(mean, 0.0, 255.0, cfg["intensity_bins"])
        s = _quantize(std, 0.0, 128.0, cfg["std_bins"])
        return _expand_ngrams([f"p0_0_m{m}_s{s}_e0"], cfg)

    cols, rows = w // ps, h // ps
    tokens = []
    for r in range(rows):
        for c in range(cols):
            patch = gray[r * ps:(r + 1) * ps, c * ps:(c + 1) * ps].astype(np.float64)
            mean = float(patch.mean())
            std = float(patch.std())
            edge = 0
            if ps > 1:
                edge = int((np.abs(np.diff(patch, axis=1)) > cfg["edge_threshold"]).sum()
                           + (np.abs(np.diff(patch, axis=0)) > cfg["edge_threshold"]).sum())
            edge_density = edge / (2.0 * ps * ps)
            m = _quantize(mean, 0.0, 255.0, cfg["intensity_bins"])
            s = _quantize(std, 0.0, 128.0, cfg["std_bins"])
            e = _quantize(edge_density, 0.0, 1.0, cfg["edge_bins"])
            tokens.append(f"p{r}_{c}_m{m}_s{s}_e{e}")
    return _expand_ngrams(tokens, cfg)


# ── Synthetic MRI generators (ported from hllset_training.rs) ─────────
def generate_synthetic(abnormality: str, size: int = 256, noise: float = 0.05,
                       seed: int = 0) -> np.ndarray:
    """Generate a synthetic grayscale knee-MRI-like image for an abnormality."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    xf = x.astype(np.float64)
    yf = y.astype(np.float64)
    # Mild shared base (kept subtle so each abnormality's pattern dominates
    # the fingerprint rather than a shared gradient).
    img = (90.0 + xf / size * 20.0 + yf / size * 15.0).copy()

    def _noise(level):
        return (rng.random((size, size)) - 0.5) * 2.0 * level * 255.0

    if abnormality == "acl_tear":
        # Dark diagonal band (torn ACL).
        cx = cy = size / 2.0
        dist = np.abs((xf - cx) - (yf - cy)) / (size * 1.4)
        img[dist < 0.06] = 60.0 + rng.random((size, size))[dist < 0.06] * 30.0
        img += _noise(noise)
    elif abnormality == "meniscal_tear":
        # Dark triangular wedge in posterior region.
        cx, cy = size * 0.6, size * 0.7
        dx = np.abs(xf - cx)
        dy = yf - cy
        mask = (dy > 0.0) & (dx < dy * 1.2) & (dy < size * 0.35)
        dist = np.minimum(dx / (dy + 1.0), 1.0)
        img[mask] = 40.0 + dist[mask] * 30.0 + rng.random((size, size))[mask] * 20.0
        img += _noise(noise)
    elif abnormality == "osteoarthritis":
        # Thin bright band (joint space) between darker subchondral zones.
        mid = size * 0.5
        dist = np.abs(yf - mid)
        img[dist < size * 0.04] = 200.0 + rng.integers(0, 40, (size, size))[dist < size * 0.04]
        img[(dist >= size * 0.04) & (dist < size * 0.15)] = (
            50.0 + rng.random((size, size))[(dist >= size * 0.04) & (dist < size * 0.15)] * 30.0
        )
        img += _noise(noise * 0.8)
    elif abnormality == "patellar_dislocation":
        # Bright spot (contusion) + dark crescent (MPFL injury).
        cx, cy = size * 0.35, size * 0.4
        dist = np.sqrt((xf - cx) ** 2 + (yf - cy) ** 2)
        img[dist < size * 0.1] = 220.0 + rng.integers(0, 30, (size, size))[dist < size * 0.1]
        cx2, cy2 = size * 0.55, size * 0.45
        dist2 = np.sqrt((xf - cx2) ** 2 + (yf - cy2) ** 2)
        crescent = (dist2 < size * 0.08) & (dist2 > size * 0.03)
        img[crescent] = 40.0 + rng.integers(0, 25, (size, size))[crescent]
        img += _noise(noise)
    elif abnormality == "bone_marrow_lesion":
        # Irregular bright blob (edema). Position is FIXED so the fingerprint
        # is stable across train/test (a random center made the pattern
        # un-learnable by the reference union).
        cx = size * 0.5
        cy = size * 0.55
        rx, ry = size * 0.20, size * 0.14
        ell = ((xf - cx) / rx) ** 2 + ((yf - cy) / ry) ** 2
        blob = ell < 1.0
        img[blob] = 180.0 + (1.0 - ell[blob]) * 60.0 + rng.random((size, size))[blob] * 20.0
        img += _noise(noise)
    elif abnormality == "chondral_defect":
        # Bright cartilage field + small dark pit.
        top = np.zeros((size, size), dtype=bool)
        top[: size // 4, :] = True
        img[top] = 180.0 + rng.integers(0, 50, (size, size))[top]
        cx, cy = size * 0.55, size * 0.25
        dist = np.sqrt((xf - cx) ** 2 + (yf - cy) ** 2)
        img[dist < size * 0.04] = 25.0 + rng.integers(0, 30, (size, size))[dist < size * 0.04]
        img += _noise(noise)
    # "normal" handled by caller (no pattern) — see generate_synthetic_normal below.

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_synthetic_normal(size: int = 256, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    """Smooth gradient with low noise (normal knee)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    img = (90.0 + x / size * 20.0 + y / size * 15.0)
    img += (rng.random((size, size)) - 0.5) * 2.0 * (noise * 0.5) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


# ── Reference library ─────────────────────────────────────────────────
def build_reference_library(size: int = 256, images_per_class: int = 8,
                            noise: float = 0.05, cfg=None) -> dict:
    """Build the reference token-set library: union of synthetic fingerprints per class."""
    cfg = cfg or DEFAULT_CONFIG
    library = {}
    for idx, ab in enumerate(ABNORMALITIES):
        union = set()
        for i in range(images_per_class):
            img = generate_synthetic(ab, size=size, noise=noise, seed=idx * 1000 + i)
            union |= fingerprint_image(img, cfg)
        library[ab] = union
    return library


# ── Classification ────────────────────────────────────────────────────
def directional_similarity(scan_tokens: set, ref_tokens: set) -> tuple:
    inter = len(scan_tokens & ref_tokens)
    denom = len(ref_tokens) or 1
    tau = inter / denom
    rho = (len(scan_tokens) - inter) / denom
    return tau, rho


def classify(gray, library, tau_threshold: float = 0.70, rho_threshold: float = 0.30,
             tau_weight: float = 0.70, rho_weight: float = 0.30, cfg=None) -> list:
    """Classify a grayscale image against the reference library.

    Returns a list of dicts (abnormality, label, severity, tau, rho,
    confidence, detected) sorted by confidence descending.
    """
    cfg = cfg or DEFAULT_CONFIG
    scan = fingerprint_image(gray, cfg)
    scores = []
    for ab in ABNORMALITIES:
        ref = library.get(ab)
        if not ref:
            continue
        tau, rho = directional_similarity(scan, ref)
        confidence = tau_weight * tau + rho_weight * (1.0 - rho)
        detected = tau >= tau_threshold and rho < rho_threshold
        scores.append({
            "abnormality": ab,
            "label": LABELS[ab],
            "severity": SEVERITY[ab],
            "tau": round(tau, 4),
            "rho": round(rho, 4),
            "confidence": round(confidence, 4),
            "detected": detected,
        })
    scores.sort(key=lambda s: s["confidence"], reverse=True)
    return scores
