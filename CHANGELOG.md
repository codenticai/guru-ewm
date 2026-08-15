# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

- NLP: keyword-centric retrieval with union-of-occurrences replies and 15-per-page pagination ("more" to continue).
- NLP: single-keyword rescue so every keyword in `nlp_keywords.csv` returns a non-fallback answer.
- Medical: ECG / X-ray / CT / knee-MRI / lab text-report matching with numeric reference ranges.
- Medical: BiomedCLIP zero-shot image classification and synthetic knee-MRI fingerprint classifier (CPU-only).
- OCR: Tesseract-based full-page and ECG header-band extraction.
- UI: three-mode chat (NLP / OCR / Diagnose), optional live CPU/RAM header badge.
- Open-source hygiene: MIT license, community docs, portable compose file.
