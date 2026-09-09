# BCI Tower Defense: Single Session Report (sub-02 | ses-02)

## 1. Executive Summary

This report analyzes the standalone performance of session **ses-02** for participant **sub-02** (`scripts/bids/bids_tower_defense/sub-02/ses-02`), consisting of **32 balanced trials** (8 trials per rhythm class: **FIRE**, **WATER**, **WIND**, **ELECTRICITY**), captured across 2 rounds of solo levels + 2 rounds of four elements.

### Core Metrics Summary
- **Total Trials Analyzed**: 32 balanced trials (8 trials / 25.0% per class).
- **Theoretical Chance Level (4-Class)**: **25.00%**.
- **Theoretical Chance Level (Pairwise 2-Class)**: **50.00%**.
- **Zero-Shot Perception-to-Imagery Transfer (Listen -> Imagine)**: **40.62%** (Remarkable zero-shot generalization, +15.62% above chance!).
- **Peak Auditory Perception Decoding (5-Fold CV)**: **34.28% ± 11.53%** (`Riemannian_TangentSpace_SVM`).
- **Peak Imagery Decoding (5-Fold CV)**: **30.95% ± 21.40%** (`Welch_PSD_ShrinkageLDA`).
- **Top Separable Binary Pair (Imagery)**: **WATER vs. WIND (68.3%)** and **WATER vs. ELECTRICITY (66.7%)**.
- **Top Separable Binary Pair (Perception)**: **FIRE vs. ELECTRICITY (76.7%)** and **WATER vs. ELECTRICITY (75.0%)**.
- **Representational Similarity Alignment (RSA)**: Spearman $\rho = 0.600$ ($p = 0.2080$).
