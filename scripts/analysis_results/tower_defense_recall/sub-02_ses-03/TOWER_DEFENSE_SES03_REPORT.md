# BCI Tower Defense: Single Session Report (sub-02 | ses-03)

## 1. Executive Summary

This report analyzes the standalone performance of session **ses-03** for participant **sub-02** (`scripts/bids/bids_tower_defense/sub-02/ses-03`), consisting of **68 trials** (16 FIRE, 18 WATER, 18 WIND, 16 ELECTRICITY), recorded with optimized electrode gel adjustments (restoring `Cz` and `F4`).

### Core Metrics Summary
- **Total Trials Analyzed**: 68 trials across all 4 elements.
- **Theoretical Chance Level (4-Class)**: **25.00%**.
- **Peak Multi-Class Imagery Decoding (5-Fold Stratified CV)**: **51.87% ± 12.17%** (`Riemannian_TangentSpace_Ridge`) and **51.65% ± 16.37%** (`CSP_ShrinkageLDA`), breaking the 50% barrier!
- **Pairwise 2-Class Rhythm Separability (Best Model)**: **73.06% Average** across all 6 pairs:
  - **WATER vs. ELECTRICITY**: **76.7%**
  - **FIRE vs. ELECTRICITY**: **75.2%**
  - **WATER vs. WIND**: **75.0%**
  - **FIRE vs. WATER**: **73.3%**
  - **WIND vs. ELECTRICITY**: **71.0%**
  - **FIRE vs. WIND**: **67.1%**
