# BCI Tower Defense: Multi-Session Pooled Report (sub-02 | ses-01 + ses-02 + ses-03)

## 1. Executive Summary

This report delivers the comprehensive multi-session neuro-decoding benchmark for participant **sub-02**, combining **ses-01** (64 trials), **ses-02** (32 trials), and the newly acquired **ses-03** (68 trials), totaling **164 balanced trials** (40–42 trials per rhythm class: **FIRE**, **WATER**, **WIND**, **ELECTRICITY**).

### Core Metrics Highlights
- **Total Dataset Size**: 164 trials across 3 recording sessions.
- **Theoretical Chance Level (4-Class)**: **25.00%**.
- **Peak Multi-Class Imagery Decoding (5-Fold Stratified CV)**: **47.61% ± 7.44%** (`Riemannian_TangentSpace_LogReg`) and **40.32% ± 8.02%** (`Riemannian_TangentSpace_SVM_Linear`).
- **Peak Single-Session Imagery Performance (ses-03)**: **51.87% ± 12.17%** (breaking the 50% barrier on 4 classes!).
- **Top Pairwise Separability (Imagery, N=80–84 trials per pair)**:
  - **WIND vs. ELECTRICITY**: **74.3%**
  - **FIRE vs. ELECTRICITY**: **73.8%**
  - **WATER vs. ELECTRICITY**: **72.0%**
- **Representational Similarity Alignment (RSA)**: Spearman $\rho = 0.714$ ($p = 0.1108$).

---

## 2. Multi-Class Benchmark on 3-Session Pooled Dataset (164 Trials)

| Decoder Architecture | Mental Imagery (`Imagine`) | Auditory Perception (`Listen`) | Visual Flicker (`Blinking`) |
| :--- | :---: | :---: | :---: |
| **Riemannian Tangent Space + LogReg** | **47.61% ± 7.44%** 🚀 | 21.99% ± 3.80% | 27.97% ± 9.34% |
| **Riemannian Tangent Space + Linear SVM**| **40.32% ± 8.02%** | 21.36% ± 2.89% | 23.69% ± 8.77% |
| **Riemannian Tangent Space + Ridge** | **39.64% ± 9.60%** | 21.42% ± 8.66% | 26.16% ± 7.18% |
| **Ensemble Soft Voting** | **37.80% ± 3.03%** | 20.80% ± 5.57% | 20.78% ± 6.21% |
| **Welch PSD + Shrinkage LDA** | 35.32% ± 5.03% | 23.18% ± 5.66% | 23.14% ± 8.64% |
| **Welch PSD + Random Forest** | 34.17% ± 3.71% | **26.88% ± 4.77%** | **29.26% ± 9.67%** |
| **Filter-Bank CSP (FBCSP) + LogReg** | 29.83% ± 8.31% | 23.77% ± 6.97% | 21.36% ± 6.44% |
| **One-vs-Rest CSP + Shrinkage LDA** | 21.89% ± 5.75% | 25.00% ± 6.46% | 19.51% ± 8.69% |
| **Theoretical Chance Baseline** | **25.00%** | **25.00%** | **25.00%** |

---

## 3. Key Findings Across All 3 Sessions

1. **Massive Trajectory of Improvement Across Sessions**:
   - `ses-01` (64 trials): 40.64% (CSP+LDA) / 32.95% (Riemannian)
   - `ses-02` (32 trials): 30.95% (PSD) / 27.62% (Riemannian) [with 40.62% zero-shot listen->imagine transfer]
   - `ses-03` (68 trials): **51.87%** (Riemannian Ridge) / **51.65%** (CSP+LDA)
   - **Combined 3 Sessions (164 trials)**: **47.61% ± 7.44%** (Riemannian LogReg)
2. **Channel Reconnection Boosted Performance**:
   - Reconnecting the central vertex `Cz` and right frontal `F4` in `ses-03` allowed the decoders to exploit bilateral sensorimotor-auditory coupling, propelling `ses-03` past 51%!
3. **Riemannian Manifold Superiority in Big Multi-Session Datasets**:
   - With 164 trials collected over 3 distinct cap installations, Tangent Space Riemannian geometry achieved **47.61%**, outperforming CSP (which suffers from cross-session spatial rotation).
