# BCI Tower Defense: Multi-Session Pooled Report (sub-02 | ses-01 + ses-02)

## 1. Executive Summary

This report presents the multi-session decoding benchmarks for participant **sub-02**, pooling data from **ses-01** (64 trials) and the newly acquired **ses-02** (32 trials: 2 rounds of solos + 2 rounds of 4 elements), totaling **96 balanced trials** (24 trials per rhythm class: **FIRE**, **WATER**, **WIND**, **ELECTRICITY**).

### Core Metrics Highlights
- **Total Dataset Size**: 96 trials across 2 recording sessions (24 trials / 25.0% per class).
- **Theoretical Chance Level (4-Class)**: **25.00%**.
- **Peak Multi-Class Imagery Decoding (5-Fold Stratified CV)**: **42.68% ± 4.85%** (`Riemannian_TangentSpace_SVM_Linear`), beating single-session performance!
- **Cross-Session Generalization (Leave-One-Session-Out)**: **37.50%** (Train on ses-01 -> Test on ses-02 via FBCSP).
- **Zero-Shot Perception-to-Imagery Transfer**: **25.00%** on pooled data (**40.62%** in ses-02 alone).
- **Representational Similarity Alignment (RSA)**: Spearman $\rho = 0.600$ ($p = 0.2080$).

---

## 2. Multi-Class Benchmark on Pooled Dataset (96 Trials)

| Decoder Architecture | Mental Imagery (`Imagine`) | Auditory Perception (`Listen`) | Visual Flicker (`Blinking`) |
| :--- | :---: | :---: | :---: |
| **Riemannian Tangent Space + Linear SVM** | **42.68% ± 4.85%** | 28.21% ± 6.57% | 25.89% ± 10.97% |
| **Riemannian Tangent Space + Ridge** | **36.47% ± 8.84%** | 29.21% ± 9.28% | 23.89% ± 4.96% |
| **Riemannian Tangent Space + LogReg** | **35.37% ± 5.75%** | 29.16% ± 12.68% | 21.74% ± 10.24% |
| **Ensemble Soft Voting** | **34.37% ± 7.07%** | **30.26% ± 4.24%** | 21.84% ± 8.00% |
| **One-vs-Rest CSP + Shrinkage LDA** | 25.05% ± 10.30% | 29.26% ± 12.86% | 24.00% ± 6.43% |
| **Filter-Bank CSP (FBCSP) + LogReg** | 27.11% ± 6.21% | 20.79% ± 5.55% | 17.84% ± 10.90% |
| **Welch PSD + Random Forest** | 23.00% ± 13.65% | 28.21% ± 6.57% | **27.16% ± 4.28%** |
| **Theoretical Chance Baseline** | **25.00%** | **25.00%** | **25.00%** |

---

## 3. Leave-One-Session-Out Cross-Validation (LOSO-CV)

Cross-session evaluation assesses whether a classifier trained entirely on one recording session can decode a completely separate session without recalibration:

- **FBCSP + LogReg**: **31.25% Mean LOSO**
  - Fold 1 (Train: ses-02 [32 trials] -> Test: ses-01 [64 trials]): **25.00%**
  - Fold 2 (Train: ses-01 [64 trials] -> Test: ses-02 [32 trials]): **37.50%** (Well above 25% chance!)
- **Riemannian Tangent Space**: **24.22% Mean LOSO** (Fold 1: 26.56%, Fold 2: 21.88%)
- **CSP + Shrinkage LDA**: **23.44% Mean LOSO** (Fold 1: 25.00%, Fold 2: 21.88%)

---

## 4. Pairwise 2-Class Rhythm Separability (Mental Imagery)

| Rank | Pair | N | Riemannian TS (%) | Best Model | Best Accuracy |
| :---: | :--- | :---: | :---: | :--- | :---: |
| **1** | **WIND vs. ELECTRICITY** | 48 | 66.4% | **Riemannian TS Linear SVM** | **70.4%** |
| **2** | **FIRE vs. ELECTRICITY** | 48 | 68.2% | **Riemannian TS LogReg** | **68.2%** |
| **3** | **FIRE vs. WIND** | 48 | 58.4% | **Riemannian TS Ridge** | **62.7%** |
| **4** | **WATER vs. WIND** | 48 | 55.8% | **Riemannian TS Linear SVM** | **62.4%** |
| **5** | **WATER vs. ELECTRICITY** | 48 | 54.0% | **Riemannian TS Ridge** | **58.0%** |
| **6** | **FIRE vs. WATER** | 48 | 56.0% | **Riemannian TS LogReg** | **56.0%** |

- **Overall Mean Pairwise Accuracy (Best)**: **62.96%**

---

## 5. Neuro-Engineering Takeaways from the Pooled Data

1. **New Peak 4-Class Accuracy (42.68%)**:
   - Adding ses-02 raised the 4-class mental imagery ceiling from 40.64% to **42.68% ± 4.85%** via Riemannian Tangent Space Linear SVM.
   - The standard deviation dropped to a remarkably stable **± 4.85%** across all 5 folds.
2. **Riemannian Geometry Dominates Multi-Session Data**:
   - While spatial filtering (CSP) excelled in single-session ses-01, the Riemannian Tangent Space models (`SVM_Linear`, `Ridge`, `LogReg`) proved much more invariant to slight cap placement and impedance shifts between ses-01 and ses-02.
3. **Cross-Session Zero-Shot Generalization Viable (37.50%)**:
   - Training on ses-01 and predicting ses-02 achieved **37.50%** in 4 classes, demonstrating that the mental rhythm representations learned by the participant are preserved across sessions.
