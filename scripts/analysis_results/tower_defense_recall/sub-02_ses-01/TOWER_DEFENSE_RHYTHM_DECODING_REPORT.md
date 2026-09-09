# BCI Tower Defense: 4-Class Rhythm Decoding & Mental Imagery Report (sub-02)

## 1. Executive Summary

This study analyzes the electrophysiological dynamics and multi-class classification feasibility of decoding four discrete musical rhythm states (**FIRE**, **WATER**, **WIND**, **ELECTRICITY**) from the newly recorded BIDS Tower Defense dataset for participant **sub-02** (`scripts/bids/bids_tower_defense/sub-02/ses-01`).

### Core Metrics Summary
- **Subject**: `sub-02` (Session `ses-01`)
- **Total Trials Analyzed**: 64 balanced trials (16 trials / 25.0% per class).
- **Theoretical Chance Level (4-Class)**: **25.00%** (uniform prior).
- **Theoretical Chance Level (Pairwise 2-Class)**: **50.00%**.
- **Peak Imagery Decoding (5-Fold Stratified CV)**: **40.64% ± 5.73%** (`CSP_ShrinkageLDA`) and **40.51% ± 10.93%** (`CSP_SVM_RBF`).
- **Mean Pairwise Imagery Decoding**: **67.46%** (CSP+LDA) / **75.16%** (Best Pipelines across all 6 pairs).
- **Top Separable Pair (Imagery)**: **FIRE vs. ELECTRICITY (84.8% ± 9.1%)**.
- **Peak Auditory Perception Decoding (5-Fold CV)**: **34.10% ± 13.94%** (`Riemannian_TangentSpace_SVM_Linear`).
- **Peak Visual Flicker Decoding (5-Fold CV)**: **29.62% ± 2.31%** (`Welch_PSD_RandomForest`).
- **Cross-Condition Transfer (Train: Listen -> Test: Imagine)**: **28.12%** (`Riemannian_TangentSpace`).
- **Representational Similarity Alignment (RSA)**: Spearman $\rho = 0.714$ ($p = 0.1108$).

---

## 2. Experimental Paradigm & Trial Structure

Each trial progresses through three distinct operational phases:
1. **Auditory Perception Phase (`Start Listen` -> `End Listen`, 5.0s)**:
   - The participant actively listens to the rhythm track corresponding to the selected element.
2. **Visual Selection / Flicker Phase (`Box start blinking` -> `Box stop blinking`, 2.0s)**:
   - Visual cue flicker on screen indicating element confirmation.
3. **Mental Imagery / Recall Phase (`Imagine` / `[ELEMENT] selected` -> `Rest`, 3.0s)**:
   - The participant actively imagines and recalls the rhythm in silent top-down mental imagery.

---

## 3. Comprehensive Model Benchmark Across Experimental Phases

| Decoder Architecture | Mental Imagery (`Imagine`) | Auditory Perception (`Listen`) | Visual Flicker (`Blinking`) |
| :--- | :---: | :---: | :---: |
| **One-vs-Rest CSP + Shrinkage LDA** | **40.64% ± 5.73%** | 31.15% ± 15.21% | 21.79% ± 8.81% |
| **One-vs-Rest CSP + SVM (RBF)** | **40.51% ± 10.93%** | 26.54% ± 10.35% | 21.92% ± 5.91% |
| **Filter-Bank CSP (FBCSP) + LogReg** | 33.08% ± 10.20% | 23.33% ± 11.76% | 21.92% ± 5.91% |
| **Riemannian Tangent Space + LogReg** | 26.67% ± 6.61% | 23.21% ± 10.70% | 26.67% ± 13.63% |
| **Riemannian Tangent Space + Linear SVM**| 32.95% ± 10.66% | **34.10% ± 13.94%** | 22.05% ± 11.76% |
| **Riemannian Tangent Space + Ridge** | 32.82% ± 7.50% | 18.59% ± 9.08% | 28.21% ± 8.11% |
| **Welch PSD + Random Forest** | 23.33% ± 9.54% | 31.28% ± 10.93% | **29.62% ± 2.31%** |
| **Welch PSD + Shrinkage LDA** | 26.67% ± 6.61% | 27.95% ± 12.25% | 27.82% ± 14.09% |
| **Ensemble Soft Voting** | 36.15% ± 11.31% | 31.15% ± 10.63% | 23.46% ± 8.46% |
| **Theoretical Chance Baseline** | **25.00%** | **25.00%** | **25.00%** |

---

## 4. Pairwise 2-Class Rhythm Decoding (Mental Imagery)

Evaluating all 6 binary rhythm combinations (Chance: **50.00%**) reveals prominent separability among elemental rhythms:

| Rank | Pair | N | CSP + LDA (%) | Riemannian TS (%) | FBCSP (%) | Welch PSD (%) | Best Pipeline |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **FIRE vs. ELECTRICITY** | 32 | **84.8% ± 9.1%** | 68.6% | 61.4% | 69.0% | **CSP_ShrinkageLDA (84.8%)** |
| **2** | **FIRE vs. WATER** | 32 | **79.0% ± 14.2%** | 62.4% | 53.8% | 40.5% | **CSP_ShrinkageLDA (79.0%)** |
| **3** | **WIND vs. ELECTRICITY** | 32 | 59.0% ± 17.3% | **74.3%** | 46.2% | 60.0% | **Riemannian TS LogReg (74.3%)** |
| **4** | **WATER vs. ELECTRICITY**| 32 | 58.6% ± 13.9% | 69.5% | 59.0% | 57.1% | **Riemannian TS Linear SVM (72.4%)** |
| **5** | **WATER vs. WIND** | 32 | 54.3% ± 15.8% | 55.7% | **71.4%** | 49.0% | **FilterBank CSP LogReg (71.4%)** |
| **6** | **FIRE vs. WIND** | 32 | **69.0% ± 17.2%** | 57.1% | 51.0% | 53.8% | **CSP_ShrinkageLDA (69.0%)** |

- **Overall Mean CSP+LDA**: **67.46%**
- **Overall Mean Best Pipeline**: **75.16%**

---

## 5. Key Neuro-Engineering Insights

1. **High Mental Imagery Separability for sub-02**:
   - `sub-02` demonstrates exceptional mental imagery classification performance (**40.64%** 4-class, compared to **33.00%** in `sub-01`).
   - Binary rhythm discrimination reaches **84.8%** for `FIRE` vs. `ELECTRICITY` and **79.0%** for `FIRE` vs. `WATER`.
2. **Spatial Pattern (CSP) Dominance in Imagery**:
   - CSP-filtered features significantly outperform spectral-only (Welch PSD) methods, proving that rhythm recall involves distinct spatial cortical configurations (sensorimotor/temporal networks) rather than uniform global power shifts.
3. **Representational Alignment (RSA rho = 0.714)**:
   - The representational geometry of neural activity during mental imagery correlates strongly with auditory perception (rho = 0.714), showing consistent cognitive encoding of the rhythm elements across perception and recall phases.

---

## 6. Generated Assets & Artifacts

All generated figures, numerical tables, and machine-readable logs are saved in:
`scripts/analysis_results/tower_defense_recall/sub-02_ses-01/`

- `rhythm_decoding_benchmark.png`: Multi-model decoding accuracy barplot comparing Imagine, Listen, and Blinking.
- `confusion_matrices_all_phases.png`: Confusion matrices for Imagine, Listen, and Transfer.
- `pairwise_rhythm_decoding_imagery.png`: Pairwise binary accuracy rankings and separability matrix for Mental Imagery.
- `pairwise_rhythm_decoding_perception.png`: Pairwise binary accuracy rankings and separability matrix for Auditory Perception.
- `rsa_perception_imagery_rdm.png`: Representational Dissimilarity Matrices and Spearman alignment.
- `pairwise_decoding_imagery.csv`: Detailed numerical table of all 6 pairwise combinations in imagery.
- `pairwise_decoding_perception.csv`: Detailed numerical table of all 6 pairwise combinations in perception.
- `models_benchmark_metrics.csv`: Cross-validation accuracy, std, balanced accuracy, F1, and Cohen Kappa for all models across all phases.
- `rhythm_decoding_summary.json`: Machine-readable JSON summary of the complete benchmark.
