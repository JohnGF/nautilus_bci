# Comprehensive Multimodal & EEG Ablation Analysis Results

This directory contains the results, benchmarking metrics, statistical tables, confusion matrices, and visual artifacts for:
1. **Smartwatch Multimodal Prediction Pipeline** (`scripts/analysis/smartwatch_prediction_pipeline.py`)
2. **Comprehensive EEG Component & Channel Ablation Study** (`scripts/analysis/eeg_component_channel_ablation_study.py`)

---

## 1. Summary of Execution Scripts & Algorithms

| Analysis Pipeline | Execution Script | Primary Algorithms Evaluated | Datasets Used |
| :--- | :--- | :--- | :--- |
| **Smartwatch Prediction** | [`smartwatch_prediction_pipeline.py`](file:///C:/Users/VR/Documents/GitHub/nautilus_bci/scripts/analysis/smartwatch_prediction_pipeline.py) | Random Forest (RF), HistGradientBoosting (GBM), Support Vector Classifier (SVC-RBF), L2 Logistic Regression, Multi-Layer Perceptron (MLP) | `scripts/bids/bids_baseline/sub-01/ses-02` (IMU Motion + PPG Physio) |
| **EEG Ablation Study** | [`eeg_component_channel_ablation_study.py`](file:///C:/Users/VR/Documents/GitHub/nautilus_bci/scripts/analysis/eeg_component_channel_ablation_study.py) | Riemannian Tangent Space Logistic Regression (TS-LogReg), Riemannian TS-SVM, OvR Common Spatial Patterns + Shrinkage LDA (OvR-CSP+sLDA), Multi-Band PSD + Random Forest, Deep Learning EEGNet (PyTorch 2D+Depthwise Conv) | **Strictly Designated Only**:<br>1. `scripts/bids/bids_tower_defense`<br>2. `scripts/bids/bids_tower_defense_6_3_27`<br>3. `scripts/bids/bids_listening` |

---

## 2. Smartwatch Multimodal Prediction Findings

- **Records Check**: Investigated all directories. Found that the 3 game-specific folders (`bids_tower_defense`, `bids_tower_defense_6_3_27`, `bids_listening`) contain pure 32-channel EEG data (motion folders are empty). Smartwatch recordings (6-axis IMU + PPG optical pulse) are present under `bids_baseline` (`sub-01/ses-01` and `ses-02`) and `bids_musica`.
- **Feature Extraction**: Extracted 35 multimodal features (accelerometer 3D magnitude, jerk, energy, spectral entropy; gyroscope angular velocity; PPG cardiac pulse amplitude, heart rate BPM, HRV time-domain metrics SDNN/RMSSD/pNN50, and frequency-domain LF/HF ratio).
- **Benchmark Performance (5-Fold Stratified CV)**:
  - **Multi-Layer Perceptron (MLP Neural Net)**: **27.39% Accuracy** | **0.2747 Macro-F1** | **0.5154 ROC-AUC**
  - **Random Forest (RF)**: **25.48% Accuracy** | **0.2558 Macro-F1** | **0.5060 ROC-AUC**
  - **HistGradientBoosting (GBM)**: **24.84% Accuracy** | **0.2489 Macro-F1** | **0.4765 ROC-AUC**
  - **Top Predictive Features**: `imu_acc_mag_rms`, `imu_acc_mag_mean`, `ppg_ppg_hrv_rmssd`, `imu_acc_spectral_entropy`.
- **Generated Artifacts**:
  - [`smartwatch_prediction_results.json`](smartwatch/smartwatch_prediction_results.json)
  - [`smartwatch_prediction_metrics.csv`](smartwatch/smartwatch_prediction_metrics.csv)
  - [`smartwatch_pipeline_report.md`](smartwatch/smartwatch_pipeline_report.md)
  - Visuals: `smartwatch_model_benchmark.png`, `smartwatch_confusion_matrices.png`, `smartwatch_feature_importance.png`

---

## 3. EEG Component & Channel Ablation Study Findings (Designated Datasets Only)

Using **strictly** `scripts/bids/bids_tower_defense`, `scripts/bids/bids_tower_defense_6_3_27`, and `scripts/bids/bids_listening`:

### A. Cognitive Trial Phase / Component Ablation (152 Joint Trials, 4-Class Elemental Imagery)
1. **Imagine + Auditory Perceptual Whitening** (with `bids_listening` Riemannian reference matrix $C_{ref}$): **31.58% Best Accuracy** (Chance = 25.0%)
2. **Visual Flicker Phase Only** (`Box start blinking` SSVEP / Visual cue): **28.95% Best Accuracy**
3. **Mental Imagery Phase Only** (`Imagine` silent recall): **28.95% Best Accuracy**
4. **Multiphase Fusion** (Auditory Cue + Visual Flicker + Imagine concatenation): **28.29% Accuracy**
5. **Auditory Cue Phase Only** (`Start Listen` -> `End Listen` prompt): **28.29% Accuracy**

### B. Feature Extraction & Algorithmic Component Ablation
1. **Riemannian Tangent Space Covariance**: **28.95% Accuracy**
2. **Full Multi-Feature Early Fusion** (Riemannian + Multi-Band PSD + Time-Domain Hjorth): **28.29% Accuracy**
3. **Multi-Band Spectral PSD** (Delta, Theta, Alpha, Beta, Gamma): **26.97% Accuracy**
4. **One-vs-Rest CSP (8 Spatial Filters + sLDA)**: **25.00% Accuracy**
5. **Time-Domain & Hjorth Statistics**: **21.05% Accuracy**

### C. Spatial Preprocessing Reference Filter Ablation
1. **Robust CAR (Median-Referenced Spatial Filter)**: **25.66% TS-LogReg**
2. **Standard CAR (Common Average Reference)**: **23.03% TS-LogReg**
3. **Raw Monopolar Reference**: **20.39% TS-LogReg**
4. **Surface Laplacian**: **20.39% TS-LogReg**

### D. EEG Channel Ablation (Regional & Progressive Reduction)
- **Top Regional Cluster**: Central / Sensorimotor Subsystem (`EEG009`–`EEG016`): **26.97% Accuracy**
- **21 Active High-Quality Channels**: **25.00% Accuracy**
- **Stepwise Channel Reduction Curve**:
  - **32 Channels**: 28.95% Best (25.66% TS-LogReg)
  - **24 Channels**: 27.63% Best (22.37% TS-LogReg)
  - **16 Channels**: 27.63% Best (27.63% TS-LogReg)
  - **12 Channels**: 26.97% Best (23.68% TS-LogReg)
  - **8 Channels**: 21.71% Best (12.50% TS-LogReg)
  - **4 Channels**: 25.66% Best (21.71% TS-LogReg)
  - **2 Channels**: 25.66% Best (20.39% TS-LogReg)
  - **1 Channel**: 25.66% Best (24.34% TS-LogReg)
- **Top 5 Solo Channels**: `EEG002` (31.58%), `EEG003` (26.97%), `EEG004` (26.32%), `EEG013` (26.32%), `EEG020` (26.32%).

### E. Deep Learning Architecture & Cross-Session Transfer
- **PyTorch EEGNet (2D Conv + Depthwise Separable Conv)**: **30.37% 5-Fold CV Accuracy** | **0.2920 Macro-F1**
- **Cross-Session Generalization**:
  - Train TD_1 $\rightarrow$ Test TD_2: **18.42%**
  - Train TD_2 $\rightarrow$ Test TD_1: **30.26%**
  - Mean Cross-Session Generalization: **24.34%**

- **Generated Artifacts**:
  - [`eeg_ablation_study_results.json`](eeg_ablation/eeg_ablation_study_results.json)
  - [`eeg_ablation_study_summary.csv`](eeg_ablation/eeg_ablation_study_summary.csv)
  - [`eeg_ablation_study_report.md`](eeg_ablation/eeg_ablation_study_report.md)
  - Visuals: `eeg_component_ablation_benchmark.png`, `eeg_channel_reduction_curve.png`, `eeg_single_channel_ranking.png`
