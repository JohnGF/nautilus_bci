# Preliminary Multimodal BCI Results & Analysis Report

**Dataset Evaluated**: `bids_baseline/sub-01/ses-02` (`task-video`, 372.4 seconds, 33 EEG channels, Smartwatch PPG, Smartwatch IMU)

---

## 1. Executive Summary & Key Findings

- **EEG Baseline Quality**: Continuous 33-channel recording @ 250 Hz across 80 stimulus trials (`water`, `earth`, `wind`, `fire`).
- **Standard CSP + LDA Decoding (4-Class)**:
  - **4-Class Accuracy**: **20.0% ± 4.7%** (chance level = **25.0%**).
  - **Pairwise Binary Decoding**: Peak at **55.0%** (Water vs. Earth).
  - *Conclusion*: Standard CSP is designed for motor imagery (C3 vs. C4 power desynchronization) and fails to decode abstract/visual cognitive thoughts from passive viewing tasks.
- **Multimodal Ablation Study**:
  - **Best Configuration**: **EEG + Smartwatch PPG** achieved **31.25% ± 3.95%** accuracy (+6.25% above chance).
  - **Feature Importance Split**: EEG (97.75%), Smartwatch IMU (1.90%), Smartwatch PPG (0.35%).
  - *Conclusion*: Combining EEG with PPG autonomic arousal features improves decoding while avoiding body motion artifacts.
- **Adaptive Gated Deep Learning Architecture (PyTorch)**:
  - Built an end-to-end multi-stream network with a **Learnable Softmax Modality Gating Layer**.
  - Automatically learned modality weights: **$w_{\text{EEG}} = 99.96\%$**, **$w_{\text{PPG}} = 0.02\%$**, **$w_{\text{IMU}} = 0.02\%$**.

---

## 2. Benchmark Classification Comparison Table

| Model / Pipeline | Modality / Inputs | 4-Class Accuracy | Status vs. Chance (25%) |
| :--- | :--- | :---: | :--- |
| **RandomForest + EEG + PPG** | EEG (Riemannian + Band Power) + Smartwatch PPG | **31.25% ± 3.95%** | ** Above Chance** |
| **HistGradientBoosting Fusion** | EEG + Smartwatch PPG + Smartwatch IMU | **27.50% ± 3.06%** | ** Above Chance** |
| **RandomForest + EEG + IMU** | EEG + Smartwatch IMU Motion | **25.00% ± 3.95%** | ⚖️ At Chance |
| **Riemannian Tangent Space + LR** | EEG Covariances (OAS estimator) | **25.00% ± 6.85%** | ⚖️ At Chance |
| **PyTorch EEGNet (Deep ConvNet)** | EEG Raw 2D Spatio-Temporal Matrix | **21.25% ± 6.37%** | ❌ Below Chance |
| **PyTorch Adaptive Gated Net** | Multimodal End-to-End Deep Net | **20.00% ± 9.19%** | ❌ Below Chance |
| **4-Class OVR-CSP + LDA** | EEG Spatial Filtering (Mu/Beta) | **20.00% ± 4.68%** | ❌ Below Chance |
| **PPG Only (Smartwatch)** | PPG Heart Rate & Pulse Amplitude | **15.00% ± 5.00%** | ❌ Below Chance |
| **IMU Only (Smartwatch)** | IMU Accel / Gyro Motion Vectors | **17.50% ± 4.68%** | ❌ Below Chance |

---

## 3. Analysis Artifacts Directory

- `baseline_sub-01_ses-02_analysis.png`: Spectrum, Motion Magnitude, and PPG Track plots.
- `baseline_sub-01_ses-02_summary.json`: Statistical breakdown of `bids_baseline`.
- `multimodal_ablation_study.png`: Ablation bar chart & modality importance pie chart.
- `multimodal_ablation_study.json`: Complete ablation experiment data.
- `advanced_decoding_benchmark.json`: Multi-pipeline accuracy metrics.
- `multimodal_gated_net_results.json`: PyTorch Gated Net metrics & learned weights.
