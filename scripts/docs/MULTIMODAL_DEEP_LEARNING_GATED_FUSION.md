# Multimodal Deep Learning & Gated Fusion Architecture Documentation

This document outlines the architecture, theory, and ablation study findings for the **Multimodal Adaptive Gated Deep Learning Framework** in the Nautilus BCI Suite.

---

## 1. Motivation: Dynamic Stream Weighting

In multimodal BCI systems, combining EEG scalp data with wearable peripheral streams (e.g. Smartwatch PPG Heart Rate and 6-DOF IMU Motion) provides complementary cognitive and physiological information. 

However, naive feature concatenation presents major challenges:
- **Motion Noise**: Wrist movement introduces high-amplitude accelerometry artifacts.
- **Modality Dominance**: High-dimensional EEG features (e.g. 600+ channels/timepoints) can overwhelm low-dimensional PPG/IMU metrics (e.g. 4–20 features).

To solve this, we introduce an **End-to-End Modality Gating Network** (implemented in [multimodal_gated_network.py](file:///C:/Users/VR/Documents/GitHub/nautilus_bci/scripts/analysis/multimodal_gated_network.py)) that automatically learns to scale up high-SNR streams and scale down noisy streams during training.

---

## 2. Architecture Overview

```
                 ┌───────────────────────┐
  EEG Stream ───►│ EEG ConvNet Encoder   ├───────┐
                 └───────────────────────┘       │
                                                 │
                 ┌───────────────────────┐       ▼  Concat   ┌──────────────────────────┐
  PPG Stream ───►│ PPG 1D Signal Encoder ├───────┼──────────►│ Softmax Gating Network   │
                 └───────────────────────┘       ▲           │ [w_EEG, w_PPG, w_IMU]    │
                                                 │           └────────────┬─────────────┘
                 ┌───────────────────────┐       │                        │
  IMU Stream ───►│ IMU 1D Motion Encoder ├───────┘                        ▼
                 └───────────────────────┘              Dynamic Weighted Fusion Matrix
                                                                      │
                                                                      ▼
                                                            Multimodal Classifier Head
```

### Components:
1. **EEG Stream Encoder (`EEGStreamEncoder`)**:
   - Uses 2D spatial + temporal convolutions (EEGNet backbone: 1x32 temporal conv $\rightarrow$ 32x1 depthwise spatial conv $\rightarrow$ ELU $\rightarrow$ AvgPool).
   - Maps 32-channel continuous EEG into an embedding vector $\mathbf{h}_{\text{EEG}} \in \mathbb{R}^{64}$.
2. **PPG & IMU Encoders (`Signal1DEncoder`)**:
   - Dense multi-layer perceptron with batch normalization and ReLU activations mapping 1D physiological vectors into embeddings $\mathbf{h}_{\text{PPG}}, \mathbf{h}_{\text{IMU}} \in \mathbb{R}^{64}$.
3. **Modality Gating Mechanism (`ModalityGatingMechanism`)**:
   - Concatenates $[\mathbf{h}_{\text{EEG}}, \mathbf{h}_{\text{PPG}}, \mathbf{h}_{\text{IMU}}]$.
   - Passes through a linear bottleneck with Softmax activation:
     $$\mathbf{w} = \text{Softmax}\left(W_g [\mathbf{h}_{\text{EEG}}, \mathbf{h}_{\text{PPG}}, \mathbf{h}_{\text{IMU}}] + b_g\right)$$
   - Computes weighted fusion vector:
     $$\mathbf{h}_{\text{fused}} = w_{\text{EEG}} \mathbf{h}_{\text{EEG}} + w_{\text{PPG}} \mathbf{h}_{\text{PPG}} + w_{\text{IMU}} \mathbf{h}_{\text{IMU}}$$
4. **Classifier Head**:
   - Dense Linear layer classifying $\mathbf{h}_{\text{fused}}$ into target classes (e.g. 4-class video conditions or motor imagery targets).

---

## 3. Ablation Study & Empirical Findings

Our ablation study on `bids_baseline/sub-01/ses-02` ([multimodal_ablation_study.py](file:///C:/Users/VR/Documents/GitHub/nautilus_bci/scripts/analysis/multimodal_ablation_study.py)) yielded the following findings:

1. **EEG + PPG Fusion (+6.25% Above Chance)**:
   - Combining EEG Riemannian features with Smartwatch PPG achieved **31.25% ± 3.95%** accuracy (compared to **20.0%** for pure CSP+LDA and **25.0%** for EEG-only).
   - Heart rate modulation provides complementary autonomic context for cognitive tasks.
2. **Impact of Motion Artifacts**:
   - Including raw IMU motion slightly degraded multi-class accuracy (from **31.25%** to **22.50%**), proving the necessity of gating/suppressing motion streams during passive viewing tasks.
3. **Learned Gating Weights**:
   - On single-session data, the end-to-end network automatically learned to weight EEG at **99.96%** while suppressing raw IMU/PPG weights to **0.02%**, preventing noisy motion features from destabilizing training.

---

## 4. Code & Tool Reference

- **PyTorch Network Code**: [multimodal_gated_network.py](file:///C:/Users/VR/Documents/GitHub/nautilus_bci/scripts/analysis/multimodal_gated_network.py)
- **Ablation Study Tool**: [multimodal_ablation_study.py](file:///C:/Users/VR/Documents/GitHub/nautilus_bci/scripts/analysis/multimodal_ablation_study.py)
- **Benchmark Suite**: [advanced_decoding_suite.py](file:///C:/Users/VR/Documents/GitHub/nautilus_bci/scripts/analysis/advanced_decoding_suite.py)
- **Preliminary Results Folder**: [preliminary_results_analysis/](file:///C:/Users/VR/Documents/GitHub/nautilus_bci/scripts/preliminary_results_analysis/)
