# BCI Multi-Model Decoding & Benchmark Report
Generated from BIDS Root: `C:\Users\VR\Documents\GitHub\nautilus_bci\scripts\bids_musica` | Subject: `sub-01` | Sessions Pooled: `7`

## Dataset Characteristics
- **Total Trial Epochs**: 180 epochs
- **Channel Dimensions**: 32 EEG channels
- **Time Points per Trial**: 1001 samples (@ 250Hz = 4.0s)
- **Number of Output Classes**: 6 target tracks
- **Theoretical Random Chance**: 16.67%

## Decoding Accuracy Benchmark comparison

| Model Architecture | Feature Extraction Method | 5-Fold CV Accuracy | Performance vs. Chance |
| :--- | :--- | :---: | :---: |
| **CSP + LDA (Baseline)** | Raw Spatio-Temporal | **0.00%** | -16.67% |
| **Riemannian MDM** | Covariance | **7.78%** | -8.89% |
| **Tangent Space + SVM** | Covariance | **0.00%** | -16.67% |
| **Tangent Space + Logistic Regression** | Covariance | **0.00%** | -16.67% |
| **EEGNet Deep Learning CNN** | Raw Spatio-Temporal | **19.44%** | +2.78% |

## Brain Rhythm Power Spectral Distribution (PSD)
The average brain rhythm energy densities across all pooled trials are saved in the results directory. See the generated [aggregated_multi_session_psd.png](aggregated_multi_session_psd.png) graph.
