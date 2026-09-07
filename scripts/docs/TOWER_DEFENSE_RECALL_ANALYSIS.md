# Tower Defense BCI: Visual Blinking & Song Recall Analysis

## 1. Study Overview & Experimental Paradigm

The **Tower Defense BCI Recall Dataset** (`scripts/bids_tower_defense`) evaluates neural correlates during a multi-stage visual-auditory BCI paradigm:

1. **Visual Flicker / Selection Phase (`Box start blinking`)**:
   - The participant attends to a visually blinking element box on screen (SSVEP / Visual Cue phase, ~5.5s duration).
2. **Auditory Recall & Musical Imagery Phase (`Box stop blinking` $\rightarrow$ Song Selection)**:
   - The visual flicker stops, registering the selection of a specific element:
     - **FIRE** ($N = 22$)
     - **WATER** ($N = 22$)
     - **WIND** ($N = 22$)
     - **ELECTRICITY** ($N = 17$)
   - The participant mentally imagines / recalls the associated elemental song during a ~12.5-second silent recall window.

```
Timeline of a Single Trial:
┌───────────────────────────────┬────────────────────────────────────────────────────────┐
│ Visual Flicker Phase          │ Auditory Recall & Song Imagery Phase                   │
│ Duration: ~5.5 seconds        │ Duration: ~12.5 seconds                                │
│ Event: `Box start blinking`   │ Event: `Box stop blinking` + `[ELEMENT] selected`      │
│ Task: Look at flashing box    │ Task: Mentally recall & imagine the chosen theme song  │
└───────────────────────────────┴────────────────────────────────────────────────────────┘
```

---

## 2. Channel Quality Screening & Electrode Diagnostics

Electrophysiological signal screening on all 33 recorded channels (sampled at 250 Hz) identifies channel integrity, impedance state, and artifact levels:

| Quality Category | Count | Channels | Characteristics / Cause |
| :--- | :---: | :--- | :--- |
| **Good Active EEG** | **21** | `EEG001`, `EEG004`, `EEG006`–`EEG010`, `EEG013`, `EEG015`–`EEG016`, `EEG018`–`EEG021`, `EEG024`–`EEG025`, `EEG027`, `EEG029`–`EEG032` | Stable physiological EEG signals ($\text{Std} \approx 3\text{--}45\,\mu\text{V}$, normal spectral roll-off). |
| **Noisy / Motion Artifacts** | **9** | `EEG002`, `EEG011`, `EEG012`, `EEG014`, `EEG017`, `EEG022`, `EEG023`, `EEG026`, `EEG028` | Intermittent high-amplitude voltage excursions and movement artifacts ($\text{PTP} > 1500\,\mu\text{V}$, $\text{Std} > 50\text{--}123\,\mu\text{V}$). |
| **Near-Flatline / Poor Contact**| **2** | `EEG003`, `EEG005` | Severely attenuated voltage ($\text{Std} < 1\,\mu\text{V}$, $\text{PTP} < 35\,\mu\text{V}$), characteristic of high impedance or disconnected pins. |
| **Unconnected / Auxiliary** | **1** | `EEG033` | Complete zero flatline ($0.00\,\mu\text{V}$), standard unpopulated 33rd pin / battery channel on the 32-channel g.Nautilus cap. |

### Preprocessing & Protection Strategy
- **`EEG033` Handling**: Automatically converted to channel type `misc` and dropped from covariance estimation to prevent rank deficiency and numerical divergence.
- **Robust CAR Spatial Filtering**: Median-based spatial referencing ([`spatial_filters.py`](../analysis/spatial_filters.py)) replaces standard mean subtraction, protecting good channels from noisy electrode contamination.
- **Temporal Filtering**: Zero-phase Butterworth bandpass filter (1.0–45.0 Hz) and 50 Hz notch filter remove slow electrochemical half-cell drift and line interference.

---

## 3. Algorithmic Framework & Mathematical Foundations

### 3.1. One-vs-Rest Common Spatial Patterns (OvR-CSP)
Common Spatial Patterns finds spatial projection filters $w$ that maximize signal variance for class $A$ while minimizing it for class $B$:
$$\max_{w} \frac{w^T \Sigma_A w}{w^T \Sigma_B w}$$
For 4-class multi-class decoding (`FIRE`, `WATER`, `WIND`, `ELECTRICITY`), the problem is decomposed into 4 binary sub-problems:
1. `FIRE` vs. `[WATER, WIND, ELECTRICITY]`
2. `WATER` vs. `[FIRE, WIND, ELECTRICITY]`
3. `WIND` vs. `[FIRE, WATER, ELECTRICITY]`
4. `ELECTRICITY` vs. `[FIRE, WATER, WIND]`

The resulting log-variance spatial power features are concatenated into a joint representation $\mathbf{x} \in \mathbb{R}^{4 \times 2k}$.

### 3.2. Shrinkage Linear Discriminant Analysis (Shrinkage LDA)
Standard sample covariance matrices $\hat{\Sigma}$ in BCI are often ill-conditioned due to high feature dimensionality relative to trial count ($N = 83$). Shrinkage LDA replaces $\hat{\Sigma}$ with a convex combination of the empirical covariance and an isotropic target:
$$\Sigma_{\text{shrunk}} = (1 - \gamma) \hat{\Sigma} + \gamma \nu I$$
The shrinkage parameter $\gamma \in [0, 1]$ is computed analytically via the Ledoit-Wolf / Oracle Approximating Shrinkage (OAS) formula, ensuring a well-conditioned, invertible matrix and mitigating overfitting.

### 3.3. Riemannian Manifold Geometry
- **Covariances**: Estimates regularized OAS covariance matrices $C_i \in \text{SPD}(n)$ for each epoch.
- **Tangent Space Projection**: Projects covariance matrices from the Riemannian manifold into Euclidean tangent space at the geometric Riemannian mean $\mathcal{G}$:
  $$S_i = \text{Log}_{\mathcal{G}}(C_i) = \mathcal{G}^{1/2} \log\left(\mathcal{G}^{-1/2} C_i \mathcal{G}^{-1/2}\right) \mathcal{G}^{1/2}$$
- **Classifiers**: Logistic Regression (L2 penalty) and Support Vector Machines with RBF kernel on the vectorized tangent space representations.
- **Riemannian MDM**: Minimum Distance to Mean classifier assigning trials to the class whose geometric centroid has the smallest Affine-Invariant Riemannian Metric (AIRM) distance.

### 3.4. Multi-Band Spectral Power (Welch PSD)
Computes relative band power across 5 classical physiological rhythms:
- **Delta**: 1.0 – 4.0 Hz
- **Theta**: 4.0 – 8.0 Hz
- **Alpha**: 8.0 – 12.0 Hz
- **Beta**: 13.0 – 30.0 Hz
- **Gamma**: 30.0 – 45.0 Hz

Features are normalized via z-score scaling and classified using an ensemble of 150 Random Forest estimators.

### 3.5. PyTorch Deep Learning (EEGNet)
Implements a 4-layer compact convolutional network (Lawhern et al., 2018):
1. **Temporal Convolution**: 1D temporal kernels capturing frequency-specific filters.
2. **Depthwise Spatial Convolution**: Spatial filters learned per temporal feature map.
3. **Separable Convolution**: Decouples spatial and temporal mixing with ELU activations and average pooling.
4. **Dense Softmax Output**: Multi-class cross-entropy loss with AdamW optimizer.

---

## 4. Benchmark Performance & Evaluation

All decoders were evaluated under **5-Fold Stratified Cross-Validation** on the 83 trials of the 4-class Song Recall task:

| Model Architecture | Mean Accuracy (5-Fold CV) | Balanced Accuracy | Macro F1-Score | Chance Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **One-vs-Rest CSP + Shrinkage LDA** | **40.74% ± 14.15%** | **39.64%** | **0.400** | 25.00% |
| **Riemannian MDM Classifier** | **27.79% ± 3.55%** | **29.14%** | **0.216** | 25.00% |
| **Multi-Band Power + Random Forest** | **27.72% ± 9.76%** | **27.14%** | **0.251** | 25.00% |
| **Riemannian Tangent Space + SVM (RBF)**| **20.51% ± 8.89%** | **19.32%** | **0.164** | 25.00% |
| **PyTorch EEGNet Deep Learning** | **20.59% ± 13.13%** | **17.05%** | **0.190** | 25.00% |
| **Riemannian Tangent Space + LogReg** | **19.19% ± 5.55%** | **19.52%** | **0.194** | 25.00% |

---

## 5. Generated Output Artifacts

The analysis pipeline automatically generates and exports all analytical figures and structured logs to `scripts/analysis_results/tower_defense_recall/`:

- **Decoding Benchmark Chart**: `decoding_benchmark_accuracy.png`
- **Spectral PSD Comparison**: `psd_spectral_comparison.png` (Blinking Phase vs. Recall Phase)
- **Confusion Matrices Grid**: `confusion_matrices_grid.png` (6-model normalized matrix comparison)
- **Metrics Summary**: `models_benchmark_metrics.csv`
- **Full Run Summary**: `recall_decoding_summary.json`

---

## 6. Reproduction & Usage Guide

To re-run the full analysis and reproduce all figures:

```bash
# Standard execution on the BIDS Tower Defense dataset
python scripts/analysis/analyze_tower_defense_recall.py --bids-root scripts/bids_tower_defense

# Custom window parameters (e.g., 0.5s to 4.5s post-stop-blinking window, robust CAR filter)
python scripts/analysis/analyze_tower_defense_recall.py \
    --bids-root scripts/bids_tower_defense \
    --sub 01 \
    --ses 01 \
    --recall-tmin 0.5 \
    --recall-tmax 4.5 \
    --spatial-filter robust_car \
    --out-dir scripts/analysis_results/tower_defense_recall
```
