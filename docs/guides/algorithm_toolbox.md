# BCI Algorithm Toolbox Reference

Three toolboxes, three jobs: **offline training** (`scripts/training/algorithms.py`),
**deep/offline benchmarks** (PyTorch nets inside `scripts/analysis/`), and
**real-time inference** (`models/assistive_inferencer.py` + hierarchical decoder).
All classifiers take EEG epochs of shape `(n_epochs, n_channels, n_samples)`
(or a single live window `(n_channels, n_samples)`) and output 4-class
probabilities over `[FIRE, WATER, WIND, ELECTRICITY]`. Standard window:
32 channels @ 250 Hz × 3.0 s = 750 samples.

Run any classical model with:
```bash
cd scripts
uv run python training/train_and_run.py --dataset bids_tower_defense --sub 02 --ses all --alg <key>
```
(`--alg all` benchmarks every key below and keeps the winner.)

---

## 1. Classical toolbox — `scripts/training/algorithms.py`

Shared feature routines: `compute_ovr_csp` (:57), `project_csp_features`
log-variance (:92), `compute_covariance_matrices` (:103),
`project_to_riemannian_tangent_space` (:140), `extract_welch_bandpower_features`
(:166, relative PSD over Theta 4–8, Alpha 8–12, Low-Beta 12–20, High-Beta 20–32,
Gamma 32–45 Hz — see `DEFAULT_BANDS`, :45).

| `--alg` key | Class (:line) | Idea in one line | Features in | Loss / objective | Output |
|---|---|---|---|---|---|
| `csp_lda` | `CSP_ShrinkageLDA` (:202) | One-vs-rest CSP spatial filters, Ledoit-Wolf shrinkage LDA | log-var of 4 filters/class | LDA closed-form (no iterative loss) | `predict_proba` |
| `csp_svm` | `CSP_SVM_RBF` (:234) | Same CSP features, non-linear boundary | standardized log-var CSP | Hinge loss, `C=1.0`, RBF kernel, Platt-scaled (`probability=True`) | calibrated probs |
| `fbcsp_logreg` | `FilterBank_CSP_LogReg` (:272) | CSP per frequency band, then linear readout | concat log-var CSP × 5 bands | L2-regularized cross-entropy, `C=0.5` | probs |
| `riemann_logreg` | `Riemannian_TangentSpace_LogReg` (:343) | Trial covariances → Fréchet-mean SPD manifold → tangent vectors | standardized tangent-space vector | L2 cross-entropy, `C=0.1`, no intercept | probs |
| `riemann_svm_lin` | `Riemannian_TangentSpace_SVM_Linear` (:389) | Same manifold features, max-margin readout | tangent-space vector | Hinge loss, `C=0.1`, linear kernel, Platt-scaled | calibrated probs |
| `riemann_ridge` | `Riemannian_TangentSpace_Ridge` (:427) | Same manifold features, least-squares readout | tangent-space vector | Squared loss + L2 (`alpha=10`), 3-fold `CalibratedClassifierCV` | calibrated probs |
| `riemann_svm_rbf` | `Riemannian_TangentSpace_SVM_RBF` (:466) | Same manifold features, curved boundary | tangent-space vector | Hinge loss, `C=1.0`, RBF kernel, Platt-scaled | calibrated probs |
| `welch_rf` | `Welch_PSD_RandomForest` (:504) | No spatial filtering; spectral fingerprints | relative bandpower/channel | Gini impurity, 150 trees, `max_depth=6` | vote-fraction probs |
| `welch_lda` | `Welch_PSD_ShrinkageLDA` (:539) | Same spectral features, linear readout | relative bandpower/channel | LDA closed-form with shrinkage | probs |
| `ensemble_voting` | `Ensemble_Voting` (:569) | Soft vote over CSP-LDA + FBCSP-LogReg + Riemannian-LogReg | all three above | mean of the three probability vectors | averaged probs |

Notes:
- `Riemannian_TangentSpace_LogReg.recalibrate_reference()` (:380) re-estimates
  the Fréchet mean on new calibration trials — the few-shot adaptation hook.
- Factory: `create_classifier(alg_key, **kwargs)` (:661); registry `ALGORITHMS` (:607).
- Analyses also use `pyriemann`'s MDM (minimum-distance-to-mean) and `RegCovariances`
  (OAS shrinkage, `scripts/analysis/analyze_fnf_bci.py:90`) in the same tangent-space family.

## 2. Deep toolbox — PyTorch nets in `scripts/analysis/`

All trained with **`nn.CrossEntropyLoss`** (multi-class cross-entropy) unless noted.
Canonical EEGNet: temporal conv → depthwise spatial conv → separable conv,
BatchNorm + Dropout, linear head (`benchmark_deep_learning.py:37`,
`F1=8, D=2, F2=16`).

| Net | File | Input | Loss | Notes |
|---|---|---|---|---|
| `EEGNet` (+ `ShallowFBCSPNet`) | `analysis/benchmark_deep_learning.py:37,106` | `(1, 32, 750)` | CrossEntropy | Pretrained weights: `models/pretrained_eegnet_sub02.pt` |
| `EEG_Spatial_BiLSTM` | `analysis/benchmark_rf_lstm_architectures.py:49` | `(32, T)` → 1×1 spatial conv → BiLSTM | CrossEntropy | `hidden=32, layers=2, dropout=0.3` |
| `FFTSpatialNet`, `STFTConvNet` | `analysis/benchmark_fft_deep_learning.py:32,100` | FFT/STFT spectral maps | CrossEntropy | Frequency-domain baselines |
| `MultimodalAdaptiveGatedNet` | `analysis/multimodal_gated_network.py:114` | EEG `(32, 626)` + PPG (4-dim) + IMU (20-dim) | CrossEntropy | `ModalityGatingMechanism` (:88) learns softmax weights `[w_EEG, w_PPG, w_IMU]`; encoders `EEGStreamEncoder` (:39), `Signal1DEncoder` (:71) |

## 3. Real-time inference — `models/assistive_inferencer.py`

- `HierarchicalAssistiveDecoder` (`scripts/analysis/hierarchical_assistive_decoder.py:34`):
  2-tier Riemannian tangent-space tree. `predict_single_epoch` (:85) scores one
  window; `accumulate_and_decide` (:107) fires only when posterior ≥ threshold
  (default 0.85, eval uses 0.75–0.80) or after `max_steps=3` windows.
- `AssistiveBCIInferencer` (`models/assistive_inferencer.py:33`): streaming wrapper.
  `push_chunk` (:118) appends 50–100 ms chunks, runs a Bayesian log-odds update
  per full window, returns `{intent, class_id, confidence, is_confirmed, probabilities}`.
  `adapt_online` (:183) keeps a 45-trial replay buffer and refits the hierarchical
  nodes once ≥12 trials from ≥2 classes exist (counters dry-electrode drift).
- Ship-ready weights: `models/hierarchical_riemannian_sub02.joblib`,
  `models/pretrained_eegnet_sub02.pt`, channel/class/threshold metadata in
  `models/model_metadata.json` (exported by `scripts/analysis/export_models.py`).

## Which to use when

- Few trials, dry-cap drift → `riemann_logreg` (+ `recalibrate_reference`) or `ensemble_voting`.
- Plenty of clean trials → `fbcsp_logreg`, `csp_svm`, or EEGNet.
- Noisy single channels, quick check → `welch_rf` (no spatial filter to break).
- Live game / assistive output → hierarchical decoder + `AssistiveBCIInferencer`
  (never raw `predict`; always go through evidence accumulation).
