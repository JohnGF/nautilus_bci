# BCI Tower Defense — Rhythm Decoding Training Studio & Real-Time Bridge

This folder (`nautilus_bci/scripts/training`) contains the complete model training suite for decoding mental rhythms (4 classes: **FIRE**, **WATER**, **WIND**, **ELECTRICITY**) in the **Tower Defense BCI** game.

It lets you select BIDS datasets (with different music and recording rooms), choose the subjects and sessions to include in training, train with any of the 10 decoding algorithms with stratified cross-validation (5-fold CV), save the model artifacts, and immediately start the real-time pipeline connected to Godot.

---

## 🖥️ How to Launch the Graphical App (GUI)

To open the interactive GUI:

```bash
# From the nautilus_bci root or the training folder:
uv run python scripts/training/train_and_run.py
```
*(or directly)*:
```bash
uv run python scripts/training/gui.py
```

### GUI Features:
1. **Dataset & Music Variation Selection**:
   - `bids_tower_defense`: Standard and modern sessions (sub-02 with rock/pop: Kiss, What's Up, Thunderstruck, etc.; sub-01 with classical pieces and water variant).
   - `bids_tower_defense(old)`: Original sessions with pure classical music (Für Elise, Bach Prelude, Vivaldi Spring, Tchaikovsky Waltz).
   - `bids_tower_defense_6_3_27`: Single-recording variation (ses-01).
2. **Subject & Session Selection**:
   - Automatic subject detection (`sub-01`, `sub-02`).
   - Individual per-session checkboxes (`ses-01`, `ses-02`, etc.) with pre-counted detected trials.
   - *"Select All"* and *"Clear All"* convenience buttons.
3. **10 Decoding Algorithms**:
   - Dropdown with every algorithm evaluated in `analyze_tower_defense_rhythm_decoding.py`.
   - Special **"⚡ Benchmark All"** option: evaluates all 10 models with 5-fold CV and automatically exports the most accurate one!
4. **Real-Time Metrics Panel**:
   - Training runs on a background thread without freezing the UI.
   - Metric cards: **CV Accuracy**, **Macro F1**, **Cohen's Kappa**, and **Total Trials**.
   - Confusion matrix and detailed report in the text console.
5. **Real-Time Bridge to Godot**:
   - **"🚀 LAUNCH REAL-TIME GAME PIPELINE"** button to start the pipeline.
   - Source choice: **Simulator (BIDS Replay)**, **Simulator (Synthetic)**, or **Live LSL (g.Nautilus)**.
   - Configurable activation threshold (*confidence threshold*).
   - Sends UDP commands straight to Godot (`127.0.0.1:4242`).

---

## ⚡ Command-Line Usage (CLI)

You can also run training headless or automated via the terminal:

### Examples:

#### 1. Train Riemannian Tangent Space on a single session:
```bash
uv run python scripts/training/train_and_run.py --dataset bids_tower_defense --sub 02 --ses 05 --alg riemann_logreg
```

#### 2. Train pooling multiple sessions (e.g. all sub-02 sessions):
```bash
uv run python scripts/training/train_and_run.py --dataset bids_tower_defense --sub 02 --ses 01,02,03,04,05 --alg riemann_logreg
```

#### 3. Train on the old dataset (original classical music):
```bash
uv run python scripts/training/train_and_run.py --dataset "bids_tower_defense(old)" --sub 01 --ses 01,02,03 --alg fbcsp_logreg
```

#### 4. Benchmark all algorithms and pick the best:
```bash
uv run python scripts/training/train_and_run.py --dataset bids_tower_defense --sub 02 --ses all --alg all
```

#### 5. Train and immediately launch the real-time pipeline with the simulator:
```bash
uv run python scripts/training/train_and_run.py --dataset bids_tower_defense --sub 02 --ses 05 --alg riemann_logreg --run-pipeline --source simulator --mode bids_replay
```

---

## 🧠 Available Algorithms

| Key (`--alg`) | Name | Description |
| :--- | :--- | :--- |
| `riemann_logreg` | Riemannian Tangent Space + LogReg | Riemannian geometry on SPD manifolds (Fréchet mean) + L2 Logistic Regression |
| `fbcsp_logreg` | Filter Bank CSP + LogReg | Multi-band spatial filtering (Theta, Alpha, Low/High Beta, Gamma) + LogReg |
| `csp_lda` | CSP + Shrinkage LDA | One-vs-Rest CSP spatial filters with regularized LDA (Ledoit-Wolf shrinkage) |
| `csp_svm` | CSP + SVM (RBF) | CSP combined with non-linear radial-basis-function Support Vector Machine |
| `riemann_svm_lin`| Riemannian Tangent Space + Linear SVM | Tangent Space with Linear SVM and probability calibration (Platt scaling) |
| `riemann_ridge` | Riemannian Tangent Space + Ridge | Tangent Space with L2 Ridge classifier and probability calibration |
| `riemann_svm_rbf`| Riemannian Tangent Space + RBF SVM | Tangent Space with RBF kernel for curved decision boundaries |
| `welch_rf` | Welch PSD + Random Forest | Classic EEG band power spectral density + Random Forest |
| `welch_lda` | Welch PSD + Shrinkage LDA | Relative Welch bandpower + Linear Discriminant Analysis |
| `ensemble_voting`| Ensemble Soft Voting | Weighted/soft vote across CSP+LDA, FBCSP and Riemannian Tangent Space |
| `all` | Competitive Benchmark | Evaluates all 10 models and trains the best on the full dataset |

See [Algorithm Toolbox](../../docs/guides/algorithm_toolbox.md) for inputs, loss functions, and outputs of each model.

---

## 📂 Files and Generated Structure

- `models/`: Folder where generated artifacts are saved:
  - `rhythm_model_<tag>.joblib`: Trained model, immediately ready to be consumed by `tower-defense-bci`.
  - `rhythm_report_<tag>.json`: Full report with cross-validation accuracy, standard deviation, macro F1, Cohen's kappa, and confusion matrix.
- `algorithms.py`: Scikit-learn compatible estimator implementations for real-time per-window `(32, 750)` inference.
- `dataset.py`: BIDS folder locator, subject/session catalog, and preprocessing/epoch-extraction routines.
- `gui.py`: Modern desktop GUI (Tkinter).
- `train_and_run.py`: Unified entry point for GUI and CLI.
