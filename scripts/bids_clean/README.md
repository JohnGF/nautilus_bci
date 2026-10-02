# Clean BIDS Neuro-Gaming Datasets

This directory contains the consolidated, standardized, and fully validated BIDS (v1.8.0) datasets for the g.Nautilus 32-channel EEG BCI project.

All datasets have been validated with `mne_bids.read_raw_bids()` (100% compliance across all 26 core runs).

> **Assistive Technology Focus**: For the clinical protocol, pure mental imagery rationale for paralyzed individuals, and real-time inference engine, see the [Assistive BCI Clinical & Technical Guide](ASSISTIVE_BCI_GUIDE.md).

---

## 1. Directory Structure & Paradigms

```text
scripts/bids_clean/
├── bids_td_classical/              # Subject 1: Classical Music Tower Defense
│   └── sub-01/
│       ├── ses-01 to ses-06/       # 6 active spell recall sessions (194 trials)
│       ├── ses-listening/          # Continuous classical listening stream (551 epochs)
│       ├── sub-01_sessions.tsv     # Recording dates, lab location, cap installation notes
│       └── sub-01_sessions.json    # Schema definitions for sessions metadata
│
├── bids_td_modern/                 # Subject 2: Modern Pop/Rock Tower Defense
│   └── sub-02/
│       ├── ses-01 to ses-05/       # 5 active recall sessions (282 trials, ses-03: 51.87%)
│       ├── ses-listening/          # Consolidated 4-song acoustic prior (379 epochs)
│       │   ├── run-01 (It's Raining Men - Water element)
│       │   ├── run-02 (What's Up - Wind element)
│       │   ├── run-03 (Thunderstruck - Electricity element)
│       │   └── run-04 (Kiss - Fire element)
│       ├── sub-02_sessions.tsv     # Recording dates, Cz/F4 electrode gel restoration notes
│       └── sub-02_sessions.json    # Schema definitions for sessions metadata
│
├── bids_fnf/                       # Friday Night Funkin' Directional Rhythm Game
│   ├── sub-01/                     # 7 sessions (motor imagery & rhythm game benchmark)
│   └── sub-03/                     # 3 sessions (4-direction rhythm arrow gameplay)
│
└── bids_auxiliary_baseline/        # Archived Signal Quality & Calibration Checks
    └── sub-01/                     # Video viewing resting state & hand motor imagery
```

---

## 2. Experimental Paradigms

### A. Tower Defense Classical (`bids_td_classical`, Subject 1)
- **Acoustic Prior**: Continuous listening to orchestral classical pieces (Bach, Beethoven, Joplin, Mozart, Vivaldi, Tchaikovsky) recorded under `ses-listening`.
- **Active Task**: 4-Class silent mental imagery / spell recall (`FIRE`, `WATER`, `WIND`, `ELECTRICITY`) across 6 recording sessions (`ses-01` to `ses-06`).
- **Trial Structure**: 5-second silent mental imagery window following element selection.

### B. Tower Defense Modern Pop/Rock (`bids_td_modern`, Subject 2)
- **Acoustic Prior**: Continuous listening to 4 modern songs under `ses-listening`:
  - `run-01`: *It's Raining Men* (Water element)
  - `run-02`: *What's Up* (Wind element)
  - `run-03`: *Thunderstruck* (Electricity element)
  - `run-04`: *Kiss* (Fire element)
- **Active Task**: 4-Class modern song recall (`recallSongsIIWT`) across 5 distinct recording days/installations (`ses-01` to `ses-05`).
- **Electrode Contact Note**: `ses-03` was recorded with optimized dry electrode pin contact restoring vertex `Cz` and right-frontal `F4` (pins mechanically parted through hair and firmly seated), achieving **51.87%** single-session 4-class decoding.

### C. Friday Night Funkin' (`bids_fnf`, Subjects 1 & 3)
- **Active Task**: 4-Directional arrow rhythm game (`Left`, `Right`, `Up`, `Down`) during gameplay hit windows.
- **Subject 1**: 7 sessions tracking progression from baseline motor imagery to full gameplay (peak benchmark: 59.46%).
- **Subject 3**: 3 consecutive gameplay calibration sessions.

---

## 3. Quick-Start: Loading the Data in Python

A dedicated data loader is available at [`scripts/analysis/load_td_data.py`](../analysis/load_td_data.py):

```python
from load_td_data import load_classical_tower_defense, load_modern_tower_defense

# 1. Load Subject 1 (Classical Paradigm)
(X_recall, y_recall), X_listen = load_classical_tower_defense()
print(f"Classical Recall: {X_recall.shape} trials, Listening Prior: {X_listen.shape} epochs")

# 2. Load Subject 2 (Modern Pop/Rock Paradigm)
(X_recall, y_recall, session_tags), (X_listen, y_listen) = load_modern_tower_defense()
print(f"Modern Recall: {X_recall.shape} trials, Listening Prior: {X_listen.shape} epochs")

# 3. Load a specific session (e.g. Session 03 which reached 51.87%)
(X_s03, y_s03, _), _ = load_modern_tower_defense(sessions=["03"])
```

---

## 4. Metadata & Session Details

Each subject folder contains a standardized BIDS `*_sessions.tsv` file detailing:
- `session_id`: Unique session folder identifier.
- `recording_date`: Calendar date of recording.
- `location`: Physical laboratory room.
- `activity_type`: Cognitive/perceptual state (`active_mental_imagery` vs `continuous_passive_listening`).
- `cap_installation`: Specific cap placement / day reinstall index.
- `impedance_notes`: Electrode contact notes (e.g. `optimized_dry_pins_Cz_F4_reseated`).
- `hardware_montage`: `dry_electrodes_32ch` across all sessions and subjects.
- `recommended_for_analysis`: Boolean flag indicating whether the session is recommended for model training or should be pruned.
- `pruning_rationale`: Scientific and empirical justification for inclusion or pruning.
- `experimental_notes`: Quantitative benchmarks and protocol observations.

---

## 5. Hardware Montage: 32 Dry EEG Channels & Pruning Dynamics

All recordings across all subjects (`sub-01`, `sub-02`, `sub-03`) and paradigms (Tower Defense Classical, Modern Pop/Rock, and FNF) utilize a **32-channel dry EEG electrode setup** (e.g. g.SAHARA multi-pin gold alloy electrodes without conductive gel or abrasive paste).

### Physical Implications of the Dry Setup:
1. **Electrode-Skin Impedance**: Dry pins have significantly higher baseline contact impedance (~10–50 kΩ) compared to wet gel (<5 kΩ). Proper contact depends on pin pressure and hair parting.
2. **Cap Re-Installation & Inter-Day Spatial Shift**: When the dry cap is removed and re-installed on subsequent days (`installation_1` vs `installation_4`), small millimeter shifts in pin angle and contact impedance rotate the spatial covariance manifold across sessions.
3. **Session Pruning Dynamics**:
   - **Modern Paradigm (`sub-02`)**:
     - `ses-03 Alone`: **51.87%** (4-class), **72.9%** (pairwise).
     - `ses-01 + ses-02 + ses-03` (Days 1 & 2): **47.61%** (4-class), **74.3%** (pairwise).
     - **Adding Day 3 (`ses-04` + `ses-05`)**: Performance drops to **26.27%** (near chance) due to dry pin contact degradation.
     - *Recommendation*: Train on `ses-01`, `ses-02`, `ses-03`; prune `ses-04`, `ses-05`.
   - **Classical Paradigm (`sub-01`)**:
     - `ses-01` (83 trials): **44.19%** (4-class), **84.8%** (`FIRE vs ELECTRICITY`).
     - `ses-04` (16 trials): **50.0%** (4-class), **75.0%** with Riemannian whitening.
     - `ses-02`, `ses-03`: Sample-limited (12–24 trials).
     - `ses-05` (56 trials): Extended recording dry pin contact drift (25.0% chance).
     - *Recommendation*: Train on `ses-01` + `ses-04` + `ses-listening`; prune `ses-02`, `ses-03`, `ses-05`.
4. **Listening Prior Advantage**: Because obtaining long active dry-cap recordings is tiring and prone to contact degradation over time, using the passive listening session (`ses-listening`) as a covariance prior allows **few-shot adaptation** ($k=1$ to $5$ active trials) to match or exceed prolonged active calibration.
