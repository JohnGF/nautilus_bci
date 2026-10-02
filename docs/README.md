# Project Documentation

This directory consolidates all `nautilus_bci` documentation:

- `guides/` — beginner guides: system overview, BIDS standard, signal-processing theory, [algorithm toolbox reference](guides/algorithm_toolbox.md).
- `science-notes/` — experiment notes, milestones, and music-perception/transfer findings.
- `vendor-manuals/` — third-party hardware manuals (g.tec, Unicorn) for reference only.

---

## Recommended Reading Path for Beginners

### Step 1: Beginner's Guide & Codebase Map (`guides/getting_started_bci_guide.md`)
- High-level introduction to BCI concept, signal acquisition, and real-time streaming.
- Diagram of the 5-stage system architecture (Sensors -> LSL Bridges -> Tasks -> BIDS Recorders -> ML Decoding).
- Direct reference map linking every key script to its role in the `scripts/` folder.
- Quickstart batch file execution guide.

### Step 2: BIDS Dataset Standard & Usage (`guides/bids_standard_and_usage.md`)
- Introduction to the Brain Imaging Data Structure (BIDS) specification.
- Directory hierarchy, European Data Format (EDF) continuous recordings, JSON metadata sidecars, and TSV event tables.
- Real-time multimodal LSL stream polling (`gNautilus` EEG, `Smartwatch_IMU`, `Smartwatch_PPG`).
- Time-stamping, MNE `RawArray` construction, and automated BIDS dataset exporting via `multimodal_bids_recorder.py`.
- Dataset loading and analysis using `analyze_bids_dataset.py`.

### Step 3: BCI Signal Processing Theory (`guides/bci_signal_processing.md`)
- Plain English intuition explanations for newcomers on CSP and LDA.
- Detailed mathematical formulations for Common Spatial Patterns (CSP) spatial filtering.
- Derivation of Linear Discriminant Analysis (LDA) decision hyperplanes.
- Bandpass filtering (mu/beta sensorimotor rhythms) and log-variance feature extraction.
- Codebase implementations (`scripts/analysis/eeg_features.py`, `scripts/analysis/compare_bci_paradigms.py`).

---

## Experimental Hardware Manuals
Hardware user manuals and datasheets for g.tec amplifiers, electrode grids, and sensors live in `vendor-manuals/`, e.g.:
- `vendor-manuals/g.Nautilus PRO Instructions for Use 1.25.02.pdf`
- `vendor-manuals/g.Sensor 8fNIRS Instructions for Use 1.20.01.pdf`
- `vendor-manuals/g.Sensors Utilities Instructions for Use 1.25.01.pdf`
