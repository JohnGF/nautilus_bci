# Project Documentation & Beginner's Guide

This directory contains technical documentation for the `nautilus_bci` project, covering signal processing theory, experimental task paradigms, and dataset storage standards.

---

## Recommended Reading Path for Beginners

### Step 1: Beginner's Guide & Codebase Map (`getting_started_bci_guide.md`)
- High-level introduction to BCI concept, signal acquisition, and real-time streaming.
- Diagram of the 5-stage system architecture (Sensors -> LSL Bridges -> Tasks -> BIDS Recorders -> ML Decoding).
- Direct reference map linking every key script to its role in the `scripts/` folder.
- Quickstart batch file execution guide.

### Step 2: BIDS Dataset Standard & Usage (`bids_standard_and_usage.md`)
- Introduction to the Brain Imaging Data Structure (BIDS) specification.
- Directory hierarchy, European Data Format (EDF) continuous recordings, JSON metadata sidecars, and TSV event tables.
- Real-time multimodal LSL stream polling (`gNautilus` EEG, `Smartwatch_IMU`, `Smartwatch_PPG`).
- Time-stamping, MNE `RawArray` construction, and automated BIDS dataset exporting via `multimodal_bids_recorder.py`.
- Dataset loading and analysis using `analyze_bids_dataset.py`.

### Step 3: BCI Signal Processing Theory (`bci_signal_processing.md`)
- Plain English intuition explanations for newcomers on CSP and LDA.
- Detailed mathematical formulations for Common Spatial Patterns (CSP) spatial filtering.
- Derivation of Linear Discriminant Analysis (LDA) decision hyperplanes.
- Bandpass filtering (mu/beta sensorimotor rhythms) and log-variance feature extraction.
- Codebase implementations (`scripts/analysis/eeg_features.py`, `scripts/analysis/compare_bci_paradigms.py`).

---

## Experimental Hardware Manuals
Hardware user manuals and datasheets for g.tec amplifiers, electrode positioning caps, and smartwatch streaming setups are located in the top-level project folder:
- `manuals/g.Nautilus PRO Manual.pdf`
- `manuals/g.Nautilus research Manual.pdf`
- `manuals/g.NEEDaccess API Manual.pdf`
