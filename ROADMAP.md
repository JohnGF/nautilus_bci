# 🗺️ Multimodal BCI Toolbox — Roadmap & Development Priorities

This document outlines the planned architecture, real-time DSP pipelines, and experimental paradigms inspired by battle-tested tools (**BCI2000**, **OpenViBE**, **EEGLAB/LSL**) reimagined for a modern Python / PySide6 / BIDS ecosystem.

---

## 1. Hardware & Driver Management *(g.Nautilus & Sensors)*

- [x] **Wet vs. Dry (g.SAHARA) Auto-Configuration** (`gds_to_lsl.py`):
  - Hardware bandpass auto-selection (`0.5–30 Hz` / `0.1–100 Hz`) to eliminate DC contact potentials and prevent ADC rail saturation.
  - Dry-calibrated impedance scales (`<100 kΩ` Good, `<200 kΩ` Acceptable).
- [ ] **Saved Hardware Configuration Profiles (`config.json`)**:
  - Store and reload device presets (sampling rates, sensitivities, notch/bandpass filters, and radio channel) per participant/montage without manual CLI steps.
- [ ] **RF Radio Channel Scanner & Hopping**:
  - Expose `GDS_GNAUTILUS_GetNetworkChannel()` / `SetNetworkChannel()` to scan and select clean 2.4 GHz wireless channels if local WiFi causes packet loss.
- [ ] **Electrode Impedance Trend Tracking**:
  - Periodic background contact tracking to visualize impedance changes (e.g. gel drying out during long experimental runs).

---

## 2. Real-Time Signal Processing & Spatial Filters

- [x] **Modular Spatial Referencing Engine** (`analysis/spatial_filters.py`):
  - **Robust CAR (Median)**: Immune to outlier/railed pins contaminating other channels.
  - **Surface Laplacian (Hjorth)**: Local 10–20 spatial neighborhood derivatives for focal motor ERD/ERS.
  - **Standard CAR (Mean)**: Traditional common average reference.
  - **Raw / Mastoid**: Unreferenced baseline.
- [x] **Live Bad-Channel & Rail Detection**:
  - Real-time detection of saturated ADCs, flatlines, and extreme noise spikes.
- [ ] **Adaptive EOG / Blink Artifact Regression**:
  - Real-time adaptive LMS / RLS filter using frontal electrodes (`Fp1`, `Fp2`) to remove blink artifacts from central motor channels (`C3`, `C4`, `Cz`).
- [ ] **Live 2D Topographic Band-Power Animation**:
  - Real-time animated 10–20 scalp heatmaps for Alpha (8–12 Hz), Beta (12–30 Hz), and Theta (4–8 Hz) power.

---

## 3. Task Paradigms & Event Synchronization

- [x] **LSL Marker Synchronization Protocol**:
  - High-precision timestamped event streaming aligned with BIDS `events.tsv`.
- [ ] **Motor Imagery Paradigm Suite**:
  - Left Hand, Right Hand, Both Hands, Feet, and Tongue visual cues with jittered fixation and audio triggers.
- [ ] **P300 Speller / Oddball Paradigm**:
  - Visual row/column flashing grid with synchronized marker timestamps.
- [ ] **Resting State Baseline Runner**:
  - Standardized Eyes-Open / Eyes-Closed audio-prompted calibration recording.

---

## 4. Online Decoding & Neurofeedback

- [ ] **Real-Time Streaming Classifier Worker**:
  - Background thread calculating live Riemannian MDM / CSP + LDA probabilities on sliding LSL epochs.
- [ ] **Live Visual Neurofeedback Widgets**:
  - Real-time motor imagery confidence bars and continuous 1D/2D cursor control.
- [ ] **Calibration & Training Wizard**:
  - 1-Click workflow: *Record Training Paradigm $\rightarrow$ Fit Decoder Model $\rightarrow$ Launch Live Feedback*.

---

## 5. Multimodal BIDS Data Management

- [x] **Multimodal BIDS Dataset Recorder** (`recorders/multimodal_bids_recorder.py`):
  - Aligns and packages 32-channel EEG, Smartwatch PPG, and 6-DOF IMU into standard BIDS structure (`sub-XX/ses-YY/eeg/`).
- [ ] **Automated Session Summary Reports**:
  - Auto-generate post-session HTML/Markdown quality reports (SNR, channel impedance summary, battery discharge rate, trial accuracy).
