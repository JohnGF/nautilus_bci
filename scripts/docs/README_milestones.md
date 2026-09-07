# BCI g.Nautilus EEG Streaming and Real-Time Visualization

This project sets up a high-performance pipeline to stream 32 channels of EEG data from a g.Nautilus BCI device (`NP-2026.05.01`) to the Lab Streaming Layer (LSL), perform real-time signal processing, and visualize brainwave activity.

---

## Quick Start (How to Resume)

To run the full suite, open a terminal window and run:

```powershell
cd "C:\Users\VR\Documents\GitHub\nautilus_bci\scripts"
uv run python run_bci_suite.py
```

### Alternative Visualizer Options

* **Option A: Real-Time 32-Channel Waveform Plot (Python)**
  Displays all 32 channels stacked vertically with a gain/spacing slider.
  ```powershell
  uv run python lsl_viewer.py
  ```

* **Option B: Brain Wave Band Feature Analyzer (Python)**
  Displays relative powers (Delta, Theta, Alpha, Beta) and a rolling concentration index.
  ```powershell
  uv run python eeg_features.py
  ```

* **Option C: Interactive Brain Rhythm Explorer & Simulator (Python)**
  An educational visualizer designed specifically for students to explore brain rhythms (Delta, Theta, Alpha, Beta, Gamma) in real time. Can stream live from g.Nautilus LSL, or run in Simulation Mode.
  ```powershell
  uv run python rhythm_visualizer.py
  ```

* **Option D: C# WinForms Visualizer**
  ```powershell
  cd "..\EegVisualizer32"
  dotnet run
  ```

---

## Milestones Achieved

### 1. Environment and Dependency Setup
* Initialized a python environment managed by `uv` inside the repository directory.
* Configured dependencies: `pylsl`, `numpy`, `scipy`, `pyqtgraph`, `mne`, `mne-bids`, `scikit-learn`, and `PySide6`.
* Configured and compiled the C# WinForms project (`EegVisualizer32`) targeting `.NET 10`.

### 2. Resilient LSL Streamer (`gds_to_lsl.py`)
* Connects to the g.Nautilus device using non-exclusive locks (`open_exclusively=False`) so the stream recovers gracefully after crashes without locking up the GDS service.
* Dynamically unpacks g.Nautilus channel labels (e.g. `Fp1`, `Fp2`, `Cz`) and streams them to LSL.

### 3. Real-Time 32-Ch Signal Plot (`lsl_viewer.py`)
* Renders 32 channels of live EEG in a vertically stacked layout with custom channel spacing.
* Integrated a 4th-order Butterworth bandpass filter (2-45 Hz) and a 50 Hz Notch filter.

### 4. Vectorized Feature Analyzer (`eeg_features.py`)
* Computes live relative powers for Delta (0.5-4Hz), Theta (4-8Hz), Alpha (8-12Hz), and Beta (12-30Hz) bands.
* Vectorized pipeline using NumPy FFT operations (`np.fft.rfft(..., axis=0)`).

### 5. Interactive Rhythm Explorer & Simulator (`rhythm_visualizer.py`)
* Dark-themed Qt6 application to visualize relative power of brainwaves (Delta, Theta, Alpha, Beta, Gamma).
* Dual source design: resolves live LSL EEG streams or falls back to Interactive Simulator Mode.

### 6. BIDS Recorder and Dataset Pipeline (`bids_recorder.py`)
* Records continuous 32-channel EEG and high-precision LSL markers directly into Brain Imaging Data Structure (BIDS) standard datasets.

### 7. Task Paradigms (`left_right_task.py` and `music_memory_task.py`)
* Implemented 4-Direction Motor Imagery (Top, Bottom, Left, Right) with eyes-closed audio cue tones.
* Implemented 6-Track Music Memory Recall paradigm with classical and jazz compositions.

### 8. Machine Learning Pipeline (`analyze_bids_dataset.py`)
* Filter Bank Common Spatial Patterns (FBCSP) + ExtraTrees / Random Forest achieving 65.6% 4-class classification accuracy.

---

## Core Files Summary

| File Path | Language | Purpose |
| :--- | :--- | :--- |
| `scripts/run_bci_suite.py` | Python | Master Control Panel with live mini visualizer and task launcher |
| `scripts/gds_to_lsl.py` | Python | Stream BCI data to LSL |
| `scripts/bids_recorder.py` | Python | Record EEG and markers into BIDS dataset format |
| `scripts/left_right_task.py` | Python | 4-Direction motor imagery task presentation interface |
| `scripts/music_memory_task.py` | Python | 6-Track music memory recall presentation interface |
| `scripts/analyze_bids_dataset.py` | Python | BIDS dataset signal analysis, FBCSP feature extraction, and ML scoring |
| `scripts/lsl_viewer.py` | Python | 32-Channel Stacked waveform viewer |
| `scripts/eeg_features.py` | Python | Vectorized Alpha/Beta band power analyzer |
| `scripts/rhythm_visualizer.py` | Python | Interactive brain rhythm visualizer and simulator dashboard |
