# Nautilus BCI: g.Nautilus EEG Streaming, BIDS Dataset Suite, and Task Paradigms

A python suite for streaming 32 channels of EEG data from the g.Nautilus BCI headset (`NP-2026.05.01`), recording standardized BIDS datasets, running real-time signal processing and visualization, and executing motor imagery and auditory memory BCI task paradigms.

---

## System Overview

This codebase provides an end-to-end BCI research pipeline:
- **EEG Hardware Streaming**: Interface with g.Nautilus via `gds_to_lsl.py` with non-exclusive locks and automatic reconnect capability.
- **Master Control Panel**: `run_bci_suite.py` provides a single GUI window managing hardware streaming, live signal visualization, dataset recording, and task paradigms.
- **Embedded Signal Monitor**: Live 30 FPS oscilloscope (Cz, C3, C4, O1) and relative band power meters (Alpha, Beta, Delta) embedded directly in the control panel.
- **BIDS Dataset Recording**: `bids_recorder.py` streams 32-channel EEG and LSL marker timestamps directly into Brain Imaging Data Structure (BIDS) standard format.
- **4-Direction Motor Imagery Paradigm**: `left_right_task.py` presents randomized Top, Bottom, Left, and Right motor imagery cues with dual-layer audio tones for eyes-closed testing.
- **6-Track Music Memory Recall Paradigm**: `music_memory_task.py` presents 6 classical and jazz compositions (Beethoven, Joplin, Bach, Mozart, Vivaldi, Tchaikovsky) for auditory imagery and mental memory recall.
- **Advanced Machine Learning Pipeline**: `analyze_bids_dataset.py` extracts Filter Bank Common Spatial Patterns (FBCSP) across Theta (4-8 Hz), Alpha (8-12 Hz), and Beta (13-30 Hz) bands, achieving 65.6% 4-class accuracy using ExtraTrees and Random Forest classifiers.

---

## Quick Start Guide

### Launching the Master Control Panel
Run the all-in-one control panel:

```powershell
cd "C:\Users\VR\Documents\GitHub\nautilus_bci\scripts"
uv run python run_bci_suite.py
```

From the control panel interface:
1. Click **Start EEG Streamer** to connect the g.Nautilus headset (or start the simulated mock streamer).
2. Observe the **Live Mini Signal Scope** to confirm active EEG signals and brain rhythms.
3. Click **Start BIDS Recording** to begin logging session data.
4. Click **4-Direction Motor Imagery Task** or **6-Track Music Memory Recall Task** to launch the presentation interface.

---

## Core Task Paradigms

### 1. 4-Direction Motor Imagery Paradigm (`left_right_task.py`)
- **Directions**: Top (Up), Bottom (Down), Left (Hand), Right (Hand).
- **Presentation Modes**:
  - Eyes-Closed Audio Mode: High Tone (1000 Hz) = Top, Low Tone (320 Hz) = Bottom, Mid-Low (520 Hz) = Left, Mid-High (780 Hz) = Right.
  - Audio + Visual Mode: On-screen directional arrows with overlay audio cues.
  - Visual-Only Mode: On-screen directional arrows.
- **Dual Audio Engine**: Overlay audio cue tones play seamlessly over background music.

### 2. 6-Track Music Memory Recall Paradigm (`music_memory_task.py`)
- **Compositions Included (Full Real Master MP3 Recordings)**:
  - Track 1: Ludwig van Beethoven - Für Elise (Full Piano Master, 5.25 MB)
  - Track 2: Scott Joplin - The Entertainer (Full Ragtime/Jazz Piano Master, 5.57 MB)
  - Track 3: Johann Sebastian Bach - Prelude in C Major (Full Piano Master, 3.04 MB)
  - Track 4: Wolfgang Amadeus Mozart - Eine kleine Nachtmusik (Full Orchestral Master, 7.74 MB)
  - Track 5: Antonio Vivaldi - The Four Seasons: Spring (Full Concerto Master, 4.47 MB)
  - Track 6: Pyotr Ilyich Tchaikovsky - Waltz of the Flowers (Full Symphony Master, 10.33 MB)
- **Structure**: 3.0s Auditory Cue Sample -> 5.0s Mental Music Imagery & Memory Recall (Silent EEG epoch) -> 2.5s Rest.

### 3. Full-Length Continuous Music Listening Paradigm (`music_full_track_task.py`)
- **Start-to-Finish Playback**: Plays complete un-cut compositions (e.g. 2 to 7+ minutes per track) for continuous long-stream brainwave and neural entrainment recording.
- **Dynamic Track Discovery**: Select individual tracks or all tracks across `real`, `classical`, `jazz_ragtime`, and `synthetic_beats`.
- **EEG-Optimized Presentation**: Minimalist Fixation Cross (`+`), Rhythmic Pulsing Visualizer, or Modern Timecode mode to minimize ocular and saccade artifacts.
- **LSL Markers**: Synchronizes session start, baseline rest, track start with exact duration, 30s progress checkpoints, track end, and inter-track rest periods.
- **Quick Launch**: Run `scripts/start_music_listening_task.bat` or launch from the Task Selector (`start_task_selector.bat`).

---

## Machine Learning and Analysis

### Analyzing BIDS Sessions (`analyze_bids_dataset.py`)
Run analysis on recorded BIDS session datasets:

```powershell
uv run python analyze_bids_dataset.py --sub 01 --ses 02
```

### Benchmark Classification Performance (4-Class Paradigm)
Across pooled Session 01 and Session 02 datasets (384 trials total):

| Model Architecture | Feature Extraction Method | 4-Class Accuracy | Baseline Chance |
| :--- | :--- | :---: | :---: |
| **FBCSP + ExtraTrees** | Filter Bank CSP (Theta, Alpha, Beta) | **65.6% +- 2.9%** | 25.0% |
| **FBCSP + Random Forest** | Filter Bank CSP (Theta, Alpha, Beta) | **62.0% +- 4.5%** | 25.0% |
| **FBCSP + Gradient Boosting** | Filter Bank CSP (Theta, Alpha, Beta) | **60.4% +- 3.3%** | 25.0% |
| **CSP + Random Forest** | Single-Band CSP (8-30 Hz) | **58.9% +- 5.4%** | 25.0% |
| **CSP + LDA** | Single-Band CSP (8-30 Hz) | **46.4% +- 5.0%** | 25.0% |

### Temporal vs. Central Cortex Analysis
- **Central Channels (C3, Cz, C4)**: Primary Motor Cortex (M1) activity for motor imagery decoding.
- **Temporal Channels (T7, T8)**: Primary Auditory Cortex (A1) activity displaying 3.7x power surge during music entrainment.

---

## Hardware Emergency Service Reset

If the manufacturer's GDS service (`g.NEEDaccess`) locks up the USB receiver:
- Click **Emergency Reset GDS Service** inside `run_bci_suite.py`.
- This restarts the background Windows service if it fails reboot PC.

---

## Script Index

| Script File | Language | Description |
| :--- | :--- | :--- |
| `run_bci_suite.py` | Python (PySide6) | Master Control Center with live mini visualizer and task launcher |
| `gds_to_lsl.py` | Python | Streams 32 channels of g.Nautilus EEG data to LSL |
| `bids_recorder.py` | Python | Records LSL EEG and markers into BIDS dataset format |
| `left_right_task.py` | Python (PySide6) | 4-Direction motor imagery task presentation interface |
| `music_memory_task.py` | Python (PySide6) | 6-Track music memory recall presentation interface |
| `analyze_bids_dataset.py` | Python (MNE/scikit-learn) | BIDS dataset signal analysis, FBCSP feature extraction, and ML scoring |
| `lsl_viewer.py` | Python (pyqtgraph) | Fullscreen 32-channel stacked waveform plot |
| `eeg_features.py` | Python | Vectorized relative band power feature analyzer |
| `rhythm_visualizer.py` | Python (PySide6) | Brain rhythm explorer and simulator dashboard |
