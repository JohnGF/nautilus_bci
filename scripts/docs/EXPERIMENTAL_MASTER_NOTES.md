# Master Experimental Notes and Research Log

This document consolidates all hardware discoveries, experimental paradigms, neuroscience insights, machine learning benchmarks, and session histories for the g.Nautilus 32-Channel BCI Project.

---

## 1. Hardware Infrastructure and Technical Discoveries

### Device Profile
- **Headset Model**: g.Nautilus 32-Channel Wireless EEG (`NP-2026.05.01`).
- **Sampling Rate**: 250.0 Hz (4.0 ms per sample interval).
- **Data Channels**: 33 Channels total (32 Scalp EEG Channels + 1 Battery Telemetry Channel).
- **Montage Mapping**: 10-20 Standard System (Channel 1 = Cz, Channel 5 = C3, Channel 6 = C4, Channel 13 = T7, Channel 14 = T8).

### Hardware Lock and GDS Service Management
- **Problem**: The manufacturer's g.NEEDaccess server executable (`g.server.exe` / Windows service `g.NEEDaccess Server`) can retain exclusive USB locks on the receiver if a script crashes or exits abruptly.
- **Solution**: Set `open_exclusively=False` in `gds_to_lsl.py` and wrap device handles in `try...finally: device.Close()`.
- **Emergency Recovery**: Restart the Windows service via PowerShell in 2 seconds without rebooting Windows:
  ```powershell
  Restart-Service -Name 'g.NEEDaccess Server'
  ```
  This command is integrated into the Master Control GUI (`run_bci_suite.py`) via the Emergency Reset GDS Service button.

### Electrode Impedance Checks
- **Behavior**: Interactive impedance measurements can hang automated pipelines.
- **Fix**: Default `run_impedance_loop` to `No (Skip)` in interactive mode and automatically bypass impedance checks in non-interactive background mode.

---

## 2. Neuroscience Insights and Cortical Mapping

### Cortical Channel Differentiation

```
                   Frontal (Fp1, Fp2) 
                   [Executive Focus / Eye Blinks]
                                |
Temporal (T7) <--- Central (C3, Cz, C4) ---> Temporal (T8)
[Auditory Cortex / Music]  [Motor Cortex / Movement]   [Auditory Cortex / Music]
                                |
                   Occipital (O1, O2)
                   [Visual Alpha / Eyes Closed]
```

#### Central Channels (C3, Cz, C4) - Primary Motor Cortex (M1)
- Measures sensorimotor rhythms (Mu 8-12 Hz, Beta 13-30 Hz).
- Displays Event-Related Desynchronization (ERD) during contralateral motor execution and imagery (e.g. imagining Left Hand movement suppresses C4 alpha power).

#### Temporal Channels (T7, T8, TP9, TP10) - Primary Auditory Cortex (A1)
- Positioned over Superior Temporal Gyrus (STG).
- Measures acoustic feature processing, pitch, tempo, and rhythm entrainment.
- Displays a **3.7x power surge** during active music listening compared to silent baselines.

---

## 3. Experimental Paradigms and Music Interaction

### The Auditory-Motor Competition Effect
- **Finding**: Background music reduces pure motor imagery decoding accuracy from **64.7% down to 35.7%** (across 4 classes).
- **Mechanism**: Auditory cortical activation ($>100,000\,\mu\text{V}^2/\text{Hz}$ power surge) introduces neural noise into sensorimotor channels, masking subtle Mu/Beta ERD signals.

### The "Music Context Switch" Architecture (Midas Touch Solution)
- **Concept**: Use music presence as a passive neural gate to solve the BCI "Midas Touch" problem (preventing accidental commands when resting).
- **State A (Music Playing)**: High temporal power ($T_7, T_8$). The BCI locks motor outputs and remains in **Standby / Idle Mode**.
- **State B (Silence / Cue)**: Temporal noise drops, unlocking clean sensorimotor rhythms ($C_3, C_4$). The BCI switches to **Active Command Mode** (**65.6% 4-Class Accuracy**).

---

## 4. Task Paradigm Definitions

### Paradigm 1: 4-Direction Motor Imagery (`left_right_task.py`)
- **Classes**: Top (Up), Bottom (Down), Left (Hand), Right (Hand).
- **Modes**:
  - Eyes-Closed Audio Mode: High Tone (1000 Hz) = Top, Low Tone (320 Hz) = Bottom, Mid-Low (520 Hz) = Left, Mid-High (780 Hz) = Right.
  - Audio + Visual Mode: Directional screen arrows + overlay audio cue tones.
  - Visual-Only Mode: Directional screen arrows.
- **Dual Audio Engine**: Overlay audio cue tones play at 95% volume while background music plays at 80% volume without mutual interruption.

### Paradigm 2: 6-Track Music Memory Recall (`music_memory_task.py`)
- **Objective**: Evaluates auditory memory recall and mental music imagery.
- **Master Composition Catalog & Smart Slices**:
  1. **Beethoven - Für Elise**: 5.0s seek offset (Main piano theme).
  2. **Scott Joplin - The Entertainer**: 10.0s seek offset (Main ragtime theme).
  3. **J.S. Bach - Prelude in C Major**: 8.0s seek offset (16th-note arpeggio loop).
  4. **W.A. Mozart - Eine kleine Nachtmusik**: 2.0s seek offset (Allegro motif).
  5. **Antonio Vivaldi - The Four Seasons: Spring**: 12.0s seek offset (Main string concerto).
  6. **Pyotr Tchaikovsky - Waltz of the Flowers**: 45.0s seek offset (Main waltz theme, bypassing harp intro).
- **Trial Structure**: 3.0s Cue Audio Sample -> 5.0s Mental Memory Recall Epoch (Eyes Closed, Silent EEG) -> 2.5s Rest.

---

## 5. Machine Learning Benchmarks and Session History

### Session Summary History

| Session ID | Total Duration | Recorded Trials | Conditions Evaluated | Primary Findings |
| :--- | :--- | :--- | :--- | :--- |
| **Session 01** | 488.7 s | 120 trials | 4-Class Motor Imagery Baseline | Established initial 4-class event marker pipeline. |
| **Session 02** | 978.3 s | 264 trials | Music (20 trials) vs No-Music (68 trials) | Proved silence yields 64.7% accuracy vs 35.7% with music. |
| **Session 03** | 620.0 s | 150 trials | Eyes-Closed 4-Direction Motor Imagery | Confirmed zero eye-blink artifact advantage. |
| **Session 04** | Active | 180 trials (Planned) | 6-Track Music Memory Recall (yt-dlp real MP3s) | Evaluating multi-track auditory imagery separability. |

### Pooled Machine Learning Benchmarks (Sessions 01 + 02, 384 Trials Total)

All models evaluated using 5-Fold Stratified Cross-Validation across 4 classes (Top, Bottom, Left, Right). Chance baseline is 25.0%.

| Model Pipeline | Feature Extraction Method | 4-Class Accuracy | Performance vs. Chance (25.0%) |
| :--- | :--- | :---: | :--- |
| **FBCSP + ExtraTrees Classifier** | Filter Bank CSP (Theta, Alpha, Beta) | **65.6% +- 2.9%** | **> 2.6x Above Chance** |
| **FBCSP + Random Forest** | Filter Bank CSP (Theta, Alpha, Beta) | **62.0% +- 4.5%** | **> 2.4x Above Chance** |
| **FBCSP + Gradient Boosting** | Filter Bank CSP (Theta, Alpha, Beta) | **60.4% +- 3.3%** | **> 2.4x Above Chance** |
| **CSP + Random Forest** | Single Band CSP (8-30 Hz) | **58.9% +- 5.4%** | **> 2.3x Above Chance** |
| **CSP + LDA** | Single Band CSP (8-30 Hz) | **46.4% +- 5.0%** | **> 1.8x Above Chance** |
| **CSP + SVM (RBF Kernel)** | Single Band CSP (8-30 Hz) | **28.4% +- 2.1%** | Baseline |

---

## 6. Software Architecture Index

| Script File | Language | Primary Purpose |
| :--- | :--- | :--- |
| `run_bci_suite.py` | Python (PySide6) | Master Control Center with live mini visualizer and task launcher |
| `gds_to_lsl.py` | Python | Streams 32 channels of g.Nautilus EEG data to LSL |
| `bids_recorder.py` | Python | Records LSL EEG and markers into BIDS dataset format |
| `left_right_task.py` | Python (PySide6) | 4-Direction motor imagery task presentation interface |
| `music_memory_task.py` | Python (PySide6) | 6-Track music memory recall presentation interface |
| `analyze_bids_dataset.py` | Python (MNE/scikit-learn) | BIDS dataset signal analysis, FBCSP feature extraction, and ML scoring |
| `lsl_viewer.py` | Python (pyqtgraph) | Fullscreen 32-channel stacked waveform plot |
| `eeg_features.py` | Python | Vectorized relative band power feature analyzer |
| `rhythm_visualizer.py` | Python (PySide6) | Interactive brain rhythm visualizer and simulator dashboard |
