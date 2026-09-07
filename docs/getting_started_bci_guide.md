# Beginner's Guide and Codebase Architecture Map

Welcome to the `nautilus_bci` project! This guide is designed for newcomers to Brain-Computer Interfaces (BCI), real-time physiological signal streaming, and machine learning for EEG. It provides an intuitive introduction to the core concepts and links them directly to the corresponding source files in the codebase.

---

## 1. What is a Brain-Computer Interface (BCI)?

A Brain-Computer Interface (BCI) establishes a direct communication pathway between the human brain and external devices. 

In this repository, we record electrical brain signals using Electroencephalography (EEG) caps (such as the 32-channel g.Nautilus PRO wireless system) alongside autonomic physiological data (PPG heart rate and 6-DOF IMU motion vectors from a Samsung Galaxy Watch).

### The 5 Core Steps of a BCI Experiment

1. **Signal Acquisition**: Electrodes on the scalp measure microvolt-level ($uV$) electrical activity from the brain's cerebral cortex.
2. **Real-Time Streaming**: Hardware streams raw samples over Wi-Fi or Bluetooth into Lab Streaming Layer (LSL) network outlets.
3. **Stimulus Presentation**: The participant performs tasks (e.g., imagining hand movement, watching video clips, or listening to audio cues) while LSL event markers are timestamped.
4. **Data Standardization**: Continuous signals and timestamped event markers are saved into standardized BIDS datasets.
5. **Feature Extraction & Decoding**: Signal processing (CSP bandpass filtering) and machine learning (LDA classification) decode the participant's intent or mental state.

---

## 2. End-to-End System Architecture

```
+-----------------------------------------------------------------------+
|                         HARDWARE SENSORS                              |
|  g.Nautilus EEG (32 Ch @ 250 Hz)    Galaxy Watch (IMU + PPG @ 50/1 Hz)|
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                         REAL-TIME LSL BRIDGES                         |
|  scripts/bridges/gds_to_lsl.py       scripts/bridges/smartwatch_lsl_bridge.py |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                    EXPERIMENTAL TASK STIMULI & MARKS                  |
|  scripts/tasks/task_launcher.py (Task Selector App)                   |
|  ├── scripts/tasks/video_dataset_task.py   (Video Paradigm)           |
|  ├── scripts/tasks/left_right_task.py      (Motor Imagery)           |
|  └── scripts/tasks/music_memory_task.py    (Auditory Recall)         |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                    RECORDING & MONITORING DASHBOARDS                  |
|  scripts/visualizers/multimodal_bci_dashboard.py (Live Scope & Quality)|
|  scripts/recorders/multimodal_bids_recorder.py   (BIDS Export Engine) |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                       OFFLINE ANALYSIS & DECODING                     |
|  scripts/analysis/analyze_bids_dataset.py    (BIDS Trial Epoching)    |
|  scripts/analysis/eeg_features.py            (CSP Filtering + LDA)    |
+-----------------------------------------------------------------------+
```

---

## 3. Codebase File Map & Reference Guide

This table links high-level BCI concepts to their exact script locations in `scripts/`:

| BCI Module / Component | Script Location | Primary Responsibility |
|---|---|---|
| **Task Launcher Studio** | `scripts/tasks/task_launcher.py` | GUI selector that dynamically discovers and runs task paradigms. |
| **Video Stimulus Task** | `scripts/tasks/video_dataset_task.py` | Displays 3s video stimuli, optional normalized WAV audio, 1.5s rest intervals, and emits LSL markers. |
| **Motor Imagery Task** | `scripts/tasks/left_right_task.py` | Presents Left Hand vs. Right Hand cues for motor cortex sensorimotor rhythm training. |
| **Auditory Recall Task** | `scripts/tasks/music_memory_task.py` | Audio cue presentation for music memory and auditory BCI experiments. |
| **Master Dashboard** | `scripts/visualizers/multimodal_bci_dashboard.py` | Master GUI displaying live 2D scalp headmap, channel quality tables, motor waves, smartwatch PPG/IMU, and 1-click BIDS recording. |
| **Electrode Quality View** | `scripts/visualizers/eeg_headmap_quality_visualizer.py` | Renders 10-20 system electrode locations with green/yellow/red quality dots. |
| **Hardware EEG Bridge** | `scripts/bridges/gds_to_lsl.py` | Connects to g.tec g.NEEDaccess C++ API and broadcasts raw EEG data onto LSL network. |
| **Smartwatch LSL Bridge** | `scripts/bridges/smartwatch_lsl_bridge.py` | Listens for UDP packets from Galaxy Watch and streams IMU/PPG via LSL outlets. |
| **BIDS Recorder Engine** | `scripts/recorders/multimodal_bids_recorder.py` | Synchronizes continuous LSL streams and outputs compliant BIDS datasets. |
| **Feature Extraction & ML** | `scripts/analysis/eeg_features.py` | Preprocesses EEG, computes CSP spatial filters, extracts log-variance features, and trains LDA classifiers. |

---

## 4. Quickstart Execution Guide

To get started on your local system, run any of the following batch scripts located in `scripts/`:

- **Run Task Launcher**: Double-click `scripts/start_task_selector.bat` or run `uv run python tasks/task_launcher.py`.
- **Run Master Dashboard**: Double-click `scripts/start_master_dashboard.bat` or run `uv run python visualizers/multimodal_bci_dashboard.py`.
- **Run Master BCI Suite**: Double-click `scripts/start_bci_suite.bat` or run `uv run python run_bci_suite.py`.
- **Run Smartwatch Bridge**: Double-click `scripts/start_smartwatch_lsl.bat` or run `uv run python bridges/smartwatch_lsl_bridge.py`.
