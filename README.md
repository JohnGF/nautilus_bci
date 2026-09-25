# Nautilus BCI: End-to-End Brain-Computer Interface Suite

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Welcome to the **Nautilus BCI** project. This is a comprehensive Python suite for streaming 32 channels of EEG data from the g.Nautilus BCI headset (`NP-2026.05.01`), synchronizing physiological data from smartwatches, running experimental task paradigms, and performing offline Machine Learning decoding on BIDS datasets.

---

## 📖 Documentation & Guides

- [Beginner's Guide & System Architecture](docs/getting_started_bci_guide.md)
- [BIDS Dataset Standard & Usage](docs/bids_standard_and_usage.md)
- [BCI Signal Processing Theory](docs/bci_signal_processing.md)
- [Scripts Directory README](scripts/README.md)

---

## ⚡ Core Features

- **Real-Time Streaming**: Native C++ API bridging to LSL for g.Nautilus EEG and UDP-to-LSL for Galaxy Watch IMU/PPG.
- **Multimodal BIDS Recording**: Automatically synchronizes and saves continuous signals and event markers into the standardized Brain Imaging Data Structure (BIDS).
- **Master Dashboard**: A single PySide6 GUI (`run_bci_suite.py`) to manage hardware, visualize 30FPS brain rhythms, start recordings, and launch tasks.
- **Experimental Paradigms**:
  - *Motor Imagery*: 4-Direction tasks for sensorimotor rhythm training.
  - *Auditory Recall*: 6-Track music memory tasks and full continuous track listening.
- **Machine Learning**: End-to-end extraction of Filter Bank Common Spatial Patterns (FBCSP) and Classification (ExtraTrees, RF, LDA) yielding benchmark accuracies > 65% on 4-class MI.

---

## 🚀 Quick Start

To launch the Master Control Panel (ensure you are using `uv` for dependency management):

```powershell
cd scripts
uv run python run_bci_suite.py
```

From the control panel, you can start the EEG streamer, observe the live signal scope, start BIDS recording, and launch any of the task paradigms.

---

## 🏗️ System Architecture & Interactive Pipelines

*Click on any node in the graphs below to navigate directly to the relevant source code or directory!*

### 1. Real-Time Data Collection Pipeline

```mermaid
flowchart TD
    %% Define Styles
    classDef hardware fill:#f9d0c4,stroke:#e88365,stroke-width:2px,color:black;
    classDef bridge fill:#c4e1f9,stroke:#5c9ad6,stroke-width:2px,color:black;
    classDef control fill:#d6f9c4,stroke:#7cd65c,stroke-width:2px,color:black;
    classDef task fill:#f9e8c4,stroke:#d6ad5c,stroke-width:2px,color:black;
    classDef output fill:#e2c4f9,stroke:#b15cd6,stroke-width:2px,color:black;

    %% Hardware Nodes
    EEG["🧠 g.Nautilus EEG\n(32 Ch @ 250 Hz)"]:::hardware
    Watch["⌚ Galaxy Watch\n(IMU + PPG)"]:::hardware

    %% Bridges
    EEGBridge["🔌 EEG to LSL Bridge\n(gds_to_lsl.py)"]:::bridge
    WatchBridge["🔌 Smartwatch Bridge\n(smartwatch_lsl_bridge.py)"]:::bridge

    %% Master Control
    Master["🎛️ Master Control Dashboard\n(run_bci_suite.py)"]:::control

    %% Tasks
    Tasks["🎮 Experimental Paradigms\n(scripts/tasks/)"]:::task

    %% Recording
    Recorder["💾 BIDS Recorder Engine\n(bids_recorder.py)"]:::output
    Dataset[/"📁 BIDS Dataset Output\n(scripts/bids/)"/]:::output

    %% Connections
    EEG --> EEGBridge
    Watch --> WatchBridge

    EEGBridge -- "Raw EEG Stream" --> Master
    WatchBridge -- "Physio Stream" --> Master

    Master -- "Launches" --> Tasks
    Tasks -- "Event Markers" --> Recorder
    EEGBridge -- "Continuous EEG" --> Recorder
    WatchBridge -- "Continuous Physio" --> Recorder

    Recorder -- "Saves to" --> Dataset

    %% Click Events
    click EEGBridge "scripts/bridges/gds_to_lsl.py" "View EEG Bridge Source"
    click WatchBridge "scripts/bridges/smartwatch_lsl_bridge.py" "View Smartwatch Bridge Source"
    click Master "scripts/run_bci_suite.py" "View Master Dashboard Source"
    click Tasks "scripts/tasks/" "View Task Paradigms Folder"
    click Recorder "scripts/recorders/multimodal_bids_recorder.py" "View BIDS Recorder Source"
    click Dataset "scripts/bids/" "View BIDS Output Directory"
```

### 2. Data Analysis & Machine Learning Pipeline

```mermaid
flowchart TD
    %% Define Styles
    classDef data fill:#e2c4f9,stroke:#b15cd6,stroke-width:2px,color:black;
    classDef process fill:#c4f9e8,stroke:#5cd6b1,stroke-width:2px,color:black;
    classDef ml fill:#f9c4d6,stroke:#d65c7c,stroke-width:2px,color:black;
    classDef result fill:#f9f5c4,stroke:#d6c45c,stroke-width:2px,color:black;

    %% Nodes
    InputData[/"📁 Saved BIDS Dataset\n(scripts/bids/)"/]:::data
    Loader["🔄 Epoching & Loading\n(analyze_bids_dataset.py)"]:::process
    FeatureExtraction["📊 Feature Extraction (FBCSP/LogVar)\n(eeg_features.py)"]:::process
    Training["🤖 Model Training\n(ExtraTrees, RF, LDA)"]:::ml
    Results(("🏆 Benchmark Results\n(65.6% Accuracy)")):::result

    %% Connections
    InputData --> Loader
    Loader --> FeatureExtraction
    FeatureExtraction --> Training
    Training --> Results

    %% Click Events
    click InputData "scripts/bids/" "View BIDS Directory"
    click Loader "scripts/analysis/analyze_bids_dataset.py" "View Dataset Analyzer Source"
    click FeatureExtraction "scripts/analysis/eeg_features.py" "View Feature Extraction Source"
    click Training "scripts/analysis/analyze_bids_dataset.py" "View Training Implementations"
```
