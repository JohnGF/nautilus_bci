# Brain Imaging Data Structure (BIDS) Standard and Project Usage

This document describes the Brain Imaging Data Structure (BIDS) specification, its data model, and how it is implemented and integrated within the `nautilus_bci` multimodal recording and analysis ecosystem.

---

## 1. Overview of BIDS

### 1.1 What is BIDS?
The Brain Imaging Data Structure (BIDS) is an international standard for organizing, naming, and describing neuroimaging and physiological data. By enforcing uniform file naming conventions and metadata schemas, BIDS facilitates data sharing, reproducibility, automated execution of processing pipelines, and long-term archival.

Originally developed for magnetic resonance imaging (MRI), BIDS has been extended to electroencephalography (EEG), magnetoencephalography (MEG), functional near-infrared spectroscopy (fNIRS), and multimodal physiological streams.

---

### 1.2 BIDS File and Folder Hierarchy
A standard EEG/Multimodal BIDS dataset follows a deterministic directory tree structure:

```
bids_dataset/
├── dataset_description.json
├── README
├── sub-01/
│   └── ses-01/
│       └── eeg/
│           ├── sub-01_ses-01_task-leftright_eeg.edf
│           ├── sub-01_ses-01_task-leftright_eeg.json
│           ├── sub-01_ses-01_task-leftright_events.tsv
│           ├── sub-01_ses-01_task-leftright_events.json
│           └── sub-01_ses-01_task-leftright_channels.tsv
└── sub-02/
    └── ses-01/
        └── eeg/
            └── ...
```

---

### 1.3 Key File Components

#### 1. Dataset Description (`dataset_description.json`)
A top-level JSON sidecar containing dataset-wide metadata, including BIDS version compliance, dataset name, authors, and license.

#### 2. Signal Data File (`.edf` or `.vhdr`/`.eeg`)
Contains the continuous multi-channel signal recordings (e.g., 32-channel EEG or 32-channel EEG + 6-DOF IMU + PPG). European Data Format (EDF) is the primary format used in this project.

#### 3. EEG Metadata Sidecar (`_eeg.json`)
Specifies hardware recording parameters, such as sampling frequency (e.g., 250 Hz), reference electrode placement, hardware filters, power line frequency (50/60 Hz), and channel count.

#### 4. Event Annotations (`_events.tsv` and `_events.json`)
Stores time-locked experimental events (stimulus onset, task cues, trial markers). 

Format of `_events.tsv`:
| onset | duration | sample | value | trial_type |
|---|---|---|---|---|
| 0.0000 | 0.0000 | 0 | 1 | Video_Dataset_Experiment_Start |
| 1.5020 | 3.0000 | 375 | 2 | Video_Start_Clip01_AUDIO_ON |
| 4.5020 | 1.5000 | 1125 | 3 | Rest_Start |

#### 5. Channel Descriptions (`_channels.tsv`)
Lists details for every recorded channel: name, physical type (EEG, IMU, PPG), unit of measurement (uV, m/s^2, bpm), sampling rate, and status (good vs. bad).

---

## 2. BIDS Implementation in `nautilus_bci`

In the `nautilus_bci` project, BIDS dataset generation is automated via high-precision continuous buffering of Lab Streaming Layer (LSL) network outlets.

---

### 2.1 Multimodal Recording Architecture (`multimodal_bids_recorder.py`)

The recorder process executes the following pipeline:

#### Step 1: LSL Stream Discovery
The recorder resolves active LSL streams over the local network:
- **EEG Stream**: `gNautilus` (32 channels @ 250 Hz)
- **Smartwatch IMU Stream**: `Smartwatch_IMU` (6 channels: Accel XYZ, Gyro XYZ @ 50 Hz)
- **Smartwatch PPG Stream**: `Smartwatch_PPG` (2 channels: Heart Rate BPM, Confidence @ 1 Hz)
- **Marker Stream**: `VideoTaskMarkers` or `MotorImageryMarkers` (String markers)

#### Step 2: Timestamp Alignment and Continuous Buffering
As data packets arrive from different hardware clocks, each sample is stamped using `pylsl.local_clock()`. The recorder maintains continuous rolling numpy buffers for each stream, resolving clock offset jitter across hardware boundaries.

#### Step 3: MNE RawArray Construction
Upon stopping a recording session, buffered signals are aligned into an `mne.io.RawArray` structure. Standard 10-20 electrode positions (Fp1, Fp2, F3, F4, C3, C4, Cz, O1, O2, etc.) are attached to EEG channels using MNE's standard head montage system (`standard_1020`).

#### Step 4: Event Marker Parsing
LSL string markers emitted by task GUIs (e.g., `left_right_task.py` or `video_dataset_task.py`) are converted into MNE `Annotations` objects with onset timestamps relative to the start of the recording session.

#### Step 5: BIDS Export via MNE-BIDS
The recorder invokes `mne_bids.write_raw_bids()` to generate a fully compliant BIDS dataset on disk:

```python
bids_path = BIDSPath(
    subject=subject_id,
    session=session_id,
    task=task_name,
    root=bids_root,
    datatype="eeg"
)
write_raw_bids(raw, bids_path=bids_path, overwrite=True)
```

---

## 3. Reading and Analyzing Recorded BIDS Datasets

Once recorded, BIDS datasets can be programmatically loaded for offline signal processing, CSP-LDA classification, and feature extraction.

### Loading a BIDS Dataset with MNE-BIDS

```python
from mne_bids import BIDSPath, read_raw_bids

bids_path = BIDSPath(
    subject="01",
    session="01",
    task="leftright",
    root="bids_dataset",
    datatype="eeg"
)

# Load continuous raw data and metadata sidecars
raw = read_raw_bids(bids_path=bids_path)
raw.load_data()

# Extract event annotations
events, event_id = mne.events_from_annotations(raw)
print(f"Loaded BIDS recording with {len(raw.ch_names)} channels and {len(events)} events.")
```

### Corresponding Codebase Modules

- **`scripts/recorders/multimodal_bids_recorder.py`**: Continuous multimodal LSL stream recorder and BIDS exporter.
- **`scripts/recorders/bids_recorder.py`**: Standalone single-stream BIDS recorder.
- **`scripts/recorders/crop_bids_dataset.py`**: Trims non-experimental baseline periods from recorded BIDS files.
- **`scripts/analysis/analyze_bids_dataset.py`**: Automated pipeline for loading BIDS datasets, epoching around trial markers, computing bandpass filtering (8-30 Hz), and performing CSP-LDA classification.
- **`scripts/analysis/analyze_music_bci.py`**: Specialised BIDS analysis pipeline for Auditory Imagery and Music Recall experiments.
