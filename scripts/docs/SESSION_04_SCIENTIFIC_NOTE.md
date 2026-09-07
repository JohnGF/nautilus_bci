# Scientific Experiment Protocol Note: Session 04

## Study Metadata
- **Protocol ID**: SUB-01_SES-04_PARADIGM_MUSIC_MEMORY_6TRACK
- **Subject ID**: sub-01
- **Session ID**: ses-04
- **Date / Timestamp**: 2026-07-27
- **Hardware**: g.Nautilus 32-Channel EEG Headset (NP-2026.05.01) @ 250 Hz
- **Paradigm**: 6-Track Auditory Imagery & Musical Memory Recall BCI Paradigm

---

## Experimental Rationale and Scientific Objectives

### Primary Objective
To investigate whether mental playback (auditory imagery and internal musical memory recall) of 6 distinct musical master compositions generates statistically separable multi-channel electroencephalographic (EEG) patterns across temporal auditory cortex (T7, T8, TP9, TP10) and sensorimotor regions (C3, Cz, C4).

### Key Scientific Hypotheses
1. **Auditory Imagery Entrainment**: Mentally recalling rhythmic musical themes with eyes closed produces frequency-specific phase locking and power spectral shifts in temporal lobe channels (T7, T8), corresponding to the tempo and meter of the acoustic cue.
2. **Genre & Meter Separability**: Classical piano (Beethoven, Bach), ragtime jazz (Joplin), and orchestral symphonies (Mozart, Vivaldi, Tchaikovsky) elicit distinct multi-band Filter Bank Common Spatial Pattern (FBCSP) distributions across Theta (4-8 Hz), Alpha (8-12 Hz), and Beta (13-30 Hz) bands.
3. **Artifact-Free Neural Data**: Closed-eye auditory imagery eliminates ocular blink and saccade artifacts, maximizing signal-to-noise ratio (SNR) for feature extraction.

---

## Acoustic Stimuli Catalog (6 Master Slices)

Each trial presents a 3.0-second cue slice taken from the most recognizable main theme of the composition, followed by a 5.0-second mental recall epoch in complete silence.

| Track ID | Composition Name & Composer | Genre & Era | Start Seek Offset | Key Acoustic Features | LSL Event Marker |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Track 1** | Ludwig van Beethoven – *Für Elise* | Solo Piano / Classical | 5.0s | 126 BPM, A minor solo piano theme | `Task_Recall_Track_1` |
| **Track 2** | Scott Joplin – *The Entertainer* | Ragtime / Jazz Piano | 10.0s | 100 BPM, C major syncopated ragtime rhythm | `Task_Recall_Track_2` |
| **Track 3** | Johann Sebastian Bach – *Prelude in C Major* | Baroque Harpsichord/Piano | 8.0s | 84 BPM, C major 16th-note arpeggiated loop | `Task_Recall_Track_3` |
| **Track 4** | Wolfgang Amadeus Mozart – *Eine kleine Nachtmusik* | Classical String Quartet | 2.0s | 140 BPM, G major allegro theme | `Task_Recall_Track_4` |
| **Track 5** | Antonio Vivaldi – *The Four Seasons: Spring* | Orchestral Concerto | 12.0s | 112 BPM, E major concerto string theme | `Task_Recall_Track_5` |
| **Track 6** | Pyotr Tchaikovsky – *Waltz of the Flowers* | Romantic Symphony | 45.0s | 144 BPM, D major 3/4 waltz theme (bypasses harp intro) | `Task_Recall_Track_6` |

---

## Trial Protocol and Timing Structure

Each trial consists of 3 distinct, high-precision timestamped phases:

```
+--------------------------------+--------------------------------------+-----------------------+
| Phase 1: Cue Audio Sample      | Phase 2: Mental Memory Recall        | Phase 3: Rest         |
| Duration: 3.0 seconds          | Duration: 5.0 seconds                | Duration: 2.5 seconds |
| Audio: 3s Master Music Slice   | Audio: Complete Silence (Eyes Closed)| Audio: Silence        |
| Task: Listen to song snippet   | Task: Mentally play back song melody | Task: Relax gaze/jaw  |
+--------------------------------+--------------------------------------+-----------------------+
```

1. **Phase 1 (Cue Sample, 3.0s)**: Subject listens to the 3.0-second master audio slice. Timestamped as `Cue_Audio_Sample_Track_X`.
2. **Phase 2 (Mental Recall Epoch, 5.0s)**: Audio cuts off completely. Subject keeps eyes closed and mentally "plays back" the melody and rhythm in their head. Timestamped as `Task_Recall_Track_X` (the primary 5-second EEG epoch evaluated by machine learning classifiers).
3. **Phase 3 (Inter-Trial Rest, 2.5s)**: Subject relaxes eyes and jaw before the next randomized trial. Timestamped as `Rest`.

---

## BIDS Data Export Specifications

Recording is processed through `bids_recorder.py` and exported directly into BIDS standard directory format:
- **Target Path**: `bids_dataset/sub-01/ses-04/eeg/`
- **Output Files**:
  - `sub-01_ses-04_task-musicmemory_eeg.vhdr` (BrainVision Header)
  - `sub-01_ses-04_task-musicmemory_eeg.vmrk` (BrainVision Markers)
  - `sub-01_ses-04_task-musicmemory_eeg.eeg` (Binary Data Stream)
  - `sub-01_ses-04_task-musicmemory_events.tsv` (BIDS Event Timestamps)

---

## Post-Recording Analysis Commands

To evaluate FBCSP multi-class machine learning accuracy on Session 04:

```powershell
cd "C:\Users\VR\Documents\GitHub\nautilus_bci\scripts"
uv run python analyze_bids_dataset.py --sub 01 --ses 04
```
