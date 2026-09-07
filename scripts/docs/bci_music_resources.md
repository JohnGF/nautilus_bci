# 🎧 Audio Track Guide: Obtaining, Converting, & Calibrating WAV Files for BCI Paradigms

This documentation guide details how to obtain, convert, calibrate, and manage audio tracks for the **Nautilus BCI 6-Track Music Memory Recall Paradigm** (`music_memory_task.py`) and **Looping Calibrator Studio** (`music_offset_calibrator.py`).

---

## ⚡ 1. Why PCM WAV Format is Required for BCI

To guarantee **zero-latency audio playback** and **sample-accurate offset seeking** during real-time BCI EEG trial sessions, all audio tracks are processed as uncompressed 16-bit 44.1kHz PCM `.wav` files.

### 🔴 The MP3 FFmpeg Stream Decoding Issue
- Compressed formats (`.mp3`, `.aac`, `.m4a`) rely on variable frame headers. When seeking to mid-song offsets (e.g. $14.5\text{s}$ into a song), PySide6 / Qt6 `QMediaPlayer` using FFmpeg must parse stream headers asynchronously, causing initial loading stalls (`MediaStatus.LoadingMedia`) and inaccurate start offsets.
- **16-bit 44.1kHz PCM `.wav` files** load synchronously in memory (`BufferedMedia`), allowing sample-accurate seek positioning down to the exact millisecond.

---

## 🛠️ 2. How to Convert Audio Files to 44.1kHz PCM WAV

### A. Converting a Single Audio File (FFmpeg)
If you have a new `.mp3`, `.flac`, or `.m4a` file, use FFmpeg in PowerShell/Command Prompt:

```powershell
# Convert MP3 to 16-bit 44.1kHz Stereo PCM WAV
ffmpeg -i input_track.mp3 -ar 44100 -ac 2 -c:a pcm_s16le output_track.wav
```

### B. Batch Converting a Folder of MP3 Files (Command Line)
To convert all `.mp3` files in a folder at once:

```cmd
:: Windows Command Prompt (cmd)
for %f in (*.mp3) do ffmpeg -i "%f" -ar 44100 -ac 2 -c:a pcm_s16le "%~nf.wav"
```

```powershell
# Windows PowerShell
Get-ChildItem -Filter *.mp3 | ForEach-Object {
    ffmpeg -i $_.FullName -ar 44100 -ac 2 -c:a pcm_s16le "$($_.DirectoryName)\$($_.BaseName).wav"
}
```

### C. Automated Converter Utility (`convert_music_tracks.py`)
The codebase includes an automated python converter script that automatically scans `music_tracks/` for `.mp3` files and converts them into pristine `.wav` files:

```powershell
cd "C:\Users\VR\Documents\GitHub\nautilus_bci\scripts"
uv run python convert_music_tracks.py
```

---

## 🎛️ 3. Calibrating Mid-Song Theme Offsets (`music_offset_calibrator.py`)

Many classical compositions and songs begin with quiet intros. To isolate iconic themes for auditory imagery BCI testing, use the **Companion Looping Calibrator Studio**:

```powershell
uv run python music_offset_calibrator.py
```

### Calibrator Features:
1. **Dynamic Track Length Caps**: The offset slider upper limit automatically scales to match each track's exact total length (up to 7+ minutes).
2. **Continuous Audio Looping**: Drag the slider while audio loops continuously to pinpoint the exact start millisecond of iconic melodies.
3. **Loop Duration Tuner**: Adjust loop snippet duration from $1.0\text{s}$ to $10.0\text{s}$.
4. **Automatic Instant Save**: Slider and input box changes automatically write to `music_offset_config.json` in real time.

---

## 📂 4. Subfolder Taxonomy & Auto-Discovery

Place `.wav` files under `music_tracks/` organized by subfolders:

```
music_tracks/
├── real/                           <-- Real Master Performances (Beethoven, Joplin, Bach, etc.)
│   ├── real_beethoven_fur_elise.wav
│   ├── real_joplin_entertainer.wav
│   └── ...
└── synthesis/                      <-- Synthesized Audio & BPM Beats
    ├── beats_only/                 <-- Pure Rhythmic Percussion
    ├── single_note_beats/          <-- A3 Tone Pulse + Beat
    └── melodic_arrangements/       <-- Synthesized Keyboard Melodies
```

Both `music_memory_task.py` and `music_offset_calibrator.py` automatically discover any subfolder added under `music_tracks/` and list its `.wav` contents dynamically without requiring hardcoded updates.
