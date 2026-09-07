import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import sys
import os
import random
import time
import wave
import json
import numpy as np

from PySide6.QtCore import Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QFormLayout, QGroupBox,
    QProgressBar, QStackedWidget, QMessageBox, QDoubleSpinBox
)
from PySide6.QtGui import QFont, QColor, QPalette
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from utils.bids_utils import get_formatted_next_session



def discover_music_subfolders(music_dir):
    """Dynamically scan music_dir for all subfolders containing uncompressed WAV audio files (0 playback lag)."""
    folders = {}
    if not os.path.exists(music_dir):
        return folders

    for root, dirs, files in os.walk(music_dir):
        # Exclusively select WAV files for zero decoding latency and instant sample-accurate seeking
        wav_files = [f for f in files if f.lower().endswith('.wav')]
        if wav_files:
            rel = os.path.relpath(root, music_dir)
            clean_rel = rel.replace('\\', '/')
            if clean_rel == '.':
                display = "Root Directory"
            else:
                display = clean_rel
            folders[clean_rel] = {
                "display": display,
                "path": root,
                "files": sorted(wav_files)
            }
    return folders


def generate_music_tracks(sound_dir, selected_subfolder='real'):
    """Dynamically load audio tracks from any discovered subfolder."""
    music_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "music_tracks"))
    config_file = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "music_offset_config.json"))

    saved_offsets = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                saved_offsets = json.load(f)
        except Exception:
            pass

    folders = discover_music_subfolders(music_dir)
    palette = ["#00E676", "#74B9FF", "#E040FB", "#FF7675", "#FFEAA7", "#FF9F43", "#00ADB5", "#6C5CE7"]

    # Target folder or all tracks
    target_files = []
    if selected_subfolder == 'all_mixed':
        for folder_rel, info in folders.items():
            for fn in info["files"]:
                target_files.append((folder_rel, fn, os.path.join(info["path"], fn)))
    elif selected_subfolder in folders:
        info = folders[selected_subfolder]
        for fn in info["files"]:
            target_files.append((selected_subfolder, fn, os.path.join(info["path"], fn)))
    else:
        # Default fallback to real folder or first available
        fallback_key = 'real' if 'real' in folders else list(folders.keys())[0] if folders else '.'
        if fallback_key in folders:
            info = folders[fallback_key]
            for fn in info["files"]:
                target_files.append((fallback_key, fn, os.path.join(info["path"], fn)))

    # Default offsets map
    default_offsets = {
        "real_beethoven_fur_elise": 5000,
        "real_joplin_entertainer": 10000,
        "real_bach_prelude": 8000,
        "real_mozart_eine_kleine": 2000,
        "real_vivaldi_spring": 12000,
        "real_tchaikovsky_waltz": 45000
    }

    out_dict = {}
    for idx, (sub_rel, fn, fullpath) in enumerate(target_files):
        # Auto-detect Audio Type tag
        if 'beats_only' in sub_rel:
            atype = "Rhythmic Beat Only"
        elif 'single_note' in sub_rel:
            atype = "Single Note Pitch & Beat"
        elif 'melodic' in sub_rel:
            atype = "Melodic Tone Synthesis"
        elif 'real' in sub_rel or 'real_' in fn:
            atype = "Real Master Performance"
        else:
            atype = "Acoustic Audio"

        clean_label = fn.replace('.wav', '').replace('.mp3', '').replace('real_', '').replace('synthetic_', '').replace('synth_', '').replace('_', ' ').title()

        if fn in saved_offsets:
            offset_ms = saved_offsets[fn]
        else:
            def_offset = 0
            for k_def, v_off in default_offsets.items():
                if k_def in fn:
                    def_offset = v_off
                    break
            offset_ms = def_offset
        color = palette[idx % len(palette)]

        out_dict[f"Track_{idx+1}"] = {
            "name": f"Track {idx+1}: {clean_label}",
            "audio_type": atype,
            "subfolder": sub_rel,
            "file": fullpath,
            "filename": fn,
            "color": color,
            "id": idx + 1,
            "offset_ms": offset_ms
        }
    return out_dict


from tasks.common.base_task import BaseTaskApp

class MusicMemoryTaskApp(BaseTaskApp):
    """BCI Paradigm App for 6-Track Auditory Memory Recall & Musical Imagery."""
    def __init__(self):
        super().__init__(marker_name='MotorImageryMarkers', source_id='Music_Memory_Markers_2026')
        self.setWindowTitle("BCI 6-Track Music Memory Recall Paradigm")
        self.resize(1080, 800)

        self.sound_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "music_tracks"))
        self.track_catalog = generate_music_tracks(self.sound_dir)
        self.bids_root = "bids_dataset_multimodal"

        # LSL Marker Outlet

        # Audio Player Layer
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.90)

        self.pending_seek_offset = 0
        self.player.mediaStatusChanged.connect(self.on_media_status_changed)

        # Timings
        self.t_sample_cue = 3.0   # seconds listening to song cue sample
        self.t_recall_task = 5.0  # seconds mentally recalling song in head (silent EEG epoch)
        self.t_rest = 2.5        # seconds rest

        self.reps_per_track = 5
        self.trials = []
        self.trial_idx = 0

        self.trial_timer = QTimer()
        self.trial_timer.setSingleShot(True)
        self.trial_timer.timeout.connect(self.advance_trial_phase)

        self.tick_timer = QTimer()
        self.tick_timer.timeout.connect(self.update_progress)
        self.phase_start_time = 0.0
        self.current_phase_duration = 1.0

        self.init_ui()

    def init_ui(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(15, 18, 25))
        palette.setColor(QPalette.WindowText, QColor(240, 240, 245))
        palette.setColor(QPalette.Base, QColor(25, 30, 42))
        palette.setColor(QPalette.Text, QColor(240, 240, 245))
        palette.setColor(QPalette.Button, QColor(35, 45, 65))
        palette.setColor(QPalette.ButtonText, QColor(240, 240, 245))
        self.setPalette(palette)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.config_screen = self.create_config_screen()
        self.stacked_widget.addWidget(self.config_screen)

        self.task_screen = self.create_task_screen()
        self.stacked_widget.addWidget(self.task_screen)

        self.stacked_widget.setCurrentWidget(self.config_screen)

    def create_config_screen(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(15)

        title = QLabel("🎵 6-Track Music Memory Recall BCI Paradigm")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #4DEEEA;")
        layout.addWidget(title)

        subtitle = QLabel("Auditory Imagery & Neural Beat Entrainment Memory Suite")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #A0A5B5;")
        layout.addWidget(subtitle)

        # Config Box
        config_group = QGroupBox("Experiment Setup")
        config_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; color: #4DEEEA; border: 1px solid #2C3545; border-radius: 8px; margin-top: 5px; padding-top: 15px; }")
        form = QFormLayout(config_group)
        form.setSpacing(12)

        self.sub_input = QLineEdit("sub-01")
        self.sub_input.setStyleSheet("background: #191E2A; color: white; padding: 6px; border-radius: 4px;")
        form.addRow("Subject ID:", self.sub_input)

        self.ses_input = QLineEdit("ses-01")
        self.ses_input.setStyleSheet("background: #191E2A; color: white; padding: 6px; border-radius: 4px;")
        form.addRow("Session ID:", self.ses_input)

        self.sub_input.textChanged.connect(self.auto_update_session)
        self.auto_update_session()
        
        self.category_combo = QComboBox()
        self.category_combo.setStyleSheet("background: #191E2A; color: white; padding: 6px; border-radius: 4px;")
        
        self.discovered_folders = discover_music_subfolders(self.sound_dir)
        self.category_combo.addItem("🌐 All Discovered Tracks Pooled", "all_mixed")
        for sub_rel, info in self.discovered_folders.items():
            disp = f"📁 {info['display']} ({len(info['files'])} tracks)"
            self.category_combo.addItem(disp, sub_rel)

        self.category_combo.currentIndexChanged.connect(self.on_category_changed)
        form.addRow("Discovered Music Subfolder:", self.category_combo)

        self.cue_window_combo = QComboBox()
        self.cue_window_combo.addItems([
            "3.0 seconds",
            "4.0 seconds",
            "5.0 seconds (Recommended)",
            "6.0 seconds",
            "8.0 seconds"
        ])
        self.cue_window_combo.setCurrentIndex(2) # Default set to 5.0s
        self.cue_window_combo.setStyleSheet("background: #191E2A; color: white; padding: 6px; border-radius: 4px;")
        form.addRow("Audio Cue Listening Window:", self.cue_window_combo)

        self.reps_combo = QComboBox()
        self.reps_combo.addItems([
            "20 per track (120 trials total)",
            "5 per track (30 trials total - Recommended)",
            "8 per track (48 trials total)"
        ])
        self.reps_combo.setCurrentIndex(1)
        self.reps_combo.setStyleSheet("background: #191E2A; color: white; padding: 6px; border-radius: 4px;")
        form.addRow("Repetitions per Track:", self.reps_combo)

        layout.addWidget(config_group)

        # Track Catalog Box
        self.cat_group = QGroupBox("🎧 Music Catalog & Audio Preview")
        self.cat_group.setStyleSheet("QGroupBox { font-size: 13px; font-weight: bold; color: #FFEAA7; border: 1px solid #2C3545; border-radius: 8px; padding: 10px; }")
        self.cat_layout = QVBoxLayout(self.cat_group)
        self.populate_track_list()

        layout.addWidget(self.cat_group)

        # Companion Calibrator Button
        btn_calib = QPushButton("🎛️ Open Companion Offset Calibrator Tool (Looping Studio)")
        btn_calib.setStyleSheet("background-color: #6C5CE7; color: white; padding: 10px 16px; border-radius: 5px; font-weight: bold;")
        btn_calib.clicked.connect(self.launch_calibrator_tool)
        layout.addWidget(btn_calib)

        # Start Button
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("🚀 Start 6-Track Music Memory Recall Session")
        self.btn_start.setFont(QFont("Arial", 13, QFont.Bold))
        self.btn_start.setStyleSheet("background-color: #00ADB5; color: white; padding: 14px 28px; border-radius: 6px;")
        self.btn_start.clicked.connect(self.start_experiment)
        btn_layout.addWidget(self.btn_start)

        layout.addLayout(btn_layout)
        return widget

    def auto_update_session(self):
        sub = self.sub_input.text().strip()
        curr_ses = self.ses_input.text().strip()
        # Look in bids_dataset, bids_music, or bids_dataset_multimodal
        bids_root = "bids_dataset"
        if not os.path.exists(bids_root) and os.path.exists("bids_music"):
            bids_root = "bids_music"
        next_ses = get_formatted_next_session(bids_root, sub, curr_ses)
        self.ses_input.blockSignals(True)
        self.ses_input.setText(next_ses)
        self.ses_input.blockSignals(False)

    def populate_track_list(self):
        # Clear existing
        while self.cat_layout.count():
            item = self.cat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for k, info in self.track_catalog.items():
            row = QHBoxLayout()
            atype = info.get('audio_type', 'Audio')
            lbl = QLabel(f"• <b>{info['name']}</b> <span style='color: #4DEEEA; font-size: 11px;'>[{atype}]</span> <span style='color: #A0A5B5; font-size: 11px;'>(Offset: {info.get('offset_ms', 0)/1000.0:.1f}s)</span>")
            lbl.setStyleSheet(f"color: {info['color']}; font-size: 13px;")
            row.addWidget(lbl)
            self.cat_layout.addLayout(row)

    def launch_calibrator_tool(self):
        import subprocess
        calib_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "music_offset_calibrator.py"))
        if os.path.exists(calib_script):
            subprocess.Popen([sys.executable, calib_script])

    def on_category_changed(self, idx):
        selected_subfolder = self.category_combo.currentData()
        if not selected_subfolder:
            selected_subfolder = 'real'
        self.track_catalog = generate_music_tracks(self.sound_dir, selected_subfolder=selected_subfolder)
        self.populate_track_list()

    def create_task_screen(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(30, 30, 30, 30)

        header = QHBoxLayout()
        self.lbl_status = QLabel("Paradigm: 6-Track Music Memory Recall")
        self.lbl_status.setFont(QFont("Arial", 14, QFont.Bold))
        self.lbl_status.setStyleSheet("color: #74B9FF;")
        header.addWidget(self.lbl_status)
        header.addStretch()

        self.lbl_trial_count = QLabel("Trial 0 / 30")
        self.lbl_trial_count.setFont(QFont("Arial", 14, QFont.Bold))
        self.lbl_trial_count.setStyleSheet("color: #A0A5B5;")
        header.addWidget(self.lbl_trial_count)
        layout.addLayout(header)

        layout.addStretch()

        # Center Visual Symbol & Instruction
        self.lbl_cue_symbol = QLabel("🎧")
        self.lbl_cue_symbol.setFont(QFont("Arial", 90, QFont.Bold))
        self.lbl_cue_symbol.setAlignment(Qt.AlignCenter)
        self.lbl_cue_symbol.setStyleSheet("color: #FFEAA7;")
        layout.addWidget(self.lbl_cue_symbol)

        self.lbl_instruction = QLabel("LISTEN TO THE MUSIC CUE SAMPLE")
        self.lbl_instruction.setFont(QFont("Arial", 22, QFont.Bold))
        self.lbl_instruction.setAlignment(Qt.AlignCenter)
        self.lbl_instruction.setStyleSheet("color: #DFE6E9; margin-top: 20px;")
        layout.addWidget(self.lbl_instruction)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: none; background-color: #191E2A; border-radius: 6px; }
            QProgressBar::chunk { background-color: #00E676; border-radius: 6px; }
        """)
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        return widget

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.stop_experiment()

    def start_experiment(self):
        # Reload latest offsets from music_offset_config.json preserving selected category
        selected_sub = self.category_combo.currentData()
        if not selected_sub:
            selected_sub = 'real'
        self.track_catalog = generate_music_tracks(self.sound_dir, selected_subfolder=selected_sub)
        self.populate_track_list()

        cue_dur_map = {0: 3.0, 1: 4.0, 2: 5.0, 3: 6.0, 4: 8.0}
        self.t_sample_cue = cue_dur_map.get(self.cue_window_combo.currentIndex(), 5.0)

        reps_map = {0: 20, 1: 5, 2: 8,}
        self.reps_per_track = reps_map[self.reps_combo.currentIndex()]

        # Generate trial list
        self.trials = []
        track_keys = list(self.track_catalog.keys())
        for k in track_keys:
            self.trials.extend([k] * self.reps_per_track)
        random.shuffle(self.trials)

        self.trial_idx = 0
        sub = self.sub_input.text()
        ses = self.ses_input.text()
        self.send_marker(f"Experiment_Start_Sub_{sub}_Ses_{ses}_Paradigm_MusicMemoryRecall_CueDur_{self.t_sample_cue:.1f}s")

        self.save_tracks_bids_metadata(sub, ses)

        self.stacked_widget.setCurrentWidget(self.task_screen)
        self.run_next_trial()

    @Slot(QMediaPlayer.MediaStatus)
    def on_media_status_changed(self, status):
        if self.pending_seek_offset > 0 and status in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
            self.player.setPosition(self.pending_seek_offset)
            print(f"[AUDIO CUE] Applied calibrated seek offset: {self.pending_seek_offset} ms ({self.pending_seek_offset/1000.0:.2f}s)")
            self.pending_seek_offset = 0

    def play_track_cue(self, file_path, offset_ms):
        if not os.path.exists(file_path):
            return
        self.player.stop()
        self.pending_seek_offset = offset_ms
        self.player.setSource(QUrl.fromLocalFile(file_path))
        self.player.play()
        if self.player.mediaStatus() in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
            self.player.setPosition(offset_ms)

    def run_next_trial(self):
        if self.trial_idx >= len(self.trials):
            self.finish_experiment()
            return

        self.current_track_key = self.trials[self.trial_idx]
        self.track_info = self.track_catalog[self.current_track_key]

        self.lbl_trial_count.setText(f"Trial {self.trial_idx + 1} / {len(self.trials)}")
        self.send_marker(f"Trial_Start_{self.trial_idx + 1}_Track_{self.current_track_key}")

        # Phase 1: Cue Audio Sample (Listening Phase)
        self.send_marker(f"Cue_Audio_Sample_{self.current_track_key}", self.t_sample_cue)
        
        # Play song cue snippet starting at exact calibrated theme offset
        if os.path.exists(self.track_info['file']):
            offset_ms = self.track_info.get('offset_ms', 0)
            self.play_track_cue(self.track_info['file'], offset_ms)

        self.set_phase("CueSample", "🔊", f"LISTEN CUE: {self.track_info['name'].upper()}", self.track_info['color'], self.t_sample_cue)

    def advance_trial_phase(self):
        if self.current_phase_name == "CueSample":
            # Stop music sample
            self.player.stop()
            
            # Phase 2: Auditory Memory Recall / Imagery
            self.send_marker(f"Task_Recall_{self.current_track_key}", self.t_recall_task)
            
            self.set_phase(
                "RecallTask",
                "🧠",
                f"MENTALLY RECALL & PLAY BACK\n'{self.track_info['name'].upper()}' IN YOUR HEAD",
                "#00E676",
                self.t_recall_task
            )

        elif self.current_phase_name == "RecallTask":
            # Phase 3: Rest
            self.send_marker("Rest", self.t_rest)
            self.set_phase("Rest", "•", "RELAX & REST", "#A0A5B5", self.t_rest)

        elif self.current_phase_name == "Rest":
            self.send_marker(f"Trial_End_{self.trial_idx + 1}")
            self.trial_idx += 1
            self.run_next_trial()

    def set_phase(self, phase_name, symbol, instruction, color, duration):
        self.current_phase_name = phase_name
        self.lbl_cue_symbol.setText(symbol)
        self.lbl_cue_symbol.setStyleSheet(f"color: {color};")
        self.lbl_instruction.setText(instruction)
        self.lbl_instruction.setStyleSheet(f"color: {color}; margin-top: 20px;")

        self.phase_start_time = time.time()
        self.current_phase_duration = duration

        self.progress_bar.setValue(0)
        self.tick_timer.start(30)
        self.trial_timer.start(int(duration * 1000))

    def update_progress(self):
        elapsed = time.time() - self.phase_start_time
        progress = min(1.0, elapsed / self.current_phase_duration)
        self.progress_bar.setValue(int(progress * 100))

    def finish_experiment(self):
        self.tick_timer.stop()
        self.trial_timer.stop()
        self.player.stop()

        self.send_marker("Experiment_End")

        self.lbl_cue_symbol.setText("🎉")
        self.lbl_cue_symbol.setStyleSheet("color: #00E676;")
        self.lbl_instruction.setText("6-TRACK MUSIC MEMORY SESSION COMPLETE!\nThank you for participating.")
        
        QTimer.singleShot(3000, self.reset_to_config)

    def save_tracks_bids_metadata(self, sub, ses):
        try:
            sub_clean = sub.replace("sub-", "")
            ses_clean = ses.replace("ses-", "")
            bids_dir = os.path.join(self.bids_root, f"sub-{sub_clean}", f"ses-{ses_clean}")
            os.makedirs(bids_dir, exist_ok=True)

            mapping = {}
            for k, info in self.track_catalog.items():
                mapping[k] = {
                    "name": info["name"],
                    "audio_type": info["audio_type"],
                    "filename": info["filename"]
                }

            out_file = os.path.join(bids_dir, f"sub-{sub_clean}_ses-{ses_clean}_tracks.json")
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, indent=4)
            print(f"[+] Saved track catalog metadata mapping to: {out_file}")
        except Exception as e:
            print(f"[-] Failed to save track BIDS metadata: {e}")

    def stop_experiment(self):
        self.tick_timer.stop()
        self.trial_timer.stop()
        self.player.stop()
        self.reset_to_config()

    def reset_to_config(self):
        self.stacked_widget.setCurrentWidget(self.config_screen)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Music Memory Task")
    parser.add_argument("--sub", type=str, default="01", help="Subject ID")
    parser.add_argument("--ses", type=str, default="01", help="Session ID")
    parser.add_argument("--bids-root", "--dataset-folder", type=str, default="bids_dataset_multimodal", help="Target BIDS dataset output directory")
    args, unknown = parser.parse_known_args()

    app = QApplication(sys.argv)
    window = MusicMemoryTaskApp()

    if args.bids_root:
        window.bids_root = args.bids_root
    if args.sub:
        window.sub_input.setText(args.sub.replace('sub-', ''))
    if args.ses:
        window.ses_input.setText(args.ses.replace('ses-', ''))

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
