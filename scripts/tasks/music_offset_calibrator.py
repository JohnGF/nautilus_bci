import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import sys
import os
import json
import time
import wave

from PySide6.QtCore import Qt, QTimer, QUrl, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QDoubleSpinBox, QGroupBox,
    QFormLayout, QMessageBox
)
from PySide6.QtGui import QFont, QColor, QPalette
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput


def get_audio_duration_seconds(file_path):
    """Calculate exact total audio duration in seconds."""
    if not os.path.exists(file_path):
        return 300.0
    try:
        if file_path.lower().endswith('.wav'):
            with wave.open(file_path, 'rb') as f:
                return f.getnframes() / float(f.getframerate())
    except Exception:
        pass
    return 300.0


from tasks.common.base_task import BaseTaskApp

class MusicOffsetCalibratorApp(BaseTaskApp):
    """Standalone Companion App for Looping Audio & Start Offset Calibration."""
    def __init__(self):
        super().__init__(marker_name='MusicCalibrationMarkers', source_id='Music_Calib_Markers_2026')
        self.setWindowTitle("🎵 Music Offset & Tempo Calibrator (Looping Companion Studio)")
        self.resize(750, 600)

        self.music_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "music_tracks"))
        self.config_file = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "music_offset_config.json"))

        self.offsets_db = self.load_saved_offsets()

        # Audio Player Layer
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)

        # Loop Timer Engine
        self.loop_timer = QTimer()
        self.loop_timer.timeout.connect(self.check_loop_boundary)

        self.is_playing = False
        self.current_file = None
        self.start_offset_sec = 0.0
        self.loop_dur_sec = 3.0
        self.total_dur_sec = 300.0

        self.init_ui()
        self.on_category_changed(0)

    def load_saved_offsets(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "real_beethoven_fur_elise.wav": 5000,
            "real_joplin_entertainer.wav": 10000,
            "real_bach_prelude.wav": 8000,
            "real_mozart_eine_kleine.wav": 2000,
            "real_vivaldi_spring.wav": 12000,
            "real_tchaikovsky_waltz.wav": 45000,
            "beethoven_fur_elise.wav": 5000,
            "bach_prelude_c_major.wav": 8000,
            "mozart_eine_kleine_nachtmusik.wav": 2000,
            "vivaldi_four_seasons_spring.wav": 12000,
            "tchaikovsky_waltz_of_the_flowers.wav": 45000,
            "scott_joplin_the_entertainer.wav": 10000
        }

    def save_offsets_to_json(self, show_msg=True):
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.offsets_db, f, indent=2)
            print(f"[AUTO-SAVE] Updated offset configuration in {self.config_file}")
            if show_msg:
                QMessageBox.information(self, "Calibration Saved", f"Successfully saved offset configuration to:\n{self.config_file}")
        except Exception as e:
            if show_msg:
                QMessageBox.critical(self, "Error Saving Config", f"Failed to save configuration:\n{e}")

    def init_ui(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(15, 18, 25))
        palette.setColor(QPalette.WindowText, QColor(240, 240, 245))
        palette.setColor(QPalette.Base, QColor(25, 30, 42))
        palette.setColor(QPalette.Text, QColor(240, 240, 245))
        palette.setColor(QPalette.Button, QColor(35, 45, 65))
        palette.setColor(QPalette.ButtonText, QColor(240, 240, 245))
        self.setPalette(palette)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(15)

        title = QLabel("🎛️ Music Start Offset & Tempo Calibrator")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #4DEEEA;")
        layout.addWidget(title)

        subtitle = QLabel("Continuous Looping Audio Companion Tool for BCI Cue Calibration")
        subtitle.setFont(QFont("Arial", 11))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #A0A5B5;")
        layout.addWidget(subtitle)

        # Controls Group
        grp_setup = QGroupBox("Track Selection")
        grp_setup.setStyleSheet("QGroupBox { font-size: 13px; font-weight: bold; color: #FFEAA7; border: 1px solid #2C3545; border-radius: 8px; padding: 12px; }")
        form = QFormLayout(grp_setup)
        form.setSpacing(10)

        self.category_combo = QComboBox()
        self.category_combo.setStyleSheet("background: #191E2A; color: white; padding: 6px; border-radius: 4px;")

        # Dynamic Subfolder Discovery
        from tasks.music_memory_task import discover_music_subfolders
        self.discovered_folders = discover_music_subfolders(self.music_dir)
        for sub_rel, info in self.discovered_folders.items():
            disp = f"📁 {info['display']} ({len(info['files'])} tracks)"
            self.category_combo.addItem(disp, sub_rel)

        self.category_combo.currentIndexChanged.connect(self.on_category_changed)
        form.addRow("Discovered Music Subfolder:", self.category_combo)

        self.track_combo = QComboBox()
        self.track_combo.setStyleSheet("background: #191E2A; color: white; padding: 6px; border-radius: 4px;")
        self.track_combo.currentIndexChanged.connect(self.on_track_changed)
        form.addRow("Select Track:", self.track_combo)

        layout.addWidget(grp_setup)

        # Looping Studio Box
        grp_studio = QGroupBox("🎧 Looping Studio Controls")
        grp_studio.setStyleSheet("QGroupBox { font-size: 13px; font-weight: bold; color: #00E676; border: 1px solid #2C3545; border-radius: 8px; padding: 15px; }")
        s_layout = QVBoxLayout(grp_studio)
        s_layout.setSpacing(15)

        # Offset Slider Row
        row_offset = QHBoxLayout()
        row_offset.addWidget(QLabel("Start Offset:"))

        self.slider_offset = QSlider(Qt.Horizontal)
        self.slider_offset.setRange(0, 3000)
        self.slider_offset.setValue(0)
        self.slider_offset.valueChanged.connect(self.on_slider_offset_changed)
        row_offset.addWidget(self.slider_offset)

        self.spin_offset = QDoubleSpinBox()
        self.spin_offset.setRange(0.0, 300.0)
        self.spin_offset.setSingleStep(0.5)
        self.spin_offset.setSuffix(" s")
        self.spin_offset.setFixedWidth(80)
        self.spin_offset.setStyleSheet("background: #191E2A; color: #4DEEEA; font-weight: bold; padding: 4px;")
        self.spin_offset.valueChanged.connect(self.on_spin_offset_changed)
        row_offset.addWidget(self.spin_offset)

        self.lbl_dur_badge = QLabel("Track Cap: 300.0s")
        self.lbl_dur_badge.setStyleSheet("color: #FFEAA7; font-size: 11px; font-weight: bold; background: #191E2A; padding: 4px 8px; border-radius: 4px;")
        row_offset.addWidget(self.lbl_dur_badge)

        s_layout.addLayout(row_offset)

        # Loop Length Row
        row_dur = QHBoxLayout()
        row_dur.addWidget(QLabel("Loop Length:"))

        self.spin_dur = QDoubleSpinBox()
        self.spin_dur.setRange(1.0, 10.0)
        self.spin_dur.setValue(3.0)
        self.spin_dur.setSingleStep(0.5)
        self.spin_dur.setSuffix(" s (Cue Length)")
        self.spin_dur.setFixedWidth(130)
        self.spin_dur.setStyleSheet("background: #191E2A; color: #FFEAA7; font-weight: bold; padding: 4px;")
        self.spin_dur.valueChanged.connect(self.on_dur_changed)
        row_dur.addWidget(self.spin_dur)
        row_dur.addStretch()

        s_layout.addLayout(row_dur)

        # Transport Buttons
        row_transport = QHBoxLayout()
        self.btn_play_loop = QPushButton("🔄 PLAY CONTINUOUS LOOP")
        self.btn_play_loop.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_play_loop.setStyleSheet("background-color: #00E676; color: #0F1219; padding: 12px 24px; border-radius: 6px;")
        self.btn_play_loop.clicked.connect(self.toggle_loop_play)
        row_transport.addWidget(self.btn_play_loop)

        self.btn_stop = QPushButton("⏹️ STOP")
        self.btn_stop.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_stop.setStyleSheet("background-color: #E17055; color: white; padding: 12px 20px; border-radius: 6px;")
        self.btn_stop.clicked.connect(self.stop_playback)
        row_transport.addWidget(self.btn_stop)

        s_layout.addLayout(row_transport)
        layout.addWidget(grp_studio)

        # Save Button
        row_save = QHBoxLayout()
        self.btn_save = QPushButton("💾 SAVE ALL CALIBRATED OFFSETS")
        self.btn_save.setFont(QFont("Arial", 13, QFont.Bold))
        self.btn_save.setStyleSheet("background-color: #6C5CE7; color: white; padding: 14px 28px; border-radius: 6px;")
        self.btn_save.clicked.connect(self.save_offsets_to_json)
        row_save.addWidget(self.btn_save)

        layout.addLayout(row_save)

    def on_category_changed(self, idx):
        sub_rel = self.category_combo.currentData()
        if not sub_rel or sub_rel not in self.discovered_folders:
            sub_rel = list(self.discovered_folders.keys())[0] if self.discovered_folders else 'real'

        info = self.discovered_folders.get(sub_rel, {})
        dir_path = info.get('path', os.path.join(self.music_dir, sub_rel))
        files = info.get('files', [])

        self.track_files = [(f, os.path.join(dir_path, f)) for f in files]

        self.track_combo.blockSignals(True)
        self.track_combo.clear()
        for fname, fullpath in self.track_files:
            clean_name = fname.replace('.wav', '').replace('.mp3', '').replace('real_', '').replace('synthetic_', '').replace('synth_', '').replace('_', ' ').title()
            self.track_combo.addItem(clean_name, fullpath)
        self.track_combo.blockSignals(False)

        if self.track_files:
            self.on_track_changed(0)

    def on_track_changed(self, idx):
        file_path = self.track_combo.currentData()
        if not file_path or not os.path.exists(file_path):
            return

        self.current_file = file_path
        filename = os.path.basename(file_path)
        offset_ms = self.offsets_db.get(filename, 0)

        # Dynamic offset cap set to exact total audio duration
        self.total_dur_sec = get_audio_duration_seconds(file_path)

        self.spin_offset.blockSignals(True)
        self.slider_offset.blockSignals(True)

        # Set range dynamically to track length (in 0.1s steps for slider)
        self.slider_offset.setRange(0, int(self.total_dur_sec * 10))
        self.spin_offset.setRange(0.0, round(self.total_dur_sec, 1))

        self.start_offset_sec = min(offset_ms / 1000.0, self.total_dur_sec)
        self.spin_offset.setValue(self.start_offset_sec)
        self.slider_offset.setValue(int(self.start_offset_sec * 10))

        self.spin_offset.blockSignals(False)
        self.slider_offset.blockSignals(False)

        if hasattr(self, 'lbl_dur_badge'):
            self.lbl_dur_badge.setText(f"Track Cap: {self.total_dur_sec:.1f}s ({self.total_dur_sec/60.0:.2f} m)")

        if self.is_playing:
            self.restart_loop()

    def on_slider_offset_changed(self, val):
        self.start_offset_sec = val / 10.0
        self.spin_offset.blockSignals(True)
        self.spin_offset.setValue(self.start_offset_sec)
        self.spin_offset.blockSignals(False)
        self.update_current_offset_db()
        if self.is_playing:
            self.player.setPosition(int(self.start_offset_sec * 1000))

    def on_spin_offset_changed(self, val):
        self.start_offset_sec = val
        self.slider_offset.blockSignals(True)
        self.slider_offset.setValue(int(val * 10))
        self.slider_offset.blockSignals(False)
        self.update_current_offset_db()
        if self.is_playing:
            self.player.setPosition(int(self.start_offset_sec * 1000))

    def on_dur_changed(self, val):
        self.loop_dur_sec = val

    def update_current_offset_db(self):
        if self.current_file:
            filename = os.path.basename(self.current_file)
            self.offsets_db[filename] = int(self.start_offset_sec * 1000)
            self.save_offsets_to_json(show_msg=False)

    def toggle_loop_play(self):
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_loop_play()

    def start_loop_play(self):
        if not self.current_file or not os.path.exists(self.current_file):
            return

        self.is_playing = True
        self.btn_play_loop.setText("⏸️ PAUSE LOOP")
        self.btn_play_loop.setStyleSheet("background-color: #FFEAA7; color: #0F1219; padding: 12px 24px; border-radius: 6px;")

        self.restart_loop()
        self.loop_timer.start(50)

    def restart_loop(self):
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(self.current_file))
        self.player.setPosition(int(self.start_offset_sec * 1000))
        self.player.play()

    def check_loop_boundary(self):
        if not self.is_playing:
            return

        target_end_ms = int((self.start_offset_sec + self.loop_dur_sec) * 1000)
        if self.player.position() >= target_end_ms or self.player.mediaStatus() == QMediaPlayer.EndOfMedia:
            self.player.setPosition(int(self.start_offset_sec * 1000))
            if self.player.playbackState() != QMediaPlayer.PlayingState:
                self.player.play()

    def stop_playback(self):
        self.is_playing = False
        self.loop_timer.stop()
        self.player.stop()
        self.btn_play_loop.setText("🔄 PLAY CONTINUOUS LOOP")
        self.btn_play_loop.setStyleSheet("background-color: #00E676; color: #0F1219; padding: 12px 24px; border-radius: 6px;")

    def closeEvent(self, event):
        self.stop_playback()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MusicOffsetCalibratorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
