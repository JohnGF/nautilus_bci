"""
Full-Length Music Listening & Neural Entrainment BCI Task Paradigm
===================================================================
Plays complete music tracks from start to finish for continuous brainwave (EEG) 
and multimodal signal recording (IMU, PPG, fNIRS).

Features:
  - Complete, un-cut playback of selected audio tracks (WAV / MP3).
  - Dynamic folder discovery & custom tracklist selector with duration calculation.
  - Configurable Initial Baseline Rest, Inter-Track Rest, and Post-Session Rest.
  - High-precision LSL marker synchronization (Track Start, 30s Section checkpoints, Track End, Rest).
  - Clean EEG-friendly visual modes: Minimalist Fixation Cross, Rhythmic Visualizer, or Modern Timecode.
  - Integrated BIDS Multimodal Recorder control widget for 1-click recording & BIDS export.
"""

import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

import json
import math
import random
import time
import wave

from PySide6.QtCore import Qt, QTimer, QUrl, Slot, QPointF
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QFormLayout, QGroupBox,
    QProgressBar, QStackedWidget, QMessageBox, QDoubleSpinBox, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QSlider,
    QFileDialog, QRadioButton, QButtonGroup, QFrame
)
from PySide6.QtGui import QFont, QColor, QPalette, QPainter, QBrush, QPen, QRadialGradient
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from tasks.common.base_task import BaseTaskApp
from recorders.bids_recorder_widget import BIDSRecorderControlWidget
from utils.bids_utils import get_formatted_next_session


def get_wav_duration_seconds(file_path):
    """Accurately calculates duration of a WAV file in seconds using the wave module."""
    try:
        with wave.open(file_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return float(frames) / float(rate)
    except Exception:
        pass
    return 0.0


def format_duration(seconds):
    """Format seconds into MM:SS or HH:MM:SS."""
    sec = int(round(seconds))
    mins = sec // 60
    secs = sec % 60
    if mins >= 60:
        hrs = mins // 60
        mins = mins % 60
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def discover_music_folders(base_music_dir):
    """Scan base_music_dir for subdirectories containing audio tracks."""
    folders = {}
    if not os.path.exists(base_music_dir):
        return folders

    for root, dirs, files in os.walk(base_music_dir):
        audio_files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
        if audio_files:
            rel = os.path.relpath(root, base_music_dir)
            clean_rel = rel.replace('\\', '/')
            display = "Root Directory" if clean_rel == '.' else clean_rel
            folders[clean_rel] = {
                "display": display,
                "path": root,
                "files": sorted(audio_files)
            }
    return folders


class RhythmicPulseVisualizer(QWidget):
    """Subtle smooth animated pulsing circle to accompany music listening without sharp saccade triggers."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.phase = 0.0
        self.is_playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate_step)
        self.timer.start(33)  # ~30 FPS

    def set_playing(self, playing):
        self.is_playing = playing
        self.update()

    def animate_step(self):
        if self.is_playing:
            self.phase += 0.05
            if self.phase > 2 * math.pi:
                self.phase -= 2 * math.pi
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        cx = w / 2.0
        cy = h / 2.0

        if not self.is_playing:
            # Subtle resting circle
            painter.setPen(QPen(QColor(100, 110, 130, 80), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), 60, 60)
            # Center Fixation Dot
            painter.setBrush(QBrush(QColor(77, 238, 234)))
            painter.setPen(Qt.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), 4, 4)
            return

        # Pulsing rhythmic ring
        pulse_scale = 1.0 + 0.18 * math.sin(self.phase)
        base_radius = min(w, h) * 0.22
        r = base_radius * pulse_scale

        gradient = QRadialGradient(cx, cy, r * 1.3)
        gradient.setColorAt(0.0, QColor(0, 230, 118, 40))
        gradient.setColorAt(0.7, QColor(77, 238, 234, 90))
        gradient.setColorAt(1.0, QColor(116, 185, 255, 0))

        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), r * 1.3, r * 1.3)

        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(77, 238, 234, 180), 3))
        painter.drawEllipse(QPointF(cx, cy), r, r)

        # Secondary outer wave ring
        r2 = base_radius * (1.0 + 0.12 * math.cos(self.phase * 1.5)) * 1.35
        painter.setPen(QPen(QColor(116, 185, 255, 100), 1.5))
        painter.drawEllipse(QPointF(cx, cy), r2, r2)

        # Fixation Point at dead center
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(Qt.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), 5, 5)


class MusicFullTrackTaskApp(BaseTaskApp):
    """
    BCI Full-Length Music Listening Paradigm Application.
    Plays complete music tracks sequentially with synchronized LSL markers.
    """
    def __init__(self):
        super().__init__(marker_name='MusicListeningMarkers', source_id='Music_FullTrack_Markers_2026')
        self.setWindowTitle("BCI Full-Length Music Listening & Entrainment Paradigm")
        self.resize(1120, 840)

        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.music_root = os.path.abspath(os.path.join(base_dir, "music_tracks"))

        # Audio Player Layer
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.90)

        # Player signals
        self.player.positionChanged.connect(self.on_player_position_changed)
        self.player.durationChanged.connect(self.on_player_duration_changed)
        self.player.mediaStatusChanged.connect(self.on_player_media_status)

        # State Variables
        self.playlist = []  # List of dicts: {'name': str, 'path': str, 'dur': float, 'id': int, 'subfolder': str}
        self.current_track_idx = 0
        self.current_phase = "IDLE"  # "IDLE", "BASELINE", "TRACK", "REST", "POST_REST", "FINISHED"
        self.phase_start_time = 0.0
        self.phase_duration = 0.0
        self.is_paused = False
        self.last_section_marker_sec = 0

        # Timing Timers
        self.phase_timer = QTimer()
        self.phase_timer.setSingleShot(True)
        self.phase_timer.timeout.connect(self.on_phase_timer_timeout)

        self.tick_timer = QTimer()
        self.tick_timer.timeout.connect(self.update_tick_ui)

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
        layout.setContentsMargins(36, 24, 36, 24)
        layout.setSpacing(12)

        # Header
        title = QLabel("🎵 Full-Length Music Listening BCI Paradigm")
        title.setFont(QFont("Arial", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #4DEEEA;")
        layout.addWidget(title)

        subtitle = QLabel("Continuous Start-to-Finish Music Stream • Neural Entrainment • Multimodal LSL Synchronized")
        subtitle.setFont(QFont("Arial", 11))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #A0A5B5;")
        layout.addWidget(subtitle)

        # Integrated BIDS Recorder Control Widget
        self.recorder_widget = BIDSRecorderControlWidget(default_task="musiclistening", default_bids_root="bids_dataset_multimodal")
        layout.addWidget(self.recorder_widget)
        self.sub_input = self.recorder_widget.txt_sub
        self.ses_input = self.recorder_widget.txt_ses

        # Settings Box
        settings_group = QGroupBox("Session & Timing Parameters")
        settings_group.setFont(QFont("Arial", 11, QFont.Bold))
        settings_group.setStyleSheet("QGroupBox { color: #4DEEEA; border: 1px solid #2C354A; border-radius: 8px; margin-top: 4px; padding: 10px; }")
        form_layout = QFormLayout(settings_group)
        form_layout.setSpacing(8)

        # Folder selector
        folder_layout = QHBoxLayout()
        self.combo_folders = QComboBox()
        self.combo_folders.setStyleSheet("background: #191E2A; color: white; padding: 5px; border-radius: 4px;")
        self.discovered_folders = discover_music_folders(self.music_root)

        # Prioritize 'real' folder if present
        default_index = 0
        idx_count = 0
        self.combo_folders.addItem("🌐 All Folders Combined (Full Catalog)", "all")
        for sub_rel, info in self.discovered_folders.items():
            idx_count += 1
            disp = f"📁 {info['display']} ({len(info['files'])} tracks)"
            self.combo_folders.addItem(disp, sub_rel)
            if sub_rel == 'real':
                default_index = idx_count

        self.combo_folders.setCurrentIndex(default_index)
        self.combo_folders.currentIndexChanged.connect(self.populate_tracks_table)
        folder_layout.addWidget(self.combo_folders, stretch=3)

        btn_browse_custom = QPushButton("Custom Folder...")
        btn_browse_custom.setStyleSheet("background: #2C354A; color: white; padding: 5px 12px; border-radius: 4px;")
        btn_browse_custom.clicked.connect(self.browse_custom_music_dir)
        folder_layout.addWidget(btn_browse_custom, stretch=1)
        form_layout.addRow("Music Folder / Catalog:", folder_layout)

        # Baseline Rest Duration (before 1st track)
        self.spn_baseline_dur = QDoubleSpinBox()
        self.spn_baseline_dur.setRange(0.0, 300.0)
        self.spn_baseline_dur.setValue(10.0)
        self.spn_baseline_dur.setSingleStep(5.0)
        self.spn_baseline_dur.setSuffix(" sec (Fixation Cross baseline before music)")
        self.spn_baseline_dur.setStyleSheet("background: #191E2A; color: white; padding: 4px;")
        form_layout.addRow("Initial Baseline Rest:", self.spn_baseline_dur)

        # Inter-Track Rest Duration
        self.spn_rest_dur = QDoubleSpinBox()
        self.spn_rest_dur.setRange(0.0, 120.0)
        self.spn_rest_dur.setValue(5.0)
        self.spn_rest_dur.setSingleStep(1.0)
        self.spn_rest_dur.setSuffix(" sec (Rest interval between songs)")
        self.spn_rest_dur.setStyleSheet("background: #191E2A; color: white; padding: 4px;")
        form_layout.addRow("Inter-Track Rest:", self.spn_rest_dur)

        # Playback Order & Repetitions
        order_layout = QHBoxLayout()
        self.chk_shuffle = QCheckBox("🔀 Shuffle Track Order")
        self.chk_shuffle.setStyleSheet("color: #FFEAA7; font-weight: bold;")
        order_layout.addWidget(self.chk_shuffle)

        self.chk_section_markers = QCheckBox("⏱️ Send 30s Checkpoint Markers during playback")
        self.chk_section_markers.setChecked(True)
        self.chk_section_markers.setStyleSheet("color: #74B9FF; font-weight: bold;")
        self.chk_section_markers.setToolTip("Emits an LSL event marker every 30 seconds of playback for easier EEG epoch segmentation.")
        order_layout.addWidget(self.chk_section_markers)
        form_layout.addRow("Session Options:", order_layout)

        # Visual Display Mode
        vis_layout = QHBoxLayout()
        self.btn_vis_fixation = QRadioButton("➕ Minimalist Fixation Cross (Optimal for EEG)")
        self.btn_vis_pulse = QRadioButton("💫 Rhythmic Pulse Visualizer")
        self.btn_vis_info = QRadioButton("📊 Info & Big Digital Timecode")
        self.btn_vis_fixation.setChecked(True)

        self.vis_group = QButtonGroup()
        self.vis_group.addButton(self.btn_vis_fixation, 1)
        self.vis_group.addButton(self.btn_vis_pulse, 2)
        self.vis_group.addButton(self.btn_vis_info, 3)

        vis_layout.addWidget(self.btn_vis_fixation)
        vis_layout.addWidget(self.btn_vis_pulse)
        vis_layout.addWidget(self.btn_vis_info)
        form_layout.addRow("Display Mode:", vis_layout)

        layout.addWidget(settings_group)

        # Tracklist Selection Table Box
        table_group = QGroupBox("🎧 Playlist Tracks (Select tracks to include in this session)")
        table_group.setFont(QFont("Arial", 11, QFont.Bold))
        table_group.setStyleSheet("QGroupBox { color: #FFEAA7; border: 1px solid #2C354A; border-radius: 8px; margin-top: 4px; padding: 10px; }")
        table_layout = QVBoxLayout(table_group)

        self.track_table = QTableWidget()
        self.track_table.setColumnCount(4)
        self.track_table.setHorizontalHeaderLabels(["Include", "Track Name", "Folder / Source", "Duration"])
        self.track_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.track_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.track_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.track_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.track_table.setStyleSheet("""
            QTableWidget { background-color: #191E2A; color: white; gridline-color: #2C354A; border: none; }
            QHeaderView::section { background-color: #232D42; color: #4DEEEA; font-weight: bold; padding: 4px; border: 1px solid #191E2A; }
        """)
        table_layout.addWidget(self.track_table)

        # Table Control Buttons & Summary
        tbl_bottom = QHBoxLayout()
        btn_sel_all = QPushButton("Select All")
        btn_sel_all.setStyleSheet("background: #2C354A; color: white; padding: 4px 10px; border-radius: 4px;")
        btn_sel_all.clicked.connect(self.select_all_tracks)
        tbl_bottom.addWidget(btn_sel_all)

        btn_desel_all = QPushButton("Deselect All")
        btn_desel_all.setStyleSheet("background: #2C354A; color: white; padding: 4px 10px; border-radius: 4px;")
        btn_desel_all.clicked.connect(self.deselect_all_tracks)
        tbl_bottom.addWidget(btn_desel_all)

        tbl_bottom.addStretch()

        self.lbl_playlist_summary = QLabel("Total Duration: 0 tracks (00:00)")
        self.lbl_playlist_summary.setFont(QFont("Arial", 11, QFont.Bold))
        self.lbl_playlist_summary.setStyleSheet("color: #00E676;")
        tbl_bottom.addWidget(self.lbl_playlist_summary)
        table_layout.addLayout(tbl_bottom)

        layout.addWidget(table_group)

        # Start Button
        self.btn_start_session = QPushButton("🚀 Start Full-Length Music Session Playback")
        self.btn_start_session.setFont(QFont("Arial", 13, QFont.Bold))
        self.btn_start_session.setStyleSheet("background-color: #00ADB5; color: white; padding: 14px 28px; border-radius: 6px;")
        self.btn_start_session.clicked.connect(self.start_session)
        layout.addWidget(self.btn_start_session)

        self.populate_tracks_table()
        return widget

    def browse_custom_music_dir(self):
        chosen_dir = QFileDialog.getExistingDirectory(self, "Select Music Directory", self.music_root)
        if chosen_dir and os.path.exists(chosen_dir):
            custom_key = f"custom_{os.path.basename(chosen_dir)}"
            files = [f for f in sorted(os.listdir(chosen_dir)) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
            self.discovered_folders[custom_key] = {
                "display": f"Custom: {os.path.basename(chosen_dir)}",
                "path": chosen_dir,
                "files": files
            }
            self.combo_folders.addItem(f"📁 Custom: {os.path.basename(chosen_dir)} ({len(files)} tracks)", custom_key)
            self.combo_folders.setCurrentIndex(self.combo_folders.count() - 1)

    def populate_tracks_table(self):
        selected_key = self.combo_folders.currentData()
        all_files = []

        if selected_key == "all":
            for sub_rel, info in self.discovered_folders.items():
                for fn in info["files"]:
                    all_files.append((sub_rel, fn, os.path.join(info["path"], fn)))
        elif selected_key in self.discovered_folders:
            info = self.discovered_folders[selected_key]
            for fn in info["files"]:
                all_files.append((selected_key, fn, os.path.join(info["path"], fn)))
        else:
            # Fallback to real folder if available
            fallback = 'real' if 'real' in self.discovered_folders else list(self.discovered_folders.keys())[0] if self.discovered_folders else None
            if fallback:
                info = self.discovered_folders[fallback]
                for fn in info["files"]:
                    all_files.append((fallback, fn, os.path.join(info["path"], fn)))

        # Fill table
        self.track_table.setRowCount(len(all_files))
        for row, (sub_rel, fn, full_path) in enumerate(all_files):
            clean_name = fn.replace('.wav', '').replace('.mp3', '').replace('.ogg', '').replace('real_', '').replace('synthetic_', '').replace('synth_', '').replace('_', ' ').title()

            # Duration calculation
            dur = get_wav_duration_seconds(full_path)
            dur_str = format_duration(dur) if dur > 0 else "--:--"

            # Checkbox item
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk_item.setCheckState(Qt.Checked)
            chk_item.setData(Qt.UserRole, {
                'id': row + 1,
                'name': clean_name,
                'filename': fn,
                'path': full_path,
                'dur': dur,
                'subfolder': sub_rel
            })
            self.track_table.setItem(row, 0, chk_item)

            name_item = QTableWidgetItem(clean_name)
            name_item.setFlags(Qt.ItemIsEnabled)
            self.track_table.setItem(row, 1, name_item)

            src_item = QTableWidgetItem(sub_rel)
            src_item.setFlags(Qt.ItemIsEnabled)
            self.track_table.setItem(row, 2, src_item)

            dur_item = QTableWidgetItem(dur_str)
            dur_item.setFlags(Qt.ItemIsEnabled)
            dur_item.setTextAlignment(Qt.AlignCenter)
            self.track_table.setItem(row, 3, dur_item)

        self.track_table.itemChanged.connect(self.update_playlist_summary)
        self.update_playlist_summary()

    def select_all_tracks(self):
        for r in range(self.track_table.rowCount()):
            item = self.track_table.item(r, 0)
            if item:
                item.setCheckState(Qt.Checked)
        self.update_playlist_summary()

    def deselect_all_tracks(self):
        for r in range(self.track_table.rowCount()):
            item = self.track_table.item(r, 0)
            if item:
                item.setCheckState(Qt.Unchecked)
        self.update_playlist_summary()

    def update_playlist_summary(self):
        selected_count = 0
        total_seconds = 0.0
        for r in range(self.track_table.rowCount()):
            item = self.track_table.item(r, 0)
            if item and item.checkState() == Qt.Checked:
                data = item.data(Qt.UserRole)
                selected_count += 1
                if data and 'dur' in data:
                    total_seconds += data['dur']

        baseline = self.spn_baseline_dur.value()
        inter_rest = self.spn_rest_dur.value()
        if selected_count > 0:
            total_seconds += baseline + (max(0, selected_count - 1) * inter_rest)

        dur_str = format_duration(total_seconds)
        self.lbl_playlist_summary.setText(f"Session Total: {selected_count} tracks (~{dur_str})")

    def create_task_screen(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(40, 30, 40, 30)

        # Top Bar: Session & Progress info
        top_bar = QHBoxLayout()
        self.lbl_session_title = QLabel("🎵 Paradigm: Full-Length Music Listening")
        self.lbl_session_title.setFont(QFont("Arial", 14, QFont.Bold))
        self.lbl_session_title.setStyleSheet("color: #74B9FF;")
        top_bar.addWidget(self.lbl_session_title)

        top_bar.addStretch()

        self.lbl_track_counter = QLabel("Track 0 / 0")
        self.lbl_track_counter.setFont(QFont("Arial", 14, QFont.Bold))
        self.lbl_track_counter.setStyleSheet("color: #A0A5B5;")
        top_bar.addWidget(self.lbl_track_counter)
        layout.addLayout(top_bar)

        layout.addStretch()

        # Center Area: Visual Stimulus / Fixation Cross / Pulse Visualizer
        self.center_container = QStackedWidget()

        # Page 0: Fixation Cross (+)
        page_fix = QWidget()
        layout_fix = QVBoxLayout(page_fix)
        self.lbl_fixation = QLabel("+")
        self.lbl_fixation.setFont(QFont("Arial", 85, QFont.Bold))
        self.lbl_fixation.setAlignment(Qt.AlignCenter)
        self.lbl_fixation.setStyleSheet("color: #FFEAA7;")
        layout_fix.addWidget(self.lbl_fixation)
        self.center_container.addWidget(page_fix)

        # Page 1: Rhythmic Pulse Visualizer
        self.pulse_widget = RhythmicPulseVisualizer()
        self.center_container.addWidget(self.pulse_widget)

        # Page 2: Big Info Mode
        page_info = QWidget()
        layout_info = QVBoxLayout(page_info)
        self.lbl_big_icon = QLabel("🎧")
        self.lbl_big_icon.setFont(QFont("Arial", 75))
        self.lbl_big_icon.setAlignment(Qt.AlignCenter)
        layout_info.addWidget(self.lbl_big_icon)
        self.center_container.addWidget(page_info)

        layout.addWidget(self.center_container)

        # Large Track Name & Phase Instruction
        self.lbl_track_name = QLabel("PREPARING BASELINE REST...")
        self.lbl_track_name.setFont(QFont("Arial", 24, QFont.Bold))
        self.lbl_track_name.setAlignment(Qt.AlignCenter)
        self.lbl_track_name.setStyleSheet("color: #FFFFFF; margin-top: 15px;")
        layout.addWidget(self.lbl_track_name)

        self.lbl_sub_status = QLabel("Please relax and listen naturally")
        self.lbl_sub_status.setFont(QFont("Arial", 13))
        self.lbl_sub_status.setAlignment(Qt.AlignCenter)
        self.lbl_sub_status.setStyleSheet("color: #4DEEEA; margin-bottom: 20px;")
        layout.addWidget(self.lbl_sub_status)

        # Big Digital Timecode
        self.lbl_timecode = QLabel("00:00 / 00:00")
        self.lbl_timecode.setFont(QFont("Arial", 20, QFont.Bold))
        self.lbl_timecode.setAlignment(Qt.AlignCenter)
        self.lbl_timecode.setStyleSheet("color: #DFE6E9;")
        layout.addWidget(self.lbl_timecode)

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

        # Bottom Controls: Pause, Skip, Stop, Volume Slider
        bottom_bar = QHBoxLayout()
        self.btn_pause = QPushButton("⏸️ Pause")
        self.btn_pause.setStyleSheet("background-color: #2C354A; color: white; padding: 8px 16px; border-radius: 4px; font-weight: bold;")
        self.btn_pause.clicked.connect(self.toggle_pause)
        bottom_bar.addWidget(self.btn_pause)

        self.btn_skip = QPushButton("⏭️ Skip Track")
        self.btn_skip.setStyleSheet("background-color: #2C354A; color: white; padding: 8px 16px; border-radius: 4px; font-weight: bold;")
        self.btn_skip.clicked.connect(self.skip_current_track)
        bottom_bar.addWidget(self.btn_skip)

        bottom_bar.addSpacing(20)

        # Volume control
        bottom_bar.addWidget(QLabel("🔊 Volume:"))
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(90)
        self.vol_slider.setFixedWidth(120)
        self.vol_slider.valueChanged.connect(lambda v: self.audio_output.setVolume(v / 100.0))
        bottom_bar.addWidget(self.vol_slider)

        bottom_bar.addStretch()

        self.btn_stop = QPushButton("⏹️ Finish & Stop Session")
        self.btn_stop.setStyleSheet("background-color: #FF7675; color: white; padding: 8px 18px; border-radius: 4px; font-weight: bold;")
        self.btn_stop.clicked.connect(self.stop_session)
        bottom_bar.addWidget(self.btn_stop)

        layout.addLayout(bottom_bar)
        return widget

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.stop_session()
        elif event.key() == Qt.Key_Space:
            self.toggle_pause()
        elif event.key() == Qt.Key_N:
            self.skip_current_track()

    def start_session(self):
        # Build active playlist from selected items
        self.playlist = []
        for r in range(self.track_table.rowCount()):
            item = self.track_table.item(r, 0)
            if item and item.checkState() == Qt.Checked:
                data = item.data(Qt.UserRole)
                if data:
                    self.playlist.append(data)

        if not self.playlist:
            QMessageBox.warning(self, "No Tracks Selected", "Please select at least one music track to start the session.")
            return

        if self.chk_shuffle.isChecked():
            random.shuffle(self.playlist)

        # Re-assign sequential IDs
        for idx, trk in enumerate(self.playlist):
            trk['session_order_id'] = idx + 1

        self.current_track_idx = 0
        self.is_paused = False

        # Set visual mode page
        vis_mode_id = self.vis_group.checkedId()
        if vis_mode_id == 1:
            self.center_container.setCurrentIndex(0)  # Fixation cross
        elif vis_mode_id == 2:
            self.center_container.setCurrentIndex(1)  # Rhythmic pulse
        else:
            self.center_container.setCurrentIndex(2)  # Big info

        sub = self.sub_input.text().strip()
        ses = self.ses_input.text().strip()
        self.send_marker(f"Experiment_Start_Sub_{sub}_Ses_{ses}_Paradigm_FullLengthMusic_Tracks_{len(self.playlist)}")

        self.stacked_widget.setCurrentWidget(self.task_screen)
        self.tick_timer.start(50)  # 20 Hz tick update

        # Check for Initial Baseline Rest
        baseline_dur = self.spn_baseline_dur.value()
        if baseline_dur > 0:
            self.start_phase_baseline(baseline_dur)
        else:
            self.start_track_playback()

    def start_phase_baseline(self, duration_sec):
        self.current_phase = "BASELINE"
        self.phase_duration = duration_sec
        self.phase_start_time = time.time()
        self.last_section_marker_sec = 0

        self.lbl_session_title.setText("🎵 Paradigm: Full-Length Music Listening (Baseline Rest)")
        self.lbl_track_counter.setText(f"Rest Phase (0 / {len(self.playlist)})")
        self.lbl_track_name.setText("INITIAL RESTING BASELINE")
        self.lbl_sub_status.setText("Please relax with eyes fixed on center point (+)")
        self.pulse_widget.set_playing(False)

        self.send_marker(f"Baseline_Rest_Start_dur_{duration_sec:.1f}s", duration_sec)
        self.phase_timer.start(int(duration_sec * 1000))

    def start_track_playback(self):
        if self.current_track_idx >= len(self.playlist):
            self.finish_session()
            return

        self.current_phase = "TRACK"
        self.last_section_marker_sec = 0
        trk = self.playlist[self.current_track_idx]
        total_tracks = len(self.playlist)

        self.lbl_session_title.setText("🎵 Paradigm: Full-Length Music Listening")
        self.lbl_track_counter.setText(f"Track {self.current_track_idx + 1} / {total_tracks}")
        self.lbl_track_name.setText(f"▶ {trk['name']}")
        self.lbl_sub_status.setText(f"Playing from start to finish • Folder: {trk['subfolder']}")

        # Estimate duration
        trk_dur = trk.get('dur', 0.0)
        self.phase_duration = trk_dur
        self.phase_start_time = time.time()

        # Send LSL Marker
        clean_marker_name = trk['name'].replace(' ', '_')
        self.send_marker(f"Track_Start_id_{trk['session_order_id']}_name_{clean_marker_name}_dur_{trk_dur:.1f}s", trk_dur)

        # Load and play audio
        self.pulse_widget.set_playing(True)
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(trk['path']))
        self.player.play()

    @Slot(int)
    def on_player_position_changed(self, pos_ms):
        if self.current_phase != "TRACK":
            return
        pos_sec = pos_ms / 1000.0
        dur_ms = self.player.duration()
        dur_sec = (dur_ms / 1000.0) if dur_ms > 0 else self.phase_duration

        self.lbl_timecode.setText(f"{format_duration(pos_sec)} / {format_duration(dur_sec)}")
        if dur_sec > 0:
            pct = int((pos_sec / dur_sec) * 100)
            self.progress_bar.setValue(min(100, pct))

        # Checkpoint marker every 30 seconds if enabled
        if self.chk_section_markers.isChecked():
            current_30s_block = int(pos_sec // 30)
            if current_30s_block > self.last_section_marker_sec and current_30s_block > 0:
                self.last_section_marker_sec = current_30s_block
                trk = self.playlist[self.current_track_idx]
                sec_mark = current_30s_block * 30
                self.send_marker(f"Track_Checkpoint_id_{trk['session_order_id']}_sec_{sec_mark}")

    @Slot(int)
    def on_player_duration_changed(self, dur_ms):
        if dur_ms > 0:
            self.phase_duration = dur_ms / 1000.0

    @Slot(QMediaPlayer.MediaStatus)
    def on_player_media_status(self, status):
        if self.current_phase == "TRACK" and status == QMediaPlayer.MediaStatus.EndOfMedia:
            print(f"[AUDIO FINISHED] Track {self.current_track_idx + 1} finished naturally.")
            self.on_track_finished()

    def on_track_finished(self):
        trk = self.playlist[self.current_track_idx]
        clean_marker_name = trk['name'].replace(' ', '_')
        self.send_marker(f"Track_End_id_{trk['session_order_id']}_name_{clean_marker_name}")

        self.player.stop()
        self.pulse_widget.set_playing(False)
        self.current_track_idx += 1

        if self.current_track_idx >= len(self.playlist):
            self.finish_session()
            return

        # Inter-track rest period
        rest_dur = self.spn_rest_dur.value()
        if rest_dur > 0:
            self.start_phase_inter_rest(rest_dur)
        else:
            self.start_track_playback()

    def start_phase_inter_rest(self, duration_sec):
        self.current_phase = "REST"
        self.phase_duration = duration_sec
        self.phase_start_time = time.time()

        next_trk = self.playlist[self.current_track_idx]
        self.lbl_session_title.setText("🎵 Paradigm: Full-Length Music Listening (Inter-Track Rest)")
        self.lbl_track_name.setText("REST INTERVAL")
        self.lbl_sub_status.setText(f"Next Up: {next_trk['name']} • Relax & breathe naturally")
        self.pulse_widget.set_playing(False)

        self.send_marker(f"InterTrack_Rest_Start_dur_{duration_sec:.1f}s", duration_sec)
        self.phase_timer.start(int(duration_sec * 1000))

    def on_phase_timer_timeout(self):
        if self.current_phase == "BASELINE":
            self.send_marker("Baseline_Rest_End")
            self.start_track_playback()
        elif self.current_phase == "REST":
            self.send_marker("InterTrack_Rest_End")
            self.start_track_playback()

    def update_tick_ui(self):
        if self.current_phase in ("BASELINE", "REST"):
            elapsed = time.time() - self.phase_start_time
            rem = max(0.0, self.phase_duration - elapsed)
            self.lbl_timecode.setText(f"Rest: {rem:.1f}s remaining")
            if self.phase_duration > 0:
                pct = int((elapsed / self.phase_duration) * 100)
                self.progress_bar.setValue(min(100, pct))

    def toggle_pause(self):
        if self.current_phase != "TRACK":
            return
        if self.is_paused:
            self.player.play()
            self.pulse_widget.set_playing(True)
            self.btn_pause.setText("⏸️ Pause")
            self.is_paused = False
            self.send_marker("Playback_Resumed")
        else:
            self.player.pause()
            self.pulse_widget.set_playing(False)
            self.btn_pause.setText("▶️ Resume")
            self.is_paused = True
            self.send_marker("Playback_Paused")

    def skip_current_track(self):
        if self.current_phase == "TRACK":
            self.send_marker("Track_Skipped_By_User")
            self.on_track_finished()
        elif self.current_phase in ("BASELINE", "REST"):
            self.phase_timer.stop()
            self.on_phase_timer_timeout()

    def stop_session(self):
        self.phase_timer.stop()
        self.tick_timer.stop()
        self.player.stop()
        self.pulse_widget.set_playing(False)
        self.send_marker("Session_Stopped_Early")
        self.finish_session(stopped_early=True)

    def finish_session(self, stopped_early=False):
        self.phase_timer.stop()
        self.tick_timer.stop()
        self.player.stop()
        self.pulse_widget.set_playing(False)
        self.current_phase = "FINISHED"

        sub = self.sub_input.text().strip()
        ses = self.ses_input.text().strip()
        self.send_marker(f"Session_End_Sub_{sub}_Ses_{ses}")

        # Stop recorder if it was started from this GUI
        if hasattr(self, 'recorder_widget') and self.recorder_widget and self.recorder_widget.recorder and self.recorder_widget.recorder.is_recording:
            self.recorder_widget.toggle_recording()

        if not stopped_early:
            QMessageBox.information(self, "Session Completed", "🎉 Full-Length Music Session completed successfully!\nAll tracks played from start to finish with aligned LSL markers.")
        
        self.stacked_widget.setCurrentWidget(self.config_screen)
        self.auto_update_session()

    def auto_update_session(self):
        if hasattr(self, 'recorder_widget') and self.recorder_widget:
            self.recorder_widget.auto_update_session()


def main():
    app = QApplication(sys.argv)
    window = MusicFullTrackTaskApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
