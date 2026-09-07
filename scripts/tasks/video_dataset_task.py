"""
Video Dataset Presentation Task for BCI / Multimodal Data Collection
======================================================================
Displays video stimuli for 3 seconds (configurable), toggles audio output,
supports optional external WAV audio files (in videos/audio/), provides Peak 
Audio Normalization (-1.0 dB), applies 1.5s rest intervals, displays video title 
prominently on top, and broadcasts LSL event markers for BIDS synchronization.
"""

import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import random
import time
import wave
import numpy as np

from PySide6.QtCore import Qt, QTimer, QUrl, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFormLayout, QGroupBox,
    QProgressBar, QStackedWidget, QMessageBox, QCheckBox, QFileDialog,
    QDoubleSpinBox, QSpinBox, QComboBox
)
from PySide6.QtGui import QFont, QColor, QPalette, QPixmap
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget


from recorders.bids_recorder_widget import BIDSRecorderControlWidget


def normalize_wav_file(input_path, output_path, target_peak_dB=-1.0):
    """Normalize a PCM WAV audio file to target peak dB (-1.0 dBFS by default)."""
    try:
        with wave.open(input_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)

        if n_frames == 0 or len(raw_bytes) == 0:
            return input_path

        dtype = np.int16 if sampwidth == 2 else (np.int32 if sampwidth == 4 else np.uint8)
        data = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float32)

        max_val = np.max(np.abs(data))
        if max_val > 0:
            target_linear = 10.0 ** (target_peak_dB / 20.0)
            if dtype == np.int16:
                max_possible = 32767.0
            elif dtype == np.int32:
                max_possible = 2147483647.0
            else:
                max_possible = 127.5
                data = data - 127.5

            scale = (target_linear * max_possible) / max_val
            normalized_data = data * scale
            normalized_data = np.clip(normalized_data, -max_possible, max_possible)

            if dtype == np.int16:
                out_bytes = normalized_data.astype(np.int16).tobytes()
            elif dtype == np.int32:
                out_bytes = normalized_data.astype(np.int32).tobytes()
            else:
                out_bytes = (normalized_data + 127.5).astype(np.uint8).tobytes()
        else:
            out_bytes = raw_bytes

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(out_bytes)
        return output_path
    except Exception as e:
        print(f"[-] WAV Normalization error for {input_path}: {e}")
        return input_path


from tasks.common.base_task import BaseTaskApp

class VideoDatasetTaskApp(BaseTaskApp):
    def __init__(self):
        super().__init__(marker_name='VideoTaskMarkers', source_id='Video_Task_Markers_2026')
        self.setWindowTitle("BCI Video Dataset Presentation Suite")
        self.resize(1080, 840)

        # Default paths
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.video_dir = os.path.abspath(os.path.join(base_dir, "videos"))
        self.audio_dir = os.path.abspath(os.path.join(self.video_dir, "audio"))
        self.norm_cache_dir = os.path.abspath(os.path.join(self.audio_dir, "normalized"))
        
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.norm_cache_dir, exist_ok=True)

        # LSL Marker Outlet setup

        # Dual Media Engine Setup (Video Player + External WAV Audio Player)
        self.video_player = QMediaPlayer()
        self.video_audio = QAudioOutput()
        self.video_player.setAudioOutput(self.video_audio)
        self.video_audio.setVolume(0.85)

        self.custom_audio_player = QMediaPlayer()
        self.custom_audio = QAudioOutput()
        self.custom_audio_player.setAudioOutput(self.custom_audio)
        self.custom_audio.setVolume(0.85)

        # Paradigm Parameters (Defaults: Video 3.0s, Rest 1.5s)
        self.t_video = 3.0   # seconds
        self.t_rest = 1.5    # seconds
        self.audio_enabled = True
        self.normalize_audio = True
        self.use_external_audio = True
        self.repetitions = 1
        self.randomize_order = True

        # State tracking
        self.video_files = []
        self.trials = []  # List of dicts: {'video': path, 'audio': path_or_None}
        self.current_trial_idx = 0
        self.current_phase = "INIT"  # "VIDEO" or "REST"

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
        layout.setContentsMargins(40, 25, 40, 25)
        layout.setSpacing(12)

        title = QLabel("Video & Image Slideshow Dataset Presentation Task")
        title.setFont(QFont("Arial", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #4DEEEA;")
        layout.addWidget(title)

        subtitle = QLabel("Video & Image Stimuli • Audio Toggle & Peak Normalization • Rest Interval • LSL Synchronized")
        subtitle.setFont(QFont("Arial", 11))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #A0A5B5;")
        layout.addWidget(subtitle)

        # Integrated BIDS Recording & Participant Metadata Box
        self.recorder_widget = BIDSRecorderControlWidget(default_task="video", default_bids_root="bids_dataset")
        layout.addWidget(self.recorder_widget)

        self.sub_input = self.recorder_widget.txt_sub
        self.ses_input = self.recorder_widget.txt_ses

        # Form Group
        form_group = QGroupBox("Experiment & Paradigm Settings")
        form_group.setFont(QFont("Arial", 11, QFont.Bold))
        form_group.setStyleSheet("QGroupBox { color: #4DEEEA; border: 1px solid #2C354A; border-radius: 8px; margin-top: 5px; padding: 12px; }")
        form_layout = QFormLayout(form_group)
        form_layout.setSpacing(10)

        # Media Type Mode Selector (Videos, Static Images / Slideshow, or All)
        self.cmb_media_mode = QComboBox()
        self.cmb_media_mode.addItem("🌐 All Media (Videos + Static Images / Slideshow)", "all")
        self.cmb_media_mode.addItem("📹 Video Files Only (.mp4, .avi, .mov, .webm)", "video_only")
        self.cmb_media_mode.addItem("🖼️ Static Images / Slideshow Only (.png, .jpg, .jpeg, .bmp)", "image_only")
        self.cmb_media_mode.setStyleSheet("background: #191E2A; color: white; padding: 5px; border-radius: 4px;")
        self.cmb_media_mode.currentIndexChanged.connect(self.scan_video_directory)
        form_layout.addRow("Media Presentation Mode:", self.cmb_media_mode)

        # Video/Image directory selector
        dir_layout = QHBoxLayout()
        self.lbl_video_dir = QLineEdit(self.video_dir)
        btn_browse = QPushButton("Browse...")
        btn_browse.setStyleSheet("background-color: #2C354A; color: white; padding: 5px 12px; border-radius: 4px;")
        btn_browse.clicked.connect(self.browse_video_dir)
        dir_layout.addWidget(self.lbl_video_dir)
        dir_layout.addWidget(btn_browse)
        form_layout.addRow("Media Folder:", dir_layout)

        # Audio Options
        self.chk_audio = QCheckBox("Enable Audio Output during Media Playback")
        self.chk_audio.setChecked(True)
        self.chk_audio.setStyleSheet("color: #FFEAA7; font-weight: bold;")
        form_layout.addRow("Audio Master Switch:", self.chk_audio)

        self.chk_normalize = QCheckBox("Normalize Audio Volume (-1.0 dB Peak)")
        self.chk_normalize.setChecked(True)
        self.chk_normalize.setStyleSheet("color: #00E676; font-weight: bold;")
        self.chk_normalize.setToolTip("Normalizes WAV audio files to -1.0 dBFS peak amplitude to eliminate volume discrepancies.")
        form_layout.addRow("Audio Normalization:", self.chk_normalize)

        self.chk_ext_audio = QCheckBox("Use External Audio Folder (videos/audio/*.wav)")
        self.chk_ext_audio.setChecked(True)
        self.chk_ext_audio.setStyleSheet("color: #74B9FF; font-weight: bold;")
        self.chk_ext_audio.setToolTip("If a matching .wav file is found in videos/audio/, it will play in sync with the video/image.")
        form_layout.addRow("External Audio Source:", self.chk_ext_audio)

        # Video/Image Duration (default 3.0s)
        self.spn_video_dur = QDoubleSpinBox()
        self.spn_video_dur.setRange(1.0, 60.0)
        self.spn_video_dur.setValue(3.0)
        self.spn_video_dur.setSingleStep(0.5)
        self.spn_video_dur.setSuffix(" sec")
        form_layout.addRow("Stimulus Presentation Time:", self.spn_video_dur)

        # Rest Duration (default 1.5s)
        self.spn_rest_dur = QDoubleSpinBox()
        self.spn_rest_dur.setRange(0.5, 30.0)
        self.spn_rest_dur.setValue(1.5)
        self.spn_rest_dur.setSingleStep(0.5)
        self.spn_rest_dur.setSuffix(" sec")
        form_layout.addRow("Rest Interval Duration:", self.spn_rest_dur)

        # Repetitions
        self.spn_reps = QSpinBox()
        self.spn_reps.setRange(1, 20)
        self.spn_reps.setValue(1)
        form_layout.addRow("Repetitions per Media Item:", self.spn_reps)

        # Randomize Order
        self.chk_random = QCheckBox("Randomize Media Playback Order")
        self.chk_random.setChecked(True)
        form_layout.addRow("Order:", self.chk_random)

        layout.addWidget(form_group)

        # Video scan status label
        self.lbl_scan_status = QLabel("Scanning media directory...")
        self.lbl_scan_status.setStyleSheet("color: #74B9FF;")
        layout.addWidget(self.lbl_scan_status)
        self.scan_video_directory()

        # Start Button
        btn_start = QPushButton("▶ START MEDIA DATASET EXPERIMENT")
        btn_start.setFont(QFont("Arial", 14, QFont.Bold))
        btn_start.setStyleSheet("QPushButton { background-color: #00B894; color: white; padding: 12px; border-radius: 6px; } QPushButton:hover { background-color: #00ECB5; }")
        btn_start.clicked.connect(self.start_experiment)
        layout.addWidget(btn_start)

        return widget

    def browse_video_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Video Stimulus Directory", self.video_dir)
        if folder:
            self.video_dir = folder
            self.audio_dir = os.path.abspath(os.path.join(self.video_dir, "audio"))
            self.norm_cache_dir = os.path.abspath(os.path.join(self.audio_dir, "normalized"))
            os.makedirs(self.audio_dir, exist_ok=True)
            os.makedirs(self.norm_cache_dir, exist_ok=True)

            self.lbl_video_dir.setText(folder)
            self.scan_video_directory()

    def find_matching_audios(self, media_path):
        """
        Smart Naming & Audio Pairing Convention for Images & Videos:
        Looks for matching audio tracks in videos/audio/ or alongside the media file.
        Supports:
          1. Exact match: 'water_01.png' matches 'water_01.wav'
          2. Multiple variations: 'water_01.png' matches 'water_01_soundA.wav', 'water_01_soundB.wav'
          3. Concept prefix match: 'water_01.png' or 'water.png' matches 'water.wav', 'water_track1.wav'
        """
        if not self.chk_ext_audio.isChecked():
            return []

        stem = os.path.splitext(os.path.basename(media_path))[0].lower()
        
        # Concept prefix (e.g. 'water_01' -> 'water', 'fire02' -> 'fire')
        if '_' in stem:
            concept_prefix = stem.split('_')[0]
        else:
            concept_prefix = stem.rstrip('0123456789')
            if not concept_prefix:
                concept_prefix = stem

        audio_dirs = [self.audio_dir, os.path.dirname(media_path)]
        audio_exts = ('.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg')

        found_audios = set()

        for a_dir in audio_dirs:
            if not os.path.exists(a_dir):
                continue
            try:
                for fname in os.listdir(a_dir):
                    if fname.lower().endswith(audio_exts):
                        a_stem = os.path.splitext(fname)[0].lower()
                        full_p = os.path.abspath(os.path.join(a_dir, fname))

                        # Priority match conditions:
                        if a_stem == stem:
                            found_audios.add((1, full_p))
                        elif a_stem.startswith(stem + "_") or a_stem.startswith(stem + "-"):
                            found_audios.add((2, full_p))
                        elif a_stem == concept_prefix or a_stem.startswith(concept_prefix + "_") or a_stem.startswith(concept_prefix + "-"):
                            found_audios.add((3, full_p))
            except Exception:
                pass

        if not found_audios:
            return []

        sorted_matches = sorted(list(found_audios), key=lambda x: (x[0], x[1]))
        return [p for _, p in sorted_matches]

    def find_matching_audio(self, media_path):
        audios = self.find_matching_audios(media_path)
        return audios[0] if audios else None

    def scan_video_directory(self):
        target_dir = self.lbl_video_dir.text().strip()
        if not os.path.exists(target_dir):
            self.lbl_scan_status.setText(f"⚠️ Folder path does not exist: {target_dir}")
            self.video_files = []
            return

        mode = self.cmb_media_mode.currentData() if hasattr(self, 'cmb_media_mode') else "all"
        video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        image_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

        if mode == "video_only":
            valid_exts = video_exts
        elif mode == "image_only":
            valid_exts = image_exts
        else:
            valid_exts = video_exts + image_exts

        self.video_files = [
            os.path.join(target_dir, f) for f in sorted(os.listdir(target_dir))
            if f.lower().endswith(valid_exts)
        ]

        if self.video_files:
            v_cnt = sum(1 for f in self.video_files if f.lower().endswith(video_exts))
            i_cnt = sum(1 for f in self.video_files if f.lower().endswith(image_exts))
            paired_audio_count = sum(1 for v in self.video_files if len(self.find_matching_audios(v)) > 0)

            msg = f"✅ Discovered {len(self.video_files)} media item(s) ({v_cnt} video(s), {i_cnt} static image(s))."
            if paired_audio_count > 0:
                msg += f" 🎵 {paired_audio_count} paired audio track(s) detected in videos/audio/."
            else:
                msg += f" (Optional audio tracks can be placed in: {self.audio_dir})"

            self.lbl_scan_status.setText(msg)
            self.lbl_scan_status.setStyleSheet("color: #00E676;")
        else:
            self.lbl_scan_status.setText(f"⚠️ No matching media files (.mp4, .png, .jpg) found in {target_dir}.")
            self.lbl_scan_status.setStyleSheet("color: #FF7675;")

    def create_task_screen(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Video/Image Title Header
        self.lbl_video_title = QLabel("MEDIA STIMULUS TITLE")
        self.lbl_video_title.setFont(QFont("Arial", 20, QFont.Bold))
        self.lbl_video_title.setAlignment(Qt.AlignCenter)
        self.lbl_video_title.setStyleSheet("color: #FFEAA7; background-color: #191E2A; padding: 10px; border-radius: 6px;")
        layout.addWidget(self.lbl_video_title)

        # Center Display Stack (Video Widget vs. Image Widget vs. Rest Fixation Screen)
        self.display_stack = QStackedWidget()
        
        # 1. Video Player Widget
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: #000000; border-radius: 8px;")
        self.video_player.setVideoOutput(self.video_widget)
        self.display_stack.addWidget(self.video_widget)

        # 2. Static Image / Slideshow Widget
        self.image_widget = QLabel()
        self.image_widget.setAlignment(Qt.AlignCenter)
        self.image_widget.setStyleSheet("background-color: #000000; border-radius: 8px;")
        self.display_stack.addWidget(self.image_widget)

        # 3. Rest / Fixation Screen
        self.rest_widget = QWidget()
        rest_layout = QVBoxLayout(self.rest_widget)
        rest_layout.setAlignment(Qt.AlignCenter)

        self.lbl_fixation = QLabel("+")
        self.lbl_fixation.setFont(QFont("Arial", 90, QFont.Bold))
        self.lbl_fixation.setAlignment(Qt.AlignCenter)
        self.lbl_fixation.setStyleSheet("color: #4DEEEA;")

        self.lbl_rest_sub = QLabel("REST (1.5s)")
        self.lbl_rest_sub.setFont(QFont("Arial", 16, QFont.Bold))
        self.lbl_rest_sub.setAlignment(Qt.AlignCenter)
        self.lbl_rest_sub.setStyleSheet("color: #A0A5B5;")

        rest_layout.addWidget(self.lbl_fixation)
        rest_layout.addWidget(self.lbl_rest_sub)
        self.display_stack.addWidget(self.rest_widget)

        layout.addWidget(self.display_stack, stretch=1)

        # Bottom Progress Bar & Info
        bottom_box = QHBoxLayout()
        
        self.lbl_status = QLabel("Trial 0 / 0")
        self.lbl_status.setFont(QFont("Arial", 12, QFont.Bold))
        self.lbl_status.setStyleSheet("color: #00E676;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { background-color: #191E2A; border-radius: 6px; } QProgressBar::chunk { background-color: #4DEEEA; }")

        btn_stop = QPushButton("⏹ STOP EXPERIMENT")
        btn_stop.setStyleSheet("background-color: #D63031; color: white; font-weight: bold; padding: 6px 14px; border-radius: 4px;")
        btn_stop.clicked.connect(self.stop_experiment)

        bottom_box.addWidget(self.lbl_status)
        bottom_box.addWidget(self.progress_bar, stretch=1)
        bottom_box.addWidget(btn_stop)

        layout.addLayout(bottom_box)
        return widget

    def start_experiment(self):
        self.scan_video_directory()
        if not self.video_files:
            QMessageBox.warning(self, "No Videos Found", f"No video files found in:\n{self.lbl_video_dir.text()}\n\nPlease place video files (.mp4, .avi, .mov) in this directory.")
            return

        self.t_video = self.spn_video_dur.value()
        self.t_rest = self.spn_rest_dur.value()
        self.audio_enabled = self.chk_audio.isChecked()
        self.normalize_audio = self.chk_normalize.isChecked()
        self.use_external_audio = self.chk_ext_audio.isChecked()
        self.repetitions = self.spn_reps.value()
        self.randomize_order = self.chk_random.isChecked()

        # Build Trial List and process Audio Normalization if enabled
        base_trials = []
        for v_path in self.video_files:
            matched_audios = self.find_matching_audios(v_path)
            if matched_audios:
                for audio_path in matched_audios:
                    # Apply Normalization to WAV audio if enabled
                    if audio_path and audio_path.lower().endswith('.wav') and self.normalize_audio:
                        base_name = os.path.splitext(os.path.basename(audio_path))[0]
                        norm_path = os.path.join(self.norm_cache_dir, f"{base_name}_norm.wav")
                        audio_path = normalize_wav_file(audio_path, norm_path, target_peak_dB=-1.0)
                    base_trials.append({'video': v_path, 'audio': audio_path})
            else:
                base_trials.append({'video': v_path, 'audio': None})

        self.trials = []
        for _ in range(self.repetitions):
            v_list = list(base_trials)
            if self.randomize_order:
                random.shuffle(v_list)
            self.trials.extend(v_list)

        self.current_trial_idx = 0
        self.stacked_widget.setCurrentWidget(self.task_screen)
        
        self.send_marker("Video_Dataset_Experiment_Start")
        self.send_marker(f"Audio_Enabled_{self.audio_enabled}")
        self.send_marker(f"Audio_Normalized_{self.normalize_audio}")
        
        # Start initial rest phase (1.5s) before first video
        self.start_rest_phase()

    def start_video_phase(self):
        if self.current_trial_idx >= len(self.trials):
            self.finish_experiment()
            return

        self.current_phase = "VIDEO"
        trial = self.trials[self.current_trial_idx]
        video_path = trial['video']
        audio_path = trial['audio']

        is_image = video_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        file_name = os.path.splitext(os.path.basename(video_path))[0]
        icon = "🖼️" if is_image else "📹"
        media_label = "Static Image" if is_image else "Video"

        self.lbl_video_title.setText(f"{icon} {file_name}")
        self.lbl_status.setText(f"Trial {self.current_trial_idx + 1} / {len(self.trials)}: Presenting {media_label}")

        if is_image:
            pix = QPixmap(video_path)
            if not pix.isNull():
                disp_size = self.display_stack.size()
                if disp_size.width() < 50:
                    disp_size = self.size()
                scaled_pix = pix.scaled(disp_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_widget.setPixmap(scaled_pix)
            self.display_stack.setCurrentWidget(self.image_widget)
        else:
            self.display_stack.setCurrentWidget(self.video_widget)

        # Configure Audio Outputs
        if not self.audio_enabled:
            self.video_audio.setMuted(True)
            self.custom_audio.setMuted(True)
            audio_source_str = "AUDIO_MUTED"
        elif audio_path is not None:
            # Custom external audio track exists
            self.video_audio.setMuted(True)
            self.custom_audio.setMuted(False)
            self.custom_audio_player.setSource(QUrl.fromLocalFile(audio_path))
            audio_source_str = f"EXTERNAL_AUDIO_{os.path.basename(audio_path)}"
        else:
            # Native embedded video audio
            self.video_audio.setMuted(False)
            self.custom_audio.setMuted(True)
            audio_source_str = "NATIVE_VIDEO_AUDIO"

        # Broadcast LSL Markers
        norm_str = "NORM_ACTIVE" if (self.normalize_audio and audio_path) else "NORM_OFF"
        self.send_marker(f"Trial_Start_{self.current_trial_idx + 1}")
        self.send_marker(f"Media_Start_{file_name}_{audio_source_str}_{norm_str}")

        # Start Playback
        if not is_image:
            self.video_player.setSource(QUrl.fromLocalFile(video_path))
            self.video_player.play()
        if self.audio_enabled and audio_path is not None:
            self.custom_audio_player.play()

        # Schedule timer for configured duration
        self.phase_start_time = time.time()
        self.current_phase_duration = self.t_video
        self.trial_timer.start(int(self.t_video * 1000))
        self.tick_timer.start(30)

    def start_rest_phase(self):
        self.current_phase = "REST"
        self.video_player.stop()
        self.custom_audio_player.stop()

        self.lbl_video_title.setText("REST INTERVAL")
        self.lbl_rest_sub.setText(f"REST ({self.t_rest:.1f}s)")
        self.lbl_status.setText(f"Trial {self.current_trial_idx + 1} / {len(self.trials)}: Rest")
        self.display_stack.setCurrentWidget(self.rest_widget)

        self.send_marker("Rest_Start")

        self.phase_start_time = time.time()
        self.current_phase_duration = self.t_rest
        self.trial_timer.start(int(self.t_rest * 1000))
        self.tick_timer.start(30)

    def advance_trial_phase(self):
        self.tick_timer.stop()
        if self.current_phase == "VIDEO":
            trial = self.trials[self.current_trial_idx]
            file_name = os.path.splitext(os.path.basename(trial['video']))[0]
            
            self.video_player.stop()
            self.custom_audio_player.stop()

            self.send_marker(f"Video_End_{file_name}")
            self.send_marker(f"Trial_End_{self.current_trial_idx + 1}")
            self.current_trial_idx += 1
            self.start_rest_phase()
        elif self.current_phase == "REST":
            self.send_marker("Rest_End")
            if self.current_trial_idx < len(self.trials):
                self.start_video_phase()
            else:
                self.finish_experiment()

    def update_progress(self):
        elapsed = time.time() - self.phase_start_time
        pct = min(100, int((elapsed / max(0.01, self.current_phase_duration)) * 100))
        self.progress_bar.setValue(pct)

    def finish_experiment(self):
        self.trial_timer.stop()
        self.tick_timer.stop()
        self.video_player.stop()
        self.custom_audio_player.stop()
        self.send_marker("Video_Dataset_Experiment_Complete")
        
        QMessageBox.information(self, "Experiment Complete", "All video trials have been completed successfully!")
        self.stacked_widget.setCurrentWidget(self.config_screen)

    def stop_experiment(self):
        self.trial_timer.stop()
        self.tick_timer.stop()
        self.video_player.stop()
        self.custom_audio_player.stop()
        self.send_marker("Video_Dataset_Experiment_Aborted")
        self.stacked_widget.setCurrentWidget(self.config_screen)

    def closeEvent(self, event):
        self.trial_timer.stop()
        self.tick_timer.stop()
        self.video_player.stop()
        self.custom_audio_player.stop()
        event.accept()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Video Stimulus Dataset Task")
    parser.add_argument("--sub", type=str, default="01", help="Subject ID")
    parser.add_argument("--ses", type=str, default="01", help="Session ID")
    parser.add_argument("--bids-root", "--dataset-folder", type=str, default="bids_dataset", help="Target BIDS dataset output directory")
    args, unknown = parser.parse_known_args()

    app = QApplication(sys.argv)
    window = VideoDatasetTaskApp()

    if hasattr(window, 'recorder_widget') and window.recorder_widget:
        if args.sub:
            window.recorder_widget.txt_sub.setText(args.sub.replace('sub-', ''))
        if args.ses:
            window.recorder_widget.txt_ses.setText(args.ses.replace('ses-', ''))
        if args.bids_root:
            window.recorder_widget.txt_outdir.setText(args.bids_root)

    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
