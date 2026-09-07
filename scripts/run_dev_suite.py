import sys
import os
import time
import subprocess
import threading
import numpy as np
import scipy.signal as signal

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFormLayout, QGroupBox,
    QRadioButton, QButtonGroup, QTextEdit, QMessageBox, QFrame,
    QProgressBar, QSplitter, QFileDialog
)
from PySide6.QtGui import QFont, QColor, QPalette

import pyqtgraph as pg

try:
    from pylsl import resolve_byprop, StreamInlet, resolve_streams
    HAS_LSL = True
except ImportError:
    HAS_LSL = False

base_dir = os.path.dirname(__file__)
for sub in ['recorders', 'tasks', 'bridges', 'visualizers', 'analysis', 'utils']:
    p = os.path.join(base_dir, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from recorders.multimodal_bids_recorder import MultimodalBIDSRecorder
from tasks.left_right_task import LeftRightTaskApp
from tasks.music_memory_task import MusicMemoryTaskApp
from utils.bids_utils import get_formatted_next_session


class MiniEEGVisualizerWidget(QWidget):
    """Embedded live mini EEG scope & band power monitor with dual-source live rendering."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.inlet = None
        self.fs = 250.0
        self.num_channels = 32

        # 3-second rolling buffer
        self.buf_seconds = 3.0
        self.buf_samples = int(self.fs * self.buf_seconds)
        self.eeg_data = np.zeros((self.buf_samples, self.num_channels))
        self.sim_sample_count = 0

        # Filter design (2-45 Hz bandpass)
        nyq = 0.5 * self.fs
        self.b_bp, self.a_bp = signal.butter(2, [2.0 / nyq, 45.0 / nyq], btype='band')

        self.init_ui()

        # LSL background discovery thread & render timers
        self.lsl_timer = QTimer(self)
        self.lsl_timer.timeout.connect(self.check_lsl_connection)
        self.lsl_timer.start(1000)

        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.update_live_plot)
        self.render_timer.start(33)  # ~30 FPS smooth rendering loop

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Left Panel: PyQtGraph Live Waveform Scope (Cz, C3, C4, O1)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground('#0D1117')
        layout.addWidget(self.glw, stretch=3)

        self.plot = self.glw.addPlot(title="Live EEG Signal Waves (Cz, C3, C4, O1)")
        self.plot.setLabel('bottom', 'Time (s)')
        self.plot.setLabel('left', 'Amplitude (µV)')
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        self.ch_indices = [0, 1, 2, 3]  # Representative channels
        self.ch_labels = ["Cz", "C3", "C4", "O1"]
        self.colors = ['#00E676', '#74B9FF', '#E040FB', '#FFEAA7']
        self.curves = []

        self.t_vector = np.linspace(-self.buf_seconds, 0, self.buf_samples)
        for i, color in enumerate(self.colors):
            curve = self.plot.plot(pen=pg.mkPen(color=color, width=2.0), name=self.ch_labels[i])
            self.curves.append(curve)

        # Right Panel: Brain Rhythm Band Meters & RMS Health
        right_panel = QVBoxLayout()
        right_panel.setSpacing(6)

        info_hdr = QLabel("<b>Brain Rhythm Power</b>")
        info_hdr.setStyleSheet("color: #4DEEEA; font-size: 12px;")
        right_panel.addWidget(info_hdr)

        # Alpha Bar (8-12 Hz)
        right_panel.addWidget(QLabel("<font color='#00E676'>Alpha (8-12Hz)</font>"))
        self.bar_alpha = QProgressBar()
        self.bar_alpha.setFixedHeight(10)
        self.bar_alpha.setTextVisible(False)
        self.bar_alpha.setStyleSheet("QProgressBar { border: none; background: #191E2A; border-radius: 4px; } QProgressBar::chunk { background: #00E676; }")
        right_panel.addWidget(self.bar_alpha)

        # Beta Bar (12-30 Hz)
        right_panel.addWidget(QLabel("<font color='#74B9FF'>Beta (12-30Hz)</font>"))
        self.bar_beta = QProgressBar()
        self.bar_beta.setFixedHeight(10)
        self.bar_beta.setTextVisible(False)
        self.bar_beta.setStyleSheet("QProgressBar { border: none; background: #191E2A; border-radius: 4px; } QProgressBar::chunk { background: #74B9FF; }")
        right_panel.addWidget(self.bar_beta)

        # Delta Bar (0.5-4 Hz)
        right_panel.addWidget(QLabel("<font color='#FFEAA7'>Delta (0.5-4Hz)</font>"))
        self.bar_delta = QProgressBar()
        self.bar_delta.setFixedHeight(10)
        self.bar_delta.setTextVisible(False)
        self.bar_delta.setStyleSheet("QProgressBar { border: none; background: #191E2A; border-radius: 4px; } QProgressBar::chunk { background: #FFEAA7; }")
        right_panel.addWidget(self.bar_delta)

        # Status Label
        self.lbl_signal_quality = QLabel("Initialising Signal Engine...")
        self.lbl_signal_quality.setFont(QFont("Arial", 9, QFont.Bold))
        self.lbl_signal_quality.setStyleSheet("color: #A0A5B5; margin-top: 5px;")
        right_panel.addWidget(self.lbl_signal_quality)

        right_panel.addStretch()
        layout.addLayout(right_panel, stretch=1)

    def check_lsl_connection(self):
        if not HAS_LSL:
            return

        if self.inlet is None:
            try:
                streams = resolve_streams(wait_time=0.2)
                eeg_streams = [s for s in streams if s.type().upper() == 'EEG' or 'NAUTILUS' in s.name().upper()]
                if eeg_streams:
                    self.inlet = StreamInlet(eeg_streams[0], max_buflen=360)
                    info = self.inlet.info()
                    self.fs = info.nominal_srate() if info.nominal_srate() > 0 else 250.0
                    new_ch = info.channel_count()
                    if new_ch > 0 and new_ch != self.num_channels:
                        self.num_channels = new_ch
                        self.eeg_data = np.zeros((self.buf_samples, self.num_channels))
            except Exception:
                self.inlet = None

    def update_live_plot(self):
        pulled = False

        # 1. Attempt to pull live data from LSL stream
        if self.inlet is not None:
            try:
                samples, _ = self.inlet.pull_chunk(max_samples=250)
                if samples and len(samples) > 0:
                    new_data = np.array(samples, dtype=np.float64)

                    # Auto-scale Volts to Microvolts if data is in Volts (e.g. max < 0.01 V)
                    if np.max(np.abs(new_data)) < 0.01:
                        new_data *= 1e6

                    n_new = len(new_data)
                    n_cols = new_data.shape[1]

                    if n_cols != self.eeg_data.shape[1]:
                        self.num_channels = n_cols
                        self.eeg_data = np.zeros((self.buf_samples, self.num_channels))

                    if n_new >= self.buf_samples:
                        self.eeg_data = new_data[-self.buf_samples:, :]
                    else:
                        self.eeg_data = np.roll(self.eeg_data, -n_new, axis=0)
                        self.eeg_data[-n_new:, :] = new_data
                    pulled = True
            except Exception:
                self.inlet = None

        # 2. If no LSL data yet, generate smooth preview waveforms
        if not pulled:
            n_new = 8  # 8 samples per 33ms tick = ~240 Hz
            t_sim = (self.sim_sample_count + np.arange(n_new)) / self.fs
            self.sim_sample_count += n_new

            sim_chunk = np.zeros((n_new, self.num_channels))
            for c in range(self.num_channels):
                f_alpha = 10.0 + (c % 3) * 0.5
                f_beta = 18.0 + (c % 4) * 1.2
                wave_alpha = 14.0 * np.sin(2 * np.pi * f_alpha * t_sim)
                wave_beta = 7.0 * np.sin(2 * np.pi * f_beta * t_sim)
                noise = np.random.normal(0, 2.0, n_new)
                sim_chunk[:, c] = wave_alpha + wave_beta + noise

            self.eeg_data = np.roll(self.eeg_data, -n_new, axis=0)
            self.eeg_data[-n_new:, :] = sim_chunk

        # 3. Filter and Plot Waveforms
        filt_data = self.eeg_data - np.mean(self.eeg_data, axis=0)
        try:
            filt_data = signal.filtfilt(self.b_bp, self.a_bp, filt_data, axis=0)
        except Exception:
            pass

        # Dynamic gain scaling so small microvolt waves are always clearly visible
        current_std = np.std(filt_data)
        spacing = max(20.0, current_std * 3.5)
        offsets = [0, spacing, spacing * 2, spacing * 3]

        for i, ch_idx in enumerate(self.ch_indices):
            if ch_idx < filt_data.shape[1]:
                y_val = filt_data[:, ch_idx] + offsets[i]
                self.curves[i].setData(self.t_vector, y_val)

        self.plot.setYRange(-spacing * 0.8, spacing * 3.8)

        # 4. Calculate Live Band Powers (FFT)
        if len(filt_data) >= 64:
            fft_vals = np.abs(np.fft.rfft(filt_data, axis=0))
            freqs = np.fft.rfftfreq(len(filt_data), 1.0 / self.fs)

            alpha_idx = (freqs >= 8) & (freqs <= 12)
            beta_idx = (freqs >= 12) & (freqs <= 30)
            delta_idx = (freqs >= 0.5) & (freqs <= 4)

            p_alpha = np.mean(fft_vals[alpha_idx, :]) if np.any(alpha_idx) else 0.0
            p_beta = np.mean(fft_vals[beta_idx, :]) if np.any(beta_idx) else 0.0
            p_delta = np.mean(fft_vals[delta_idx, :]) if np.any(delta_idx) else 0.0

            total_p = p_alpha + p_beta + p_delta + 1e-6
            self.bar_alpha.setValue(int(min(100, (p_alpha / total_p) * 100)))
            self.bar_beta.setValue(int(min(100, (p_beta / total_p) * 100)))
            self.bar_delta.setValue(int(min(100, (p_delta / total_p) * 100)))

            rms = np.sqrt(np.mean(filt_data**2))
            if pulled:
                self.lbl_signal_quality.setText(f"LIVE LSL STREAM | RMS: {rms:.1f}µV")
                self.lbl_signal_quality.setStyleSheet("color: #00E676;")
            else:
                self.lbl_signal_quality.setText(f"PREVIEW (Click Start Streamer) | RMS: {rms:.1f}µV")
                self.lbl_signal_quality.setStyleSheet("color: #FFEAA7;")


class BCISuiteControlCenter(QMainWindow):
    def __init__(self, initial_bids_root="bids_dataset"):
        super().__init__()
        self.setWindowTitle("BCI Motor Imagery & BIDS Studio - Master Control Center")
        self.resize(920, 800)

        self.initial_bids_root = initial_bids_root
        self.streamer_process = None
        self.recorder = None
        self.task_window = None
        self.music_memory_window = None
        self.calibrator_window = None
        self.video_task_window = None

        self.init_ui()

        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.update_status)
        self.monitor_timer.start(1000)

    def init_ui(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(15, 18, 25))
        palette.setColor(QPalette.WindowText, QColor(240, 240, 245))
        palette.setColor(QPalette.Base, QColor(25, 30, 42))
        palette.setColor(QPalette.Text, QColor(240, 240, 245))
        palette.setColor(QPalette.Button, QColor(35, 45, 65))
        palette.setColor(QPalette.ButtonText, QColor(240, 240, 245))
        self.setPalette(palette)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(12)

        # Title Header
        title = QLabel("🧠 BCI Motor Imagery & BIDS Dataset Studio")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #4DEEEA;")
        main_layout.addWidget(title)

        subtitle = QLabel("All-in-One Control Panel with Embedded Live Signal & Brain Rhythm Monitor")
        subtitle.setFont(QFont("Arial", 11))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #A0A5B5;")
        main_layout.addWidget(subtitle)

        # ----------------------------------------------------
        # Box 1: EEG Streamer Manager
        # ----------------------------------------------------
        stream_box = QGroupBox("Step 1: EEG Streamer Control")
        stream_box.setStyleSheet("QGroupBox { font-size: 13px; font-weight: bold; color: #4DEEEA; border: 1px solid #2C3545; border-radius: 8px; margin-top: 5px; padding-top: 12px; }")
        stream_layout = QVBoxLayout(stream_box)

        stream_opts = QHBoxLayout()
        self.radio_mock = QRadioButton("Simulated Mock Streamer (mock_lsl_streamer.py)")
        self.radio_mock.setChecked(True)

        self.stream_btn_group = QButtonGroup()
        self.stream_btn_group.addButton(self.radio_mock)

        stream_opts.addWidget(self.radio_mock)
        stream_layout.addLayout(stream_opts)

        stream_ctrl_row = QHBoxLayout()
        self.btn_toggle_streamer = QPushButton("▶ Start Mock EEG Streamer")
        self.btn_toggle_streamer.setFont(QFont("Arial", 11, QFont.Bold))
        self.btn_toggle_streamer.setStyleSheet("background-color: #00ADB5; color: white; padding: 8px 16px; border-radius: 5px;")
        self.btn_toggle_streamer.clicked.connect(self.toggle_streamer)
        stream_ctrl_row.addWidget(self.btn_toggle_streamer)

        self.lbl_stream_status = QLabel("STATUS: [OFFLINE]")
        self.lbl_stream_status.setFont(QFont("Arial", 11, QFont.Bold))
        self.lbl_stream_status.setStyleSheet("color: #FF7675;")
        stream_ctrl_row.addWidget(self.lbl_stream_status)
        stream_ctrl_row.addStretch()

        stream_layout.addLayout(stream_ctrl_row)
        main_layout.addWidget(stream_box)

        # ----------------------------------------------------
        # Box 2: Embedded Mini Live Signal Monitor
        # ----------------------------------------------------
        mini_vis_box = QGroupBox("📡 Live Mini Signal Scope & Brain Rhythm Monitor")
        mini_vis_box.setStyleSheet("QGroupBox { font-size: 13px; font-weight: bold; color: #FFEAA7; border: 1px solid #2C3545; border-radius: 8px; margin-top: 5px; padding-top: 10px; }")
        mini_vis_layout = QVBoxLayout(mini_vis_box)

        self.mini_visualizer = MiniEEGVisualizerWidget()
        self.mini_visualizer.setFixedHeight(190)
        mini_vis_layout.addWidget(self.mini_visualizer)

        main_layout.addWidget(mini_vis_box)

        # ----------------------------------------------------
        # Box 3: BIDS Recorder Manager
        # ----------------------------------------------------
        rec_box = QGroupBox("Step 2: BIDS Dataset Recording")
        rec_box.setStyleSheet("QGroupBox { font-size: 13px; font-weight: bold; color: #4DEEEA; border: 1px solid #2C3545; border-radius: 8px; margin-top: 5px; padding-top: 12px; }")
        rec_layout = QVBoxLayout(rec_box)

        form_layout = QHBoxLayout()
        self.txt_sub = QLineEdit("01")
        self.txt_sub.setFixedWidth(80)
        self.txt_sub.setStyleSheet("background: #191E2A; color: white; padding: 5px; border-radius: 4px;")

        self.txt_ses = QLineEdit("01")
        self.txt_ses.setFixedWidth(80)
        self.txt_ses.setStyleSheet("background: #191E2A; color: white; padding: 5px; border-radius: 4px;")

        self.txt_outdir = QLineEdit(self.initial_bids_root)
        self.txt_outdir.setStyleSheet("background: #191E2A; color: white; padding: 5px; border-radius: 4px;")

        self.btn_browse_bids = QPushButton("Browse...")
        self.btn_browse_bids.setStyleSheet("background-color: #2C354A; color: white; padding: 5px 10px; border-radius: 4px;")
        self.btn_browse_bids.clicked.connect(self.browse_bids_folder)

        form_layout.addWidget(QLabel("Subject ID: sub-"))
        form_layout.addWidget(self.txt_sub)
        form_layout.addWidget(QLabel("Session ID: ses-"))
        form_layout.addWidget(self.txt_ses)
        form_layout.addWidget(QLabel("BIDS Folder:"))
        form_layout.addWidget(self.txt_outdir)
        form_layout.addWidget(self.btn_browse_bids)
        rec_layout.addLayout(form_layout)

        # Auto-update session based on existing sessions for subject/bids root
        self.txt_sub.textChanged.connect(self.auto_update_session)
        self.txt_outdir.textChanged.connect(self.auto_update_session)
        self.auto_update_session()

        rec_ctrl_row = QHBoxLayout()
        self.btn_toggle_recording = QPushButton("🔴 Start BIDS Recording")
        self.btn_toggle_recording.setFont(QFont("Arial", 11, QFont.Bold))
        self.btn_toggle_recording.setStyleSheet("background-color: #E17055; color: white; padding: 8px 16px; border-radius: 5px;")
        self.btn_toggle_recording.clicked.connect(self.toggle_recording)
        rec_ctrl_row.addWidget(self.btn_toggle_recording)

        self.lbl_rec_status = QLabel("RECORDER: [STANDBY]")
        self.lbl_rec_status.setFont(QFont("Arial", 11, QFont.Bold))
        self.lbl_rec_status.setStyleSheet("color: #A0A5B5;")
        rec_ctrl_row.addWidget(self.lbl_rec_status)
        rec_ctrl_row.addStretch()

        rec_layout.addLayout(rec_ctrl_row)
        main_layout.addWidget(rec_box)

        # ----------------------------------------------------
        # Box 4: Task GUI Paradigm Launcher (Segregated Cards)
        # ----------------------------------------------------
        task_box = QGroupBox("Step 3: Audio-Visual Task Presentation & Calibration Studio")
        task_box.setStyleSheet("QGroupBox { font-size: 13px; font-weight: bold; color: #4DEEEA; border: 1px solid #2C3545; border-radius: 8px; margin-top: 5px; padding-top: 12px; }")
        task_layout = QHBoxLayout(task_box)
        task_layout.setSpacing(12)

        # Card A: Motor Imagery Task (Sessions 01-03)
        card_mi = QGroupBox("🎮 Motor Imagery Paradigm")
        card_mi.setStyleSheet("QGroupBox { font-size: 11px; font-weight: bold; color: #A0A5B5; border: 1px solid #2C3545; border-radius: 6px; padding: 8px; }")
        layout_mi = QVBoxLayout(card_mi)
        lbl_mi = QLabel("4-Class Limb Movement\n(Left, Right, Feet, Tongue)")
        lbl_mi.setStyleSheet("color: #74B9FF; font-size: 10px;")
        layout_mi.addWidget(lbl_mi)

        self.btn_launch_task = QPushButton("🎮 Launch Motor Imagery")
        self.btn_launch_task.setFont(QFont("Arial", 10, QFont.Bold))
        self.btn_launch_task.setStyleSheet("background-color: #6C5CE7; color: white; padding: 8px 12px; border-radius: 5px;")
        self.btn_launch_task.clicked.connect(self.launch_task_gui)
        layout_mi.addWidget(self.btn_launch_task)
        task_layout.addWidget(card_mi)

        # Card B: 6-Track Music Memory Recall Task (Session 04)
        card_music = QGroupBox("🎵 6-Track Music Recall Paradigm")
        card_music.setStyleSheet("QGroupBox { font-size: 11px; font-weight: bold; color: #A0A5B5; border: 1px solid #2C3545; border-radius: 6px; padding: 8px; }")
        layout_music = QVBoxLayout(card_music)
        lbl_music = QLabel("Auditory Imagery & Recall\n(6 Master WAV Compositions)")
        lbl_music.setStyleSheet("color: #00ADB5; font-size: 10px;")
        layout_music.addWidget(lbl_music)

        self.btn_launch_music_memory = QPushButton("🎵 Launch Music Memory Task")
        self.btn_launch_music_memory.setFont(QFont("Arial", 10, QFont.Bold))
        self.btn_launch_music_memory.setStyleSheet("background-color: #00ADB5; color: white; padding: 8px 12px; border-radius: 5px;")
        self.btn_launch_music_memory.clicked.connect(self.launch_music_memory_gui)
        layout_music.addWidget(self.btn_launch_music_memory)
        task_layout.addWidget(card_music)

        # Card C: Standalone Companion Calibrator Tool
        card_calib = QGroupBox("🎛️ Companion Offset Studio")
        card_calib.setStyleSheet("QGroupBox { font-size: 11px; font-weight: bold; color: #A0A5B5; border: 1px solid #2C3545; border-radius: 6px; padding: 8px; }")
        layout_calib = QVBoxLayout(card_calib)
        lbl_calib = QLabel("Continuous Audio Looping\n& Live Offset Seek Studio")
        lbl_calib.setStyleSheet("color: #FFEAA7; font-size: 10px;")
        layout_calib.addWidget(lbl_calib)

        self.btn_launch_calibrator = QPushButton("🎛️ Launch Companion Calibrator")
        self.btn_launch_calibrator.setFont(QFont("Arial", 10, QFont.Bold))
        self.btn_launch_calibrator.setStyleSheet("background-color: #E17055; color: white; padding: 8px 12px; border-radius: 5px;")
        self.btn_launch_calibrator.clicked.connect(self.launch_calibrator_gui)
        layout_calib.addWidget(self.btn_launch_calibrator)
        task_layout.addWidget(card_calib)

        # Card D: Video & Image Slideshow Dataset Task
        card_video = QGroupBox("📹 Video & Image Slideshow Paradigm")
        card_video.setStyleSheet("QGroupBox { font-size: 11px; font-weight: bold; color: #A0A5B5; border: 1px solid #2C354A; border-radius: 6px; padding: 8px; }")
        layout_video = QVBoxLayout(card_video)
        lbl_video = QLabel("Video Clips & Static Image Stimuli\n(Audio Sync & Rest Interval)")
        lbl_video.setStyleSheet("color: #4DEEEA; font-size: 10px;")
        layout_video.addWidget(lbl_video)

        self.btn_launch_video_task = QPushButton("🖼️ Launch Video/Image Task")
        self.btn_launch_video_task.setFont(QFont("Arial", 10, QFont.Bold))
        self.btn_launch_video_task.setStyleSheet("background-color: #00B894; color: white; padding: 8px 12px; border-radius: 5px;")
        self.btn_launch_video_task.clicked.connect(self.launch_video_task_gui)
        layout_video.addWidget(self.btn_launch_video_task)
        task_layout.addWidget(card_video)

        main_layout.addWidget(task_box)

        # Console Log Box
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(100)
        self.log_box.setStyleSheet("background-color: #0D1117; color: #7EE787; font-family: Consolas, monospace; font-size: 11px; border-radius: 6px;")
        self.log_box.append("[SYSTEM] BCI Motor Imagery & BIDS Studio Initialized.")
        main_layout.addWidget(self.log_box)

    def log(self, text):
        self.log_box.append(f"[{time.strftime('%H:%M:%S')}] {text}")

    def toggle_streamer(self):
        if self.streamer_process is None:
            script_name = "mock_lsl_streamer.py"
            base_dir = os.path.dirname(__file__)
            script_path = os.path.join(base_dir, "bridges", script_name)
            if not os.path.exists(script_path):
                script_path = os.path.join(base_dir, script_name)

            self.log(f"Starting Mock EEG Streamer ({script_name})...")
            args = [sys.executable, script_path]

            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            self.streamer_process = subprocess.Popen(args, creationflags=creationflags)
            self.btn_toggle_streamer.setText("⏹ Stop Mock EEG Streamer")
            self.btn_toggle_streamer.setStyleSheet("background-color: #D63031; color: white; padding: 8px 16px; border-radius: 5px;")
            self.lbl_stream_status.setText("STATUS: [RUNNING]")
            self.lbl_stream_status.setStyleSheet("color: #55E6C1;")
        else:
            self.log("Stopping Mock EEG Streamer...")
            self.stop_streamer_gracefully()
            self.btn_toggle_streamer.setText("▶ Start Mock EEG Streamer")
            self.btn_toggle_streamer.setStyleSheet("background-color: #00ADB5; color: white; padding: 8px 16px; border-radius: 5px;")
            self.lbl_stream_status.setText("STATUS: [OFFLINE]")
            self.lbl_stream_status.setStyleSheet("color: #FF7675;")

    def stop_streamer_gracefully(self):
        if self.streamer_process:
            try:
                if sys.platform == "win32":
                    import signal
                    self.streamer_process.send_signal(signal.CTRL_C_EVENT)
                else:
                    self.streamer_process.terminate()
                self.streamer_process.wait(timeout=3)
            except Exception:
                try:
                    self.streamer_process.terminate()
                except Exception:
                    pass
            self.streamer_process = None

    def toggle_recording(self):
        if self.recorder is None or not self.recorder.is_recording:
            try:
                outdir = self.txt_outdir.text().strip()
                self.recorder = MultimodalBIDSRecorder(bids_root=outdir)
                self.log("Discovering all LSL streams (EEG, Smartwatch IMU/PPG, Markers)...")
                connected = self.recorder.discover_and_connect_streams(timeout=3.0)
                if not connected:
                    raise RuntimeError("No active LSL streams discovered.")
                self.recorder.start_recording()

                self.btn_toggle_recording.setText("⏹ Stop & Export Multimodal BIDS Dataset")
                self.btn_toggle_recording.setStyleSheet("background-color: #D63031; color: white; padding: 8px 16px; border-radius: 5px;")
                self.lbl_rec_status.setText("RECORDER: [RECORDING MULTIMODAL...]")
                self.lbl_rec_status.setStyleSheet("color: #FF7675;")
                self.log("[RECORDING STARTED] Continuous EEG, Smartwatch PPG/IMU, and Markers are being recorded.")
            except Exception as e:
                self.log(f"Error starting recorder: {e}")
                QMessageBox.critical(self, "Recording Error", f"Could not connect to LSL streams:\n{e}\n\nMake sure LSL streamers are running!")
                self.recorder = None
        else:
            self.log("Stopping recording and compiling Multimodal BIDS dataset...")
            sub = self.txt_sub.text().strip()
            ses = self.txt_ses.text().strip()
            try:
                self.recorder.stop_and_export_bids(subject_id=sub, session_id=ses)
                out_path = self.txt_outdir.text().strip()
                self.log(f"[SUCCESS] Multimodal BIDS Dataset exported to:\n{out_path}")
                self.btn_toggle_recording.setText("🔴 Start Multimodal BIDS Recording")
                self.btn_toggle_recording.setStyleSheet("background-color: #E74C3C; color: white; padding: 8px 16px; border-radius: 5px;")
                self.lbl_rec_status.setText("RECORDER: [STANDBY]")
                self.lbl_rec_status.setStyleSheet("color: #A0A5B5;")
                QMessageBox.information(self, "BIDS Export Complete", f"Successfully saved Multimodal BIDS dataset to:\n{out_path}")
                self.auto_update_session()
            except Exception as e:
                self.log(f"Error exporting BIDS: {e}")
                QMessageBox.critical(self, "BIDS Export Error", f"Failed to export BIDS dataset:\n{e}")

            self.recorder = None
            self.btn_toggle_recording.setText("🔴 Start BIDS Recording")
            self.btn_toggle_recording.setStyleSheet("background-color: #E17055; color: white; padding: 8px 16px; border-radius: 5px;")
            self.lbl_rec_status.setText("RECORDER: [STANDBY]")
            self.lbl_rec_status.setStyleSheet("color: #A0A5B5;")

    def auto_update_session(self):
        sub = self.txt_sub.text().strip()
        outdir = self.txt_outdir.text().strip()
        curr_ses = self.txt_ses.text().strip()
        next_ses = get_formatted_next_session(outdir, sub, curr_ses)
        self.txt_ses.blockSignals(True)
        self.txt_ses.setText(next_ses)
        self.txt_ses.blockSignals(False)

    def launch_task_gui(self):
        if self.task_window is None or not self.task_window.isVisible():
            self.log("Launching Audio-Visual 4-Direction Task Presentation Window...")
            from tasks import left_right_task as visual_motor_imagery_task
            import importlib
            importlib.reload(visual_motor_imagery_task)
            self.task_window = LeftRightTaskApp()
            self.task_window.sub_input.setText(f"sub-{self.txt_sub.text().strip()}")
            self.task_window.ses_input.setText(f"ses-{self.txt_ses.text().strip()}")
            self.task_window.movement_axis_combo.setCurrentIndex(0)
            self.task_window.show()
        else:
            self.task_window.activateWindow()

    def launch_music_memory_gui(self):
        if self.music_memory_window is None or not self.music_memory_window.isVisible():
            self.log("Launching 6-Track Music Memory Recall Presentation Window...")
            from tasks import music_memory_task
            import importlib
            importlib.reload(music_memory_task)
            self.music_memory_window = music_memory_task.MusicMemoryTaskApp()
            self.music_memory_window.sub_input.setText(f"sub-{self.txt_sub.text().strip()}")
            self.music_memory_window.ses_input.setText(f"ses-{self.txt_ses.text().strip()}")
            self.music_memory_window.show()
        else:
            self.music_memory_window.activateWindow()

    def launch_calibrator_gui(self):
        if self.calibrator_window is None or not self.calibrator_window.isVisible():
            self.log("Launching Companion Music Offset & Tempo Calibrator Studio...")
            from tasks import music_offset_calibrator
            import importlib
            importlib.reload(music_offset_calibrator)
            self.calibrator_window = music_offset_calibrator.MusicOffsetCalibratorApp()
            self.calibrator_window.show()
        else:
            self.calibrator_window.activateWindow()

    def launch_video_task_gui(self):
        if self.video_task_window is None or not self.video_task_window.isVisible():
            self.log("Launching Video Stimulus Dataset Task Presentation Window...")
            from tasks import video_dataset_task
            import importlib
            importlib.reload(video_dataset_task)
            self.video_task_window = video_dataset_task.VideoDatasetTaskApp()
            self.video_task_window.sub_input.setText(f"sub-{self.txt_sub.text().strip()}")
            self.video_task_window.ses_input.setText(f"ses-{self.txt_ses.text().strip()}")
            self.video_task_window.show()
        else:
            self.video_task_window.activateWindow()

    def update_status(self):
        try:
            if self.recorder and self.recorder.is_recording:
                n_eeg = len(self.recorder.eeg_samples)
                n_mrk = len(self.recorder.marker_events)
                self.lbl_rec_status.setText(f"RECORDER: [RECORDING LIVE | {n_eeg} EEG samples | {n_mrk} Events]")
        except Exception:
            pass

    def browse_bids_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select BIDS Target Dataset Directory", self.txt_outdir.text().strip())
        if folder:
            self.txt_outdir.setText(folder)

    def closeEvent(self, event):
        if self.recorder and self.recorder.is_recording:
            self.recorder.is_recording = False
        self.stop_streamer_gracefully()
        event.accept()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BCI Master Control Center")
    parser.add_argument("--bids-root", "--dataset-folder", type=str, default="bids_dataset", help="Target BIDS dataset output directory")
    args, unknown = parser.parse_known_args()

    try:
        app = QApplication(sys.argv)
        window = BCISuiteControlCenter(initial_bids_root=args.bids_root)
        window.show()
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("[*] Control Center closed.")


if __name__ == "__main__":
    main()
