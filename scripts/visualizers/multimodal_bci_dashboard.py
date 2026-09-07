"""
Multimodal BCI Master Dashboard & Data Collection Studio
=========================================================

Unified GUI application displaying:
  1. 🧠 EEG Scalp Quality (2D 10-20 Headmap & Channel Goodness Table)
  2. ⚡ Key EEG Waves & Motor Cortex Signals (C3, C4, Cz, O1, O2 + Alpha/Beta/Delta Power)
  3. ⌚ Smartwatch PPG & 6-DOF IMU Motion Vectors (Heart Rate BPM, Accel XYZ, Gyro XYZ, Motion Magnitude)
  4. 🔴 1-Click Multimodal BIDS Recording Engine

Usage:
  uv run python visualizers/multimodal_bci_dashboard.py
"""

import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

import time
import subprocess
import numpy as np
import scipy.signal as signal

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

pg.setConfigOption('background', '#0D1117')
pg.setConfigOption('foreground', '#ECF0F1')
pg.setConfigOptions(antialias=True)

try:
    from pylsl import StreamInlet, resolve_streams
    HAS_LSL = True
except ImportError:
    HAS_LSL = False

from visualizers.eeg_headmap_quality_visualizer import HeadMapCanvas, ELECTRODE_POSITIONS_1020
from analysis.spatial_filters import apply_spatial_filter


def resolve_script_path(script_name):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cand = os.path.join(base_dir, script_name)
    if os.path.exists(cand):
        return cand
    for sub in ['bridges', 'tasks', 'recorders', 'visualizers', 'analysis', 'utils']:
        cand = os.path.join(base_dir, sub, script_name)
        if os.path.exists(cand):
            return cand
    return os.path.join(base_dir, script_name)


class MultimodalBCIDashboard(QtWidgets.QMainWindow):
    def __init__(self, initial_bids_root="bids_dataset_multimodal"):
        super().__init__()
        self.setWindowTitle("Multimodal BCI Master Dashboard (EEG + Smartwatch PPG/IMU + BIDS)")
        self.resize(1300, 850)

        self.initial_bids_root = initial_bids_root
        # Process handles
        self.streamer_process = None
        self.mock_streamer_process = None
        self.watch_bridge_process = None

        # LSL Stream Inlets
        self.eeg_inlet = None
        self.imu_inlet = None
        self.ppg_inlet = None

        # Data Buffers
        self.fs_eeg = 250.0
        self.buf_samples_eeg = int(self.fs_eeg * 3.0)
        self.eeg_buffer = np.zeros((self.buf_samples_eeg, 33))  # 32 channels + Battery
        self.eeg_ch_names = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
            'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 
            'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 
            'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10', 'Battery'
        ]

        self.buf_samples_imu = 150  # 3 seconds @ 50 Hz
        self.imu_buffer = np.zeros((self.buf_samples_imu, 6))  # Accel XYZ, Gyro XYZ
        
        self.buf_samples_ppg = 30  # 30 seconds @ 1 Hz
        self.ppg_history = np.zeros(self.buf_samples_ppg)

        # Filters for EEG
        nyq = 0.5 * self.fs_eeg
        self.b_bp, self.a_bp = signal.butter(2, [2.0 / nyq, 45.0 / nyq], btype='band')

        self.init_ui()

        # Timers
        self.lsl_discover_timer = QtCore.QTimer()
        self.lsl_discover_timer.timeout.connect(self.discover_lsl_streams)
        self.lsl_discover_timer.start(1000)

        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.poll_and_render_all)
        self.update_timer.start(33)  # ~30 FPS

    def init_ui(self):
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(15, 18, 25))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(240, 240, 245))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 30, 42))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(240, 240, 245))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor(35, 45, 65))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(240, 240, 245))
        self.setPalette(palette)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0F1219;
                color: #F0F0F5;
            }
            QGroupBox {
                font-size: 12px;
                font-weight: bold;
                color: #4DEEEA;
                border: 1px solid #2C354A;
                border-radius: 8px;
                margin-top: 6px;
                padding: 10px;
            }
            QTabWidget::pane {
                border: 1px solid #2C354A;
                background-color: #151924;
            }
            QTabBar::tab {
                background-color: #232D41;
                color: #A0A5B5;
                padding: 8px 18px;
                font-weight: bold;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #00ADB5;
                color: #FFFFFF;
            }
            QTableWidget {
                background-color: #191E2A;
                color: #F0F0F5;
                gridline-color: #2C354A;
                border: 1px solid #2C354A;
                border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #232D41;
                color: #4DEEEA;
                font-weight: bold;
                padding: 4px;
                border: 1px solid #2C354A;
            }
            QLineEdit {
                background-color: #191E2A;
                color: #FFFFFF;
                border: 1px solid #2C354A;
                border-radius: 4px;
                padding: 4px;
            }
        """)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # Top Control Bar
        top_bar = QtWidgets.QGroupBox("🎛️ Master Control & Stream Launcher")
        top_layout = QtWidgets.QHBoxLayout(top_bar)

        # Cap Mode Selector
        top_layout.addWidget(QtWidgets.QLabel("Cap Mode:"))
        self.combo_cap_type = QtWidgets.QComboBox()
        self.combo_cap_type.addItem("💧 Wet (Gel / Ag-AgCl)", "wet")
        self.combo_cap_type.addItem("🌵 Dry (g.SAHARA Pins)", "dry")
        self.combo_cap_type.setStyleSheet("background-color: #191E2A; color: #4DEEEA; font-weight: bold; padding: 4px 8px; border: 1px solid #2C354A; border-radius: 4px;")
        top_layout.addWidget(self.combo_cap_type)

        # Spatial Referencing Selector
        top_layout.addWidget(QtWidgets.QLabel("Spatial Ref:"))
        self.combo_spatial_ref = QtWidgets.QComboBox()
        self.combo_spatial_ref.addItem("Robust CAR (Median)", "robust_car")
        self.combo_spatial_ref.addItem("Standard CAR (Mean)", "car")
        self.combo_spatial_ref.addItem("Surface Laplacian (Hjorth)", "laplacian")
        self.combo_spatial_ref.addItem("Raw / Mastoid (None)", "raw")
        self.combo_spatial_ref.setStyleSheet("background-color: #191E2A; color: #4DEEEA; font-weight: bold; padding: 4px 8px; border: 1px solid #2C354A; border-radius: 4px;")
        top_layout.addWidget(self.combo_spatial_ref)

        self.btn_streamer = QtWidgets.QPushButton("▶ Launch Hardware EEG Streamer")
        self.btn_streamer.setStyleSheet("background-color: #2980B9; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_streamer.clicked.connect(self.toggle_eeg_streamer)
        top_layout.addWidget(self.btn_streamer)

        self.btn_mock_streamer = QtWidgets.QPushButton("🧪 Mock EEG Streamer")
        self.btn_mock_streamer.setStyleSheet("background-color: #27AE60; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_mock_streamer.clicked.connect(self.toggle_mock_streamer)
        top_layout.addWidget(self.btn_mock_streamer)

        self.btn_watch_bridge = QtWidgets.QPushButton("⌚ Launch Watch Bridge")
        self.btn_watch_bridge.setStyleSheet("background-color: #8E44AD; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_watch_bridge.clicked.connect(self.toggle_watch_bridge)
        top_layout.addWidget(self.btn_watch_bridge)

        self.lbl_status_eeg = QtWidgets.QLabel("EEG: 🔴 Offline")
        self.lbl_status_eeg.setStyleSheet("color: #E74C3C; font-weight: bold;")
        top_layout.addWidget(self.lbl_status_eeg)

        self.lbl_status_watch = QtWidgets.QLabel("Watch: 🔴 Offline")
        self.lbl_status_watch.setStyleSheet("color: #E74C3C; font-weight: bold;")
        top_layout.addWidget(self.lbl_status_watch)

        main_layout.addWidget(top_bar)

        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tab_headmap = self.build_headmap_tab()
        self.tabs.addTab(self.tab_headmap, "🧠 10-20 Headmap & Electrode Quality")

        self.tab_waves = self.build_waves_tab()
        self.tabs.addTab(self.tab_waves, "⚡ Motor Waves & Brain Rhythms")

        self.tab_watch = self.build_watch_tab()
        self.tabs.addTab(self.tab_watch, "⌚ Smartwatch PPG & IMU Vectors")

        main_layout.addWidget(self.tabs)

    def build_headmap_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)

        self.headmap_canvas = HeadMapCanvas(self.eeg_ch_names)
        layout.addWidget(self.headmap_canvas, stretch=3)

        right_panel = QtWidgets.QVBoxLayout()
        self.lbl_headmap_summary = QtWidgets.QLabel("Active Channels: Waiting for EEG stream...")
        self.lbl_headmap_summary.setStyleSheet("font-size: 15px; font-weight: bold; color: #ECF0F1;")
        right_panel.addWidget(self.lbl_headmap_summary)

        legend_layout = QtWidgets.QHBoxLayout()
        legend_layout.addWidget(self.create_legend_dot("POSITIONED / GOOD", "#2ECC71"))
        legend_layout.addWidget(self.create_legend_dot("HIGH NOISE / ARTIFACT", "#F1C40F"))
        legend_layout.addWidget(self.create_legend_dot("NO CONTACT / FLATLINE", "#E74C3C"))
        right_panel.addLayout(legend_layout)

        self.table_quality = QtWidgets.QTableWidget(len(self.eeg_ch_names) - 1, 3)
        self.table_quality.setHorizontalHeaderLabels(["Electrode", "Live Voltage Metrics", "Status"])
        self.table_quality.horizontalHeader().setStretchLastSection(True)

        for i, ch_name in enumerate(self.eeg_ch_names[:-1]):
            item_name = QtWidgets.QTableWidgetItem(ch_name)
            item_val = QtWidgets.QTableWidgetItem("std: 0.0uV | p-p: 0.0uV")
            item_status = QtWidgets.QTableWidgetItem("FLATLINE")
            item_status.setForeground(QtGui.QColor("#E74C3C"))

            self.table_quality.setItem(i, 0, item_name)
            self.table_quality.setItem(i, 1, item_val)
            self.table_quality.setItem(i, 2, item_status)

        right_panel.addWidget(self.table_quality)
        layout.addLayout(right_panel, stretch=2)
        return widget

    def create_legend_dot(self, text, color_hex):
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet(f"background-color: {color_hex}; color: #000; font-weight: bold; border-radius: 4px; padding: 4px;")
        return lbl

    def build_waves_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)

        self.glw_waves = pg.GraphicsLayoutWidget()
        self.glw_waves.setBackground('#0D1117')
        layout.addWidget(self.glw_waves, stretch=3)

        self.plot_waves = self.glw_waves.addPlot(title="Motor Cortex & Occipital Waves (Cz, C3, C4, O1, O2)")
        self.plot_waves.setLabel('bottom', 'Time (s)')
        self.plot_waves.setLabel('left', 'Amplitude (uV)')
        self.plot_waves.showGrid(x=True, y=True, alpha=0.3)

        self.target_wave_channels = ['Cz', 'C3', 'C4', 'O1', 'O2']
        self.wave_colors = ['#00E676', '#74B9FF', '#E040FB', '#FFEAA7', '#FF7675']
        self.wave_curves = []
        self.t_vec_eeg = np.linspace(-3.0, 0, self.buf_samples_eeg)

        for color, name in zip(self.wave_colors, self.target_wave_channels):
            c = self.plot_waves.plot(pen=pg.mkPen(color=color, width=2.0), name=name)
            self.wave_curves.append(c)

        right_panel = QtWidgets.QVBoxLayout()
        right_panel.setSpacing(10)
        right_panel.addWidget(QtWidgets.QLabel("<b>Live Brain Rhythm Power</b>"))

        right_panel.addWidget(QtWidgets.QLabel("<font color='#00E676'>Alpha (8-12 Hz) - Relaxation</font>"))
        self.bar_alpha = QtWidgets.QProgressBar()
        self.bar_alpha.setStyleSheet("QProgressBar::chunk { background: #00E676; }")
        right_panel.addWidget(self.bar_alpha)

        right_panel.addWidget(QtWidgets.QLabel("<font color='#74B9FF'>Beta (12-30 Hz) - Active Focus</font>"))
        self.bar_beta = QtWidgets.QProgressBar()
        self.bar_beta.setStyleSheet("QProgressBar::chunk { background: #74B9FF; }")
        right_panel.addWidget(self.bar_beta)

        right_panel.addWidget(QtWidgets.QLabel("<font color='#FFEAA7'>Delta (0.5-4 Hz) - Baseline</font>"))
        self.bar_delta = QtWidgets.QProgressBar()
        self.bar_delta.setStyleSheet("QProgressBar::chunk { background: #FFEAA7; }")
        right_panel.addWidget(self.bar_delta)

        right_panel.addStretch()
        layout.addLayout(right_panel, stretch=1)
        return widget

    def build_watch_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        hr_box = QtWidgets.QGroupBox("🫀 Live Smartwatch Heart Rate (PPG)")
        hr_layout = QtWidgets.QHBoxLayout(hr_box)

        self.lbl_hr_val = QtWidgets.QLabel("-- BPM")
        self.lbl_hr_val.setStyleSheet("font-size: 32px; font-weight: bold; color: #E74C3C;")
        hr_layout.addWidget(self.lbl_hr_val)

        self.lbl_hr_status = QtWidgets.QLabel("Status: Waiting for PPG stream...")
        self.lbl_hr_status.setStyleSheet("color: #BDC3C7;")
        hr_layout.addWidget(self.lbl_hr_status)
        layout.addWidget(hr_box)

        imu_box = QtWidgets.QGroupBox("⌚ 6-DOF IMU Motion Vectors (Accelerometer & Gyroscope)")
        imu_layout = QtWidgets.QVBoxLayout(imu_box)

        self.glw_imu = pg.GraphicsLayoutWidget()
        self.glw_imu.setBackground('#0D1117')
        imu_layout.addWidget(self.glw_imu)

        self.plot_accel = self.glw_imu.addPlot(row=0, col=0, title="3-Axis Accelerometer (m/s²)")
        self.plot_accel.showGrid(x=True, y=True, alpha=0.3)
        self.t_vec_imu = np.linspace(-3.0, 0, self.buf_samples_imu)

        self.curve_ax = self.plot_accel.plot(pen=pg.mkPen('#E74C3C', width=2), name="Accel X")
        self.curve_ay = self.plot_accel.plot(pen=pg.mkPen('#2ECC71', width=2), name="Accel Y")
        self.curve_az = self.plot_accel.plot(pen=pg.mkPen('#3498DB', width=2), name="Accel Z")

        self.plot_mag = self.glw_imu.addPlot(row=0, col=1, title="Total Motion Magnitude |A|")
        self.plot_mag.showGrid(x=True, y=True, alpha=0.3)
        self.curve_mag = self.plot_mag.plot(pen=pg.mkPen('#F1C40F', width=2), name="Magnitude")

        layout.addWidget(imu_box)
        return widget

    def toggle_eeg_streamer(self):
        if self.streamer_process is None:
            script_path = resolve_script_path("gds_to_lsl.py")
            selected_cap = self.combo_cap_type.currentData() or "wet"
            self.streamer_process = subprocess.Popen([sys.executable, script_path, "--non-interactive", "--cap-type", selected_cap])
            self.btn_streamer.setText("⏹ Stop Hardware Streamer")
            self.btn_streamer.setStyleSheet("background-color: #C0392B; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        else:
            self.streamer_process.terminate()
            self.streamer_process = None
            self.btn_streamer.setText("▶ Launch Hardware EEG Streamer")
            self.btn_streamer.setStyleSheet("background-color: #2980B9; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")

    def toggle_mock_streamer(self):
        if self.mock_streamer_process is None:
            script_path = resolve_script_path("mock_lsl_streamer.py")
            self.mock_streamer_process = subprocess.Popen([sys.executable, script_path])
            self.btn_mock_streamer.setText("⏹ Stop Mock Streamer")
            self.btn_mock_streamer.setStyleSheet("background-color: #C0392B; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        else:
            self.mock_streamer_process.terminate()
            self.mock_streamer_process = None
            self.btn_mock_streamer.setText("🧪 Mock EEG Streamer")
            self.btn_mock_streamer.setStyleSheet("background-color: #27AE60; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")

    def toggle_watch_bridge(self):
        if self.watch_bridge_process is None:
            script_path = resolve_script_path("smartwatch_lsl_bridge.py")
            self.watch_bridge_process = subprocess.Popen([sys.executable, script_path, "--mode", "udp", "--port", "5005"])
            self.btn_watch_bridge.setText("⏹ Stop Watch Bridge")
            self.btn_watch_bridge.setStyleSheet("background-color: #C0392B; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        else:
            self.watch_bridge_process.terminate()
            self.watch_bridge_process = None
            self.btn_watch_bridge.setText("⌚ Launch Watch Bridge")
            self.btn_watch_bridge.setStyleSheet("background-color: #8E44AD; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")

    # BIDS recording methods removed (managed by standalone recorder GUI)

    def discover_lsl_streams(self):
        if not HAS_LSL:
            return

        streams = resolve_streams(wait_time=0.2)
        for s in streams:
            stype = s.type().upper()
            sname = s.name()

            if (stype == 'EEG' or 'gNautilus' in sname) and self.eeg_inlet is None:
                try:
                    self.eeg_inlet = StreamInlet(s, max_buflen=360)
                except Exception:
                    pass

            if (stype == 'IMU' or 'Smartwatch_IMU' in sname) and self.imu_inlet is None:
                try:
                    self.imu_inlet = StreamInlet(s, max_buflen=360)
                except Exception:
                    pass

            if (stype == 'PPG' or 'Smartwatch_PPG' in sname) and self.ppg_inlet is None:
                try:
                    self.ppg_inlet = StreamInlet(s, max_buflen=360)
                except Exception:
                    pass

        if self.eeg_inlet:
            self.lbl_status_eeg.setText("EEG: 🟢 Connected")
            self.lbl_status_eeg.setStyleSheet("color: #2ECC71; font-weight: bold;")
        else:
            self.lbl_status_eeg.setText("EEG: 🔴 Offline")
            self.lbl_status_eeg.setStyleSheet("color: #E74C3C; font-weight: bold;")

        if self.imu_inlet or self.ppg_inlet:
            self.lbl_status_watch.setText("Watch: 🟢 Connected")
            self.lbl_status_watch.setStyleSheet("color: #2ECC71; font-weight: bold;")
        else:
            self.lbl_status_watch.setText("Watch: 🔴 Offline")
            self.lbl_status_watch.setStyleSheet("color: #E74C3C; font-weight: bold;")

    def poll_and_render_all(self):
        # 1. Poll EEG
        if self.eeg_inlet:
            try:
                samples, _ = self.eeg_inlet.pull_chunk(max_samples=250)
                if samples:
                    chunk = np.array(samples, dtype=np.float64)
                    if np.max(np.abs(chunk)) < 0.01:
                        chunk *= 1e6  # Volts to uV
                    n = len(chunk)
                    if chunk.shape[1] == self.eeg_buffer.shape[1]:
                        self.eeg_buffer = np.roll(self.eeg_buffer, -n, axis=0)
                        self.eeg_buffer[-n:, :] = chunk
            except Exception:
                self.eeg_inlet = None

        # 2. Poll IMU
        if self.imu_inlet:
            try:
                samples, _ = self.imu_inlet.pull_chunk(max_samples=100)
                if samples:
                    chunk = np.array(samples, dtype=np.float64)
                    n = len(chunk)
                    if chunk.shape[1] >= 6:
                        self.imu_buffer = np.roll(self.imu_buffer, -n, axis=0)
                        self.imu_buffer[-n:, :] = chunk[:, :6]
            except Exception:
                self.imu_inlet = None

        # 3. Poll PPG / Heart Rate
        if self.ppg_inlet:
            try:
                samples, _ = self.ppg_inlet.pull_chunk(max_samples=10)
                if samples:
                    hr_val = samples[-1][0]
                    if hr_val > 0:
                        self.lbl_hr_val.setText(f"{hr_val:.0f} BPM")
                        self.lbl_hr_status.setText("Status: Live PPG Stream Active 🟢")
                    else:
                        self.lbl_hr_val.setText("Locking...")
                        self.lbl_hr_status.setText("Status: PPG Active — Searching for Pulse Lock 🟡")
            except Exception:
                self.ppg_inlet = None

        # Render active tab
        idx = self.tabs.currentIndex()
        if idx == 0:
            self.render_headmap_tab()
        elif idx == 1:
            self.render_waves_tab()
        elif idx == 2:
            self.render_watch_tab()

    def render_headmap_tab(self):
        # Apply bandpass (AC component) to assess electrode scalp coupling independently of DC offset
        ac_eeg = self.eeg_buffer - np.mean(self.eeg_buffer, axis=0)
        try:
            ac_eeg = signal.filtfilt(self.b_bp, self.a_bp, ac_eeg, axis=0)
        except Exception:
            pass

        stds = np.std(ac_eeg, axis=0)
        ranges = np.ptp(ac_eeg, axis=0)

        channel_stds = {}
        channel_maxs = {}
        good_count = 0
        flat_count = 0
        is_dry = (self.combo_cap_type.currentData() == "dry")

        # Thresholds adapted for dry pins vs wet gel
        rail_mean_thresh = 600.0 if is_dry else 300.0
        rail_ptp_thresh = 1000.0 if is_dry else 500.0
        noise_std_thresh = 120.0 if is_dry else 80.0

        # Raw metrics directly from API buffer (DC offset & true raw range)
        raw_offsets = np.mean(self.eeg_buffer, axis=0)
        raw_ptps = np.ptp(self.eeg_buffer, axis=0)

        for i, name in enumerate(self.eeg_ch_names[:-1]):
            std_val = stds[i]
            ptp_val = ranges[i]
            raw_dc = raw_offsets[i]
            raw_p2p = raw_ptps[i]

            channel_stds[name] = std_val
            channel_maxs[name] = ptp_val

            item_val = self.table_quality.item(i, 1)
            if item_val:
                item_val.setText(f"DC: {raw_dc:+.1f} uV | Raw p-p: {raw_p2p:.1f} uV | AC RMS: {std_val:.1f} uV")

            if std_val < 0.2:
                item = QtWidgets.QTableWidgetItem("NO CONTACT / FLAT")
                item.setForeground(QtGui.QColor("#E74C3C"))
                flat_count += 1
            elif abs(raw_dc) > (rail_mean_thresh * 1000.0) or raw_p2p > (rail_ptp_thresh * 1000.0):
                item = QtWidgets.QTableWidgetItem("RAILED/SATURATED")
                item.setForeground(QtGui.QColor("#E67E22"))
            elif std_val > noise_std_thresh:
                item = QtWidgets.QTableWidgetItem("HIGH NOISE / ARTIFACT")
                item.setForeground(QtGui.QColor("#F1C40F"))
            else:
                item = QtWidgets.QTableWidgetItem("POSITIONED / OK")
                item.setForeground(QtGui.QColor("#2ECC71"))
                good_count += 1

            self.table_quality.setItem(i, 2, item)

        self.headmap_canvas.update_channel_status(channel_stds, channel_maxs)
        cap_name = "Dry (SAHARA)" if is_dry else "Wet"
        self.lbl_headmap_summary.setText(f"Active Channels ({cap_name}): {good_count}/32 Positioned ({flat_count} Disconnected)")

    def render_waves_tab(self):
        # 1. Apply selected Spatial Reference (Robust CAR, Surface Laplacian, Standard CAR, or Raw)
        ref_mode = self.combo_spatial_ref.currentData() or "robust_car"
        cap_mode = self.combo_cap_type.currentData() or "wet"
        filt = apply_spatial_filter(self.eeg_buffer, self.eeg_ch_names, mode=ref_mode, cap_type=cap_mode)

        # 2. Apply Temporal Bandpass Filter
        try:
            filt = signal.filtfilt(self.b_bp, self.a_bp, filt, axis=0)
        except Exception:
            pass

        spacing = max(20.0, np.std(filt) * 3.5)
        offsets = [0, spacing, spacing * 2, spacing * 3, spacing * 4]

        for i, name in enumerate(self.target_wave_channels):
            if name in self.eeg_ch_names:
                ch_idx = self.eeg_ch_names.index(name)
                y = filt[:, ch_idx] + offsets[i]
                self.wave_curves[i].setData(self.t_vec_eeg, y)

        self.plot_waves.setYRange(-spacing * 0.8, spacing * 4.8)

        if len(filt) >= 64:
            fft_vals = np.abs(np.fft.rfft(filt, axis=0))
            freqs = np.fft.rfftfreq(len(filt), 1.0 / self.fs_eeg)

            alpha = np.mean(fft_vals[(freqs >= 8) & (freqs <= 12), :]) if np.any((freqs >= 8) & (freqs <= 12)) else 0
            beta = np.mean(fft_vals[(freqs >= 12) & (freqs <= 30), :]) if np.any((freqs >= 12) & (freqs <= 30)) else 0
            delta = np.mean(fft_vals[(freqs >= 0.5) & (freqs <= 4), :]) if np.any((freqs >= 0.5) & (freqs <= 4)) else 0

            tot = alpha + beta + delta + 1e-6
            self.bar_alpha.setValue(int(min(100, (alpha / tot) * 100)))
            self.bar_beta.setValue(int(min(100, (beta / tot) * 100)))
            self.bar_delta.setValue(int(min(100, (delta / tot) * 100)))

    def render_watch_tab(self):
        ax = self.imu_buffer[:, 0]
        ay = self.imu_buffer[:, 1]
        az = self.imu_buffer[:, 2]
        mag = np.sqrt(ax**2 + ay**2 + az**2)

        self.curve_ax.setData(self.t_vec_imu, ax)
        self.curve_ay.setData(self.t_vec_imu, ay)
        self.curve_az.setData(self.t_vec_imu, az)
        self.curve_mag.setData(self.t_vec_imu, mag)

    def closeEvent(self, event):
        if self.streamer_process:
            self.streamer_process.terminate()
        if self.watch_bridge_process:
            self.watch_bridge_process.terminate()
        event.accept()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Multimodal BCI Master Dashboard")
    parser.add_argument("--bids-root", "--dataset-folder", type=str, default="bids_dataset_multimodal", help="Target BIDS dataset output directory")
    args, unknown = parser.parse_known_args()

    app = QtWidgets.QApplication(sys.argv)
    window = MultimodalBCIDashboard(initial_bids_root=args.bids_root)
    window.show()
    sys.exit(app.exec())
