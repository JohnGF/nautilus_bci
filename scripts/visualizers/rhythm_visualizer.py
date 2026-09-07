import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

import os
import sys
import time
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
from pylsl import StreamInlet, resolve_byprop
import scipy.signal as signal
import wave
import winsound
import threading
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler

# ==============================================================================
# DESIGN CONFIGURATION & PALETTE
# ==============================================================================
COLOR_BG = "#0f172a"          # slate-900 (main window background)
COLOR_CARD = "#1e293b"        # slate-800 (panels, controls)
COLOR_TEXT = "#f8fafc"        # slate-50 (primary text)
COLOR_MUTED = "#94a3b8"       # slate-400 (secondary text/labels)
COLOR_BORDER = "#334155"      # slate-700 (borders)

COLOR_DELTA = "#3b82f6"       # Bright Blue (1-4 Hz)
COLOR_THETA = "#10b981"       # Emerald Green (4-8 Hz)
COLOR_ALPHA = "#f59e0b"       # Amber Gold (8-12 Hz)
COLOR_BETA = "#ef4444"        # Rose Red (12-30 Hz)
COLOR_GAMMA = "#8b5cf6"       # Violet Purple (30-45 Hz)

QSS_STYLESHEET = f"""
QMainWindow {{
    background-color: {COLOR_BG};
}}

QWidget {{
    color: {COLOR_TEXT};
    font-family: 'Segoe UI', Arial, sans-serif;
}}

QGroupBox {{
    background-color: {COLOR_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    margin-top: 15px;
    font-weight: bold;
    font-size: 13px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    left: 10px;
    color: {COLOR_ALPHA};
}}

QLabel {{
    color: {COLOR_TEXT};
}}

QLabel#muted {{
    color: {COLOR_MUTED};
    font-size: 11px;
}}

QComboBox {{
    background-color: {COLOR_BG};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    padding: 5px;
    min-width: 100px;
    color: {COLOR_TEXT};
}}

QComboBox::drop-down {{
    border: none;
}}

QCheckBox {{
    spacing: 5px;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {COLOR_BORDER};
    border-radius: 3px;
    background-color: {COLOR_BG};
}}

QCheckBox::indicator:checked {{
    background-color: {COLOR_THETA};
    border: 1px solid {COLOR_THETA};
}}

QPushButton {{
    background-color: {COLOR_BORDER};
    border: 1px solid {COLOR_BORDER};
    border-radius: 5px;
    padding: 6px 12px;
    font-weight: bold;
    min-height: 20px;
}}

QPushButton:hover {{
    background-color: #475569;
}}

QPushButton:pressed {{
    background-color: #1e293b;
}}

QPushButton#action-btn {{
    background-color: {COLOR_THETA};
    color: #ffffff;
    border: none;
}}

QPushButton#action-btn:hover {{
    background-color: #059669;
}}

QPushButton#blink-btn {{
    background-color: {COLOR_DELTA};
    color: white;
}}

QPushButton#blink-btn:hover {{
    background-color: #2563eb;
}}

QPushButton#muscle-btn {{
    background-color: {COLOR_GAMMA};
    color: white;
}}

QPushButton#muscle-btn:hover {{
    background-color: #7c3aed;
}}

QSlider::groove:horizontal {{
    border: 1px solid {COLOR_BORDER};
    height: 6px;
    background: {COLOR_BG};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {COLOR_ALPHA};
    border: none;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}}

QSlider::groove:vertical {{
    border: 1px solid {COLOR_BORDER};
    width: 6px;
    background: {COLOR_BG};
    border-radius: 3px;
}}

QSlider::handle:vertical {{
    background: {COLOR_ALPHA};
    border: none;
    width: 14px;
    height: 14px;
    margin: 0 -4px;
    border-radius: 7px;
}}

QProgressBar {{
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    background-color: {COLOR_BG};
    text-align: center;
    font-weight: bold;
}}

QProgressBar::chunk {{
    border-radius: 3px;
}}
"""

DEFAULT_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
    'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 
    'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 
    'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10'
]

OCCIPITAL_INDICES = [8, 9, 18, 19, 16, 17] # O1, O2, Pz, Oz, Fz, Cz
FRONTAL_INDICES = [0, 1, 2, 3, 10, 11] # Fp1, Fp2, F3, F4, F7, F8
TEMPORAL_INDICES = [12, 13, 28, 29] # T7, T8, FT9, FT10

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

class WebServerSignals(QtCore.QObject):
    trigger_record = QtCore.Signal()
    trigger_play = QtCore.Signal()
    trigger_stop = QtCore.Signal()

class RemoteRequestHandler(BaseHTTPRequestHandler):
    signals = None
    app_ref = None
    
    def log_message(self, format, *args):
        # Suppress logging server traffic to keep terminal clean
        pass
        
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            html = self.app_ref.get_mobile_html()
            self.wfile.write(html.encode('utf-8'))
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            import json
            l_idx = self.app_ref.rec_channel_combo_l.currentIndex()
            r_idx = self.app_ref.rec_channel_combo_r.currentIndex()
            l_name = self.app_ref.rec_channel_combo_l.itemText(l_idx)
            r_name = self.app_ref.rec_channel_combo_r.itemText(r_idx)
            status = {
                'recording': self.app_ref.recording_active,
                'playing': self.app_ref.playback_timer.isActive(),
                'duration': self.app_ref.recording_duration,
                'left_channel': l_name,
                'right_channel': r_name
            }
            self.wfile.write(json.dumps(status).encode('utf-8'))
        elif self.path == '/api/record':
            self.signals.trigger_record.emit()
            self.send_ok_response()
        elif self.path == '/api/play':
            self.signals.trigger_play.emit()
            self.send_ok_response()
        elif self.path == '/api/stop':
            self.signals.trigger_stop.emit()
            self.send_ok_response()
        else:
            self.send_error(404, "Not Found")
            
    def send_ok_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

class RhythmVisualizerApp(QtWidgets.QMainWindow):
    def __init__(self, inlet=None):
        super().__init__()
        self.inlet = inlet
        
        # Internal configuration
        self.fs = 250.0 # Default sampling rate for g.Nautilus
        self.num_channels = len(DEFAULT_CHANNELS)
        self.channel_names = DEFAULT_CHANNELS
        
        # Check if we have an LSL inlet
        if self.inlet is not None:
            info = self.inlet.info()
            self.fs = info.nominal_srate()
            lsl_chans = info.channel_count()
            temp_names = []
            ch = info.desc().child("channels").child("channel")
            for i in range(lsl_chans):
                if ch.empty():
                    temp_names.append(f"CH {i+1}")
                else:
                    temp_names.append(ch.child_value("label"))
                    ch = ch.next_sibling()
            
            if len(temp_names) > 0 and temp_names[-1].upper() == 'BATTERY':
                self.channel_names = temp_names[:-1]
                self.num_channels = len(self.channel_names)
            else:
                self.channel_names = temp_names
                self.num_channels = len(temp_names)
            
            self.simulation_mode = False
        else:
            self.simulation_mode = True

        # Signal Buffers
        self.buffer_duration = 3.0 # 3 seconds buffer for FFT
        self.buffer_samples = int(self.fs * self.buffer_duration)
        self.data_buffer = np.zeros((self.buffer_samples, self.num_channels))
        self.time_axis = np.linspace(-self.buffer_duration, 0, self.buffer_samples)
        
        # History buffers for challenges and rolling plots
        self.history_len = 100
        self.alpha_history = np.zeros(self.history_len)
        self.beta_history = np.zeros(self.history_len)
        
        # Current active channel
        self.current_channel_idx = 0
        
        # Simulator state variables
        self.sim_sample_count = 0
        self.sim_alpha_amp = 15.0 # uV
        self.sim_beta_amp = 8.0 # uV
        self.sim_noise_amp = 2.0 # uV
        self.sim_powerline_active = True
        self.blink_active_samples = 0
        self.muscle_active_samples = 0
        
        # Challenge / Game variables
        self.current_challenge = "Alpha Relaxation"
        self.challenge_score = 0.0
        self.challenge_success_threshold = 5.0
        self.challenge_completed = False
        self.last_challenge_update = time.time()
        
        # Recording status
        self.recording_active = False
        self.recording_data = []
        self.recording_duration = 10.0 # default seconds
        self.recording_target_samples = int(self.fs * self.recording_duration)
        self.recording_filename = ""
        self.raw_recorded_data = None  # Cache for raw multi-channel data
        self.rec_idx_l = 0
        self.rec_idx_r = 0
        
        # Initialize DSP Filters (4th-order Butterworth, notch 50Hz)
        self.filters_enabled = True
        self.notch_enabled = True
        self.setup_filters()
        
        # Set up GUI
        self.init_ui()
        self.setStyleSheet(QSS_STYLESHEET)
        
        # Set up update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.on_timer_tick)
        self.timer.start(40) # 25 Hz update rate
        
        # Set up playback timer (playhead sync)
        self.playback_timer = QtCore.QTimer()
        self.playback_timer.timeout.connect(self.update_playhead)
        
        # Start Remote Control HTTP Server
        self.web_signals = WebServerSignals()
        self.web_signals.trigger_record.connect(self.toggle_recording)
        self.web_signals.trigger_play.connect(self.play_recorded_audio)
        self.web_signals.trigger_stop.connect(self.stop_recorded_audio)
        self.start_web_server()
        
    def setup_filters(self):
        nyq = 0.5 * self.fs
        # 1.0 - 45.0 Hz Bandpass
        self.b_band, self.a_band = signal.butter(4, [1.0 / nyq, 45.0 / nyq], btype='band')
        # 50 Hz Notch filter
        self.b_notch, self.a_notch = signal.iirnotch(50.0, 30.0, self.fs)
        
    def start_web_server(self):
        RemoteRequestHandler.signals = self.web_signals
        RemoteRequestHandler.app_ref = self
        
        self.web_port = 8080
        self.local_ip = get_local_ip()
        
        def run_server():
            try:
                self.httpd = HTTPServer(('0.0.0.0', self.web_port), RemoteRequestHandler)
                print(f"\n[Remote Control] Server active! Connect your phone to: http://{self.local_ip}:{self.web_port}")
                print(f"[Remote Control] Allows triggering recordings remotely from 1.5+ meters away.\n")
                self.httpd.serve_forever()
            except Exception as e:
                print(f"[-] Remote Control Web Server failed to start: {e}")
                
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

    def get_mobile_html(self):
        return """<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>BCI Remote Control</title>
    <style>
        body {
            background-color: #0f172a;
            color: #f8fafc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 90vh;
        }
        .container {
            width: 100%;
            max-width: 400px;
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 24px;
            box-sizing: border-box;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        }
        h1 {
            font-size: 20px;
            margin-top: 0;
            margin-bottom: 8px;
            text-align: center;
            color: #f59e0b;
        }
        .status-badge {
            background-color: #334155;
            padding: 8px 16px;
            border-radius: 9999px;
            font-size: 14px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 24px;
            color: #94a3b8;
            transition: all 0.3s ease;
        }
        .btn {
            display: block;
            width: 100%;
            padding: 18px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            color: white;
            cursor: pointer;
            margin-bottom: 12px;
            text-align: center;
            box-sizing: border-box;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            transition: transform 0.1s, filter 0.2s;
        }
        .btn:active {
            transform: scale(0.98);
        }
        .btn-record {
            background-color: #ef4444;
        }
        .btn-record.active {
            background-color: #dc2626;
            animation: pulse 1.5s infinite;
        }
        .btn-play {
            background-color: #10b981;
        }
        .btn-stop {
            background-color: #475569;
        }
        .info-panel {
            background-color: #0f172a;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 12px;
            margin-top: 16px;
            font-size: 13px;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
        }
        .info-row:last-child {
            margin-bottom: 0;
        }
        .info-label {
            color: #94a3b8;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.6; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 BCI Remote Controller</h1>
        <div id="status" class="status-badge">CONNECTED</div>
        
        <button id="record-btn" class="btn btn-record" onclick="trigger('/api/record')">🔴 START RECORDING</button>
        <button id="play-btn" class="btn btn-play" onclick="trigger('/api/play')">▶️ PLAY AUDIO</button>
        <button id="stop-btn" class="btn btn-stop" onclick="trigger('/api/stop')">⏹️ STOP PLAYBACK</button>
        
        <div class="info-panel">
            <div class="info-row">
                <span class="info-label">Left Channel:</span>
                <span id="info-l">-</span>
            </div>
            <div class="info-row">
                <span class="info-label">Right Channel:</span>
                <span id="info-r">-</span>
            </div>
            <div class="info-row">
                <span class="info-label">Duration:</span>
                <span id="info-dur">-</span>
            </div>
        </div>
    </div>

    <script>
        function trigger(endpoint) {
            fetch(endpoint)
                .then(res => res.json())
                .then(data => {
                    pollStatus();
                });
        }
        
        function pollStatus() {
            fetch('/api/status')
                .then(res => res.json())
                .then(data => {
                    const statusBadge = document.getElementById('status');
                    const recordBtn = document.getElementById('record-btn');
                    
                    if (data.recording) {
                        statusBadge.innerText = '🔴 RECORDING ACTIVE';
                        statusBadge.style.color = '#ffffff';
                        statusBadge.style.backgroundColor = '#ef4444';
                        recordBtn.innerText = '⏹️ STOP RECORDING';
                        recordBtn.classList.add('active');
                    } else {
                        statusBadge.innerText = data.playing ? '🔊 PLAYING AUDIO' : '📡 STANDBY';
                        statusBadge.style.color = data.playing ? '#ffffff' : '#94a3b8';
                        statusBadge.style.backgroundColor = data.playing ? '#10b981' : '#334155';
                        recordBtn.innerText = '🔴 START RECORDING';
                        recordBtn.classList.remove('active');
                    }
                    
                    document.getElementById('info-l').innerText = data.left_channel;
                    document.getElementById('info-r').innerText = data.right_channel;
                    document.getElementById('info-dur').innerText = data.duration + 's';
                })
                .catch(err => {
                    document.getElementById('status').innerText = '❌ DISCONNECTED';
                    document.getElementById('status').style.color = '#ef4444';
                    document.getElementById('status').style.backgroundColor = '#450a0a';
                });
        }
        
        setInterval(pollStatus, 800);
        pollStatus();
    </script>
</body>
</html>"""

    def init_ui(self):
        self.setWindowTitle("g.Nautilus Brainwave Rhythm Explorer & Simulator")
        self.resize(1350, 850)
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # ======================================================================
        # 1. Header Bar
        # ======================================================================
        header_widget = QtWidgets.QWidget()
        header_widget.setStyleSheet(f"background-color: {COLOR_CARD}; border-radius: 8px; border: 1px solid {COLOR_BORDER};")
        header_layout = QtWidgets.QHBoxLayout(header_widget)
        header_layout.setContentsMargins(15, 10, 15, 10)
        
        title_label = QtWidgets.QLabel("🧠 Brain Rhythm Explorer & Recorder")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff;")
        header_layout.addWidget(title_label)
        
        subtitle_label = QtWidgets.QLabel("Interactive BCI Lab Tool for Students")
        subtitle_label.setStyleSheet(f"color: {COLOR_MUTED}; font-size: 12px; margin-left: 10px;")
        header_layout.addWidget(subtitle_label)
        
        header_layout.addStretch()
        
        self.status_indicator = QtWidgets.QLabel()
        self.update_status_indicator()
        header_layout.addWidget(self.status_indicator)
        
        self.source_checkbox = QtWidgets.QCheckBox("Simulation Mode")
        self.source_checkbox.setChecked(self.simulation_mode)
        self.source_checkbox.stateChanged.connect(self.on_source_changed)
        header_layout.addWidget(self.source_checkbox)
        
        main_layout.addWidget(header_widget)
        
        # ======================================================================
        # 2. Main Tab Widget
        # ======================================================================
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {COLOR_BORDER};
                background: {COLOR_BG};
                border-radius: 8px;
            }}
            QTabBar::tab {{
                background: {COLOR_CARD};
                border: 1px solid {COLOR_BORDER};
                padding: 10px 20px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
                font-size: 13px;
                color: {COLOR_MUTED};
            }}
            QTabBar::tab:selected, QTabBar::tab:hover {{
                background: {COLOR_BG};
                color: {COLOR_TEXT};
                border-bottom-color: {COLOR_BG};
            }}
        """)
        main_layout.addWidget(self.tabs)
        
        # ----------------------------------------------------------------------
        # TAB 1: Live Rhythm Explorer
        # ----------------------------------------------------------------------
        tab1 = QtWidgets.QWidget()
        tab1_layout = QtWidgets.QHBoxLayout(tab1)
        tab1_layout.setContentsMargins(5, 5, 5, 5)
        tab1_layout.setSpacing(10)
        
        # Left panel: controls
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(280)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        # Card 1: Channel & Processing Controls
        proc_group = QtWidgets.QGroupBox("Channel & Signal Settings")
        proc_layout = QtWidgets.QVBoxLayout(proc_group)
        proc_layout.setContentsMargins(12, 15, 12, 12)
        proc_layout.setSpacing(10)
        
        proc_layout.addWidget(QtWidgets.QLabel("Select Electrode / Channel:"))
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems(self.channel_names)
        self.channel_combo.currentIndexChanged.connect(self.on_channel_changed)
        proc_layout.addWidget(self.channel_combo)
        
        self.filter_band_cb = QtWidgets.QCheckBox("Bandpass Filter (1-45 Hz)")
        self.filter_band_cb.setChecked(self.filters_enabled)
        self.filter_band_cb.stateChanged.connect(self.on_filter_toggled)
        proc_layout.addWidget(self.filter_band_cb)
        
        self.filter_notch_cb = QtWidgets.QCheckBox("Notch Filter (50 Hz)")
        self.filter_notch_cb.setChecked(self.notch_enabled)
        self.filter_notch_cb.stateChanged.connect(self.on_filter_toggled)
        proc_layout.addWidget(self.filter_notch_cb)
        
        proc_layout.addWidget(QtWidgets.QLabel("Waveform Vertical Zoom:"))
        self.zoom_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.zoom_slider.setRange(10, 200)
        self.zoom_slider.setValue(60)
        proc_layout.addWidget(self.zoom_slider)
        
        left_layout.addWidget(proc_group)
        
        # Card 2: Interactive EEG Simulator
        self.sim_group = QtWidgets.QGroupBox("Interactive BCI Simulator")
        sim_layout = QtWidgets.QVBoxLayout(self.sim_group)
        sim_layout.setContentsMargins(12, 15, 12, 12)
        sim_layout.setSpacing(8)
        
        sim_info = QtWidgets.QLabel("Alter brain rhythms below to see how they impact the frequency spectrum and classifier.")
        sim_info.setWordWrap(True)
        sim_info.setObjectName("muted")
        sim_layout.addWidget(sim_info)
        
        sim_layout.addWidget(QtWidgets.QLabel("Alpha Rhythm (Relaxation/Closed Eyes):"))
        self.sim_alpha_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sim_alpha_slider.setRange(0, 50)
        self.sim_alpha_slider.setValue(int(self.sim_alpha_amp))
        self.sim_alpha_slider.valueChanged.connect(self.on_sim_params_changed)
        sim_layout.addWidget(self.sim_alpha_slider)
        
        sim_layout.addWidget(QtWidgets.QLabel("Beta Rhythm (Active Concentration):"))
        self.sim_beta_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sim_beta_slider.setRange(0, 50)
        self.sim_beta_slider.setValue(int(self.sim_beta_amp))
        self.sim_beta_slider.valueChanged.connect(self.on_sim_params_changed)
        sim_layout.addWidget(self.sim_beta_slider)
        
        sim_layout.addWidget(QtWidgets.QLabel("Baseline Neural Noise:"))
        self.sim_noise_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sim_noise_slider.setRange(1, 20)
        self.sim_noise_slider.setValue(int(self.sim_noise_amp * 2))
        self.sim_noise_slider.valueChanged.connect(self.on_sim_params_changed)
        sim_layout.addWidget(self.sim_noise_slider)
        
        self.sim_powerline_cb = QtWidgets.QCheckBox("Add 50 Hz Powerline Noise")
        self.sim_powerline_cb.setChecked(self.sim_powerline_active)
        self.sim_powerline_cb.stateChanged.connect(self.on_sim_params_changed)
        sim_layout.addWidget(self.sim_powerline_cb)
        
        sim_layout.addWidget(QtWidgets.QLabel("Trigger Artifacts:"))
        
        self.blink_btn = QtWidgets.QPushButton("Trigger Eye Blink (Delta)")
        self.blink_btn.setObjectName("blink-btn")
        self.blink_btn.clicked.connect(self.trigger_eye_blink)
        sim_layout.addWidget(self.blink_btn)
        
        self.muscle_btn = QtWidgets.QPushButton("Trigger Muscle Clench (Gamma)")
        self.muscle_btn.setObjectName("muscle-btn")
        self.muscle_btn.clicked.connect(self.trigger_muscle_clench)
        sim_layout.addWidget(self.muscle_btn)
        
        left_layout.addWidget(self.sim_group)
        left_layout.addStretch()
        tab1_layout.addWidget(left_panel)
        
        # Center panel: plots
        center_panel = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(10)
        
        self.win = pg.GraphicsLayoutWidget(title="Brainwave Signal Analysis")
        self.win.setBackground(COLOR_BG)
        center_layout.addWidget(self.win)
        
        self.plot_time = self.win.addPlot(row=0, col=0, title="Selected Channel Brainwaves (uV)")
        self.plot_time.setLabel('bottom', 'Time', units='s')
        self.plot_time.setLabel('left', 'Amplitude', units='uV')
        self.plot_time.showGrid(x=True, y=True, alpha=0.2)
        self.plot_time.setXRange(-self.buffer_duration, 0)
        
        self.curve_raw = self.plot_time.plot(pen=pg.mkPen('#e74c3c', width=0.8), name="Raw")
        self.curve_filtered = self.plot_time.plot(pen=pg.mkPen('#2ecc71', width=1.5), name="Filtered")
        self.plot_time.addLegend(offset=(-30, 30))
        
        self.plot_freq = self.win.addPlot(row=1, col=0, title="Frequency Spectrum (FFT Power)")
        self.plot_freq.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_freq.setLabel('left', 'Power Spectral Density')
        self.plot_freq.showGrid(x=True, y=True, alpha=0.2)
        self.plot_freq.setXRange(0.5, 45.0)
        
        self.band_regions = []
        bands_def = [
            ("Delta", 0.5, 4, COLOR_DELTA),
            ("Theta", 4, 8, COLOR_THETA),
            ("Alpha", 8, 12, COLOR_ALPHA),
            ("Beta", 12, 30, COLOR_BETA),
            ("Gamma", 30, 45, COLOR_GAMMA)
        ]
        for name, low, high, color in bands_def:
            lr = pg.LinearRegionItem(values=[low, high], brush=pg.mkBrush(color + "15"), movable=False)
            self.plot_freq.addItem(lr)
            text = pg.TextItem(name, color=color, anchor=(0.5, 0))
            text.setPos((low + high) / 2.0, 50.0)
            self.plot_freq.addItem(text)
            self.band_regions.append((low, high, text))
            
        self.curve_fft = self.plot_freq.plot(pen=pg.mkPen('#ffffff', width=1.5))
        tab1_layout.addWidget(center_panel, stretch=2)
        
        # Right panel: bars and biofeedback game
        right_panel = QtWidgets.QWidget()
        right_panel.setFixedWidth(300)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        
        power_group = QtWidgets.QGroupBox("Live Brain Rhythms Ratio (%)")
        power_layout = QtWidgets.QVBoxLayout(power_group)
        power_layout.setContentsMargins(15, 20, 15, 15)
        power_layout.setSpacing(10)
        
        self.bars = {}
        rhythms = [
            ("Delta", "Deep Sleep / Blinking (1-4 Hz)", COLOR_DELTA),
            ("Theta", "Drowsiness / Creative (4-8 Hz)", COLOR_THETA),
            ("Alpha", "Relaxed / Calm (8-12 Hz)", COLOR_ALPHA),
            ("Beta", "Focused / Thinking (12-30 Hz)", COLOR_BETA),
            ("Gamma", "High Alert / Muscle Tension (30-45 Hz)", COLOR_GAMMA)
        ]
        for key, desc, color in rhythms:
            row_layout = QtWidgets.QVBoxLayout()
            row_layout.setSpacing(2)
            
            lbl_layout = QtWidgets.QHBoxLayout()
            lbl_title = QtWidgets.QLabel(f"<b>{key}</b>")
            lbl_title.setStyleSheet(f"color: {color}; font-size: 12px;")
            lbl_desc = QtWidgets.QLabel(desc)
            lbl_desc.setObjectName("muted")
            
            lbl_layout.addWidget(lbl_title)
            lbl_layout.addWidget(lbl_desc)
            lbl_layout.addStretch()
            
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(14)
            bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
            
            row_layout.addLayout(lbl_layout)
            row_layout.addWidget(bar)
            power_layout.addLayout(row_layout)
            
            self.bars[key] = bar
            
        right_layout.addWidget(power_group)
        
        state_group = QtWidgets.QGroupBox("Cognitive State Interpretation")
        state_layout = QtWidgets.QVBoxLayout(state_group)
        state_layout.setContentsMargins(15, 18, 15, 15)
        
        self.state_card = QtWidgets.QWidget()
        self.state_card.setStyleSheet(f"background-color: {COLOR_BG}; border-radius: 6px; border: 1px solid {COLOR_BORDER};")
        self.state_card_layout = QtWidgets.QVBoxLayout(self.state_card)
        self.state_card_layout.setContentsMargins(15, 15, 15, 15)
        
        self.state_title_lbl = QtWidgets.QLabel("ANALYZING...")
        self.state_title_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        self.state_card_layout.addWidget(self.state_title_lbl)
        
        self.state_desc_lbl = QtWidgets.QLabel("Acquiring brain signals to classify cognitive state...")
        self.state_desc_lbl.setWordWrap(True)
        self.state_desc_lbl.setStyleSheet(f"color: {COLOR_MUTED}; font-size: 11px;")
        self.state_card_layout.addWidget(self.state_desc_lbl)
        
        state_layout.addWidget(self.state_card)
        right_layout.addWidget(state_group)
        
        game_group = QtWidgets.QGroupBox("Zen Biofeedback Training")
        game_layout = QtWidgets.QVBoxLayout(game_group)
        game_layout.setContentsMargins(15, 15, 15, 15)
        game_layout.setSpacing(8)
        
        game_layout.addWidget(QtWidgets.QLabel("Select Biofeedback Challenge:"))
        self.challenge_combo = QtWidgets.QComboBox()
        self.challenge_combo.addItems(["Alpha Relaxation", "Beta Concentration Focus", "Perfect Signal Stability"])
        self.challenge_combo.currentIndexChanged.connect(self.on_challenge_changed)
        game_layout.addWidget(self.challenge_combo)
        
        self.game_instruction = QtWidgets.QLabel("Close eyes and relax to trigger Alpha waves. Hold above 35% for 5 seconds.")
        self.game_instruction.setWordWrap(True)
        self.game_instruction.setObjectName("muted")
        game_layout.addWidget(self.game_instruction)
        
        self.game_progress_bar = QtWidgets.QProgressBar()
        self.game_progress_bar.setRange(0, 100)
        self.game_progress_bar.setValue(0)
        self.game_progress_bar.setFormat("Target: 0% / Current: 0%")
        self.game_progress_bar.setFixedHeight(20)
        game_layout.addWidget(self.game_progress_bar)
        
        self.score_label = QtWidgets.QLabel("Time Held: 0.0s / 5.0s")
        self.score_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #ffffff;")
        self.score_label.setAlignment(QtCore.Qt.AlignCenter)
        game_layout.addWidget(self.score_label)
        
        self.success_widget = QtWidgets.QWidget()
        self.success_widget.setStyleSheet("background-color: #064e3b; border-radius: 4px; border: 1px solid #047857;")
        self.success_layout = QtWidgets.QHBoxLayout(self.success_widget)
        self.success_layout.setContentsMargins(8, 8, 8, 8)
        self.success_lbl = QtWidgets.QLabel("🏆 Challenge Completed!")
        self.success_lbl.setStyleSheet("font-weight: bold; color: #34d399;")
        self.success_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.success_layout.addWidget(self.success_lbl)
        self.success_widget.setVisible(False)
        game_layout.addWidget(self.success_widget)
        
        self.reset_game_btn = QtWidgets.QPushButton("Reset Challenge")
        self.reset_game_btn.clicked.connect(self.reset_challenge)
        game_layout.addWidget(self.reset_game_btn)
        
        right_layout.addWidget(game_group)
        right_layout.addStretch()
        tab1_layout.addWidget(right_panel)
        
        self.tabs.addTab(tab1, "📈 Live Rhythm Explorer")

        # ----------------------------------------------------------------------
        # TAB 2: Rhythm Sonifier & Recorder
        # ----------------------------------------------------------------------
        tab2 = QtWidgets.QWidget()
        tab2_layout = QtWidgets.QHBoxLayout(tab2)
        tab2_layout.setContentsMargins(5, 5, 5, 5)
        tab2_layout.setSpacing(10)
        
        # Tab 2 Left panel: Settings
        tab2_left = QtWidgets.QWidget()
        tab2_left.setFixedWidth(300)
        tab2_left_layout = QtWidgets.QVBoxLayout(tab2_left)
        tab2_left_layout.setContentsMargins(0, 0, 0, 0)
        tab2_left_layout.setSpacing(10)
        
        settings_group = QtWidgets.QGroupBox("Audio Generation Settings")
        settings_layout = QtWidgets.QVBoxLayout(settings_group)
        settings_layout.setContentsMargins(12, 15, 12, 12)
        settings_layout.setSpacing(10)
        
        settings_layout.addWidget(QtWidgets.QLabel("Left Ear Channel (Stereo L):"))
        self.rec_channel_combo_l = QtWidgets.QComboBox()
        self.rec_channel_combo_l.addItem("✨ Auto-Detect Left")
        self.rec_channel_combo_l.addItems(self.channel_names)
        settings_layout.addWidget(self.rec_channel_combo_l)
        
        settings_layout.addWidget(QtWidgets.QLabel("Right Ear Channel (Stereo R):"))
        self.rec_channel_combo_r = QtWidgets.QComboBox()
        self.rec_channel_combo_r.addItem("✨ Auto-Detect Right")
        self.rec_channel_combo_r.addItems(self.channel_names)
        self.rec_channel_combo_r.setCurrentIndex(2) # Default to Cz / second index
        settings_layout.addWidget(self.rec_channel_combo_r)
        
        settings_layout.addWidget(QtWidgets.QLabel("Recording Duration:"))
        self.rec_duration_combo = QtWidgets.QComboBox()
        self.rec_duration_combo.addItems(["5 Seconds", "10 Seconds", "15 Seconds", "30 Seconds"])
        self.rec_duration_combo.setCurrentIndex(1) # Default 10s
        settings_layout.addWidget(self.rec_duration_combo)
        
        settings_layout.addWidget(QtWidgets.QLabel("Sonification Mode:"))
        self.sonifier_mode_combo = QtWidgets.QComboBox()
        self.sonifier_mode_combo.addItems([
            "Vibrato (FM - Pitch modulation)", 
            "Tremolo (AM - Volume modulation)",
            "Drum Beats (Clicks at wave peaks)"
        ])
        settings_layout.addWidget(self.sonifier_mode_combo)

        # Tempo guidance controls
        self.guide_tempo_cb = QtWidgets.QCheckBox("Guide with Known Tempo (BPM)")
        self.guide_tempo_cb.setChecked(False)
        self.guide_tempo_cb.stateChanged.connect(self.on_guide_tempo_toggled)
        settings_layout.addWidget(self.guide_tempo_cb)
        
        self.tempo_widget = QtWidgets.QWidget()
        tempo_h_layout = QtWidgets.QHBoxLayout(self.tempo_widget)
        tempo_h_layout.setContentsMargins(0, 0, 0, 0)
        tempo_h_layout.addWidget(QtWidgets.QLabel("Target Tempo:"))
        self.tempo_spin = QtWidgets.QSpinBox()
        self.tempo_spin.setRange(50, 220)
        self.tempo_spin.setValue(120)
        self.tempo_spin.setSuffix(" BPM")
        tempo_h_layout.addWidget(self.tempo_spin)
        self.tempo_widget.setEnabled(False)
        settings_layout.addWidget(self.tempo_widget)
        
        settings_layout.addWidget(QtWidgets.QLabel("Base Tone Carrier Frequency:"))
        self.carrier_freq_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.carrier_freq_slider.setRange(100, 880)
        self.carrier_freq_slider.setValue(220)
        self.carrier_freq_slider.valueChanged.connect(self.on_carrier_slider_changed)
        settings_layout.addWidget(self.carrier_freq_slider)
        
        self.carrier_freq_lbl = QtWidgets.QLabel("220 Hz (Closest to A3 - Bass)")
        self.carrier_freq_lbl.setObjectName("muted")
        settings_layout.addWidget(self.carrier_freq_lbl)
        
        settings_layout.addWidget(QtWidgets.QLabel("Modulation Depth / Strength:"))
        self.mod_depth_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.mod_depth_slider.setRange(10, 100)
        self.mod_depth_slider.setValue(40)
        settings_layout.addWidget(self.mod_depth_slider)
        
        tab2_left_layout.addWidget(settings_group)
        
        explanation_group = QtWidgets.QGroupBox("How It Works")
        explanation_layout = QtWidgets.QVBoxLayout(explanation_group)
        explanation_layout.setContentsMargins(12, 15, 12, 12)
        
        exp_lbl = QtWidgets.QLabel(
            "EEG waves (1-45 Hz) are too low for direct human hearing. This tool <b>sonifies</b> them: "
            "we modulate an audible carrier wave with your brainwaves.\n\n"
            "• <b>Stereo Sonification</b>: Tracks different electrodes for your Left and Right ears (e.g. motor cortex "
            "asymmetry or occipital relaxation).\n\n"
            "• <b>Auto-Detect (L/R)</b>: Scans the respective hemispheres and isolates the strongest, cleanest rhythm.\n\n"
            "• <b>On-the-Fly Adjustability</b>: You can record once, then change sonification modes, target tempos, or frequencies, "
            "and immediately click Play to hear the new audio without having to re-record!"
        )
        exp_lbl.setWordWrap(True)
        exp_lbl.setObjectName("muted")
        explanation_layout.addWidget(exp_lbl)
        
        tab2_left_layout.addWidget(explanation_group)
        tab2_left_layout.addStretch()
        tab2_layout.addWidget(tab2_left)
        
        # Tab 2 Right panel: Recording controls and Visual feedback
        tab2_right = QtWidgets.QWidget()
        tab2_right_layout = QtWidgets.QVBoxLayout(tab2_right)
        tab2_right_layout.setContentsMargins(0, 0, 0, 0)
        tab2_right_layout.setSpacing(10)
        
        rec_ctrl_group = QtWidgets.QWidget()
        rec_ctrl_group.setStyleSheet(f"background-color: {COLOR_CARD}; border-radius: 8px; border: 1px solid {COLOR_BORDER};")
        rec_ctrl_layout = QtWidgets.QHBoxLayout(rec_ctrl_group)
        rec_ctrl_layout.setContentsMargins(15, 15, 15, 15)
        rec_ctrl_layout.setSpacing(15)
        
        self.record_btn = QtWidgets.QPushButton("🔴 START EEG RECORDING")
        self.record_btn.setStyleSheet("font-size: 14px; padding: 10px 20px; background-color: #ef4444; color: white;")
        self.record_btn.clicked.connect(self.toggle_recording)
        rec_ctrl_layout.addWidget(self.record_btn)
        
        self.rec_progress_bar = QtWidgets.QProgressBar()
        self.rec_progress_bar.setRange(0, 100)
        self.rec_progress_bar.setValue(0)
        self.rec_progress_bar.setFormat("Ready to record")
        self.rec_progress_bar.setFixedHeight(25)
        rec_ctrl_layout.addWidget(self.rec_progress_bar)
        
        self.play_audio_btn = QtWidgets.QPushButton("▶️ PLAY AUDIO")
        self.play_audio_btn.setStyleSheet("font-size: 14px; padding: 10px 20px; background-color: #10b981; color: white;")
        self.play_audio_btn.setEnabled(False)
        self.play_audio_btn.clicked.connect(self.play_recorded_audio)
        rec_ctrl_layout.addWidget(self.play_audio_btn)
        
        self.stop_audio_btn = QtWidgets.QPushButton("⏹️ STOP")
        self.stop_audio_btn.setStyleSheet("font-size: 14px; padding: 10px 20px; background-color: #475569; color: white;")
        self.stop_audio_btn.setEnabled(False)
        self.stop_audio_btn.clicked.connect(self.stop_recorded_audio)
        rec_ctrl_layout.addWidget(self.stop_audio_btn)
        
        tab2_right_layout.addWidget(rec_ctrl_group)
        
        self.rec_win = pg.GraphicsLayoutWidget(title="Recording Analysis")
        self.rec_win.setBackground(COLOR_BG)
        tab2_right_layout.addWidget(self.rec_win)
        
        self.plot_rec_eeg = self.rec_win.addPlot(row=0, col=0, title="Recorded Brainwave Signal (Filtered 1 Hz HP & 50 Hz Notch)")
        self.plot_rec_eeg.setLabel('bottom', 'Time', units='s')
        self.plot_rec_eeg.setLabel('left', 'Amplitude', units='uV')
        self.plot_rec_eeg.showGrid(x=True, y=True, alpha=0.2)
        
        # Two curves for stereo L/R
        self.curve_rec_eeg_l = self.plot_rec_eeg.plot(pen=pg.mkPen(COLOR_ALPHA, width=1.5), name="Left Ear Channel")
        self.curve_rec_eeg_r = self.plot_rec_eeg.plot(pen=pg.mkPen(COLOR_THETA, width=1.5), name="Right Ear Channel")
        self.plot_rec_eeg.addLegend(offset=(-30, 30))
        
        # Spectrogram plot to show rhythm frequency evolution over time
        self.plot_rec_spec = self.rec_win.addPlot(row=1, col=0, title="Rhythm Frequency Evolution Over Time (Spectrogram)")
        self.plot_rec_spec.setLabel('bottom', 'Time', units='s')
        self.plot_rec_spec.setLabel('left', 'Frequency', units='Hz')
        self.plot_rec_spec.setYRange(0.5, 30.0)
        self.plot_rec_spec.showGrid(x=True, y=True, alpha=0.2)
        
        self.spec_image = pg.ImageItem()
        self.plot_rec_spec.addItem(self.spec_image)
        
        # Color map for spectrogram (Inferno looks premium and beautiful!)
        colormap = pg.colormap.get('inferno')
        self.spec_colorbar = pg.ColorBarItem(values=(0, 1))
        self.spec_colorbar.setColorMap(colormap)
        self.spec_colorbar.setImageItem(self.spec_image)
        self.rec_win.addItem(self.spec_colorbar, row=1, col=1)
        
        # Playhead indicator lines
        self.playhead_eeg = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#ef4444', width=2.0))
        self.playhead_spec = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#ef4444', width=2.0))
        self.plot_rec_eeg.addItem(self.playhead_eeg)
        self.plot_rec_spec.addItem(self.playhead_spec)
        
        self.saved_file_lbl = QtWidgets.QLabel("No recording saved yet.")
        self.saved_file_lbl.setStyleSheet(f"color: {COLOR_MUTED}; font-size: 12px;")
        tab2_right_layout.addWidget(self.saved_file_lbl)
        
        tab2_layout.addWidget(tab2_right, stretch=2)
        
        self.tabs.addTab(tab2, "🎵 Rhythm Sonifier & Recorder")
        
        self.on_source_changed(self.simulation_mode)
        
    def update_status_indicator(self):
        if self.simulation_mode:
            self.status_indicator.setText("⚙️ SIMULATION ACTIVE")
            self.status_indicator.setStyleSheet("font-weight: bold; color: #f59e0b; border: 1px solid #f59e0b; padding: 4px 10px; border-radius: 4px; font-size: 11px;")
        else:
            self.status_indicator.setText(f"📡 STREAM CONNECTED ({self.fs:.0f}Hz)")
            self.status_indicator.setStyleSheet("font-weight: bold; color: #10b981; border: 1px solid #10b981; padding: 4px 10px; border-radius: 4px; font-size: 11px;")
            
    def on_source_changed(self, state):
        self.simulation_mode = bool(state)
        self.sim_group.setEnabled(self.simulation_mode)
        self.update_status_indicator()
        
        if not self.simulation_mode:
            print("[*] Reconnecting LSL EEG stream...")
            streams = resolve_byprop('type', 'EEG', timeout=1.0)
            if streams:
                self.inlet = StreamInlet(streams[0])
                info = self.inlet.info()
                self.fs = info.nominal_srate()
                lsl_chans = info.channel_count()
                
                temp_names = []
                ch = info.desc().child("channels").child("channel")
                for i in range(lsl_chans):
                    if ch.empty():
                        temp_names.append(f"CH {i+1}")
                    else:
                        temp_names.append(ch.child_value("label"))
                        ch = ch.next_sibling()
                
                if len(temp_names) > 0 and temp_names[-1].upper() == 'BATTERY':
                    self.channel_names = temp_names[:-1]
                else:
                    self.channel_names = temp_names
                    
                self.num_channels = len(self.channel_names)
                
                # Update combo boxes
                self.channel_combo.clear()
                self.channel_combo.addItems(self.channel_names)
                
                self.rec_channel_combo_l.clear()
                self.rec_channel_combo_l.addItem("✨ Auto-Detect Left")
                self.rec_channel_combo_l.addItems(self.channel_names)
                
                self.rec_channel_combo_r.clear()
                self.rec_channel_combo_r.addItem("✨ Auto-Detect Right")
                self.rec_channel_combo_r.addItems(self.channel_names)
                
                self.buffer_samples = int(self.fs * self.buffer_duration)
                self.data_buffer = np.zeros((self.buffer_samples, self.num_channels))
                self.time_axis = np.linspace(-self.buffer_duration, 0, self.buffer_samples)
                self.setup_filters()
                print(f"[+] LSL Connection Restored: {info.name()} ({self.fs} Hz, {self.num_channels} channels)")
            else:
                print("[-] LSL Warning: No stream found. Keeping simulation active.")
                self.simulation_mode = True
                self.source_checkbox.setChecked(True)
                self.sim_group.setEnabled(True)
                self.update_status_indicator()
                
        self.reset_challenge()
        
    def on_channel_changed(self, idx):
        if idx >= 0 and idx < self.num_channels:
            self.current_channel_idx = idx
            
    def on_filter_toggled(self):
        self.filters_enabled = self.filter_band_cb.isChecked()
        self.notch_enabled = self.filter_notch_cb.isChecked()
        
    def on_sim_params_changed(self):
        self.sim_alpha_amp = float(self.sim_alpha_slider.value())
        self.sim_beta_amp = float(self.sim_beta_slider.value())
        self.sim_noise_amp = float(self.sim_noise_slider.value()) / 2.0
        self.sim_powerline_active = self.sim_powerline_cb.isChecked()
        
    def trigger_eye_blink(self):
        self.blink_active_samples = int(self.fs * 0.4)
        
    def trigger_muscle_clench(self):
        self.muscle_active_samples = int(self.fs * 0.8)
        
    def on_carrier_slider_changed(self):
        val = self.carrier_freq_slider.value()
        notes = [
            (110.0, "A2 (Very Low)"),
            (130.8, "C3"),
            (146.8, "D3"),
            (164.8, "E3"),
            (196.0, "G3"),
            (220.0, "A3 (Bass)"),
            (261.6, "C4 (Middle C)"),
            (293.7, "D4"),
            (329.6, "E4"),
            (392.0, "G4"),
            (440.0, "A4 (Standard pitch)"),
            (523.3, "C5"),
            (587.3, "D5"),
            (659.3, "E5"),
            (784.0, "G5"),
            (880.0, "A5 (High pitch)")
        ]
        nearest_note = min(notes, key=lambda x: abs(x[0] - val))
        self.carrier_freq_lbl.setText(f"{val} Hz (Closest to {nearest_note[1]})")
        
    def on_guide_tempo_toggled(self, state):
        self.tempo_widget.setEnabled(bool(state))
        
    def set_settings_enabled(self, enabled):
        self.rec_channel_combo_l.setEnabled(enabled)
        self.rec_channel_combo_r.setEnabled(enabled)
        self.rec_duration_combo.setEnabled(enabled)
        self.sonifier_mode_combo.setEnabled(enabled)
        self.carrier_freq_slider.setEnabled(enabled)
        self.mod_depth_slider.setEnabled(enabled)
        self.source_checkbox.setEnabled(enabled)
        self.channel_combo.setEnabled(enabled)
        self.guide_tempo_cb.setEnabled(enabled)
        if enabled:
            self.tempo_widget.setEnabled(self.guide_tempo_cb.isChecked())
        else:
            self.tempo_widget.setEnabled(False)

    def toggle_recording(self):
        if self.recording_active:
            # Stop prematurely
            self.recording_active = False
            self.record_btn.setText("🔴 START EEG RECORDING")
            self.rec_progress_bar.setFormat("Recording cancelled")
            self.rec_progress_bar.setValue(0)
            self.set_settings_enabled(True)
        else:
            # Start
            self.recording_active = True
            self.recording_data = []
            self.raw_recorded_data = None # clear previous cache
            
            dur_text = self.rec_duration_combo.currentText()
            self.recording_duration = float(dur_text.split()[0])
            self.recording_target_samples = int(self.fs * self.recording_duration)
            
            self.record_btn.setText("⏹️ STOP RECORDING")
            self.play_audio_btn.setEnabled(False)
            self.stop_audio_btn.setEnabled(False)
            self.rec_progress_bar.setValue(0)
            self.rec_progress_bar.setFormat("Recording... 0%")
            
            self.rec_idx_l = self.rec_channel_combo_l.currentIndex()
            self.rec_idx_r = self.rec_channel_combo_r.currentIndex()
            print(f"[Recording] Started: {self.recording_duration}s (All 32 channels recorded in parallel)")
            self.set_settings_enabled(False)
            
    def play_recorded_audio(self):
        # Verify we have cached data
        if self.raw_recorded_data is None:
            self.rec_progress_bar.setFormat("❌ No recorded data to play!")
            return
            
        # RE-SONIFY ON-THE-FLY using the CURRENT settings in the UI
        self.process_and_save_recording()
        
        if self.recording_filename and os.path.exists(self.recording_filename):
            print(f"[Audio] Playing: {self.recording_filename}")
            winsound.PlaySound(None, winsound.SND_PURGE) # Stop current first
            
            # Start playhead sweep
            self.playback_start_time = time.time()
            self.playback_timer.start(30) # 33 fps
            
            winsound.PlaySound(self.recording_filename, winsound.SND_FILENAME | winsound.SND_ASYNC)
            
    def stop_recorded_audio(self):
        winsound.PlaySound(None, winsound.SND_PURGE)
        self.playback_timer.stop()
        self.playhead_eeg.setValue(0)
        self.playhead_spec.setValue(0)
        print("[Audio] Playback stopped.")
        
    def update_playhead(self):
        elapsed = time.time() - self.playback_start_time
        if elapsed >= self.recording_duration:
            self.stop_recorded_audio()
        else:
            self.playhead_eeg.setValue(elapsed)
            self.playhead_spec.setValue(elapsed)

    def process_and_save_recording(self):
        # 1. Load data from raw cache (or read from recording_data if just completed)
        if self.raw_recorded_data is None:
            # We just completed a recording, cache it
            self.raw_recorded_data = np.vstack(self.recording_data)[:self.recording_target_samples]
            
        raw_data = self.raw_recorded_data
        
        # Retrieve L/R combo selections
        self.rec_idx_l = self.rec_channel_combo_l.currentIndex()
        self.rec_idx_r = self.rec_channel_combo_r.currentIndex()
        
        # Setup temporary filters for evaluation
        nyq = 0.5 * self.fs
        b_hp, a_hp = signal.butter(4, 1.0 / nyq, btype='high')
        b_notch, a_notch = signal.iirnotch(50.0, 30.0, self.fs)
        
        # 2. Evaluate Left Channel
        best_idx_l = 0
        if self.rec_idx_l == 0:
            # Auto-Detect Left Hemisphere (odd channel numbers & central)
            left_chans = [i for i, name in enumerate(self.channel_names) if any(char in name for char in ['1','3','5','7','9']) or name in ['Fz','Cz','Pz','Oz']]
            scores_l = []
            for ch in left_chans:
                filt_ch = signal.lfilter(b_hp, a_hp, raw_data[:, ch])
                filt_ch = signal.lfilter(b_notch, a_notch, filt_ch)
                
                std_val = np.std(filt_ch)
                raw_std_val = np.std(signal.detrend(raw_data[:, ch]))
                
                if raw_std_val > 150.0 or std_val < 1.0 or std_val > 100.0:
                    score = 0.0
                else:
                    n_fft = len(filt_ch)
                    freqs = np.fft.rfftfreq(n_fft, d=1.0/self.fs)
                    fft_vals = np.abs(np.fft.rfft(filt_ch)) / (n_fft / 2.0)
                    interest_mask = (freqs >= 1.0) & (freqs <= 30.0)
                    if np.any(interest_mask):
                        score = np.max(fft_vals[interest_mask]) / (np.mean(fft_vals[interest_mask]) + 1e-6)
                        if std_val > 40.0: score *= (1.0 - (std_val / 150.0))
                    else:
                        score = 0.0
                scores_l.append(score)
            best_idx_l = left_chans[int(np.argmax(scores_l))]
        else:
            best_idx_l = self.rec_idx_l - 1
            
        # 3. Evaluate Right Channel
        best_idx_r = 0
        if self.rec_idx_r == 0:
            # Auto-Detect Right Hemisphere (even channel numbers & central)
            right_chans = [i for i, name in enumerate(self.channel_names) if any(char in name for char in ['2','4','6','8','10']) or name in ['Fz','Cz','Pz','Oz']]
            scores_r = []
            for ch in right_chans:
                filt_ch = signal.lfilter(b_hp, a_hp, raw_data[:, ch])
                filt_ch = signal.lfilter(b_notch, a_notch, filt_ch)
                
                std_val = np.std(filt_ch)
                raw_std_val = np.std(signal.detrend(raw_data[:, ch]))
                
                if raw_std_val > 150.0 or std_val < 1.0 or std_val > 100.0:
                    score = 0.0
                else:
                    n_fft = len(filt_ch)
                    freqs = np.fft.rfftfreq(n_fft, d=1.0/self.fs)
                    fft_vals = np.abs(np.fft.rfft(filt_ch)) / (n_fft / 2.0)
                    interest_mask = (freqs >= 1.0) & (freqs <= 30.0)
                    if np.any(interest_mask):
                        score = np.max(fft_vals[interest_mask]) / (np.mean(fft_vals[interest_mask]) + 1e-6)
                        if std_val > 40.0: score *= (1.0 - (std_val / 150.0))
                    else:
                        score = 0.0
                scores_r.append(score)
            best_idx_r = right_chans[int(np.argmax(scores_r))]
        else:
            best_idx_r = self.rec_idx_r - 1
            
        # Extract and filter EEG for both channels (starting fresh from raw cache)
        raw_eeg_l = raw_data[:, best_idx_l]
        raw_eeg_r = raw_data[:, best_idx_r]
        
        try:
            filtered_eeg_l = signal.filtfilt(b_hp, a_hp, raw_eeg_l)
            filtered_eeg_l = signal.filtfilt(b_notch, a_notch, filtered_eeg_l)
            filtered_eeg_r = signal.filtfilt(b_hp, a_hp, raw_eeg_r)
            filtered_eeg_r = signal.filtfilt(b_notch, a_notch, filtered_eeg_r)
        except Exception:
            filtered_eeg_l = signal.lfilter(b_hp, a_hp, raw_eeg_l)
            filtered_eeg_l = signal.lfilter(b_notch, a_notch, filtered_eeg_l)
            filtered_eeg_r = signal.lfilter(b_hp, a_hp, raw_eeg_r)
            filtered_eeg_r = signal.lfilter(b_notch, a_notch, filtered_eeg_r)
            
        # 4. Update visual waveform plot (both left and right)
        rec_time_axis = np.linspace(0, self.recording_duration, len(filtered_eeg_l))
        self.curve_rec_eeg_l.setData(rec_time_axis, filtered_eeg_l)
        self.curve_rec_eeg_r.setData(rec_time_axis, filtered_eeg_r)
        self.plot_rec_eeg.setXRange(0, self.recording_duration)
        
        # Display selected channel names in plot title
        l_name = self.channel_names[best_idx_l]
        r_name = self.channel_names[best_idx_r]
        self.plot_rec_eeg.setTitle(f"Recorded Brainwave Signals | L: {l_name} (Yellow) | R: {r_name} (Green)")
        
        # 5. Resample to 44100 Hz for Audio Generation
        audio_fs = 44100
        num_audio_samples = int(self.recording_duration * audio_fs)
        audio_time_axis = np.linspace(0, self.recording_duration, num_audio_samples)
        
        interpolated_l = np.interp(audio_time_axis, rec_time_axis, filtered_eeg_l)
        interpolated_r = np.interp(audio_time_axis, rec_time_axis, filtered_eeg_r)
        
        # Normalize both channels
        interpolated_l = signal.detrend(interpolated_l)
        max_l = np.max(np.abs(interpolated_l))
        norm_l = interpolated_l / max_l if max_l > 0 else interpolated_l
        
        interpolated_r = signal.detrend(interpolated_r)
        max_r = np.max(np.abs(interpolated_r))
        norm_r = interpolated_r / max_r if max_r > 0 else interpolated_r
        
        # 6. Generate Audio for Left and Right Channels
        fc = float(self.carrier_freq_slider.value())
        mode = self.sonifier_mode_combo.currentText()
        depth = float(self.mod_depth_slider.value())
        
        audio_l = np.zeros(num_audio_samples)
        audio_r = np.zeros(num_audio_samples)
        
        if "Vibrato" in mode:
            # FM Modulation (Vibrato)
            phase_l = 2 * np.pi * np.cumsum(fc + depth * norm_l) / audio_fs
            phase_r = 2 * np.pi * np.cumsum(fc + depth * norm_r) / audio_fs
            audio_l = np.sin(phase_l)
            audio_r = np.sin(phase_r)
        elif "Tremolo" in mode:
            # AM Modulation (Tremolo)
            depth_factor = depth / 100.0
            carrier = np.sin(2 * np.pi * fc * audio_time_axis)
            envelope_l = 1.0 - (depth_factor * 0.5) + (depth_factor * 0.5 * norm_l)
            envelope_r = 1.0 - (depth_factor * 0.5) + (depth_factor * 0.5 * norm_r)
            audio_l = envelope_l * carrier
            audio_r = envelope_r * carrier
        else:
            # Drum Beats (Clicks at wave peaks)
            # Apply tempo filter L/R in Delta/Theta band
            if self.guide_tempo_cb.isChecked():
                bpm = self.tempo_spin.value()
                f_beat = bpm / 60.0
                low_d = max(0.4, f_beat - 0.35)
                high_d = f_beat + 0.35
                b_d, a_d = signal.butter(4, [low_d / nyq, high_d / nyq], btype='band')
                min_dist_d = int(self.fs * (60.0 / bpm) * 0.75)
                
                f_sub = 2.0 * f_beat
                low_t = max(0.4, f_sub - 0.6)
                high_t = min(nyq - 1.0, f_sub + 0.6)
                b_t, a_t = signal.butter(4, [low_t / nyq, high_t / nyq], btype='band')
                min_dist_t = int(self.fs * (30.0 / bpm) * 0.75)
            else:
                b_d, a_d = signal.butter(4, [1.0 / nyq, 4.0 / nyq], btype='band')
                min_dist_d = int(self.fs * 0.25)
                b_t, a_t = signal.butter(4, [4.0 / nyq, 8.0 / nyq], btype='band')
                min_dist_t = int(self.fs * 0.12)
                
            try:
                # Left Channel triggers drum beats
                eeg_d_l = signal.filtfilt(b_d, a_d, raw_eeg_l)
                eeg_t_l = signal.filtfilt(b_t, a_t, raw_eeg_l)
                # Right Channel triggers drum beats
                eeg_d_r = signal.filtfilt(b_d, a_d, raw_eeg_r)
                eeg_t_r = signal.filtfilt(b_t, a_t, raw_eeg_r)
            except Exception:
                eeg_d_l = signal.lfilter(b_d, a_d, raw_eeg_l)
                eeg_t_l = signal.lfilter(b_t, a_t, raw_eeg_l)
                eeg_d_r = signal.lfilter(b_d, a_d, raw_eeg_r)
                eeg_t_r = signal.lfilter(b_t, a_t, raw_eeg_r)
                
            # Calculate rise-time derivative envelopes to target the beat onset phase
            rise_d_l = np.concatenate(([0], np.maximum(0, np.diff(eeg_d_l))))
            rise_t_l = np.concatenate(([0], np.maximum(0, np.diff(eeg_t_l))))
            rise_d_r = np.concatenate(([0], np.maximum(0, np.diff(eeg_d_r))))
            rise_t_r = np.concatenate(([0], np.maximum(0, np.diff(eeg_t_r))))
            
            max_d_l = np.max(rise_d_l) if np.max(rise_d_l) > 0 else 1.0
            max_t_l = np.max(rise_t_l) if np.max(rise_t_l) > 0 else 1.0
            max_d_r = np.max(rise_d_r) if np.max(rise_d_r) > 0 else 1.0
            max_t_r = np.max(rise_t_r) if np.max(rise_t_r) > 0 else 1.0
            
            peaks_d_l, _ = signal.find_peaks(rise_d_l, height=0.25 * max_d_l, distance=min_dist_d)
            peaks_t_l, _ = signal.find_peaks(rise_t_l, height=0.25 * max_t_l, distance=min_dist_t)
            peaks_d_r, _ = signal.find_peaks(rise_d_r, height=0.25 * max_d_r, distance=min_dist_d)
            peaks_t_r, _ = signal.find_peaks(rise_t_r, height=0.25 * max_t_r, distance=min_dist_t)
            
            # Sound templates
            bd_len = int(audio_fs * 0.05)
            bd_t = np.linspace(0, 0.05, bd_len)
            bd_freq = 50.0 + (140.0 - 50.0) * np.exp(-bd_t / 0.012)
            bd_wave = np.sin(2 * np.pi * bd_freq * bd_t) * np.exp(-bd_t / 0.015)
            
            cym_len = int(audio_fs * 0.02)
            cym_t = np.linspace(0, 0.02, cym_len)
            cym_wave = np.sin(2 * np.pi * 1200.0 * cym_t) * np.exp(-cym_t / 0.003)
            
            # Left track assembly
            for p in peaks_d_l:
                p_audio = int(p * (audio_fs / self.fs))
                end_idx = min(p_audio + bd_len, num_audio_samples)
                span = end_idx - p_audio
                if span > 0: audio_l[p_audio:end_idx] += bd_wave[:span]
            for p in peaks_t_l:
                p_audio = int(p * (audio_fs / self.fs))
                end_idx = min(p_audio + cym_len, num_audio_samples)
                span = end_idx - p_audio
                if span > 0: audio_l[p_audio:end_idx] += 0.7 * cym_wave[:span]
                
            # Right track assembly
            for p in peaks_d_r:
                p_audio = int(p * (audio_fs / self.fs))
                end_idx = min(p_audio + bd_len, num_audio_samples)
                span = end_idx - p_audio
                if span > 0: audio_r[p_audio:end_idx] += bd_wave[:span]
            for p in peaks_t_r:
                p_audio = int(p * (audio_fs / self.fs))
                end_idx = min(p_audio + cym_len, num_audio_samples)
                span = end_idx - p_audio
                if span > 0: audio_r[p_audio:end_idx] += 0.7 * cym_wave[:span]
                
        # 7. Save Stereo WAV file
        self.recording_filename = os.path.join(os.getcwd(), "brain_rhythm_audio.wav")
        try:
            audio_l_clipped = np.clip(audio_l, -1.0, 1.0)
            audio_r_clipped = np.clip(audio_r, -1.0, 1.0)
            
            # Interleave L and R channels
            stereo_data = np.empty((num_audio_samples * 2,), dtype=np.float32)
            stereo_data[0::2] = audio_l_clipped
            stereo_data[1::2] = audio_r_clipped
            
            audio_int16 = (stereo_data * 32767).astype(np.int16)
            
            with wave.open(self.recording_filename, 'wb') as wav_file:
                wav_file.setnchannels(2) # STEREO
                wav_file.setsampwidth(2) # 16-bit
                wav_file.setframerate(audio_fs)
                wav_file.writeframes(audio_int16.tobytes())
                
            self.saved_file_lbl.setText(f"✓ Saved Stereo WAV to: {self.recording_filename}")
            self.play_audio_btn.setEnabled(True)
            self.stop_audio_btn.setEnabled(True)
            
            # Print status message
            status_msg = f"✓ Isolated L: {l_name} | R: {r_name} in Stereo!"
            self.rec_progress_bar.setFormat(status_msg)
        except Exception as e:
            self.saved_file_lbl.setText(f"Error saving audio: {e}")
            self.rec_progress_bar.setFormat("❌ Audio generation failed")
            
        # 8. Calculate and display the Spectrogram of L channel for frequency evolution
        try:
            nperseg = min(len(filtered_eeg_l), int(self.fs * 1.5))
            noverlap = min(len(filtered_eeg_l) - 1, int(self.fs * 1.35))
            f_spec, t_spec, Sxx = signal.spectrogram(filtered_eeg_l, fs=self.fs, nperseg=nperseg, noverlap=noverlap)
            
            freq_mask = (f_spec >= 0.5) & (f_spec <= 30.0)
            f_masked = f_spec[freq_mask]
            Sxx_masked = Sxx[freq_mask, :]
            
            Sxx_log = 10.0 * np.log10(Sxx_masked + 1e-8)
            
            self.spec_image.setImage(Sxx_log.T)
            self.spec_image.setRect(QtCore.QRectF(0, f_masked[0], self.recording_duration, f_masked[-1] - f_masked[0]))
            
            p5, p95 = np.percentile(Sxx_log, [5, 95])
            if p95 > p5:
                self.spec_image.setLevels([p5, p95])
                self.spec_colorbar.setLevels([p5, p95])
                
            self.plot_rec_spec.setTitle(f"Rhythm Frequency Evolution Over Time (L: {l_name})")
        except Exception as e:
            print(f"[-] Spectrogram computation error: {e}")
            
        # Reset playheads to start
        self.playhead_eeg.setValue(0)
        self.playhead_spec.setValue(0)
        
        self.set_settings_enabled(True)

    def generate_simulated_chunk(self, size):
        chunk = []
        for s in range(size):
            t = (self.sim_sample_count + s) / self.fs
            sample = []
            
            # Active blink envelope
            blink_val = 0.0
            if self.blink_active_samples > 0:
                progress = (int(self.fs * 0.4) - self.blink_active_samples) / (self.fs * 0.4)
                blink_val = 90.0 * np.sin(np.pi * progress)
                self.blink_active_samples -= 1
                
            # Active muscle envelope
            muscle_val = 0.0
            if self.muscle_active_samples > 0:
                muscle_val = np.random.normal(0, 30.0)
                self.muscle_active_samples -= 1
            
            for ch in range(self.num_channels):
                noise = np.random.normal(0, self.sim_noise_amp)
                
                # Add Alpha wave (Strongest on occipital index 8/9/18/19 O1/O2/Oz/Pz)
                alpha_ch_factor = 1.0 if ch in OCCIPITAL_INDICES else 0.2
                alpha_freq = 10.0 + 0.2 * np.sin(2 * np.pi * 0.1 * t)
                alpha = alpha_ch_factor * self.sim_alpha_amp * np.sin(2 * np.pi * alpha_freq * t)
                
                # Add Beta wave (Strongest on frontal index 0/1/2/3 Fp1/Fp2/F3/F4)
                beta_ch_factor = 1.0 if ch in FRONTAL_INDICES else 0.3
                beta_freq = 18.0 + 0.5 * np.cos(2 * np.pi * 0.15 * t)
                beta = beta_ch_factor * self.sim_beta_amp * np.sin(2 * np.pi * beta_freq * t)
                
                # Add 50 Hz powerline interference
                line_noise = 0.0
                if self.sim_powerline_active:
                    line_noise = 15.0 * np.sin(2 * np.pi * 50.0 * t)
                    
                # Delta drift / Blink
                blink_factor = 1.0 if ch in FRONTAL_INDICES else 0.15
                delta_drift = 5.0 * np.sin(2 * np.pi * 0.3 * t) + (blink_val * blink_factor)
                
                # Muscle tension (Gamma)
                muscle_factor = 1.0 if ch in TEMPORAL_INDICES else 0.2
                muscle = muscle_val * muscle_factor
                
                dc_offset = (ch % 7 - 3.5) * 80.0
                val = dc_offset + noise + alpha + beta + line_noise + delta_drift + muscle
                sample.append(val)
                
            chunk.append(sample)
        
        self.sim_sample_count += size
        return chunk
        
    def on_timer_tick(self):
        if self.simulation_mode:
            chunk_size = int(self.fs * 0.04)
            if chunk_size < 1:
                chunk_size = 1
            chunk = self.generate_simulated_chunk(chunk_size)
        else:
            if self.inlet is None:
                return
            chunk, timestamps = self.inlet.pull_chunk()
            if not chunk:
                return
                
        chunk = np.array(chunk)
        num_samples = chunk.shape[0]
        
        if num_samples == 0:
            return
            
        self.data_buffer = np.roll(self.data_buffer, -num_samples, axis=0)
        self.data_buffer[-num_samples:, :] = chunk[:, :self.num_channels]
        
        # Accumulate recording samples
        if self.recording_active:
            # Always record all channels
            self.recording_data.append(chunk[:, :self.num_channels])
            
            accumulated_samples = sum(len(c) for c in self.recording_data)
            progress = accumulated_samples / self.recording_target_samples
            percent = int(progress * 100)
            self.rec_progress_bar.setValue(min(100, percent))
            self.rec_progress_bar.setFormat(f"Recording... {percent}% ({accumulated_samples / self.fs:.1f}s / {self.recording_duration:.0f}s)")
            
            if accumulated_samples >= self.recording_target_samples:
                self.recording_active = False
                self.record_btn.setText("🔴 START EEG RECORDING")
                self.rec_progress_bar.setFormat("Processing recording...")
                self.process_and_save_recording()
        
        raw_sig = self.data_buffer[:, self.current_channel_idx]
        detrended_sig = signal.detrend(raw_sig)
        
        filtered_sig = detrended_sig.copy()
        if self.filters_enabled:
            filtered_sig = signal.lfilter(self.b_band, self.a_band, filtered_sig)
        if self.notch_enabled:
            filtered_sig = signal.lfilter(self.b_notch, self.a_notch, filtered_sig)
            
        zoom_range = float(self.zoom_slider.value())
        self.plot_time.setYRange(-zoom_range, zoom_range)
        
        self.curve_raw.setData(self.time_axis, detrended_sig)
        self.curve_filtered.setData(self.time_axis, filtered_sig)
        
        n_fft = self.buffer_samples
        freqs = np.fft.rfftfreq(n_fft, d=1.0/self.fs)
        
        fft_vals = np.abs(np.fft.rfft(filtered_sig)) / (n_fft / 2.0)
        
        fft_smoothed = signal.savgol_filter(fft_vals, window_length=9, polyorder=2) if len(fft_vals) > 9 else fft_vals
        self.curve_fft.setData(freqs, fft_smoothed)
        
        max_visible_power = np.max(fft_smoothed[(freqs >= 0.5) & (freqs <= 45.0)]) if len(freqs) > 0 else 1.0
        max_y_limit = max(0.5, max_visible_power * 1.2)
        self.plot_freq.setYRange(0, max_y_limit)
        
        for low, high, text in self.band_regions:
            text.setPos((low + high) / 2.0, max_y_limit * 0.85)
            
        idx_delta = (freqs >= 0.5) & (freqs < 4.0)
        idx_theta = (freqs >= 4.0) & (freqs < 8.0)
        idx_alpha = (freqs >= 8.0) & (freqs < 12.0)
        idx_beta = (freqs >= 12.0) & (freqs < 30.0)
        idx_gamma = (freqs >= 30.0) & (freqs <= 45.0)
        
        p_delta = np.sum(fft_smoothed[idx_delta])
        p_theta = np.sum(fft_smoothed[idx_theta])
        p_alpha = np.sum(fft_smoothed[idx_alpha])
        p_beta = np.sum(fft_smoothed[idx_beta])
        p_gamma = np.sum(fft_smoothed[idx_gamma])
        
        total_p = p_delta + p_theta + p_alpha + p_beta + p_gamma
        if total_p == 0.0:
            total_p = 1.0
            
        rel_delta = (p_delta / total_p) * 100
        rel_theta = (p_theta / total_p) * 100
        rel_alpha = (p_alpha / total_p) * 100
        rel_beta = (p_beta / total_p) * 100
        rel_gamma = (p_gamma / total_p) * 100
        
        self.bars["Delta"].setValue(int(rel_delta))
        self.bars["Theta"].setValue(int(rel_theta))
        self.bars["Alpha"].setValue(int(rel_alpha))
        self.bars["Beta"].setValue(int(rel_beta))
        self.bars["Gamma"].setValue(int(rel_gamma))
        
        std_dev = np.std(filtered_sig)
        raw_std_dev = np.std(detrended_sig)
        
        if raw_std_dev > 180.0:
            state_text = "❌ RAILED / DISCONNECTED"
            state_desc = "Standard deviation is extremely high. The electrode contact may be broken, or there is massive movement noise."
            bg_color = "#450a0a"
            border_color = "#991b1b"
        elif rel_delta > 55.0 and std_dev > 25.0:
            state_text = "👁️ EYE BLINK / MOVE (Delta)"
            state_desc = "Massive low-frequency pulses detected on frontal channels. Indicative of eye blinking or facial movement."
            bg_color = "#172554"
            border_color = "#1e40af"
        elif rel_gamma > 45.0 and std_dev > 20.0:
            state_text = "💪 MUSCLE TENSION (Gamma)"
            state_desc = "High-frequency activity detected. Likely jaw clenching, frowning, or neck muscle activation."
            bg_color = "#2e1065"
            border_color = "#5b21b6"
        else:
            powers = [rel_delta, rel_theta, rel_alpha, rel_beta, rel_gamma]
            max_idx = np.argmax(powers)
            
            if max_idx == 0:
                state_text = "😴 DEEP SLEEP / DROWSY"
                state_desc = "Delta rhythms are dominant. Typical of deep sleep or very heavy drowsiness in awake subjects."
                bg_color = "#1e3a8a"
                border_color = "#3b82f6"
            elif max_idx == 1:
                state_text = "🧘 MEDITATIVE / DEEP CALM"
                state_desc = "Theta waves are dominant. Associated with deep meditation, hypnagogic states, or daydreaming."
                bg_color = "#064e3b"
                border_color = "#10b981"
            elif max_idx == 2:
                state_text = "😌 CALM & RELAXED (Alpha)"
                state_desc = "Strong Alpha peaks (8-12 Hz) in occipital region. Occurs when eyes are closed or the mind is relaxed."
                bg_color = "#78350f"
                border_color = "#f59e0b"
            elif max_idx == 3:
                state_text = "⚡ CONCENTRATING & FOCUS"
                state_desc = "Active Beta rhythm (12-30 Hz). Occurs during active concentration, thinking, problem solving, or motor tasks."
                bg_color = "#7f1d1d"
                border_color = "#ef4444"
            else:
                state_text = "🔥 HIGHLY ALERT / COGNITIVE"
                state_desc = "Gamma waves are active. Associated with multi-sensory binding, high cognitive load, or sudden insights."
                bg_color = "#4c1d95"
                border_color = "#8b5cf6"
                
        self.state_title_lbl.setText(state_text)
        self.state_desc_lbl.setText(state_desc)
        self.state_card.setStyleSheet(f"background-color: {bg_color}; border: 1px solid {border_color}; border-radius: 6px;")
        
        self.update_challenge(rel_delta, rel_theta, rel_alpha, rel_beta, std_dev)
        
    def on_challenge_changed(self, idx):
        challenges = [
            "Alpha Relaxation",
            "Beta Concentration Focus",
            "Perfect Signal Stability"
        ]
        self.current_challenge = challenges[idx]
        
        if self.current_challenge == "Alpha Relaxation":
            self.game_instruction.setText("Relax your mind and close your eyes. Generate Alpha (8-12Hz) rhythm. Target: Alpha > 35%.")
        elif self.current_challenge == "Beta Concentration Focus":
            self.game_instruction.setText("Solve math puzzles or focus deeply to generate Beta (12-30Hz) rhythms. Target: Beta > 30%.")
        elif self.current_challenge == "Perfect Signal Stability":
            self.game_instruction.setText("Sit completely still, relax your jaw, and keep eyes open without blinking. Target: Signal StdDev < 5 uV.")
            
        self.reset_challenge()
        
    def update_challenge(self, delta, theta, alpha, beta, std_dev):
        if self.challenge_completed:
            return
            
        now = time.time()
        dt = now - self.last_challenge_update
        self.last_challenge_update = now
        
        target_met = False
        current_val = 0.0
        target_val = 0.0
        
        if self.current_challenge == "Alpha Relaxation":
            current_val = alpha
            target_val = 35.0
            target_met = alpha >= target_val and std_dev < 30.0
            self.game_progress_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {COLOR_ALPHA}; }}")
            self.game_progress_bar.setFormat(f"Goal: >{target_val:.0f}% | Current: {current_val:.1f}%")
            
        elif self.current_challenge == "Beta Concentration Focus":
            current_val = beta
            target_val = 30.0
            target_met = beta >= target_val and std_dev < 30.0
            self.game_progress_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {COLOR_BETA}; }}")
            self.game_progress_bar.setFormat(f"Goal: >{target_val:.0f}% | Current: {current_val:.1f}%")
            
        elif self.current_challenge == "Perfect Signal Stability":
            current_val = std_dev
            target_val = 6.0
            target_met = std_dev <= target_val
            
            self.game_progress_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {COLOR_THETA}; }}")
            self.game_progress_bar.setFormat(f"Goal: <{target_val:.1f} uV | Current: {current_val:.1f} uV")
            
        if self.current_challenge == "Perfect Signal Stability":
            prog = max(0, min(100, int((20.0 - current_val) / (20.0 - target_val) * 100))) if current_val < 20.0 else 0
        else:
            prog = max(0, min(100, int((current_val / target_val) * 100)))
            
        self.game_progress_bar.setValue(prog)
        
        if target_met:
            self.challenge_score += dt
            if self.challenge_score >= self.challenge_success_threshold:
                self.challenge_score = self.challenge_success_threshold
                self.challenge_completed = True
                self.success_widget.setVisible(True)
        else:
            self.challenge_score = max(0.0, self.challenge_score - dt * 1.5)
            
        self.score_label.setText(f"Time Held: {self.challenge_score:.1f}s / {self.challenge_success_threshold:.1f}s")
        
    def reset_challenge(self):
        self.challenge_score = 0.0
        self.challenge_completed = False
        self.success_widget.setVisible(False)
        self.last_challenge_update = time.time()
        self.score_label.setText(f"Time Held: 0.0s / {self.challenge_success_threshold:.1f}s")
        self.game_progress_bar.setValue(0)
        
    def closeEvent(self, event):
        self.timer.stop()
        self.playback_timer.stop()
        if hasattr(self, 'httpd'):
            self.httpd.shutdown()
        winsound.PlaySound(None, winsound.SND_PURGE) # Stop sound if playing
        event.accept()

def main():
    print("=" * 80)
    print("        g.Nautilus Brainwave Rhythm Explorer & Simulator (PySide6)")
    print("=" * 80)
    
    force_sim = any(arg in sys.argv for arg in ['--fake', '--mock', '--sim'])
    inlet = None
    
    if force_sim:
        print("[*] Starting directly in Interactive Simulation Mode (Skipping LSL scan)...")
    else:
        print("[*] Checking for live LSL EEG streams on the network...")
        try:
            streams = resolve_byprop('type', 'EEG', timeout=1.2)
            if streams:
                print(f"[+] Found live LSL Stream: {streams[0].name()}")
                inlet = StreamInlet(streams[0])
            else:
                print("[-] No live LSL EEG streams detected. Starting in Interactive Simulation Mode.")
                print("    (You can toggle Simulation Mode off once gds_to_lsl.py starts streaming.)")
        except Exception as e:
            print(f"[-] Error searching for LSL stream: {e}. Defaulting to Simulation Mode.")
        
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(COLOR_BG))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(COLOR_TEXT))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(COLOR_CARD))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(COLOR_BG))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(COLOR_TEXT))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(COLOR_TEXT))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(COLOR_TEXT))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(COLOR_CARD))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(COLOR_TEXT))
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(COLOR_THETA))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)
    app.setPalette(palette)
    
    visualizer = RhythmVisualizerApp(inlet)
    visualizer.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
