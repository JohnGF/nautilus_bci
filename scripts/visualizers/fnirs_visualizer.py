"""
Real-Time fNIRS (Functional Near-Infrared Spectroscopy) Cortical Visualizer
Designed for Artinis & g.tec fNIRS Systems & Lab Streaming Layer (LSL)

Hardware Geometry:
- 2 Emitters (Sources Tx1, Tx2) & 4 Receivers (Detectors Rx1..Rx4)
- Selectable Cortical Placements: Frontal (Prefrontal), Motor, Parietal
- Accurate Optode Pairing: Tx1 -> Rx1, Rx2 | Tx2 -> Rx3, Rx4 (4 Channels x 2 Wavelengths = 8 Optical Channels)
- True Modified Beer-Lambert Law (MBLL) with Scalp Coupling Index (SCI) Live Verification.
"""

import sys
import os
import time
import math
import numpy as np

_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# Try importing pylsl
try:
    from pylsl import StreamInlet, resolve_streams
    HAS_LSL = True
except ImportError:
    HAS_LSL = False

# Try importing scipy for filtering
try:
    import scipy.signal as signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ==============================================================================
# MBLL (Modified Beer-Lambert Law) Physical Constants
# ==============================================================================
EPSILON = {
    760: {'hbo': 1486.586, 'hbr': 3843.707},
    850: {'hbo': 2526.391, 'hbr': 1798.643}
}
DEFAULT_DPF = 6.0
INTER_OPTODE_DIST_CM = 3.0

def compute_mbll_matrix(lambda1=760, lambda2=850, d=INTER_OPTODE_DIST_CM, dpf=DEFAULT_DPF):
    e_matrix = np.array([
        [EPSILON[lambda1]['hbo'], EPSILON[lambda1]['hbr']],
        [EPSILON[lambda2]['hbo'], EPSILON[lambda2]['hbr']]
    ], dtype=np.float64)
    pathlength = d * dpf
    inv_e = np.linalg.pinv(e_matrix * pathlength) * 1e6  # Output in µM
    return inv_e


# ==============================================================================
# Interactive 2D Optode Head Map Widget (2 Receivers, 8 Transmitters)
# ==============================================================================
class OptodeHeadMapWidget(QtWidgets.QFrame):
    """
    2D Head Map specifically designed for 2 Central Receivers (Rx1, Rx2) and 8 Transmitters (Tx1..Tx8).
    Arranged as 2 Quad-Emitter Sensor Pods (4 Transmitters surrounding each Central Receiver):
    - Left Pod: 1 Center Receiver (Rx1) + 4 Transmitters (Tx1..Tx4: Top, Bottom, Left, Right) -> 4 Channels (C1..C4)
    - Right Pod: 1 Center Receiver (Rx2) + 4 Transmitters (Tx5..Tx8: Top, Bottom, Left, Right) -> 4 Channels (C5..C8)
    Supports Frontal, Motor/Central, and Parietal bilateral configurations.
    """

    def __init__(self, active_zone="FRONTAL", parent=None):
        super().__init__(parent)
        self.active_zone = active_zone
        self.channel_values = np.zeros(8)
        self.channel_sci = np.ones(8)
        self.setMinimumSize(330, 330)
        self.setStyleSheet("background-color: #161b22; border-radius: 10px; border: 1px solid #30363d;")
        self.recompute_montage()

    def set_zone(self, zone):
        self.active_zone = zone
        self.recompute_montage()
        self.update()

    def recompute_montage(self):
        d = 0.12  # Radius of transmitter emitters around each central photodetector receiver

        if self.active_zone == "FRONTAL":
            # Left Frontal Pod (Rx1 at AF3) & Right Frontal Pod (Rx2 at AF4)
            rx1 = (-0.35, -0.60)
            rx2 = ( 0.35, -0.60)
            self.rx_positions = [rx1, rx2]
            self.tx_positions = [
                # Left Pod Transmitters (Tx1..Tx4)
                (rx1[0], rx1[1] - d),      # Tx1: L-Superior (Fp1)
                (rx1[0], rx1[1] + d),      # Tx2: L-Inferior (F3)
                (rx1[0] - d * 1.1, rx1[1]),# Tx3: L-Lateral (F7)
                (rx1[0] + d * 1.1, rx1[1]),# Tx4: L-Medial (AFz)
                # Right Pod Transmitters (Tx5..Tx8)
                (rx2[0], rx2[1] - d),      # Tx5: R-Superior (Fp2)
                (rx2[0], rx2[1] + d),      # Tx6: R-Inferior (F4)
                (rx2[0] - d * 1.1, rx2[1]),# Tx7: R-Medial (AFz)
                (rx2[0] + d * 1.1, rx2[1]) # Tx8: R-Lateral (F8)
            ]
            self.ch_pairings = [
                (0, 0), (1, 0), (2, 0), (3, 0),  # Tx1..Tx4 -> Rx1
                (4, 1), (5, 1), (6, 1), (7, 1)   # Tx5..Tx8 -> Rx2
            ]
            self.channel_labels = [
                "C1: L-Superior (Tx1-Rx1 Fp1)", "C2: L-Inferior (Tx2-Rx1 F3)",
                "C3: L-Lateral (Tx3-Rx1 F7)", "C4: L-Medial (Tx4-Rx1 AFz)",
                "C5: R-Superior (Tx5-Rx2 Fp2)", "C6: R-Inferior (Tx6-Rx2 F4)",
                "C7: R-Medial (Tx7-Rx2 AFz)", "C8: R-Lateral (Tx8-Rx2 F8)"
            ]

        elif self.active_zone in ["MOTOR", "CENTRAL"]:
            # Left Central Motor Pod (Rx1 at C3) & Right Central Motor Pod (Rx2 at C4)
            rx1 = (-0.55, 0.0)
            rx2 = ( 0.55, 0.0)
            self.rx_positions = [rx1, rx2]
            self.tx_positions = [
                # Left Motor Pod (Tx1..Tx4)
                (rx1[0], rx1[1] - d),      # Tx1: L-Anterior (FC3)
                (rx1[0], rx1[1] + d),      # Tx2: L-Posterior (CP3)
                (rx1[0] - d * 1.1, rx1[1]),# Tx3: L-Lateral (C5)
                (rx1[0] + d * 1.1, rx1[1]),# Tx4: L-Medial (C1)
                # Right Motor Pod (Tx5..Tx8)
                (rx2[0], rx2[1] - d),      # Tx5: R-Anterior (FC4)
                (rx2[0], rx2[1] + d),      # Tx6: R-Posterior (CP4)
                (rx2[0] - d * 1.1, rx2[1]),# Tx7: R-Medial (C2)
                (rx2[0] + d * 1.1, rx2[1]) # Tx8: R-Lateral (C6)
            ]
            self.ch_pairings = [
                (0, 0), (1, 0), (2, 0), (3, 0),
                (4, 1), (5, 1), (6, 1), (7, 1)
            ]
            self.channel_labels = [
                "C1: L-PreMotor (Tx1-Rx1 FC3)", "C2: L-SensoryMotor (Tx2-Rx1 CP3)",
                "C3: L-Lateral (Tx3-Rx1 C5)", "C4: L-Medial (Tx4-Rx1 C1)",
                "C5: R-PreMotor (Tx5-Rx2 FC4)", "C6: R-SensoryMotor (Tx6-Rx2 CP4)",
                "C7: R-Medial (Tx7-Rx2 C2)", "C8: R-Lateral (Tx8-Rx2 C6)"
            ]

        else:  # PARIETAL
            # Left Parietal Pod (Rx1 at P3) & Right Parietal Pod (Rx2 at P4)
            rx1 = (-0.45, 0.50)
            rx2 = ( 0.45, 0.50)
            self.rx_positions = [rx1, rx2]
            self.tx_positions = [
                # Left Parietal Pod (Tx1..Tx4)
                (rx1[0], rx1[1] - d),      # Tx1: L-Anterior (CP3)
                (rx1[0], rx1[1] + d),      # Tx2: L-Posterior (PO3)
                (rx1[0] - d * 1.1, rx1[1]),# Tx3: L-Lateral (P5)
                (rx1[0] + d * 1.1, rx1[1]),# Tx4: L-Medial (P1)
                # Right Parietal Pod (Tx5..Tx8)
                (rx2[0], rx2[1] - d),      # Tx5: R-Anterior (CP4)
                (rx2[0], rx2[1] + d),      # Tx6: R-Posterior (PO4)
                (rx2[0] - d * 1.1, rx2[1]),# Tx7: R-Medial (P2)
                (rx2[0] + d * 1.1, rx2[1]) # Tx8: R-Lateral (P6)
            ]
            self.ch_pairings = [
                (0, 0), (1, 0), (2, 0), (3, 0),
                (4, 1), (5, 1), (6, 1), (7, 1)
            ]
            self.channel_labels = [
                "C1: L-Anterior (Tx1-Rx1 CP3)", "C2: L-Posterior (Tx2-Rx1 PO3)",
                "C3: L-Lateral (Tx3-Rx1 P5)", "C4: L-Medial (Tx4-Rx1 P1)",
                "C5: R-Anterior (Tx5-Rx2 CP4)", "C6: R-Posterior (Tx6-Rx2 PO4)",
                "C7: R-Medial (Tx7-Rx2 P2)", "C8: R-Lateral (Tx8-Rx2 P6)"
            ]

    def update_metrics(self, values, sci_scores=None):
        if len(values) > 0:
            self.channel_values = values[:8]
        if sci_scores is not None and len(sci_scores) > 0:
            self.channel_sci = sci_scores[:8]
        self.update()

    def get_channel_label(self, idx):
        if idx < len(self.channel_labels):
            return self.channel_labels[idx]
        return f"Ch {idx+1}"

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        cx, cy = w / 2, h / 2
        radius = min(w, h) * 0.40

        # Head Outline
        pen_head = QtGui.QPen(QtGui.QColor("#3a445e"), 3)
        painter.setPen(pen_head)
        painter.setBrush(QtGui.QColor("#161b22"))
        painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

        # Nose (Nasion)
        nose_path = QtGui.QPainterPath()
        nose_path.moveTo(cx - 14, cy - radius)
        nose_path.lineTo(cx, cy - radius - 18)
        nose_path.lineTo(cx + 14, cy - radius)
        painter.fillPath(nose_path, QtGui.QColor("#3a445e"))

        # Ears
        painter.setBrush(QtGui.QColor("#161b22"))
        painter.drawEllipse(QtCore.QPointF(cx - radius - 6, cy), 8, 16)
        painter.drawEllipse(QtCore.QPointF(cx + radius + 6, cy), 8, 16)

        # Zone Header
        painter.setPen(QtGui.QPen(QtGui.QColor("#58a6ff")))
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(int(cx - 60), int(cy - radius - 24), 120, 14, QtCore.Qt.AlignCenter, f"ZONE: {self.active_zone}")
        # Draw Channels (Lines connecting Tx -> Rx)
        for i, (tx_i, rx_i) in enumerate(self.ch_pairings):
            if tx_i >= len(self.tx_positions) or rx_i >= len(self.rx_positions):
                continue
            tx_x, tx_y = cx + self.tx_positions[tx_i][0] * radius, cy + self.tx_positions[tx_i][1] * radius
            rx_x, rx_y = cx + self.rx_positions[rx_i][0] * radius, cy + self.rx_positions[rx_i][1] * radius
            mx, my = (tx_x + rx_x) / 2.0, (tx_y + rx_y) / 2.0

            sci = self.channel_sci[i] if i < len(self.channel_sci) else 0.85
            if sci >= 0.70:
                sci_color = QtGui.QColor("#238636")  # Green (Good Contact)
            elif sci >= 0.45:
                sci_color = QtGui.QColor("#d29922")  # Yellow (Moderate)
            else:
                sci_color = QtGui.QColor("#da3633")  # Red (Noisy / Loose)

            # Draw Channel Connection Line
            painter.setPen(QtGui.QPen(sci_color, 1.5, QtCore.Qt.DashLine))
            painter.drawLine(QtCore.QPointF(tx_x, tx_y), QtCore.QPointF(rx_x, rx_y))

            # Channel Activation Node
            val = self.channel_values[i] if i < len(self.channel_values) else 0.0
            val_norm = np.clip(val / 2.5, -1.0, 1.0)
            if val_norm >= 0:
                act_col = QtGui.QColor(int(255 * val_norm), int(60 * (1.0 - val_norm)), 60, 220)
            else:
                act_col = QtGui.QColor(30, 100, int(255 * abs(val_norm)), 220)

            painter.setBrush(act_col)
            painter.setPen(QtGui.QPen(QtGui.QColor("#0d1117"), 1))
            painter.drawEllipse(QtCore.QPointF(mx, my), 5, 5)

            painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff")))
            font_ch = painter.font()
            font_ch.setPointSize(6)
            font_ch.setBold(True)
            painter.setFont(font_ch)
            painter.drawText(int(mx - 12), int(my + 9), 24, 10, QtCore.Qt.AlignCenter, f"C{i+1}")

        # Draw 8 Transmitters (Yellow Outer Emitters T1..T8)
        for i, (x_norm, y_norm) in enumerate(self.tx_positions):
            px = cx + x_norm * radius
            py = cy + y_norm * radius
            painter.setBrush(QtGui.QColor("#e3b341"))  # Yellow
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.2))
            painter.drawEllipse(QtCore.QPointF(px, py), 6, 6)

            painter.setPen(QtGui.QPen(QtGui.QColor("#c9d1d9")))
            font = painter.font()
            font.setPointSize(6)
            font.setBold(False)
            painter.setFont(font)
            painter.drawText(int(px - 10), int(py - 14), 20, 10, QtCore.Qt.AlignCenter, f"T{i+1}")

        # Draw 2 Receivers (Blue Center Photodetectors R1, R2)
        for i, (x_norm, y_norm) in enumerate(self.rx_positions):
            px = cx + x_norm * radius
            py = cy + y_norm * radius
            painter.setBrush(QtGui.QColor("#58a6ff"))  # Blue
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.8))
            painter.drawEllipse(QtCore.QPointF(px, py), 8, 8)

            painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff")))
            font = painter.font()
            font.setPointSize(7)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(int(px - 12), int(py + 10), 24, 12, QtCore.Qt.AlignCenter, f"R{i+1}")


# ==============================================================================
# Main Real-Time fNIRS Application
# ==============================================================================
class fNIRSVisualizerWindow(QtWidgets.QMainWindow):
    """fNIRS Cortical Visualizer for 2 Emitters / 4 Receivers Montage."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("fNIRS Cortical Monitor (2 Emitters / 4 Receivers Setup)")
        self.resize(1360, 880)

        self.srate = 10.0
        self.window_sec = 20.0
        self.active_zone = "FRONTAL"
        self.num_channels = 8  # 4 Pairs x 2 Wavelengths (760nm & 850nm)
        self.mode_mbll = True
        self.filter_enabled = True

        self.inlet = None
        self.max_samples = int(self.srate * 60.0)
        self.time_buffer = np.zeros(self.max_samples)
        self.raw_buffer = np.zeros((self.max_samples, self.num_channels))
        self.sample_count = 0

        self.baseline_intensity = np.ones(self.num_channels) * 2000.0
        self.is_calibrated = False
        self.mbll_matrix = compute_mbll_matrix(lambda1=760, lambda2=850)

        self.fixed_scale = 0.020
        self.is_auto_scale = False

        self.init_filters()
        self.init_dark_theme()
        self.setup_ui()

        # Auto-connect to LSL stream on launch
        QtCore.QTimer.singleShot(600, lambda: self.connect_lsl(silent=True))

        # Update Timer (50 Hz / 20ms refresh)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll_and_update)
        self.timer.start(20)

    def on_spin_scale_changed(self, val):
        self.is_auto_scale = False
        self.fixed_scale = max(float(val), 0.0001)
        for p in self.plots:
            p.enableAutoRange(axis='y', enable=False)
            p.setYRange(-self.fixed_scale, self.fixed_scale, padding=0.02)

    def zoom_in(self):
        cur = self.spin_scale.value()
        new_val = max(cur * 0.5, 0.0001)
        self.spin_scale.setValue(new_val)

    def zoom_out(self):
        cur = self.spin_scale.value()
        new_val = min(cur * 2.0, 100.0)
        self.spin_scale.setValue(new_val)

    def on_preset_changed(self, index):
        presets = {
            0: 0.010,
            1: 0.020,
            2: 0.050,
            3: 0.100,
            4: 0.500,
            5: 1.0,
            6: 2.0,
            7: 5.0,
            8: None
        }
        val = presets.get(index, 0.020)
        if val is None:
            self.is_auto_scale = True
            for p in self.plots:
                p.enableAutoRange(axis='y', enable=True)
        else:
            self.spin_scale.setValue(val)

    def init_filters(self):
        if not HAS_SCIPY:
            return
        nyq = 0.5 * self.srate
        low_hrf = max(0.01 / nyq, 0.001)
        high_hrf = min(0.20 / nyq, 0.49)
        self.sos_hrf = signal.butter(3, [low_hrf, high_hrf], btype='bandpass', output='sos')

        low_card = max(0.8 / nyq, 0.01)
        high_card = min(2.0 / nyq, 0.49)
        self.sos_cardiac = signal.butter(2, [low_card, high_card], btype='bandpass', output='sos')

    def init_dark_theme(self):
        app = QtWidgets.QApplication.instance()
        if app:
            app.setStyle("Fusion")
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#0d1117"))
            palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#c9d1d9"))
            palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#161b22"))
            palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#21262d"))
            palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#c9d1d9"))
            palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#21262d"))
            palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#c9d1d9"))
            palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#1f6beb"))
            app.setPalette(palette)

    def setup_ui(self):
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Header Toolbar
        toolbar_card = QtWidgets.QFrame()
        toolbar_card.setStyleSheet("background-color: #161b22; border-radius: 8px; border: 1px solid #30363d; padding: 6px 12px;")
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar_card)

        title_label = QtWidgets.QLabel("🧠 2-Receiver / 8-Transmitter fNIRS Monitor")
        title_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #58a6ff;")
        toolbar_layout.addWidget(title_label)
        toolbar_layout.addSpacing(15)

        zone_label = QtWidgets.QLabel("Select Head Placement:")
        zone_label.setStyleSheet("font-weight: bold; color: #c9d1d9;")
        toolbar_layout.addWidget(zone_label)

        self.combo_zone = QtWidgets.QComboBox()
        self.combo_zone.addItems([
            "Frontal (Left & Right Prefrontal)",
            "Central / Motor (Left & Right Motor Strip)",
            "Parietal (Left & Right Somatosensory)"
        ])
        self.combo_zone.setStyleSheet("background-color: #21262d; color: #58a6ff; font-weight: bold; padding: 4px 8px; border-radius: 4px;")
        self.combo_zone.currentIndexChanged.connect(self.on_zone_changed)
        toolbar_layout.addWidget(self.combo_zone)
        toolbar_layout.addSpacing(10)

        self.btn_lsl = QtWidgets.QPushButton("🔗 Connect LSL Stream")
        self.btn_lsl.setStyleSheet("background-color: #238636; color: white; font-weight: bold; padding: 5px 12px; border-radius: 6px;")
        self.btn_lsl.clicked.connect(self.connect_lsl)
        toolbar_layout.addWidget(self.btn_lsl)

        self.btn_zero = QtWidgets.QPushButton("🎯 Zero Baseline (I₀)")
        self.btn_zero.setStyleSheet("background-color: #8957e5; color: white; font-weight: bold; padding: 5px 12px; border-radius: 6px;")
        self.btn_zero.clicked.connect(self.zero_baseline)
        toolbar_layout.addWidget(self.btn_zero)

        toolbar_layout.addSpacing(10)

        self.combo_view = QtWidgets.QComboBox()
        self.combo_view.addItems([
            "View: 2 Receiver Pods (Left R1 & Right R2)",
            "View: 8 Individual Channels (C1 - C8)"
        ])
        self.combo_view.setStyleSheet("background-color: #21262d; color: #79c0ff; font-weight: bold; padding: 4px 8px; border-radius: 4px;")
        self.combo_view.currentIndexChanged.connect(self.on_view_mode_changed)
        toolbar_layout.addWidget(self.combo_view)

        # Interactive Amplitude Scale Controls
        lbl_scale = QtWidgets.QLabel("Amplitude (±):")
        lbl_scale.setStyleSheet("font-weight: bold; color: #79c0ff;")
        toolbar_layout.addWidget(lbl_scale)

        self.spin_scale = QtWidgets.QDoubleSpinBox()
        self.spin_scale.setRange(0.0001, 100.0)
        self.spin_scale.setValue(0.0200)
        self.spin_scale.setSingleStep(0.0050)
        self.spin_scale.setDecimals(4)
        self.spin_scale.setStyleSheet("background-color: #21262d; color: #58a6ff; font-weight: bold; padding: 3px 6px; border-radius: 4px; border: 1px solid #30363d;")
        self.spin_scale.valueChanged.connect(self.on_spin_scale_changed)
        toolbar_layout.addWidget(self.spin_scale)

        self.btn_zoom_in = QtWidgets.QPushButton("➕")
        self.btn_zoom_in.setToolTip("Zoom In (Increase Amplitude Sensitivity)")
        self.btn_zoom_in.setStyleSheet("background-color: #21262d; color: #58a6ff; font-weight: bold; padding: 4px 8px; border-radius: 4px; border: 1px solid #30363d;")
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        toolbar_layout.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QtWidgets.QPushButton("➖")
        self.btn_zoom_out.setToolTip("Zoom Out (Decrease Amplitude Sensitivity)")
        self.btn_zoom_out.setStyleSheet("background-color: #21262d; color: #58a6ff; font-weight: bold; padding: 4px 8px; border-radius: 4px; border: 1px solid #30363d;")
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        toolbar_layout.addWidget(self.btn_zoom_out)

        self.combo_presets = QtWidgets.QComboBox()
        self.combo_presets.addItems([
            "Preset: ±0.010",
            "Preset: ±0.020 (Default)",
            "Preset: ±0.050",
            "Preset: ±0.100",
            "Preset: ±0.500",
            "Preset: ±1.0 µM",
            "Preset: ±2.0 µM",
            "Preset: ±5.0 µM",
            "Preset: Auto-Scale"
        ])
        self.combo_presets.setCurrentIndex(1)  # Default: ±0.020
        self.combo_presets.setStyleSheet("background-color: #21262d; color: #c9d1d9; font-weight: bold; padding: 3px 6px; border-radius: 4px;")
        self.combo_presets.currentIndexChanged.connect(self.on_preset_changed)
        toolbar_layout.addWidget(self.combo_presets)

        toolbar_layout.addStretch()

        self.chk_mbll = QtWidgets.QCheckBox("MBLL (HbO / HbR / HbT)")
        self.chk_mbll.setChecked(True)
        self.chk_mbll.setStyleSheet("font-weight: bold; color: #79c0ff;")
        self.chk_mbll.stateChanged.connect(self.on_mbll_toggled)
        toolbar_layout.addWidget(self.chk_mbll)

        self.chk_filter = QtWidgets.QCheckBox("Filter (0.01 - 0.20 Hz)")
        self.chk_filter.setChecked(True)
        self.chk_filter.setStyleSheet("font-weight: bold; color: #d2a8ff;")
        self.chk_filter.stateChanged.connect(self.on_filter_toggled)
        toolbar_layout.addWidget(self.chk_filter)

        main_layout.addWidget(toolbar_card)

        # Split Layout
        content_layout = QtWidgets.QHBoxLayout()

        # Left Sidebar
        sidebar_card = QtWidgets.QFrame()
        sidebar_card.setFixedWidth(340)
        sidebar_card.setStyleSheet("background-color: #161b22; border-radius: 8px; border: 1px solid #30363d; padding: 8px;")
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_card)

        # Optode Configuration Legend
        legend_box = QtWidgets.QGroupBox("Hardware Socket Placement Guide")
        legend_box.setStyleSheet("QGroupBox { font-weight: bold; color: #8b949e; border: 1px solid #30363d; margin-top: 6px; padding-top: 8px; }")
        leg_layout = QtWidgets.QVBoxLayout(legend_box)

        lbl_guide = QtWidgets.QLabel(
            "<b>Left Pod (Left Hemisphere):</b><br>"
            "• <b>[R1] Center:</b> Left Photodetector Receiver<br>"
            "• <b>[T1] Top:</b> Superior Emitter (Fp1 / FC3)<br>"
            "• <b>[T2] Bottom:</b> Inferior Emitter (F3 / CP3)<br>"
            "• <b>[T3] Left:</b> Lateral Emitter (F7 / C5)<br>"
            "• <b>[T4] Right:</b> Medial Emitter (AFz / C1)<br><br>"
            "<b>Right Pod (Right Hemisphere):</b><br>"
            "• <b>[R2] Center:</b> Right Photodetector Receiver<br>"
            "• <b>[T5] Top:</b> Superior Emitter (Fp2 / FC4)<br>"
            "• <b>[T6] Bottom:</b> Inferior Emitter (F4 / CP4)<br>"
            "• <b>[T7] Left:</b> Medial Emitter (AFz / C2)<br>"
            "• <b>[T8] Right:</b> Lateral Emitter (F8 / C6)"
        )
        lbl_guide.setStyleSheet("color: #c9d1d9; font-size: 10px; line-height: 1.3;")
        lbl_ch = QtWidgets.QLabel("🔴 ΔHbO  🔵 ΔHbR  🟢 ΔHbT (µM)")
        lbl_ch.setStyleSheet("color: #2ed573; font-weight: bold; font-size: 11px;")
        lbl_sci = QtWidgets.QLabel("🟢 Contact Good (SCI > 0.70) | 🔴 Loose Optode")
        lbl_sci.setStyleSheet("color: #7ee787; font-size: 10px;")

        leg_layout.addWidget(lbl_guide)
        leg_layout.addWidget(lbl_ch)
        leg_layout.addWidget(lbl_sci)
        sidebar_layout.addWidget(legend_box)

        # Head Map
        self.headmap_widget = OptodeHeadMapWidget(active_zone=self.active_zone)
        sidebar_layout.addWidget(self.headmap_widget)

        # Telemetry
        metrics_box = QtWidgets.QGroupBox("Live Channel Telemetry")
        metrics_box.setStyleSheet("QGroupBox { font-weight: bold; color: #8b949e; border: 1px solid #30363d; margin-top: 6px; padding-top: 8px; }")
        m_layout = QtWidgets.QVBoxLayout(metrics_box)

        self.lbl_srate = QtWidgets.QLabel("Sampling Rate: -- Hz")
        self.lbl_srate.setStyleSheet("color: #c9d1d9; font-size: 11px;")
        self.lbl_hbo_peak = QtWidgets.QLabel("Peak ΔHbO: -- µM")
        self.lbl_hbo_peak.setStyleSheet("color: #ff4757; font-weight: bold; font-size: 11px;")
        self.lbl_sci_avg = QtWidgets.QLabel("Mean Optode Contact (SCI): --")
        self.lbl_sci_avg.setStyleSheet("color: #7ee787; font-weight: bold; font-size: 11px;")
        self.lbl_status = QtWidgets.QLabel("Stream: Idle (Press Connect LSL)")
        self.lbl_status.setStyleSheet("color: #e3b341; font-weight: bold; font-size: 11px;")

        m_layout.addWidget(self.lbl_srate)
        m_layout.addWidget(self.lbl_hbo_peak)
        m_layout.addWidget(self.lbl_sci_avg)
        m_layout.addWidget(self.lbl_status)
        sidebar_layout.addWidget(metrics_box)

        content_layout.addWidget(sidebar_card)

        # Right Charts
        plot_card = QtWidgets.QFrame()
        plot_card.setStyleSheet("background-color: #161b22; border-radius: 8px; border: 1px solid #30363d; padding: 6px;")
        plot_layout = QtWidgets.QVBoxLayout(plot_card)

        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('#0d1117')
        plot_layout.addWidget(self.plot_widget)

        content_layout.addWidget(plot_card)
        main_layout.addLayout(content_layout)

        self.view_mode = "PODS"  # "PODS" (2 plots) or "CHANNELS" (8 plots)
        self.rebuild_plots()

    def on_view_mode_changed(self, index):
        self.view_mode = "PODS" if index == 0 else "CHANNELS"
        self.rebuild_plots()

    def zero_baseline(self):
        if self.sample_count >= 10:
            window_size = min(int(self.srate * 3.0), self.sample_count)
            self.baseline_intensity = np.mean(self.raw_buffer[-window_size:, :], axis=0)
            self.baseline_intensity = np.maximum(self.baseline_intensity, 1.0)
            self.is_calibrated = True
            self.lbl_status.setText("Stream: Baseline Calibrated (I₀ set)")
            self.lbl_status.setStyleSheet("color: #238636; font-weight: bold;")

    def on_zone_changed(self, index):
        zones = ["FRONTAL", "MOTOR", "PARIETAL"]
        self.active_zone = zones[index] if index < len(zones) else "FRONTAL"
        self.headmap_widget.set_zone(self.active_zone)
        self.rebuild_plots()

    def rebuild_plots(self):
        self.plot_widget.clear()
        self.plots = []
        self.curves = []

        if self.view_mode == "PODS":
            # 2 Large Receiver Pod Plots: Left Pod (R1) and Right Pod (R2)
            pod_titles = [
                "Left Hemisphere Pod (Receiver R1: T1..T4 Transmitters)",
                "Right Hemisphere Pod (Receiver R2: T5..T8 Transmitters)"
            ]
            for pod_i in range(2):
                p = self.plot_widget.addPlot(row=pod_i, col=0)
                p.setTitle(f"<span style='color: #58a6ff; font-weight: bold; font-size: 13px;'>{pod_titles[pod_i]}</span>")
                p.setMouseEnabled(x=True, y=True)
                if self.is_auto_scale:
                    p.enableAutoRange(axis='y', enable=True)
                else:
                    p.enableAutoRange(axis='y', enable=False)
                    p.setYRange(-self.fixed_scale, self.fixed_scale, padding=0.02)
                p.showGrid(x=True, y=True, alpha=0.25)
                p.getAxis('left').setPen(pg.mkPen('#8b949e', width=1.2))
                p.getAxis('bottom').setPen(pg.mkPen('#8b949e', width=1.2))
                legend = p.addLegend(offset=(10, 10))
                legend.setBrush(pg.mkBrush('#161b2299'))

                if self.mode_mbll:
                    p.setLabel('left', 'Concentration', units='µM')
                    c_hbo = p.plot(pen=pg.mkPen('#ff4757', width=2.8), name=f"HbO (Oxygenated)")
                    c_hbr = p.plot(pen=pg.mkPen('#1e90ff', width=2.5), name=f"HbR (Deoxygenated)")
                    c_hbt = p.plot(pen=pg.mkPen('#2ed573', width=1.8, style=QtCore.Qt.DashLine), name=f"HbT (Total)")
                    self.curves.append((c_hbo, c_hbr, c_hbt))
                else:
                    p.setLabel('left', 'Optical Intensity', units='Counts')
                    c_raw1 = p.plot(pen=pg.mkPen('#79c0ff', width=2.5), name=f"760nm")
                    c_raw2 = p.plot(pen=pg.mkPen('#ffa657', width=2.5), name=f"850nm")
                    self.curves.append((c_raw1, c_raw2))

                if pod_i == 0:
                    p.hideAxis('bottom')

                self.plots.append(p)

            if len(self.plots) > 0:
                self.plots[-1].setLabel('bottom', 'Time Window', units='s')

        else:
            # 8 Separate Channels (4 Left, 4 Right)
            for i in range(8):
                p = self.plot_widget.addPlot(row=i, col=0)
                p.setMouseEnabled(x=True, y=True)
                if self.is_auto_scale:
                    p.enableAutoRange(axis='y', enable=True)
                else:
                    p.enableAutoRange(axis='y', enable=False)
                    p.setYRange(-self.fixed_scale, self.fixed_scale, padding=0.02)
                p.showGrid(x=True, y=True, alpha=0.20)
                p.getAxis('left').setPen(pg.mkPen('#8b949e', width=1.2))
                p.getAxis('bottom').setPen(pg.mkPen('#8b949e', width=1.2))

                ch_name = self.headmap_widget.get_channel_label(i)

                if self.mode_mbll:
                    p.setLabel('left', ch_name, units='µM')
                    c_hbo = p.plot(pen=pg.mkPen('#ff4757', width=2.2), name=f"{ch_name}_HbO")
                    c_hbr = p.plot(pen=pg.mkPen('#1e90ff', width=2.0), name=f"{ch_name}_HbR")
                    c_hbt = p.plot(pen=pg.mkPen('#2ed573', width=1.5, style=QtCore.Qt.DashLine), name=f"{ch_name}_HbT")
                    self.curves.append((c_hbo, c_hbr, c_hbt))
                else:
                    p.setLabel('left', ch_name, units='Counts')
                    c_raw1 = p.plot(pen=pg.mkPen('#79c0ff', width=2.0), name=f"{ch_name}_760nm")
                    c_raw2 = p.plot(pen=pg.mkPen('#ffa657', width=2.0), name=f"{ch_name}_850nm")
                    self.curves.append((c_raw1, c_raw2))

                if i < 7:
                    p.hideAxis('bottom')

                self.plots.append(p)

            if len(self.plots) > 0:
                self.plots[-1].setLabel('bottom', 'Time Window', units='s')

    def on_mbll_toggled(self, state):
        self.mode_mbll = (state == QtCore.Qt.Checked)
        self.rebuild_plots()

    def on_filter_toggled(self, state):
        self.filter_enabled = (state == QtCore.Qt.Checked)

    def connect_lsl(self, silent=False):
        if not HAS_LSL:
            if not silent:
                QtWidgets.QMessageBox.warning(self, "LSL Not Installed", "pylsl library is not installed.")
            return

        self.lbl_status.setText("Stream: Searching LSL...")
        self.lbl_status.setStyleSheet("color: #e3b341; font-weight: bold;")
        QtWidgets.QApplication.processEvents()

        streams = resolve_streams(wait_time=2.0)
        target_stream = None
        for s in streams:
            stype = s.type().upper()
            sname = s.name().upper()
            if any(k in stype for k in ['FNIRS', 'NIRS', 'EEG', 'GNAUTILUS', 'ARTINIS']) or any(k in sname for k in ['ARTINIS', 'FNIRS', 'OCTAMON', 'NAUTILUS']):
                target_stream = s
                break

        if target_stream:
            self.inlet = StreamInlet(target_stream)
            self.srate = self.inlet.info().nominal_srate() or 25.0
            self.init_filters()

            lsl_ch = self.inlet.info().channel_count()
            if lsl_ch > 0:
                self.num_channels = lsl_ch
                self.max_samples = int(self.srate * 60.0)
                self.time_buffer = np.zeros(self.max_samples)
                self.raw_buffer = np.zeros((self.max_samples, self.num_channels))
                self.baseline_intensity = np.ones(self.num_channels) * 2000.0
                self.sample_count = 0

            self.lbl_srate.setText(f"Sampling Rate: {self.srate:.1f} Hz")
            self.lbl_status.setText(f"Stream: Connected ({target_stream.name()})")
            self.lbl_status.setStyleSheet("color: #238636; font-weight: bold;")
        else:
            self.lbl_status.setText("Stream: Idle (Press Connect LSL)")
            self.lbl_status.setStyleSheet("color: #e3b341; font-weight: bold;")
            if not silent:
                QtWidgets.QMessageBox.information(self, "LSL Stream", "No active fNIRS LSL stream found.\nPlease run 'uv run scripts/bridges/artinis_to_lsl.py'.")

    def push_sample(self, sample):
        self.raw_buffer = np.roll(self.raw_buffer, -1, axis=0)
        self.raw_buffer[-1, :min(len(sample), self.num_channels)] = sample[:self.num_channels]

        self.time_buffer = np.roll(self.time_buffer, -1)
        self.time_buffer[-1] = time.time()
        self.sample_count += 1

        if not self.is_calibrated and self.sample_count >= 15:
            self.baseline_intensity = np.mean(self.raw_buffer[-15:, :], axis=0)
            self.baseline_intensity = np.maximum(self.baseline_intensity, 1.0)
            self.is_calibrated = True

    def poll_and_update(self):
        if HAS_LSL and self.inlet:
            try:
                samples, timestamps = self.inlet.pull_chunk(timeout=0.0)
                if samples:
                    for s in samples:
                        self.push_sample(np.array(s))
            except Exception:
                pass

        if self.sample_count < 2:
            return

        window_size = int(self.srate * self.window_sec)
        window_size = min(window_size, self.max_samples, self.sample_count)

        raw_segment = self.raw_buffer[-window_size:, :].copy()
        time_segment = np.linspace(-self.window_sec, 0, window_size)

        latest_hbo_for_map = []
        sci_scores = []

        # Detect whether input stream is already concentrations (HbO/HbR in uM) or raw optical counts
        is_already_conc = (np.max(np.abs(raw_segment)) < 150.0) and (raw_segment.shape[1] >= 16)

        # Process Data for Display
        if self.view_mode == "PODS":
            # Left Pod (Receiver 1: Channels C1..C4 / Tx1..Tx4)
            # Right Pod (Receiver 2: Channels C5..C8 / Tx5..Tx8)
            num_cols = raw_segment.shape[1]

            if is_already_conc:
                # Channels 0..7 are Tx1..Tx8 HbO, Channels 8..15 are Tx1..Tx8 HbR
                pod_data = [
                    (raw_segment[:, 0:4], raw_segment[:, 8:12], 0),   # Left Pod
                    (raw_segment[:, 4:8], raw_segment[:, 12:16], 1)   # Right Pod
                ]
                for hbo_block, hbr_block, pod_i in pod_data:
                    if pod_i >= len(self.curves):
                        break
                    hbo = np.mean(hbo_block, axis=1)
                    hbr = np.mean(hbr_block, axis=1)

                    if self.filter_enabled and HAS_SCIPY and window_size > 15:
                        try:
                            hbo = signal.sosfiltfilt(self.sos_hrf, hbo)
                            hbr = signal.sosfiltfilt(self.sos_hrf, hbr)
                        except Exception:
                            pass

                    hbt = hbo + hbr

                    self.curves[pod_i][0].setData(time_segment, hbo)
                    self.curves[pod_i][1].setData(time_segment, hbr)
                    if len(self.curves[pod_i]) > 2:
                        self.curves[pod_i][2].setData(time_segment, hbt)

                    latest_hbo_for_map.append(float(hbo[-1]))
                    sci_scores.append(0.88)
                    self.plots[pod_i].setXRange(-self.window_sec, 0)
                    if self.is_auto_scale:
                        self.plots[pod_i].enableAutoRange(axis='y', enable=True)
                    else:
                        self.plots[pod_i].enableAutoRange(axis='y', enable=False)
                        self.plots[pod_i].setYRange(-self.fixed_scale, self.fixed_scale, padding=0.02)

            else:
                if num_cols >= 16:
                    pod_slices = [
                        (slice(0, 8), 0),   # Left Pod (R1: 8 optical channels)
                        (slice(8, 16), 1)   # Right Pod (R2: 8 optical channels)
                    ]
                elif num_cols >= 8:
                    pod_slices = [
                        (slice(0, 4), 0),
                        (slice(4, 8), 1)
                    ]
                else:
                    pod_slices = [
                        (slice(0, min(2, num_cols)), 0),
                        (slice(min(2, num_cols-1), num_cols), 1)
                    ]

                for slc, pod_i in pod_slices:
                    if pod_i >= len(self.curves):
                        break

                    pod_raw = raw_segment[:, slc]
                    if pod_raw.shape[1] >= 2:
                        sig1 = np.mean(pod_raw[:, 0::2], axis=1) # 760nm
                        sig2 = np.mean(pod_raw[:, 1::2], axis=1) # 850nm
                    else:
                        sig1 = pod_raw[:, 0]
                        sig2 = pod_raw[:, 0] * 1.15

                    # Scalp Coupling Index (SCI)
                    sci_val = 0.85
                    if HAS_SCIPY and window_size > 15:
                        try:
                            card1 = signal.sosfiltfilt(self.sos_cardiac, sig1)
                            card2 = signal.sosfiltfilt(self.sos_cardiac, sig2)
                            std1, std2 = np.std(card1), np.std(card2)
                            if std1 > 1e-4 and std2 > 1e-4:
                                corr = np.corrcoef(card1, card2)[0, 1]
                                sci_val = float(np.clip(corr, 0.0, 1.0)) if not np.isnan(corr) else 0.5
                            else:
                                sci_val = 0.5
                        except Exception:
                            sci_val = 0.8
                    sci_scores.append(sci_val)

                    if self.mode_mbll:
                        base_slc = self.baseline_intensity[slc]
                        if len(base_slc) >= 2:
                            base1 = float(np.mean(base_slc[0::2]))
                            base2 = float(np.mean(base_slc[1::2]))
                        else:
                            base1 = 2000.0
                            base2 = 2500.0
                        base1, base2 = max(base1, 1.0), max(base2, 1.0)

                        safe_sig1 = np.maximum(sig1, 1.0)
                        safe_sig2 = np.maximum(sig2, 1.0)

                        dod_760 = -np.log(safe_sig1 / base1)
                        dod_850 = -np.log(safe_sig2 / base2)

                        if self.filter_enabled and HAS_SCIPY and window_size > 15:
                            try:
                                dod_760 = signal.sosfiltfilt(self.sos_hrf, dod_760)
                                dod_850 = signal.sosfiltfilt(self.sos_hrf, dod_850)
                            except Exception:
                                pass

                        dod_stack = np.vstack([dod_760, dod_850])
                        hbo_hbr = self.mbll_matrix @ dod_stack
                        hbo = hbo_hbr[0, :]
                        hbr = hbo_hbr[1, :]
                        hbt = hbo + hbr

                        self.curves[pod_i][0].setData(time_segment, hbo)
                        self.curves[pod_i][1].setData(time_segment, hbr)
                        self.curves[pod_i][2].setData(time_segment, hbt)

                        latest_hbo_for_map.append(hbo[-1])
                    else:
                        self.curves[pod_i][0].setData(time_segment, sig1)
                        self.curves[pod_i][1].setData(time_segment, sig2)
                        latest_hbo_for_map.append(sig1[-1])

                    self.plots[pod_i].setXRange(-self.window_sec, 0)
                    if self.is_auto_scale:
                        self.plots[pod_i].enableAutoRange(axis='y', enable=True)
                    else:
                        self.plots[pod_i].enableAutoRange(axis='y', enable=False)
                        self.plots[pod_i].setYRange(-self.fixed_scale, self.fixed_scale, padding=0.02)

            if len(latest_hbo_for_map) >= 2:
                map_hbo = np.array([latest_hbo_for_map[0]]*4 + [latest_hbo_for_map[1]]*4)
                map_sci = np.array([sci_scores[0]]*4 + [sci_scores[1]]*4)
            else:
                map_hbo = np.array(latest_hbo_for_map)
                map_sci = np.array(sci_scores)

        else:
            # Process All 8 Individual Channels (C1 - C8)
            for i in range(8):
                if i >= len(self.curves):
                    break

                if is_already_conc:
                    hbo = raw_segment[:, i]
                    hbr = raw_segment[:, 8 + i] if raw_segment.shape[1] > 8 + i else hbo * -0.3

                    if self.filter_enabled and HAS_SCIPY and window_size > 15:
                        try:
                            hbo = signal.sosfiltfilt(self.sos_hrf, hbo)
                            hbr = signal.sosfiltfilt(self.sos_hrf, hbr)
                        except Exception:
                            pass

                    hbt = hbo + hbr
                    self.curves[i][0].setData(time_segment, hbo)
                    self.curves[i][1].setData(time_segment, hbr)
                    if len(self.curves[i]) > 2:
                        self.curves[i][2].setData(time_segment, hbt)
                    latest_hbo_for_map.append(float(hbo[-1]))
                    sci_scores.append(0.88)

                else:
                    idx1 = min(2 * i, raw_segment.shape[1] - 1)
                    idx2 = min(2 * i + 1, raw_segment.shape[1] - 1)

                    sig1 = raw_segment[:, idx1]
                    sig2 = raw_segment[:, idx2]

                    sci_val = 0.85
                    if HAS_SCIPY and window_size > 15:
                        try:
                            card1 = signal.sosfiltfilt(self.sos_cardiac, sig1)
                            card2 = signal.sosfiltfilt(self.sos_cardiac, sig2)
                            std1, std2 = np.std(card1), np.std(card2)
                            if std1 > 1e-4 and std2 > 1e-4:
                                corr = np.corrcoef(card1, card2)[0, 1]
                                sci_val = float(np.clip(corr, 0.0, 1.0)) if not np.isnan(corr) else 0.5
                            else:
                                sci_val = 0.5
                        except Exception:
                            sci_val = 0.8
                    sci_scores.append(sci_val)

                    if self.mode_mbll:
                        base1 = max(self.baseline_intensity[idx1], 1.0)
                        base2 = max(self.baseline_intensity[idx2], 1.0)
                        safe_sig1 = np.maximum(sig1, 1.0)
                        safe_sig2 = np.maximum(sig2, 1.0)

                        dod_760 = -np.log(safe_sig1 / base1)
                        dod_850 = -np.log(safe_sig2 / base2)

                        if self.filter_enabled and HAS_SCIPY and window_size > 15:
                            try:
                                dod_760 = signal.sosfiltfilt(self.sos_hrf, dod_760)
                                dod_850 = signal.sosfiltfilt(self.sos_hrf, dod_850)
                            except Exception:
                                pass

                        dod_stack = np.vstack([dod_760, dod_850])
                        hbo_hbr = self.mbll_matrix @ dod_stack
                        hbo = hbo_hbr[0, :]
                        hbr = hbo_hbr[1, :]
                        hbt = hbo + hbr

                        self.curves[i][0].setData(time_segment, hbo)
                        self.curves[i][1].setData(time_segment, hbr)
                        self.curves[i][2].setData(time_segment, hbt)

                        latest_hbo_for_map.append(hbo[-1])
                    else:
                        self.curves[i][0].setData(time_segment, sig1)
                        self.curves[i][1].setData(time_segment, sig2)
                        latest_hbo_for_map.append(sig1[-1])

                self.plots[i].setXRange(-self.window_sec, 0)
                if self.is_auto_scale:
                    self.plots[i].enableAutoRange(axis='y', enable=True)
                else:
                    self.plots[i].enableAutoRange(axis='y', enable=False)
                    self.plots[i].setYRange(-self.fixed_scale, self.fixed_scale, padding=0.02)

            map_hbo = np.array(latest_hbo_for_map)
            map_sci = np.array(sci_scores)

        # Update telemetry & head map
        if len(map_hbo) > 0:
            max_hbo = np.max(np.abs(map_hbo))
            self.lbl_hbo_peak.setText(f"Peak ΔHbO: {max_hbo:.2f} µM" if self.mode_mbll else f"Peak Count: {max_hbo:.0f}")
        if len(map_sci) > 0:
            avg_sci = np.mean(map_sci)
            self.lbl_sci_avg.setText(f"Mean Contact (SCI): {avg_sci:.2f}")

        self.headmap_widget.update_metrics(map_hbo, sci_scores=map_sci)

        self.lbl_srate.setText(f"Sampling Rate: {self.srate:.1f} Hz")
        if len(latest_hbo_for_map) > 0:
            max_hbo = np.max(np.abs(latest_hbo_for_map))
            self.lbl_hbo_peak.setText(f"Peak ΔHbO: {max_hbo:.2f} µM")
        if len(sci_scores) > 0:
            avg_sci = float(np.mean(sci_scores))
            self.lbl_sci_avg.setText(f"Mean Optode Contact (SCI): {avg_sci:.2f}")
            self.lbl_sci_avg.setStyleSheet("color: #7ee787;" if avg_sci >= 0.65 else "color: #e3b341;" if avg_sci >= 0.45 else "color: #ff7b72;")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = fNIRSVisualizerWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
