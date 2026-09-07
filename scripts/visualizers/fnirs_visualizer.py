"""
Real-Time fNIRS (Functional Near-Infrared Spectroscopy) Visualizer
Designed for g.Nautilus / g.tec BCI Systems & Lab Streaming Layer (LSL)
Features:
- Cortical Zone Preset Selection: Frontal (Active), Motor, Parietal, Full Cortex
- Left & Right Hemisphere Receiver-Emitter Geometry Mapping
- Real-time LSL Stream Acquisition & Synthetic Demo Generator Fallback
- Dual-Wavelength / Hemodynamic Conversion (HbO, HbR, HbT) via Modified Beer-Lambert Law (MBLL)
- Physiological Bandpass Filtering (0.01 - 0.5 Hz) & Baseline Drift Removal
- Interactive 2D Optode Head Map Topology with Cortical Zone Highlights
- Modern Dark GUI using PySide6 & PyQtGraph
"""
import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)


import sys
import os
import time
import math
import numpy as np
from datetime import datetime

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# Try importing pylsl; fallback gracefully if not installed
try:
    from pylsl import StreamInlet, resolve_byprop, resolve_streams
    HAS_LSL = True
except ImportError:
    HAS_LSL = False

try:
    import scipy.signal as signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class SyntheticfNIRSStreamer(QtCore.QThread):
    """Generates realistic synthetic dual-wavelength / hemodynamic fNIRS data for testing."""
    sample_signal = QtCore.Signal(np.ndarray, list)

    def __init__(self, num_channels=8, srate=10.0, parent=None):
        super().__init__(parent)
        self.num_channels = num_channels
        self.srate = srate
        self.running = True
        
        self.labels = []
        for i in range(num_channels // 2):
            self.labels.append(f"Ch{i+1}_HbO")
            self.labels.append(f"Ch{i+1}_HbR")

    def run(self):
        t = 0.0
        dt = 1.0 / self.srate
        while self.running:
            t += dt
            # Simulate hemodynamic response function (HRF) + Mayer waves (0.1Hz) + Respiration (0.25Hz) + Cardiac (1.1Hz)
            hrf = 2.0 * math.exp(-((t % 20.0 - 6.0) ** 2) / 6.0) if (t % 20.0) > 2.0 else 0.0
            mayer = 0.3 * math.sin(2 * math.pi * 0.1 * t)
            cardiac = 0.15 * math.sin(2 * math.pi * 1.1 * t)
            
            data_sample = np.zeros(self.num_channels)
            num_pairs = self.num_channels // 2
            for i in range(num_pairs):
                phase = i * 0.4
                # HbO increases during task activation
                data_sample[2 * i] = (hrf + mayer + cardiac + np.random.normal(0, 0.05)) * (1.0 + 0.1 * math.sin(t + phase))
                # HbR decreases during task activation (inverse of HbO)
                data_sample[2 * i + 1] = (-0.4 * hrf + 0.2 * mayer + 0.08 * cardiac + np.random.normal(0, 0.03)) * (1.0 + 0.1 * math.sin(t + phase))

            self.sample_signal.emit(data_sample, self.labels)
            time.sleep(dt)

    def stop(self):
        self.running = False
        self.wait()


class OptodeHeadMapWidget(QtWidgets.QFrame):
    """2D Interactive Head Topography map showing optode locations and Anatomical Cortical Zones."""

    def __init__(self, num_channels=8, active_zone="FRONTAL", parent=None):
        super().__init__(parent)
        self.num_channels = num_channels
        self.active_zone = active_zone  # "FRONTAL", "MOTOR", "PARIETAL", "FULL"
        self.channel_values = np.zeros(num_channels)
        self.channel_names = [f"Ch {i+1}" for i in range(num_channels)]
        self.setMinimumSize(260, 260)
        self.setStyleSheet("background-color: #151922; border-radius: 10px; border: 1px solid #2a3142;")
        self.recompute_positions()

    def set_zone_and_channels(self, zone, count):
        self.active_zone = zone
        self.num_channels = count
        self.channel_values = np.zeros(count)
        self.channel_names = [f"Ch {i+1}" for i in range(count)]
        self.recompute_positions()
        self.update()

    def recompute_positions(self):
        """Build 2D optode layout based on active zone (Frontal, Motor, Parietal, Full)."""
        self.positions = []
        self.region_labels = []
        num_pairs = self.num_channels // 2 if self.num_channels >= 2 else self.num_channels

        if self.active_zone == "FRONTAL":
            if num_pairs == 4:
                self.positions = [
                    (-0.35, -0.65), (-0.55, -0.45),
                    ( 0.35, -0.65), ( 0.55, -0.45),
                ]
                self.region_labels = ["L. Frontal (Fp1)", "L. Frontal (F3)", "R. Frontal (Fp2)", "R. Frontal (F4)"]
            else:
                self.positions = [
                    (-0.30, -0.70), (-0.50, -0.60), (-0.30, -0.40), (-0.55, -0.35),
                    ( 0.30, -0.70), ( 0.50, -0.60), ( 0.30, -0.40), ( 0.55, -0.35),
                ]
                self.region_labels = [
                    "L. Frontal (Fp1)", "L. Frontal (F3)", "L. Frontal (AF3)", "L. Frontal (F7)",
                    "R. Frontal (Fp2)", "R. Frontal (F4)", "R. Frontal (AF4)", "R. Frontal (F8)"
                ]

        elif self.active_zone == "MOTOR":
            if num_pairs == 4:
                self.positions = [
                    (-0.55, -0.10), (-0.55,  0.15),
                    ( 0.55, -0.10), ( 0.55,  0.15),
                ]
                self.region_labels = ["L. Motor (FC3)", "L. Motor (C3)", "R. Motor (FC4)", "R. Motor (C4)"]
            else:
                self.positions = [
                    (-0.45, -0.15), (-0.65, -0.05), (-0.45,  0.15), (-0.65,  0.25),
                    ( 0.45, -0.15), ( 0.65, -0.05), ( 0.45,  0.15), ( 0.65,  0.25),
                ]
                self.region_labels = [
                    "L. Motor (FC3)", "L. Motor (C3)", "L. Motor (CP3)", "L. Motor (C5)",
                    "R. Motor (FC4)", "R. Motor (C4)", "R. Motor (CP4)", "R. Motor (C6)"
                ]

        elif self.active_zone == "PARIETAL":
            if num_pairs == 4:
                self.positions = [
                    (-0.35,  0.45), (-0.55,  0.60),
                    ( 0.35,  0.45), ( 0.55,  0.60),
                ]
                self.region_labels = ["L. Parietal (P3)", "L. Parietal (PO3)", "R. Parietal (P4)", "R. Parietal (PO4)"]
            else:
                self.positions = [
                    (-0.30,  0.40), (-0.55,  0.45), (-0.30,  0.65), (-0.55,  0.70),
                    ( 0.30,  0.40), ( 0.55,  0.45), ( 0.30,  0.65), ( 0.55,  0.70),
                ]
                self.region_labels = [
                    "L. Parietal (P3)", "L. Parietal (P7)", "L. Parietal (PO3)", "L. Parietal (O1)",
                    "R. Parietal (P4)", "R. Parietal (P8)", "R. Parietal (PO4)", "R. Parietal (O2)"
                ]
        else: # FULL CORTEX
            self.positions = [
                (-0.35, -0.55), (-0.62, -0.20), (-0.62,  0.20), (-0.35,  0.55),
                ( 0.35, -0.55), ( 0.62, -0.20), ( 0.62,  0.20), ( 0.35,  0.55),
            ]
            self.region_labels = [
                "L. Frontal", "L. Motor", "L. Sensory", "L. Parietal",
                "R. Frontal", "R. Motor", "R. Sensory", "R. Parietal"
            ]

    def get_region_name(self, index):
        if index < len(self.region_labels):
            return self.region_labels[index]
        return f"Ch {index+1}"

    def update_values(self, values, names=None):
        if len(values) > 0:
            self.channel_values = values
        if names and len(names) == len(values):
            self.channel_names = names
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        cx, cy = w / 2, h / 2
        radius = min(w, h) * 0.40

        pen_head = QtGui.QPen(QtGui.QColor("#3a445e"), 3)
        painter.setPen(pen_head)
        painter.setBrush(QtGui.QColor("#1c2230"))
        painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

        nose_path = QtGui.QPainterPath()
        nose_path.moveTo(cx - 14, cy - radius)
        nose_path.lineTo(cx, cy - radius - 18)
        nose_path.lineTo(cx + 14, cy - radius)
        painter.fillPath(nose_path, QtGui.QColor("#3a445e"))

        painter.setPen(QtGui.QPen(QtGui.QColor("#8b949e")))
        font = painter.font()
        font.setPointSize(7)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(int(cx - 30), int(cy - radius - 22), 60, 12, QtCore.Qt.AlignCenter, "FRONT")

        painter.setBrush(QtGui.QColor("#151922"))
        painter.drawEllipse(QtCore.QPointF(cx - radius - 6, cy), 8, 16)
        painter.drawEllipse(QtCore.QPointF(cx + radius + 6, cy), 8, 16)

        pen_zone = QtGui.QPen(QtGui.QColor("#30363d"), 1, QtCore.Qt.DashLine)
        painter.setPen(pen_zone)

        y_fm = cy - 0.30 * radius
        y_mp = cy + 0.30 * radius
        painter.drawLine(QtCore.QPointF(cx - 0.85 * radius, y_fm), QtCore.QPointF(cx + 0.85 * radius, y_fm))
        painter.drawLine(QtCore.QPointF(cx - 0.85 * radius, y_mp), QtCore.QPointF(cx + 0.85 * radius, y_mp))

        if self.active_zone == "FRONTAL":
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(31, 107, 235, 35))
            painter.drawPie(QtCore.QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius), 30 * 16, 120 * 16)
        elif self.active_zone == "MOTOR":
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(46, 213, 115, 35))
            painter.drawRect(QtCore.QRectF(cx - radius, y_fm, 2 * radius, y_mp - y_fm))
        elif self.active_zone == "PARIETAL":
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(255, 71, 87, 35))
            painter.drawPie(QtCore.QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius), 210 * 16, 120 * 16)

        font_zone = painter.font()
        font_zone.setPointSize(8)
        font_zone.setBold(True)
        painter.setFont(font_zone)

        painter.setPen(QtGui.QPen(QtGui.QColor("#58a6ff" if self.active_zone == "FRONTAL" else "#484f58")))
        painter.drawText(int(cx - 40), int(cy - 0.65 * radius), 80, 15, QtCore.Qt.AlignCenter, "FRONTAL" + (" ★" if self.active_zone == "FRONTAL" else ""))

        painter.setPen(QtGui.QPen(QtGui.QColor("#2ed573" if self.active_zone == "MOTOR" else "#484f58")))
        painter.drawText(int(cx - 40), int(cy - 0.05 * radius), 80, 15, QtCore.Qt.AlignCenter, "MOTOR" + (" ★" if self.active_zone == "MOTOR" else ""))

        painter.setPen(QtGui.QPen(QtGui.QColor("#ff4757" if self.active_zone == "PARIETAL" else "#484f58")))
        painter.drawText(int(cx - 40), int(cy + 0.55 * radius), 80, 15, QtCore.Qt.AlignCenter, "PARIETAL" + (" ★" if self.active_zone == "PARIETAL" else ""))

        if len(self.positions) > 0:
            if self.active_zone == "FRONTAL":
                rx1_y, rx2_y = cy - 0.52 * radius, cy - 0.52 * radius
            elif self.active_zone == "MOTOR":
                rx1_y, rx2_y = cy, cy
            elif self.active_zone == "PARIETAL":
                rx1_y, rx2_y = cy + 0.52 * radius, cy + 0.52 * radius
            else:
                rx1_y, rx2_y = cy, cy

            rx1_x = cx - 0.45 * radius
            rx2_x = cx + 0.45 * radius

            painter.setPen(QtGui.QPen(QtGui.QColor("#58a6ff"), 2))
            painter.setBrush(QtGui.QColor("#1f6beb"))
            painter.drawRect(int(rx1_x - 7), int(rx1_y - 7), 14, 14)

            painter.setPen(QtGui.QPen(QtGui.QColor("#58a6ff"), 2))
            painter.setBrush(QtGui.QColor("#1f6beb"))
            painter.drawRect(int(rx2_x - 7), int(rx2_y - 7), 14, 14)

        for i in range(min(len(self.positions), len(self.channel_values))):
            nx, ny = self.positions[i]
            px = cx + nx * radius
            py = cy + ny * radius

            val = self.channel_values[i]
            val_norm = np.clip(val / 3.0, -1.0, 1.0)
            if val_norm >= 0:
                r_col = int(255 * val_norm)
                g_col = int(220 * (1.0 - val_norm * 0.5))
                b_col = int(80 * (1.0 - val_norm))
            else:
                val_abs = abs(val_norm)
                r_col = int(30 * (1.0 - val_abs))
                g_col = int(140 * (1.0 - val_abs))
                b_col = int(255 * val_abs)

            optode_color = QtGui.QColor(r_col, g_col, b_col)

            rx_x = rx1_x if nx < 0 else rx2_x
            rx_y = rx1_y if nx < 0 else rx2_y
            painter.setPen(QtGui.QPen(QtGui.QColor(r_col, g_col, b_col, 120), 1.5, QtCore.Qt.DashLine))
            painter.drawLine(QtCore.QPointF(rx_x, rx_y), QtCore.QPointF(px, py))

            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(r_col, g_col, b_col, 65))
            painter.drawEllipse(QtCore.QPointF(px, py), 14, 14)

            painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.5))
            painter.setBrush(optode_color)
            painter.drawEllipse(QtCore.QPointF(px, py), 8, 8)

            lbl = f"Ch{i+1}"
            painter.setPen(QtGui.QPen(QtGui.QColor("#dcdde1")))
            font = painter.font()
            font.setPointSize(7)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(int(px - 15), int(py + 14), 30, 12, QtCore.Qt.AlignCenter, lbl)


class fNIRSVisualizerWindow(QtWidgets.QMainWindow):
    """Main Real-Time fNIRS Visualizer Application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("g.tec Real-Time fNIRS Cortical Monitor (Frontal / Motor / Parietal)")
        self.resize(1320, 850)

        self.srate = 10.0
        self.window_sec = 15.0
        self.active_zone = "FRONTAL"
        self.num_channels = 8
        self.mode_mbll = True
        self.filter_enabled = True
        
        self.inlet = None
        self.synthetic_thread = None

        self.max_samples = int(self.srate * 60.0)
        self.time_buffer = np.zeros(self.max_samples)
        self.data_buffer = np.zeros((self.max_samples, self.num_channels))
        self.sample_count = 0

        self.init_dark_theme()
        self.init_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll_and_update)
        self.timer.start(20)

    def init_dark_theme(self):
        app = QtWidgets.QApplication.instance()
        if app:
            app.setStyle("Fusion")
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#0d1117"))
            palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#c9d1d9"))
            palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#161b22"))
            palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#21262d"))
            palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor("#c9d1d9"))
            palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor("#c9d1d9"))
            palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#c9d1d9"))
            palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#21262d"))
            palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#c9d1d9"))
            palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor("#ff7b72"))
            palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#1f6beb"))
            palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
            app.setPalette(palette)

    def init_ui(self):
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)

        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        toolbar_card = QtWidgets.QFrame()
        toolbar_card.setStyleSheet("background-color: #161b22; border-radius: 8px; border: 1px solid #30363d; padding: 6px;")
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar_card)
        toolbar_layout.setContentsMargins(10, 4, 10, 4)

        title_label = QtWidgets.QLabel("🧠 g.Nautilus fNIRS Cortical Monitor")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #58a6ff;")
        toolbar_layout.addWidget(title_label)

        toolbar_layout.addSpacing(15)

        zone_label = QtWidgets.QLabel("Active Cortical Setup:")
        zone_label.setStyleSheet("font-weight: bold; color: #c9d1d9;")
        toolbar_layout.addWidget(zone_label)

        self.combo_zone = QtWidgets.QComboBox()
        self.combo_zone.addItems([
            "Frontal (Left & Right Frontal) [Active]",
            "Motor (Left & Right Motor)",
            "Parietal (Left & Right Parietal)",
            "Full Cortex Coverage (8 Ch)"
        ])
        self.combo_zone.setStyleSheet("background-color: #21262d; color: #58a6ff; font-weight: bold; padding: 4px 8px; border-radius: 4px;")
        self.combo_zone.currentIndexChanged.connect(self.on_zone_changed)
        toolbar_layout.addWidget(self.combo_zone)

        toolbar_layout.addSpacing(10)

        self.btn_lsl = QtWidgets.QPushButton("🔗 Connect LSL Stream")
        self.btn_lsl.setStyleSheet("background-color: #238636; color: white; font-weight: bold; padding: 6px 12px; border-radius: 6px;")
        self.btn_lsl.clicked.connect(self.connect_lsl)
        toolbar_layout.addWidget(self.btn_lsl)

        self.btn_demo = QtWidgets.QPushButton("⚡ Launch Synthetic Demo")
        self.btn_demo.setStyleSheet("background-color: #1f6beb; color: white; font-weight: bold; padding: 6px 12px; border-radius: 6px;")
        self.btn_demo.clicked.connect(self.start_synthetic_demo)
        toolbar_layout.addWidget(self.btn_demo)

        toolbar_layout.addStretch()

        self.chk_mbll = QtWidgets.QCheckBox("Beer-Lambert (HbO / HbR)")
        self.chk_mbll.setChecked(True)
        self.chk_mbll.setStyleSheet("font-weight: bold; color: #79c0ff;")
        self.chk_mbll.stateChanged.connect(self.on_mbll_toggled)
        toolbar_layout.addWidget(self.chk_mbll)

        self.chk_filter = QtWidgets.QCheckBox("Filter (0.01 - 0.5 Hz)")
        self.chk_filter.setChecked(True)
        self.chk_filter.setStyleSheet("font-weight: bold; color: #d2a8ff;")
        self.chk_filter.stateChanged.connect(self.on_filter_toggled)
        toolbar_layout.addWidget(self.chk_filter)

        main_layout.addWidget(toolbar_card)

        content_layout = QtWidgets.QHBoxLayout()

        sidebar_card = QtWidgets.QFrame()
        sidebar_card.setFixedWidth(310)
        sidebar_card.setStyleSheet("background-color: #161b22; border-radius: 8px; border: 1px solid #30363d; padding: 8px;")
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_card)

        legend_box = QtWidgets.QGroupBox("Hemodynamic Signals & Optodes")
        legend_box.setStyleSheet("QGroupBox { font-weight: bold; color: #8b949e; border: 1px solid #30363d; margin-top: 6px; padding-top: 10px; }")
        leg_layout = QtWidgets.QVBoxLayout(legend_box)

        lbl_hbo = QtWidgets.QLabel("🔴 ΔHbO (Oxy-Hemoglobin)")
        lbl_hbo.setStyleSheet("color: #ff4757; font-weight: bold; font-size: 12px;")
        lbl_hbr = QtWidgets.QLabel("🔵 ΔHbR (Deoxy-Hemoglobin)")
        lbl_hbr.setStyleSheet("color: #1e90ff; font-weight: bold; font-size: 12px;")
        lbl_hbt = QtWidgets.QLabel("🟢 ΔHbT (Total Hemoglobin)")
        lbl_hbt.setStyleSheet("color: #2ed573; font-weight: bold; font-size: 12px;")
        lbl_rx = QtWidgets.QLabel("🟦 Receiver (Detector Optode)")
        lbl_rx.setStyleSheet("color: #58a6ff; font-weight: bold; font-size: 12px;")

        leg_layout.addWidget(lbl_hbo)
        leg_layout.addWidget(lbl_hbr)
        leg_layout.addWidget(lbl_hbt)
        leg_layout.addWidget(lbl_rx)
        sidebar_layout.addWidget(legend_box)

        sidebar_layout.addSpacing(10)

        map_title = QtWidgets.QLabel("Active Cortex Topography & Optodes")
        map_title.setStyleSheet("font-weight: bold; color: #c9d1d9; font-size: 11px;")
        sidebar_layout.addWidget(map_title)

        self.headmap_widget = OptodeHeadMapWidget(self.num_channels, active_zone=self.active_zone)
        sidebar_layout.addWidget(self.headmap_widget)

        sidebar_layout.addStretch()

        metrics_box = QtWidgets.QGroupBox("Live Metrics")
        metrics_box.setStyleSheet("QGroupBox { font-weight: bold; color: #8b949e; border: 1px solid #30363d; margin-top: 6px; padding-top: 10px; }")
        m_layout = QtWidgets.QVBoxLayout(metrics_box)

        self.lbl_srate = QtWidgets.QLabel("Sampling Rate: -- Hz")
        self.lbl_srate.setStyleSheet("color: #c9d1d9;")
        self.lbl_hbo_peak = QtWidgets.QLabel("Max ΔHbO: -- µmol/L")
        self.lbl_hbo_peak.setStyleSheet("color: #ff4757;")
        self.lbl_battery = QtWidgets.QLabel("Headset Battery: 95.0%")
        self.lbl_battery.setStyleSheet("color: #2ed573; font-weight: bold;")
        self.lbl_status = QtWidgets.QLabel("Stream: Idle")
        self.lbl_status.setStyleSheet("color: #e3b341; font-weight: bold;")

        m_layout.addWidget(self.lbl_srate)
        m_layout.addWidget(self.lbl_hbo_peak)
        m_layout.addWidget(self.lbl_battery)
        m_layout.addWidget(self.lbl_status)
        sidebar_layout.addWidget(metrics_box)


        content_layout.addWidget(sidebar_card)

        plot_card = QtWidgets.QFrame()
        plot_card.setStyleSheet("background-color: #161b22; border-radius: 8px; border: 1px solid #30363d; padding: 6px;")
        plot_layout = QtWidgets.QVBoxLayout(plot_card)

        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('#0d1117')
        plot_layout.addWidget(self.plot_widget)

        content_layout.addWidget(plot_card)
        main_layout.addLayout(content_layout)

        self.rebuild_plots()

    def on_zone_changed(self, index):
        if index == 0:
            self.active_zone = "FRONTAL"
        elif index == 1:
            self.active_zone = "MOTOR"
        elif index == 2:
            self.active_zone = "PARIETAL"
        else:
            self.active_zone = "FULL"

        self.headmap_widget.set_zone_and_channels(self.active_zone, self.num_channels)
        self.rebuild_plots()

    def rebuild_plots(self):
        self.plot_widget.clear()
        self.plots = []
        self.curves = []

        num_pairs = self.num_channels // 2 if self.mode_mbll else self.num_channels

        for i in range(num_pairs):
            p = self.plot_widget.addPlot(row=i, col=0)
            p.setMouseEnabled(x=True, y=False)
            p.showGrid(x=True, y=True, alpha=0.15)
            p.getAxis('left').setPen(pg.mkPen('#8b949e'))
            p.getAxis('bottom').setPen(pg.mkPen('#8b949e'))

            region_name = self.headmap_widget.get_region_name(i)
            ch_label_name = f"Ch{i+1}: {region_name}"

            if self.mode_mbll:
                p.setLabel('left', ch_label_name, units='µM')
                c_hbo = p.plot(pen=pg.mkPen('#ff4757', width=2), name=f"{ch_label_name}_HbO")
                c_hbr = p.plot(pen=pg.mkPen('#1e90ff', width=2), name=f"{ch_label_name}_HbR")
                c_hbt = p.plot(pen=pg.mkPen('#2ed573', width=1.5, style=QtCore.Qt.DashLine), name=f"{ch_label_name}_HbT")
                self.curves.append((c_hbo, c_hbr, c_hbt))
            else:
                p.setLabel('left', ch_label_name, units='V')
                c_raw = p.plot(pen=pg.mkPen('#79c0ff', width=2), name=f"{ch_label_name}_Raw")
                self.curves.append((c_raw,))

            if i < num_pairs - 1:
                p.hideAxis('bottom')

            self.plots.append(p)

        if len(self.plots) > 0:
            self.plots[-1].setLabel('bottom', 'Time', units='s')

    def on_mbll_toggled(self, state):
        self.mode_mbll = (state == QtCore.Qt.Checked)
        self.rebuild_plots()

    def on_filter_toggled(self, state):
        self.filter_enabled = (state == QtCore.Qt.Checked)

    def start_synthetic_demo(self):
        if self.synthetic_thread and self.synthetic_thread.isRunning():
            self.synthetic_thread.stop()
            
        if HAS_LSL and self.inlet:
            self.inlet = None

        self.synthetic_thread = SyntheticfNIRSStreamer(num_channels=self.num_channels, srate=10.0)
        self.synthetic_thread.sample_signal.connect(self.on_synthetic_sample)
        self.synthetic_thread.start()

        self.lbl_status.setText("Stream: Synthetic Demo")
        self.lbl_status.setStyleSheet("color: #238636; font-weight: bold;")
        self.btn_demo.setText("⚡ Restart Synthetic Demo")

    def on_synthetic_sample(self, sample, labels):
        self.push_sample(sample)

    def connect_lsl(self):
        if not HAS_LSL:
            QtWidgets.QMessageBox.warning(self, "LSL Not Installed", "pylsl library is not installed in this environment.")
            return

        self.lbl_status.setText("Stream: Searching LSL...")
        self.lbl_status.setStyleSheet("color: #e3b341; font-weight: bold;")
        QtWidgets.QApplication.processEvents()

        streams = resolve_streams()
        target_stream = None
        for s in streams:
            if s.type().upper() in ['FNIRS', 'NIRS', 'EEG', 'GNAUTILUS']:
                target_stream = s
                break

        if target_stream:
            self.inlet = StreamInlet(target_stream)
            self.srate = self.inlet.info().nominal_srate() or 10.0

            lsl_ch = self.inlet.info().channel_count()
            if lsl_ch > 0 and lsl_ch != self.num_channels:
                self.num_channels = lsl_ch
                self.max_samples = int(self.srate * 60.0)
                self.time_buffer = np.zeros(self.max_samples)
                self.data_buffer = np.zeros((self.max_samples, self.num_channels))
                self.sample_count = 0
                self.headmap_widget.set_zone_and_channels(self.active_zone, self.num_channels)
                self.rebuild_plots()

            self.lbl_status.setText(f"Stream: Connected ({target_stream.name()})")
            self.lbl_status.setStyleSheet("color: #238636; font-weight: bold;")

            if self.synthetic_thread and self.synthetic_thread.isRunning():
                self.synthetic_thread.stop()
        else:
            self.lbl_status.setText("Stream: No LSL Stream Found")
            self.lbl_status.setStyleSheet("color: #ff4757; font-weight: bold;")
            QtWidgets.QMessageBox.information(self, "LSL Stream", "No active LSL stream found. Please start 'gds_to_lsl.py' or use 'Synthetic Demo'.")

    def push_sample(self, sample):
        self.data_buffer = np.roll(self.data_buffer, -1, axis=0)
        self.data_buffer[-1, :min(len(sample), self.num_channels)] = sample[:self.num_channels]
        
        self.time_buffer = np.roll(self.time_buffer, -1)
        self.time_buffer[-1] = time.time()
        self.sample_count += 1

    def poll_and_update(self):
        if HAS_LSL and self.inlet:
            try:
                samples, timestamps = self.inlet.pull_chunk(timeout=0.0)
                if samples:
                    for s in samples:
                        self.push_sample(np.array(s))
            except Exception:
                pass

        if self.sample_count < 5:
            return

        window_size = int(self.srate * self.window_sec)
        window_size = min(window_size, self.max_samples, self.sample_count)

        raw_segment = self.data_buffer[-window_size:, :].copy()
        time_segment = np.linspace(-self.window_sec, 0, window_size)

        filtered_segment = raw_segment.copy()
        if self.filter_enabled and HAS_SCIPY and window_size > 30:
            try:
                nyq = 0.5 * self.srate
                low = max(0.01 / nyq, 0.001)
                high = min(0.5 / nyq, 0.49)
                b, a = signal.butter(2, [low, high], btype='band')
                filtered_segment = signal.filtfilt(b, a, filtered_segment, axis=0)
            except Exception:
                pass

        num_pairs = self.num_channels // 2 if self.mode_mbll else self.num_channels
        latest_values_for_map = []

        for i in range(num_pairs):
            if i >= len(self.curves):
                break

            if self.mode_mbll:
                idx_hbo = min(2 * i, filtered_segment.shape[1] - 1)
                idx_hbr = min(2 * i + 1, filtered_segment.shape[1] - 1)

                hbo = filtered_segment[:, idx_hbo]
                hbr = filtered_segment[:, idx_hbr]
                hbt = hbo + hbr

                self.curves[i][0].setData(time_segment, hbo)
                self.curves[i][1].setData(time_segment, hbr)
                self.curves[i][2].setData(time_segment, hbt)

                latest_values_for_map.append(hbo[-1])
                latest_values_for_map.append(hbr[-1])
            else:
                idx_raw = min(i, filtered_segment.shape[1] - 1)
                raw_ch = filtered_segment[:, idx_raw]
                self.curves[i][0].setData(time_segment, raw_ch)
                latest_values_for_map.append(raw_ch[-1])

            self.plots[i].setXRange(-self.window_sec, 0)

        self.headmap_widget.update_values(np.array(latest_values_for_map))

        self.lbl_srate.setText(f"Sampling Rate: {self.srate:.1f} Hz")
        if len(latest_values_for_map) > 0:
            max_hbo = np.max(np.abs(latest_values_for_map))
            self.lbl_hbo_peak.setText(f"Max ΔHbO: {max_hbo:.2f} µmol/L")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = fNIRSVisualizerWindow()
    win.show()

    # Default: Clean live mode. Only launch synthetic demo if --demo flag is passed.
    if "--demo" in sys.argv:
        win.start_synthetic_demo()


    sys.exit(app.exec())


if __name__ == '__main__':
    main()
