"""
Real-time 10-20 Headmap & Channel Quality Visualizer for LSL EEG
=================================================================

Renders a 2D 10-20 scalp map showing 32 electrode locations.
Electrode dots change color dynamically based on live signal quality:
  - 🟢 GREEN  : Good active EEG signal
  - 🔴 RED    : Flatline / Disconnected electrode (Low variance / std < 0.5 uV)
  - 🟡 YELLOW : Saturated / High noise / Railing (> 150 uV)

Usage:
  uv run python eeg_headmap_quality_visualizer.py
  (Make sure gds_to_lsl.py or mock_lsl_streamer.py is running)
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
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
from pylsl import StreamInlet, resolve_streams

# Standard 10-20 (x, y) 2D scalp positions mapped onto a unit circle head
ELECTRODE_POSITIONS_1020 = {
    'Fp1': (-0.30,  0.80),
    'Fp2': ( 0.30,  0.80),
    'F7':  (-0.75,  0.60),
    'F3':  (-0.40,  0.55),
    'Fz':  ( 0.00,  0.55),
    'F4':  ( 0.40,  0.55),
    'F8':  ( 0.75,  0.60),
    'FT9': (-0.90,  0.30),
    'FC5': (-0.65,  0.30),
    'FC1': (-0.25,  0.30),
    'FC2': ( 0.25,  0.30),
    'FC6': ( 0.65,  0.30),
    'FT10':( 0.90,  0.30),
    'T7':  (-0.85,  0.00),
    'C3':  (-0.45,  0.00),
    'Cz':  ( 0.00,  0.00),
    'C4':  ( 0.45,  0.00),
    'T8':  ( 0.85,  0.00),
    'TP9': (-0.90, -0.30),
    'CP5': (-0.65, -0.30),
    'CP1': (-0.25, -0.30),
    'CP2': ( 0.25, -0.30),
    'CP6': ( 0.65, -0.30),
    'TP10':( 0.90, -0.30),
    'P7':  (-0.75, -0.60),
    'P3':  (-0.40, -0.55),
    'Pz':  ( 0.00, -0.55),
    'P4':  ( 0.40, -0.55),
    'P8':  ( 0.75, -0.60),
    'O1':  (-0.30, -0.80),
    'Oz':  ( 0.00, -0.80),
    'O2':  ( 0.30, -0.80),
}


class HeadMapCanvas(QtWidgets.QGraphicsView):
    def __init__(self, channel_names):
        super().__init__()
        self._scene = QtWidgets.QGraphicsScene()
        self.setScene(self._scene)
        self.setSceneRect(-280, -280, 560, 560)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(20, 24, 33)))

        self.channel_names = channel_names
        self.dots = {}
        self.labels = {}

        self.draw_head_outline()
        self.draw_electrodes()

    def showEvent(self, event):
        super().showEvent(event)
        QtCore.QTimer.singleShot(50, lambda: self.fitInView(self._scene.sceneRect(), QtCore.Qt.KeepAspectRatio))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self._scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def draw_head_outline(self):
        # Draw Head Circle (radius 220 px)
        pen = QtGui.QPen(QtGui.QColor(100, 120, 160), 3)
        self._scene.addEllipse(-220, -220, 440, 440, pen)

        # Draw Nose
        nose_path = QtGui.QPainterPath()
        nose_path.moveTo(-25, -220)
        nose_path.lineTo(0, -255)
        nose_path.lineTo(25, -220)
        self._scene.addPath(nose_path, pen)

        # Draw Left Ear
        self._scene.addEllipse(-245, -35, 25, 70, pen)
        # Draw Right Ear
        self._scene.addEllipse(220, -35, 25, 70, pen)

    def draw_electrodes(self):
        dot_radius = 18
        for name in self.channel_names:
            if name.upper() == 'BATTERY':
                continue

            pos = ELECTRODE_POSITIONS_1020.get(name, (0.0, 0.0))
            # Map (-1..1) coords to scene coords (radius ~ 200)
            cx = pos[0] * 200
            cy = -pos[1] * 200  # Invert Y for graphics scene

            # Default Green Brush
            brush = QtGui.QBrush(QtGui.QColor(46, 204, 113))
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255), 2)
            
            ellipse = self._scene.addEllipse(cx - dot_radius, cy - dot_radius, dot_radius * 2, dot_radius * 2, pen, brush)
            
            # Label text
            text_item = self._scene.addText(name)
            text_item.setDefaultTextColor(QtGui.QColor(255, 255, 255))
            font = QtGui.QFont("Arial", 9, QtGui.QFont.Bold)
            text_item.setFont(font)
            text_item.setPos(cx - 12, cy - 12)

            self.dots[name] = ellipse
            self.labels[name] = text_item

    def update_channel_status(self, channel_stds, channel_maxs):
        """Color dots based on electrode contact / positioning status."""
        for name in self.channel_names:
            if name not in self.dots:
                continue
            
            std = channel_stds.get(name, 0.0)
            max_val = channel_maxs.get(name, 0.0)

            # Check contact/positioning status (using AC filtered variance & reasonable limits)
            if std < 0.2:
                # 🔴 NO CONTACT / DISCONNECTED
                color = QtGui.QColor(231, 76, 60)
            elif std > 250.0:
                # 🟡 LOOSE CONTACT / ARTIFACT
                color = QtGui.QColor(241, 196, 15)
            else:
                # 🟢 POSITIONED / GOOD CONTACT
                color = QtGui.QColor(46, 204, 113)

            self.dots[name].setBrush(QtGui.QBrush(color))


class HeadmapQualityVisualizerWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("g.Nautilus 32-Ch EEG Headmap & Signal Quality Visualizer")
        self.resize(1150, 750)

        # Resolve LSL EEG Stream
        print("[*] Resolving LSL EEG stream...")
        streams = resolve_streams(wait_time=3.0)
        eeg_stream = None
        for s in streams:
            if s.type().upper() == 'EEG' or 'gNautilus' in s.name():
                eeg_stream = s
                break

        if eeg_stream is None:
            QtWidgets.QMessageBox.critical(
                self, "Stream Not Found", 
                "No LSL EEG stream detected!\n\nPlease start 'gds_to_lsl.py' or 'mock_lsl_streamer.py' first."
            )
            sys.exit(1)

        self.inlet = StreamInlet(eeg_stream, max_chunklen=32)
        self.info = self.inlet.info()
        self.fs = self.info.nominal_srate()
        self.num_channels = self.info.channel_count()

        # Parse Channel Names
        self.channel_names = []
        ch = self.info.desc().child("channels").child("channel")
        for i in range(self.num_channels):
            if ch.empty():
                self.channel_names.append(f"CH{i+1}")
            else:
                self.channel_names.append(ch.child_value("label"))
                ch = ch.next_sibling()

        # Data Buffer (2 seconds window)
        self.buffer_samples = int(self.fs * 2.0)
        self.data_buffer = np.zeros((self.buffer_samples, self.num_channels))

        self.init_ui()

        # Polling Timer (update 20 times/sec)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll_lsl)
        self.timer.start(50)

    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Left Side: 2D Scalp Map
        self.headmap_canvas = HeadMapCanvas(self.channel_names)
        main_layout.addWidget(self.headmap_canvas, stretch=3)

        # Right Side: Status List & Legend
        right_panel = QtWidgets.QVBoxLayout()

        # Title / Summary
        self.lbl_summary = QtWidgets.QLabel("Active Channels: Calculating...")
        self.lbl_summary.setStyleSheet("font-size: 16px; font-weight: bold; color: #ECF0F1;")
        right_panel.addWidget(self.lbl_summary)

        # Legend
        legend_layout = QtWidgets.QHBoxLayout()
        legend_layout.addWidget(self.create_legend_dot("🟢 Good", "#2ECC71"))
        legend_layout.addWidget(self.create_legend_dot("🟡 Noisy", "#F1C40F"))
        legend_layout.addWidget(self.create_legend_dot("🟧 Railed", "#E67E22"))
        legend_layout.addWidget(self.create_legend_dot("🔴 Flatline", "#E74C3C"))
        right_panel.addLayout(legend_layout)

        # Channel Status Table
        self.table = QtWidgets.QTableWidget(len(self.channel_names), 3)
        self.table.setHorizontalHeaderLabels(["Channel", "Metrics (uV)", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setStyleSheet("background-color: #1E222B; color: #ECF0F1; gridline-color: #34495E; font-size: 12px;")

        for i, name in enumerate(self.channel_names):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem("std: 0.0 | p-p: 0.0"))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem("Checking..."))

        right_panel.addWidget(self.table)
        main_layout.addLayout(right_panel, stretch=2)

    def create_legend_dot(self, text, color_hex):
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet(f"background-color: {color_hex}; color: #000; font-weight: bold; border-radius: 4px; padding: 4px;")
        return lbl

    def poll_lsl(self):
        samples, _ = self.inlet.pull_chunk(max_samples=256)
        if not samples:
            return

        chunk = np.array(samples)
        num_new = len(chunk)

        if num_new >= self.buffer_samples:
            self.data_buffer = chunk[-self.buffer_samples:]
        else:
            self.data_buffer = np.vstack((self.data_buffer[num_new:], chunk))

        # Comprehensive Multi-Feature Quality Classification per channel:
        # 1. Std Dev (AC variance)
        # 2. Peak-to-Peak Amplitude Range (max - min)
        # 3. Mean DC Offset (abs mean)
        # 4. Zero-derivative count (consecutive duplicate values)
        stds = np.std(self.data_buffer, axis=0)
        ranges = np.ptp(self.data_buffer, axis=0)
        means = np.abs(np.mean(self.data_buffer, axis=0))

        channel_stds = {}
        channel_maxs = {}
        good_count = 0
        flat_count = 0

        for i, name in enumerate(self.channel_names):
            std_val = stds[i]
            ptp_val = ranges[i]
            mean_val = means[i]
            
            channel_stds[name] = std_val
            channel_maxs[name] = ptp_val

            # Table row display (std and range)
            self.table.item(i, 1).setText(f"std:{std_val:.1f} | p-p:{ptp_val:.1f}")

            # --- Classification Rules ---
            # 1. Flatline / Disconnected: near-zero AC noise OR peak-to-peak < 0.5 uV
            if std_val < 0.5 or ptp_val < 0.5:
                status_item = QtWidgets.QTableWidgetItem("FLATLINE")
                status_item.setForeground(QtGui.QColor("#E74C3C")) # RED
                flat_count += 1
            # 2. Railed / Saturated: huge DC offset or pegged at max ADC voltage (>300 uV)
            elif mean_val > 300.0 or ptp_val > 500.0:
                status_item = QtWidgets.QTableWidgetItem("RAILED/SATURATED")
                status_item.setForeground(QtGui.QColor("#E67E22")) # ORANGE
            # 3. High Noise / Artifact: excessive AC variance (>80 uV)
            elif std_val > 80.0:
                status_item = QtWidgets.QTableWidgetItem("HIGH NOISE")
                status_item.setForeground(QtGui.QColor("#F1C40F")) # YELLOW
            # 4. Good Signal: healthy AC variance (0.5uV - 80uV) with normal baseline
            else:
                status_item = QtWidgets.QTableWidgetItem("GOOD")
                status_item.setForeground(QtGui.QColor("#2ECC71")) # GREEN
                good_count += 1

            self.table.setItem(i, 2, status_item)

        # Update Headmap Canvas Dots
        self.headmap_canvas.update_channel_status(channel_stds, channel_maxs)

        # Update Summary Label
        total_eeg = len([n for n in self.channel_names if n.upper() != 'BATTERY'])
        self.lbl_summary.setText(f"Active Channels: {good_count}/{total_eeg} Good ({flat_count} Flat)")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = HeadmapQualityVisualizerWindow()
    win.show()
    sys.exit(app.exec())
