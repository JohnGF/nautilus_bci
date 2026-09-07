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
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
from pylsl import StreamInlet, resolve_byprop
import scipy.signal as signal

class LSLPlotter32Ch(QtWidgets.QWidget):
    def __init__(self, inlet):
        super().__init__()
        self.inlet = inlet
        self.info = inlet.info()
        self.fs = self.info.nominal_srate()
        self.num_channels = self.info.channel_count()
        
        # Default vertical spacing between channels (in microvolts)
        self.spacing = 50.0 
        
        # Resolve channel names if present in LSL metadata
        self.channel_names = []
        ch = self.info.desc().child("channels").child("channel")
        for i in range(self.num_channels):
            if ch.empty():
                self.channel_names.append(f"CH {i+1}")
            else:
                self.channel_names.append(ch.child_value("label"))
                ch = ch.next_sibling()
                
        # Detect if we have a battery channel at the end
        self.has_battery = len(self.channel_names) > 0 and self.channel_names[-1].upper() == 'BATTERY'
        if self.has_battery:
            self.plot_channels_count = self.num_channels - 1
            self.plot_channel_names = self.channel_names[:-1]
        else:
            self.plot_channels_count = self.num_channels
            self.plot_channel_names = self.channel_names
            
        # Buffer setup (last 5 seconds)
        self.buffer_duration = 5.0
        self.buffer_samples = int(self.fs * self.buffer_duration)
        self.data_buffer = np.zeros((self.buffer_samples, self.num_channels))
        self.time_axis = np.linspace(-self.buffer_duration, 0, self.buffer_samples)
        
        # Design filters (4th-order Butterworth 2-45 Hz bandpass, 50 Hz notch)
        nyq = 0.5 * self.fs
        self.b_band, self.a_band = signal.butter(4, [2.0 / nyq, 45.0 / nyq], btype='band')
        self.b_notch, self.a_notch = signal.iirnotch(50.0, 30.0, self.fs)
        
        self.init_ui()
        
        # Timer for polling LSL (every 20ms)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(20)
        
    def init_ui(self):
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        
        # Control panel at the top
        controls = QtWidgets.QHBoxLayout()
        
        spacing_label = QtWidgets.QLabel("Channel Spacing (uV):")
        spacing_label.setStyleSheet("font-weight: bold;")
        controls.addWidget(spacing_label)
        
        self.spacing_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.spacing_slider.setRange(5, 500)
        self.spacing_slider.setValue(int(self.spacing))
        self.spacing_slider.setToolTip("Adjust vertical spacing between channels")
        self.spacing_slider.valueChanged.connect(self.on_spacing_changed)
        controls.addWidget(self.spacing_slider)
        
        self.spacing_value_label = QtWidgets.QLabel(f"{self.spacing} uV")
        controls.addWidget(self.spacing_value_label)
        
        controls.addStretch()
        
        shortcut_label = QtWidgets.QLabel("Tip: Adjust slider if signals overlap")
        shortcut_label.setStyleSheet("color: gray; font-style: italic;")
        controls.addWidget(shortcut_label)
        
        # Add battery status label if present
        if self.has_battery:
            self.battery_label = QtWidgets.QLabel("Battery: --%")
            self.battery_label.setStyleSheet("font-weight: bold; color: #27ae60; border: 1px solid #27ae60; padding: 4px 8px; border-radius: 4px; font-size: 11px;")
            controls.addWidget(self.battery_label)
            
        main_layout.addLayout(controls)
        
        # PyQtGraph GraphicsLayoutWidget
        self.win = pg.GraphicsLayoutWidget(title="Live EEG LSL Multi-Channel View")
        main_layout.addWidget(self.win)
        
        # Single plot for all channels stacked vertically (Waterfall plot)
        self.plot = self.win.addPlot()
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.showGrid(x=True, y=False, alpha=0.3)
        self.plot.setXRange(-self.buffer_duration, 0)
        
        self.curves = []
        
        # Add a curve for each channel
        for i in range(self.plot_channels_count):
            # Draw channels with distinct colors
            color_index = i % 4
            colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78']
            pen = pg.mkPen(color=colors[color_index], width=1.0)
            
            curve = self.plot.plot(pen=pen)
            self.curves.append(curve)
            
        # Update Y-axis ticks to show channel names
        self.update_y_ticks()
        
        self.setWindowTitle(f"g.Nautilus LSL 32-Channel Viewer - {self.info.name()}")
        self.resize(1200, 800)
        
    def update_y_ticks(self):
        y_axis = self.plot.getAxis('left')
        ticks = []
        for i, name in enumerate(self.plot_channel_names):
            ticks.append((i * self.spacing, name))
        y_axis.setTicks([ticks])
        
        # Set static limits on Y axis to match spacing
        self.plot.setYRange(-self.spacing, self.plot_channels_count * self.spacing, padding=0.05)
        
    def on_spacing_changed(self, value):
        self.spacing = float(value)
        self.spacing_value_label.setText(f"{self.spacing} uV")
        self.update_y_ticks()
        
    def update_data(self):
        # Pull chunk from LSL
        chunk, timestamps = self.inlet.pull_chunk()
        if not chunk:
            return
            
        # chunk is list of lists, shape (samples, channels)
        chunk = np.array(chunk)
        
        # Roll buffer and insert new chunk
        num_samples = chunk.shape[0]
        self.data_buffer = np.roll(self.data_buffer, -num_samples, axis=0)
        self.data_buffer[-num_samples:, :] = chunk[:, :self.num_channels]
        
        # Update battery level display if available
        if self.has_battery:
            battery_val = np.mean(chunk[:, -1])
            self.battery_label.setText(f"Battery: {battery_val:.1f}%")
            
        # Filter the buffer to strip DC offset and noise
        # Detrend first to remove huge baseline offsets
        # Only process channels we are actually plotting (excluding Battery)
        plotting_data = self.data_buffer[:, :self.plot_channels_count]
        detrended = signal.detrend(plotting_data, axis=0)
        filtered = signal.lfilter(self.b_band, self.a_band, detrended, axis=0)
        filtered = signal.lfilter(self.b_notch, self.a_notch, filtered, axis=0)
        
        # Update curves with vertical offsets
        for i in range(self.plot_channels_count):
            # Y = data + channel_offset
            offset_y = filtered[:, i] + (i * self.spacing)
            self.curves[i].setData(self.time_axis, offset_y)

def main():
    print("=" * 70)
    print("      g.Nautilus 32-Channel Stacked EEG LSL Visualizer (PySide6)")
    print("=" * 70)
    print("[*] Resolving LSL EEG stream...")
    
    # Resolve stream of type 'EEG'
    streams = resolve_byprop('type', 'EEG', timeout=5.0)
    if not streams:
        print("[-] Error: No LSL EEG streams found on the network.")
        print("    Please make sure gds_to_lsl.py is running and streaming.")
        sys.exit(1)
        
    print(f"[+] Found LSL Stream: {streams[0].name()} (Source ID: {streams[0].source_id()})")
    inlet = StreamInlet(streams[0])
    
    app = QtWidgets.QApplication(sys.argv)
    plotter = LSLPlotter32Ch(inlet)
    plotter.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
