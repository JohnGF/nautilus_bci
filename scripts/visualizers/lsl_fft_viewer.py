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

class LSLFFTViewer(QtWidgets.QWidget):
    def __init__(self, inlet):
        super().__init__()
        self.inlet = inlet
        self.info = inlet.info()
        self.fs = self.info.nominal_srate()
        self.num_channels = self.info.channel_count()
        
        # Buffer setup (last 3 seconds for good frequency resolution)
        self.buffer_duration = 3.0
        self.buffer_samples = int(self.fs * self.buffer_duration)
        
        # We will keep a raw buffer
        self.raw_buffer = np.zeros((self.buffer_samples, self.num_channels))
        self.time_axis = np.linspace(-self.buffer_duration, 0, self.buffer_samples)
        
        # Channel names
        self.channel_names = []
        ch = self.info.desc().child("channels").child("channel")
        for i in range(self.num_channels):
            if ch.empty():
                self.channel_names.append(f"CH {i+1}")
            else:
                self.channel_names.append(ch.child_value("label"))
                ch = ch.next_sibling()
                
        # Detect battery channel
        self.has_battery = len(self.channel_names) > 0 and self.channel_names[-1].upper() == 'BATTERY'
        if self.has_battery:
            self.plot_channel_names = self.channel_names[:-1]
        else:
            self.plot_channel_names = self.channel_names
            
        self.current_channel_idx = 0
        
        # Design filters (4th-order Butterworth 2-45 Hz bandpass, 50 Hz notch)
        nyq = 0.5 * self.fs
        self.b_band, self.a_band = signal.butter(4, [2.0 / nyq, 45.0 / nyq], btype='band')
        self.b_notch, self.a_notch = signal.iirnotch(50.0, 30.0, self.fs)
        
        self.init_ui()
        
        # Timer for polling LSL and updating plots (every 50ms)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(50)
        
    def init_ui(self):
        # Layouts
        main_layout = QtWidgets.QVBoxLayout(self)
        
        # Header controls
        controls = QtWidgets.QHBoxLayout()
        
        label = QtWidgets.QLabel("Select Channel to Analyze:")
        label.setStyleSheet("font-weight: bold; font-size: 12px;")
        controls.addWidget(label)
        
        self.chan_combo = QtWidgets.QComboBox()
        self.chan_combo.addItems(self.plot_channel_names)
        self.chan_combo.currentIndexChanged.connect(self.on_channel_changed)
        controls.addWidget(self.chan_combo)
        
        controls.addStretch()
        
        info_label = QtWidgets.QLabel("Red = Raw Signal (Unfiltered) | Green = Filtered Signal (2-45 Hz, 50 Hz Notch)")
        info_label.setStyleSheet("color: gray; font-style: italic; margin-right: 10px;")
        controls.addWidget(info_label)
        
        # Add battery status label if present
        if self.has_battery:
            self.battery_label = QtWidgets.QLabel("Battery: --%")
            self.battery_label.setStyleSheet("font-weight: bold; color: #27ae60; border: 1px solid #27ae60; padding: 4px 8px; border-radius: 4px; font-size: 11px;")
            controls.addWidget(self.battery_label)
            
        main_layout.addLayout(controls)
        
        # PyQtGraph widget
        self.win = pg.GraphicsLayoutWidget(title="LSL FFT & Noise Analyzer")
        main_layout.addWidget(self.win)
        
        # Plot 1: Time Domain (Top)
        self.plot_time = self.win.addPlot(row=0, col=0, title="Time Domain Waveform (microvolts)")
        self.plot_time.setLabel('bottom', 'Time', units='s')
        self.plot_time.setLabel('left', 'Amplitude', units='uV')
        self.plot_time.showGrid(x=True, y=True, alpha=0.3)
        self.plot_time.setXRange(-self.buffer_duration, 0)
        
        # Time curves
        self.curve_raw_time = self.plot_time.plot(pen=pg.mkPen('#e74c3c', width=1.0), name="Raw")
        self.curve_filt_time = self.plot_time.plot(pen=pg.mkPen('#2ecc71', width=1.5), name="Filtered")
        
        # Plot 2: Frequency Domain (Bottom)
        self.plot_freq = self.win.addPlot(row=1, col=0, title="FFT Amplitude Spectrum (0 - 65 Hz)")
        self.plot_freq.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_freq.setLabel('left', 'Amplitude Spectrum (uV)')
        self.plot_freq.showGrid(x=True, y=True, alpha=0.3)
        self.plot_freq.setXRange(0, 65)
        
        # Freq curves
        self.curve_raw_fft = self.plot_freq.plot(pen=pg.mkPen('#e74c3c', width=1.0), name="Raw Spectrum")
        self.curve_filt_fft = self.plot_freq.plot(pen=pg.mkPen('#2ecc71', width=1.5), name="Filtered Spectrum")
        
        self.setWindowTitle(f"g.Nautilus FFT Spectrum & Noise Analyzer - {self.info.name()}")
        self.resize(1000, 700)
        
    def on_channel_changed(self, index):
        self.current_channel_idx = index
        self.plot_time.setTitle(f"Time Domain Waveform - {self.plot_channel_names[index]}")
        self.plot_freq.setTitle(f"FFT Amplitude Spectrum - {self.plot_channel_names[index]}")
        
    def update_plots(self):
        # Pull data chunk
        chunk, timestamps = self.inlet.pull_chunk()
        if not chunk:
            return
            
        chunk = np.array(chunk)
        num_samples = chunk.shape[0]
        
        # Update raw rolling buffer
        self.raw_buffer = np.roll(self.raw_buffer, -num_samples, axis=0)
        self.raw_buffer[-num_samples:, :] = chunk[:, :self.num_channels]
        
        # Update battery level display if available
        if self.has_battery:
            battery_val = np.mean(chunk[:, -1])
            self.battery_label.setText(f"Battery: {battery_val:.1f}%")
            
        # Select current channel raw data
        raw_ch = self.raw_buffer[:, self.current_channel_idx]
        
        # 1. Detrend raw channel to remove huge DC offset (essential for plotting/FFT)
        raw_detrended = signal.detrend(raw_ch)
        
        # 2. Filter raw data
        filt_ch = signal.lfilter(self.b_band, self.a_band, raw_detrended)
        filt_ch = signal.lfilter(self.b_notch, self.a_notch, filt_ch)
        
        # 3. Compute FFT for both
        n_fft = self.buffer_samples
        freqs = np.fft.rfftfreq(n_fft, d=1.0/self.fs)
        
        # Magnitude spectrums
        raw_fft_vals = np.abs(np.fft.rfft(raw_detrended)) / (n_fft / 2.0)
        filt_fft_vals = np.abs(np.fft.rfft(filt_ch)) / (n_fft / 2.0)
        
        # Update Time Domain Plot
        self.curve_raw_time.setData(self.time_axis, raw_detrended)
        self.curve_filt_time.setData(self.time_axis, filt_ch)
        
        # Auto-scale time plot limits based on signal amplitude
        max_t = np.max(np.abs(raw_detrended))
        if max_t > 5.0:
            self.plot_time.setYRange(-max_t * 1.1, max_t * 1.1)
        else:
            self.plot_time.setYRange(-5.0, 5.0)
            
        # Update Frequency Domain Plot
        self.curve_raw_fft.setData(freqs, raw_fft_vals)
        self.curve_filt_fft.setData(freqs, filt_fft_vals)
        
        # Auto-scale FFT plot limits based on frequencies of interest (2-60Hz)
        interest_mask = (freqs >= 1) & (freqs <= 65)
        if np.any(interest_mask):
            max_f = np.max(raw_fft_vals[interest_mask])
            if max_f > 0.1:
                self.plot_freq.setYRange(0, max_f * 1.1)
            else:
                self.plot_freq.setYRange(0, 0.5)

def main():
    print("=" * 70)
    print("      g.Nautilus LSL FFT Spectrum & Noise Analyzer (PySide6)")
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
    viewer = LSLFFTViewer(inlet)
    viewer.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
