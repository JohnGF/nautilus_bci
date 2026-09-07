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

class EEGFeatureExtractor(QtWidgets.QWidget):
    def __init__(self, inlet):
        super().__init__()
        self.inlet = inlet
        self.info = inlet.info()
        self.fs = self.info.nominal_srate()
        self.num_channels = self.info.channel_count()
        
        # Buffer setup (last 2 seconds for feature extraction)
        self.buffer_duration = 2.0
        self.buffer_samples = int(self.fs * self.buffer_duration)
        self.data_buffer = np.zeros((self.buffer_samples, self.num_channels))
        
        # Alpha history buffer for the rolling line plot (last 30 seconds of feature calculations)
        self.history_len = 150  # 150 points @ ~5 Hz update rate = 30 seconds
        self.alpha_history = np.zeros(self.history_len)
        self.beta_history = np.zeros(self.history_len)
        self.history_time = np.linspace(-30.0, 0, self.history_len)
        
        # Design 4th-order Butterworth bandpass filter (2-45 Hz) & 50 Hz Notch filter
        nyq = 0.5 * self.fs
        self.b_band, self.a_band = signal.butter(4, [2.0 / nyq, 45.0 / nyq], btype='band')
        self.b_notch, self.a_notch = signal.iirnotch(50.0, 30.0, self.fs)
        
        self.init_ui()
        
        # Timer for polling LSL and updating features (refresh rate of 5 Hz / 200ms is perfect for math)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process_eeg)
        self.timer.start(200)
        
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Info header
        header = QtWidgets.QLabel("EEG Real-Time Feature Analyzer (Alpha vs. Beta Bands)")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(header)
        
        # Explanation label
        explanation = QtWidgets.QLabel(
            "• Alpha Waves (8-12 Hz) indicate relaxation (increases when you close your eyes).\n"
            "• Beta Waves (12-30 Hz) indicate active concentration, attention, or mental processing."
        )
        explanation.setStyleSheet("color: #555; margin-bottom: 10px;")
        layout.addWidget(explanation)
        
        # PyQtGraph GraphicsLayoutWidget
        self.win = pg.GraphicsLayoutWidget(title="EEG Brain State Analyzer")
        layout.addWidget(self.win)
        
        # Plot 1: Bar chart of average relative band powers
        self.p_bars = self.win.addPlot(row=0, col=0, title="Live Relative Band Power (Average Across Channels)")
        self.p_bars.setLabel('left', 'Relative Power (%)')
        self.p_bars.setYRange(0, 100)
        
        # Define the 4 frequency bands
        self.bands = ['Delta (1-4Hz)\nSleep', 'Theta (4-8Hz)\nMeditation', 'Alpha (8-12Hz)\nRelaxed', 'Beta (12-30Hz)\nFocused']
        self.x_indices = np.array([1, 2, 3, 4])
        
        # Color palettes for bands
        self.bar_colors = [
            pg.mkColor('#3498db'),  # Delta (Blue)
            pg.mkColor('#2ecc71'),  # Theta (Green)
            pg.mkColor('#f1c40f'),  # Alpha (Yellow/Gold)
            pg.mkColor('#e74c3c')   # Beta (Red)
        ]
        
        # Initialize bar graph
        self.bar_graph = pg.BarGraphItem(x=self.x_indices, height=np.zeros(4), width=0.6, brushes=self.bar_colors)
        self.p_bars.addItem(self.bar_graph)
        
        # Set x-axis ticks
        ax = self.p_bars.getAxis('bottom')
        ax.setTicks([[(i+1, self.bands[i]) for i in range(len(self.bands))]])
        
        # Plot 2: Rolling state indices (Relaxation vs. Concentration)
        self.p_history = self.win.addPlot(row=0, col=1, title="Rolling Cognitive State (Last 30s)")
        self.p_history.setLabel('left', 'Relative Power (%)')
        self.p_history.setLabel('bottom', 'Time', units='s')
        self.p_history.setYRange(0, 100)
        self.p_history.addLegend()
        
        # Curves for rolling plot
        self.alpha_curve = self.p_history.plot(
            self.history_time, self.alpha_history, 
            pen=pg.mkPen('#f1c40f', width=2), name="Relaxation Index (Alpha)"
        )
        self.beta_curve = self.p_history.plot(
            self.history_time, self.beta_history, 
            pen=pg.mkPen('#e74c3c', width=2), name="Focus Index (Beta)"
        )
        
        self.setWindowTitle(f"g.Nautilus EEG Brain Wave Feature Analyzer - {self.info.name()}")
        self.resize(1100, 600)
        
    def process_eeg(self):
        # Pull all available samples from LSL
        chunk, timestamps = self.inlet.pull_chunk()
        if not chunk:
            return
            
        chunk = np.array(chunk)
        num_samples = chunk.shape[0]
        
        # Update rolling buffer
        self.data_buffer = np.roll(self.data_buffer, -num_samples, axis=0)
        self.data_buffer[-num_samples:, :] = chunk[:, :self.num_channels]
        
        # Remove baseline DC drift (vectorized along axis=0)
        detrended = signal.detrend(self.data_buffer, axis=0)
        
        # Apply 4th-order Butterworth bandpass and Notch filters
        filtered = signal.lfilter(self.b_band, self.a_band, detrended, axis=0)
        filtered = signal.lfilter(self.b_notch, self.a_notch, filtered, axis=0)
        
        # Compute FFT for all channels at once (vectorized call along axis=0)
        # Returns shape: (freq_bins, channels)
        n_fft = self.buffer_samples
        freqs = np.fft.rfftfreq(n_fft, d=1.0/self.fs)
        fft_vals = np.abs(np.fft.rfft(filtered, axis=0)) ** 2
        
        # Frequency band indices
        idx_delta = (freqs >= 1) & (freqs < 4)
        idx_theta = (freqs >= 4) & (freqs < 8)
        idx_alpha = (freqs >= 8) & (freqs < 12)
        idx_beta = (freqs >= 12) & (freqs < 30)
        
        # Calculate power per band for all channels (mean along freq axis)
        d_pow = np.mean(fft_vals[idx_delta, :], axis=0)
        t_pow = np.mean(fft_vals[idx_theta, :], axis=0)
        a_pow = np.mean(fft_vals[idx_alpha, :], axis=0)
        b_pow = np.mean(fft_vals[idx_beta, :], axis=0)
        
        # Total power per channel
        total = d_pow + t_pow + a_pow + b_pow
        total[total == 0] = 1.0  # Avoid division by zero
        
        # Relative band powers per channel
        rel_delta = d_pow / total
        rel_theta = t_pow / total
        rel_alpha = a_pow / total
        rel_beta = b_pow / total
        
        # Average relative powers across all channels
        avg_delta = np.mean(rel_delta) * 100
        avg_theta = np.mean(rel_theta) * 100
        avg_alpha = np.mean(rel_alpha) * 100
        avg_beta = np.mean(rel_beta) * 100
        
        # 1. Update live bar chart
        self.bar_graph.setOpts(height=[avg_delta, avg_theta, avg_alpha, avg_beta])
        
        # 2. Update rolling history charts
        self.alpha_history = np.roll(self.alpha_history, -1)
        self.alpha_history[-1] = avg_alpha
        self.alpha_history = np.clip(self.alpha_history, 0, 100) # clip to keep clean in grid
        
        self.beta_history = np.roll(self.beta_history, -1)
        self.beta_history[-1] = avg_beta
        self.beta_history = np.clip(self.beta_history, 0, 100)
        
        self.alpha_curve.setData(self.history_time, self.alpha_history)
        self.beta_curve.setData(self.history_time, self.beta_history)

def main():
    print("=" * 70)
    print("      g.Nautilus Live Brain Wave Feature Analyzer (PySide6)")
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
    extractor = EEGFeatureExtractor(inlet)
    extractor.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
