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
import scipy.signal as signal
import matplotlib.pyplot as plt

# Ensure g.NEEDaccess client DLL directory is added to search path
gds_dll_path = r"C:\Program Files\gtec\gNEEDaccess"
if os.path.exists(gds_dll_path):
    os.add_dll_directory(gds_dll_path)
else:
    print(f"Warning: g.NEEDaccess installation directory not found at {gds_dll_path}")

try:
    import pygds
    # Initialize with absolute path of the Client DLL
    client_dll = os.path.join(gds_dll_path, "GDSClientAPI.dll")
    pygds.Uninitialize()
    if not pygds.Initialize(gds_dll=client_dll):
        raise RuntimeError("Failed to initialize pygds library.")
except Exception as e:
    print(f"Error loading pygds or GDS client library: {e}")
    sys.exit(1)

def main():
    print("=" * 70)
    print("      g.Nautilus Live Brain-Wave & FFT Spectrum Visualizer")
    print("=" * 70)
    
    # 1. Scan and connect to the device
    print("[*] Scanning for connected BCI devices...")
    cd = pygds.ConnectedDevices()
    if len(cd) == 0:
        print("[-] No connected devices found. Please make sure the USB receiver is plugged in.")
        sys.exit(1)
        
    serial = cd[0][0]
    print(f"[+] Found device: {serial}. Connecting...")
    device = pygds.GDS(serial, open_exclusively=False)
    print("[+] Connected successfully!")
    
    try:
        # 2. Configure device parameters
        device.SamplingRate = 250  # 250 Hz is default for g.Nautilus
        for ch in device.Channels:
            ch.Acquire = 1
            ch.BandpassFilterIndex = -1  # We will do filtering in Python
            ch.NotchFilterIndex = -1
            ch.BipolarChannel = -1
            
        device.InputSignal = pygds.GNAUTILUS_INPUT_SIGNAL_ELECTRODE
        device.SetConfiguration()
        
        sampling_rate = device.SamplingRate
        channel_names = device.GetChannelNames()
        if len(channel_names) > 0 and isinstance(channel_names[0], (list, tuple)):
            channel_names = list(channel_names[0])
        num_channels = len(channel_names)
        
        # Limit plotting to first 8 channels if there are more (to keep display readable)
        plot_channels_count = min(num_channels, 8)
        print(f"[+] Device has {num_channels} channels. Visualizing the first {plot_channels_count} channels.")
        
        # 3. Setup buffers for real-time processing
        # We will keep a rolling window of 3 seconds of raw data
        buffer_duration = 3.0  # seconds
        buffer_samples = int(sampling_rate * buffer_duration)
        
        # Buffer shape: (samples, channels)
        raw_buffer = np.zeros((buffer_samples, num_channels))
        
        # Filter coefficients: Bandpass 1-40 Hz (Butterworth) & Notch 50 Hz (power line interference)
        nyq = 0.5 * sampling_rate
        b_band, a_band = signal.butter(2, [1.0 / nyq, 40.0 / nyq], btype='band')
        b_notch, a_notch = signal.iirnotch(50.0, 30.0, sampling_rate)
        
        # Live plot setup
        plt.ion() # turn on interactive mode
        fig, axes = plt.subplots(plot_channels_count, 2, figsize=(12, 10))
        fig.suptitle(f"Real-Time Brain Waves (Bandpass 1-40Hz) & FFT Spectrum | Device: {serial}", fontsize=14)
        
        time_axis = np.linspace(-buffer_duration, 0, buffer_samples)
        
        time_lines = []
        fft_lines = []
        
        for i in range(plot_channels_count):
            # Time-series plots (Left column)
            ax_time = axes[i, 0]
            line_t, = ax_time.plot(time_axis, np.zeros(buffer_samples), color='#1f77b4', lw=1.2)
            ax_time.set_ylabel(channel_names[i], fontsize=10, rotation=0, labelpad=25, fontweight='bold')
            ax_time.grid(True, linestyle='--', alpha=0.5)
            ax_time.set_xlim(-buffer_duration, 0)
            ax_time.set_ylim(-50, 50) # microvolts
            time_lines.append((ax_time, line_t))
            
            # FFT plots (Right column)
            ax_fft = axes[i, 1]
            line_f, = ax_fft.plot([], [], color='#ff7f0e', lw=1.2)
            ax_fft.grid(True, linestyle='--', alpha=0.5)
            ax_fft.set_xlim(0, 50) # Focus on 0 to 50 Hz
            ax_fft.set_ylim(0, 10) # Amplitude spectrum
            fft_lines.append((ax_fft, line_f))
            
            if i == plot_channels_count - 1:
                ax_time.set_xlabel("Time (seconds)", fontweight='bold')
                ax_fft.set_xlabel("Frequency (Hz)", fontweight='bold')
            else:
                ax_time.set_xticklabels([])
                ax_fft.set_xticklabels([])
                
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 4. Data callback definition
        block_size = 8  # read blocks of 8 samples (~32 ms)
        
        # Cache for update frequency control (limit plot refresh to ~20 Hz to avoid overhead)
        last_plot_time = [0.0]
        
        def daq_callback(data_block):
            nonlocal raw_buffer
            
            # data_block shape: (block_size, channels)
            # Shift buffer and insert new data
            raw_buffer = np.roll(raw_buffer, -block_size, axis=0)
            raw_buffer[-block_size:, :] = data_block
            
            current_time = time.time()
            if current_time - last_plot_time[0] < 0.05:
                return True # continue acquisition
                
            last_plot_time[0] = current_time
            
            # Apply filters channel by channel on the raw buffer
            # Detrend first to remove any massive DC offset/drift
            detrended = signal.detrend(raw_buffer, axis=0)
            
            # Filter
            filtered = signal.lfilter(b_band, a_band, detrended, axis=0)
            filtered = signal.lfilter(b_notch, a_notch, filtered, axis=0)
            
            # FFT analysis on the filtered data
            n_fft = buffer_samples
            freqs = np.fft.rfftfreq(n_fft, d=1.0/sampling_rate)
            
            # Update plots
            for ch_idx in range(plot_channels_count):
                # Time series update
                y_data = filtered[:, ch_idx]
                ax_t, line_t = time_lines[ch_idx]
                line_t.set_ydata(y_data)
                
                # Auto-scale time axis y-limits based on local signal amplitude
                # to prevent flat-line appearance on different electrodes
                max_val = np.max(np.abs(y_data))
                if max_val > 5:
                    ax_t.set_ylim(-max_val * 1.2, max_val * 1.2)
                else:
                    ax_t.set_ylim(-5, 5)
                
                # FFT update
                fft_vals = np.abs(np.fft.rfft(y_data)) / (n_fft / 2.0)
                ax_f, line_f = fft_lines[ch_idx]
                line_f.set_data(freqs, fft_vals)
                
                # Auto-scale FFT y-limits
                max_fft = np.max(fft_vals[(freqs >= 1) & (freqs <= 50)]) if len(freqs) > 0 else 1.0
                if max_fft > 0.5:
                    ax_f.set_ylim(0, max_fft * 1.3)
                else:
                    ax_f.set_ylim(0, 0.5)
            
            # Draw
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Keep matplotlib responsive
            plt.pause(0.001)
            
            # Check if matplotlib window is still open
            if not plt.fignum_exists(fig.number):
                print("[*] Visualizer window closed by user.")
                return False  # stop acquisition
                
            return True  # continue acquisition
            
        print("\n[+] Starting live data visualization...")
        print("    - Left Side: Time-domain EEG signals (DC offset removed via 1-40Hz filter).")
        print("    - Right Side: Frequency spectrum (FFT). You will see peaks at specific frequencies (e.g. 8-12Hz for Alpha waves).")
        print("    * CLOSE the window or press Ctrl+C in terminal to stop recording *")
        
        # Start acquisition loop
        device.GetData(block_size, daq_callback)
        
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
    except Exception as e:
        print(f"\n[-] Error in visualization loop: {e}")
    finally:
        print("[*] Disconnecting from device...")
        try:
            device.Close()
            del device
            print("[+] Disconnected successfully.")
        except:
            pass
        plt.close('all')

if __name__ == '__main__':
    main()
