import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock

def main():
    # 32 channel labels + Battery
    channel_names = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
        'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 
        'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 
        'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10',
        'Battery'
    ]
    
    num_channels = len(channel_names)
    sampling_rate = 250.0  # Hz
    serial = "SIMULATED_NAUTILUS"
    
    print("=" * 70)
    print(f" Starting Mock g.Nautilus LSL Streamer ".center(70, "="))
    print("=" * 70)
    print(f"Stream Name: gNautilus")
    print(f"Stream Type: EEG")
    print(f"Sampling Rate: {sampling_rate} Hz")
    print(f"Channels ({num_channels}): {channel_names}")
    print("=" * 70)
    
    # Create StreamInfo
    info = StreamInfo(
        name='gNautilus',
        type='EEG',
        channel_count=num_channels,
        nominal_srate=sampling_rate,
        channel_format='float32',
        source_id=f'gNautilus_{serial}'
    )
    
    # Add metadata
    channels_meta = info.desc().append_child("channels")
    for chan_name in channel_names:
        chan = channels_meta.append_child("channel")
        chan.append_child_value("label", chan_name)
        chan.append_child_value("unit", "microvolts" if chan_name != "Battery" else "percent")
        chan.append_child_value("type", "EEG" if chan_name != "Battery" else "AUX")
        
    outlet = StreamOutlet(info)
    print("[+] LSL Outlet created successfully.")
    
    # Simulation settings
    chunk_size = 10  # push 10 samples at a time
    sleep_time = chunk_size / sampling_rate  # 0.04 seconds (40 ms)
    
    # Generate frequencies for channels to show distinct waveforms
    # Occipital channels (O1, O2, Oz, Pz) will get simulated alpha bursts (10 Hz)
    occipital_indices = [8, 9, 18, 19]
    
    start_time = time.time()
    sample_count = 0
    
    print("\n[+] Mock LSL EEG stream is broadcasting!")
    print("    Open 'uv run python lsl_viewer.py' or 'uv run python eeg_features.py' to visualize.")
    print("    Press Ctrl+C to stop streaming.")
    print("-" * 70)
    
    battery_level = 100.0
    
    try:
        next_chunk_time = local_clock()
        while True:
            # Generate a chunk of data
            chunk = []
            current_time = time.time() - start_time
            
            # Slowly deplete battery
            battery_level = max(0.0, 100.0 - (current_time / 10.0))  # drop 1% every 10 seconds
            
            # Alpha burst active every 8 seconds, lasting for 3 seconds
            alpha_burst_active = (int(current_time) % 8) < 3
            
            for s in range(chunk_size):
                t = (sample_count + s) / sampling_rate
                sample = []
                
                for ch_idx in range(num_channels - 1):
                    # 1. 1/f-like baseline noise (pinkish)
                    noise = np.random.normal(0, 2.0)
                    
                    # 2. 50 Hz powerline interference (simulated)
                    powerline = 10.0 * np.sin(2 * np.pi * 50.0 * t)
                    
                    # 3. Base brainwave rhythm (e.g. 5-15 Hz depending on channel)
                    base_freq = 7.0 + (ch_idx % 5) * 1.5
                    base_wave = 5.0 * np.sin(2 * np.pi * base_freq * t)
                    
                    # 4. Occipital Alpha burst (10 Hz)
                    alpha_wave = 0.0
                    if alpha_burst_active and ch_idx in occipital_indices:
                        alpha_wave = 25.0 * np.sin(2 * np.pi * 10.0 * t)
                        
                    # Total microvolts signal
                    val = noise + powerline + base_wave + alpha_wave
                    sample.append(val)
                
                # Append battery channel
                sample.append(battery_level)
                chunk.append(sample)
                
            sample_count += chunk_size
            
            # Push to LSL
            outlet.push_chunk(chunk, next_chunk_time)
            
            # Print status periodically
            if sample_count % 500 == 0:
                print(f"[LSL STREAMING] Pushed {sample_count} samples | Battery: {battery_level:.1f}% | Alpha burst: {'ACTIVE' if alpha_burst_active else 'OFF'}")
                
            next_chunk_time += sleep_time
            
            # Sleep until the next chunk is due (precision timer)
            delay = next_chunk_time - local_clock()
            if delay > 0:
                time.sleep(delay)
            else:
                # If we fall behind, catch up
                next_chunk_time = local_clock()
                
    except KeyboardInterrupt:
        print("\n[*] Streaming stopped by user.")

if __name__ == '__main__':
    main()
