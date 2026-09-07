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

gds_dll_path = r"C:\Program Files\gtec\gNEEDaccess"
if os.path.exists(gds_dll_path):
    os.add_dll_directory(gds_dll_path)
else:
    print(f"Error: g.NEEDaccess installation directory not found at {gds_dll_path}")
    sys.exit(1)

try:
    import pygds
    client_dll = os.path.join(gds_dll_path, "GDSClientAPI.dll")
    pygds.Uninitialize()
    if not pygds.Initialize(gds_dll=client_dll):
        raise RuntimeError("Failed to initialize pygds library.")
except Exception as e:
    print(f"Error loading pygds or GDS client library: {e}")
    sys.exit(1)

print("Scanning devices...")
cd = pygds.ConnectedDevices()
if len(cd) == 0:
    print("No devices found.")
    sys.exit(1)

serial = cd[0][0]
print(f"Connecting to device: {serial}...")
try:
    device = pygds.GDS(serial, open_exclusively=False)
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)

try:
    # Get configuration info
    channels = device.GetChannelNames()
    if len(channels) > 0 and isinstance(channels[0], (list, tuple)):
        channels = list(channels[0])
    
    print("\n" + "="*50)
    print(f" Device Configuration: {serial} ")
    print("="*50)
    print(f"Sampling Rate: {device.SamplingRate} Hz")
    print(f"Channels Count: {len(channels)}")
    print(f"Channels: {channels}")
    
    # Enable raw electrode signal source
    device.InputSignal = pygds.GNAUTILUS_INPUT_SIGNAL_ELECTRODE
    device.BatteryLevel = 0
    device.AccelerationData = 0
    device.ValidationIndicator = 0
    
    # Configure raw streaming
    for ch in device.Channels:
        ch.Acquire = 1
        ch.BandpassFilterIndex = -1
        ch.NotchFilterIndex = -1
        ch.BipolarChannel = -1
        
    device.SetConfiguration()
    print("[+] Configuration set successfully to physical electrodes, raw streaming.")
    
    # Get impedances
    print("\nChecking electrode impedances...")
    impedances_list = device.GetImpedanceEx()
    if impedances_list and len(impedances_list) > 0:
        impedances = impedances_list[0]
        print(f"  {'Channel':<10} | {'Impedance (kOhm)':<18}")
        print("  " + "-"*35)
        for name, imp in zip(channels, impedances):
            imp_status = "NC/Saturated" if imp < 0 else f"{imp:.2f} kOhm"
            print(f"  {name:<10} | {imp_status:<18}")
    else:
        print("[-] Could not retrieve impedance values.")
        
    # Get 1 second of raw data
    print("\nReading 1 second of live raw EEG data (250 samples)...")
    data_samples = []
    
    def callback(data_block):
        data_samples.append(data_block)
        # We need at least 250 samples total
        total_samples = sum(len(block) for block in data_samples)
        if total_samples >= 250:
            return False  # stop streaming
        return True

    # Start data acquisition
    device.GetData(device.NumberOfScans, callback)
    
    # Combine data blocks
    if data_samples:
        all_data = np.vstack(data_samples)
        print(f"\nSuccessfully read {all_data.shape[0]} samples across {all_data.shape[1]} channels.")
        
        print("\n" + "="*70)
        print(f" {'Channel':<10} | {'Mean (uV)':<12} | {'Std Dev (uV)':<12} | {'Min (uV)':<12} | {'Max (uV)':<12} ")
        print("="*70)
        
        for i, name in enumerate(channels[:all_data.shape[1]]):
            ch_data = all_data[:, i]
            mean_val = np.mean(ch_data)
            std_val = np.std(ch_data)
            min_val = np.min(ch_data)
            max_val = np.max(ch_data)
            
            # Identify signal status
            status = ""
            if std_val < 0.1:
                status = " (FLAT-LINED/ZERO)"
            elif abs(mean_val) > 50000 or std_val > 10000:
                status = " (SATURATED/RAILED)"
                
            print(f" {name:<10} | {mean_val:<12.2f} | {std_val:<12.2f} | {min_val:<12.2f} | {max_val:<12.2f}{status}")
        print("="*70)
    else:
        print("[-] No raw data was received.")

except Exception as e:
    print(f"Error during signal check: {e}")
finally:
    device.Close()
    print("\nDisconnected successfully.")
