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

print("Scanning for connected g.tec devices...")
cd = pygds.ConnectedDevices()
if len(cd) == 0:
    print("Error: No devices found. Please ensure GDS service is started and USB receiver is plugged in.")
    sys.exit(1)

serial = cd[0][0]
print(f"Connecting to device: {serial}...")
try:
    device = pygds.GDS(serial, open_exclusively=False)
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)

try:
    # 1. Get Factory EEPROM Scaling
    factory_scaling = device.GetFactoryScaling()
    if factory_scaling and len(factory_scaling) > 0:
        # If the API returns it in a separate structure
        factory_dict = factory_scaling[0]
        if hasattr(factory_dict, '_to_python'):
            factory_dict = factory_dict._to_python()
        factory_factors = factory_dict.get('Factor', [])
        factory_offsets = factory_dict.get('Offset', [])
    else:
        # Usually GDS loads factory scaling directly into current parameters on start
        factory_factors = None
        factory_offsets = None

    # 2. Get Currently Active Scaling
    current_scaling = device.GetScaling()
    if current_scaling and len(current_scaling) > 0:
        current_dict = current_scaling[0]
        if isinstance(current_dict, dict):
            pass
        elif hasattr(current_dict, '_to_python'):
            current_dict = current_dict._to_python()
        current_factors = current_dict.get('Factor', [])
        current_offsets = current_dict.get('Offset', [])
    else:
        current_factors = []
        current_offsets = []

    channels = device.GetChannelNames()
    if len(channels) > 0 and isinstance(channels[0], (list, tuple)):
        channels = list(channels[0])

    print("\n" + "="*80)
    print(f" Calibration Parameters Comparison: {serial} ")
    print("="*80)
    if factory_factors is None:
        print(f"{'Channel':<10} | {'Current Factor':<20} | {'Current Offset (uV)':<25}")
        print("-"*80)
        for i, name in enumerate(channels):
            cur_f = current_factors[i] if i < len(current_factors) else 1.0
            cur_o = current_offsets[i] if i < len(current_offsets) else 0.0
            print(f"{name:<10} | {cur_f:<20.6f} | {cur_o:<25.2f}")
        print("="*80)
        print("[+] Note: Current scaling factors were loaded directly from the headset's EEPROM on connection.")
        print("    If scaling was neutral (uncalibrated), factors would be exactly 1.000000 and offsets 0.00.")
    else:
        all_match = True
        for i, name in enumerate(channels):
            fac_f = factory_factors[i] if i < len(factory_factors) else 1.0
            cur_f = current_factors[i] if i < len(current_factors) else 1.0
            fac_o = factory_offsets[i] if i < len(factory_offsets) else 0.0
            cur_o = current_offsets[i] if i < len(current_offsets) else 0.0

            mismatch_str = ""
            if not np.isclose(fac_f, cur_f, atol=1e-5) or not np.isclose(fac_o, cur_o, atol=1e-2):
                mismatch_str = " (MISMATCH)"
                all_match = False

            print(f"{name:<10} | {fac_f:<15.6f} | {cur_f:<15.6f} | {fac_o:<20.2f} | {cur_o:<20.2f}{mismatch_str}")
        
        print("="*80)
        if all_match:
            print("[+] SUCCESS: Currently active scaling matches the factory EEPROM calibration.")
        else:
            print("[!] WARNING: Current scaling differs from the factory EEPROM defaults.")
            print("    To restore factory defaults, run: device.SetScaling(device.GetFactoryScaling())")

except Exception as e:
    print(f"Error reading calibration: {e}")
finally:
    device.Close()
    print("\nDisconnected successfully.")
