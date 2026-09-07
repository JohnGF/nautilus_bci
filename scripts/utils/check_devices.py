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

print("Connected devices list:")
try:
    devices = pygds.ConnectedDevices()
    for i, (serial, dev_type, in_use) in enumerate(devices):
        type_str = "g.Nautilus" if dev_type == 3 else f"Type {dev_type}"
        print(f"  [{i}] Serial: {serial} | Type: {type_str} | In Use: {in_use}")
except Exception as e:
    print("Error scanning devices:", e)
