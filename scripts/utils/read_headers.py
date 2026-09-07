import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

import os

header_paths = [
    r"C:\Users\VR\Documents\gtec\gNEEDaccessClientAPI\C\GDSClientAPI.h",
    r"C:\Users\VR\Documents\gtec\gNEEDaccessClientAPI\C\GDSClientAPI_gHIamp.h",
    r"C:\Users\VR\Documents\gtec\gNEEDaccessClientAPI\C\GDSClientAPI_gNautilus.h",
    r"C:\Users\VR\Documents\gtec\gNEEDaccessClientAPI\C\GDSClientAPI_gUSBamp.h"
]

for path in header_paths:
    if os.path.exists(path):
        print(f"=== File: {path} ===")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        # Search for lines containing impedance, error, or negative definitions
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # look for constants or comments with negative values
            if 'impedance' in line_lower or 'error' in line_lower or 'status' in line_lower:
                if any(x in line for x in ['-', '0x', 'define', 'const', 'enum']):
                    print(f"Line {i+1}: {line.strip()}")
    else:
        print(f"Not found: {path}")
