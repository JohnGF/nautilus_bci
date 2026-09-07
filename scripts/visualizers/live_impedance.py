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

try:
    from colorama import init, Fore, Style
    init()
except ImportError:
    class DummyColor:
        def __getattr__(self, name):
            return ""
    Fore = DummyColor()
    Style = DummyColor()

# Scan and connect
cd = pygds.ConnectedDevices()
if len(cd) == 0:
    print("Error: No connected devices found. Please make sure the USB receiver is plugged in.")
    sys.exit(1)

serial = cd[0][0]
print(f"Connecting to device: {serial}...")
try:
    device = pygds.GDS(serial, open_exclusively=False)
except Exception as e:
    print(f"Error connecting to headset: {e}")
    sys.exit(1)

print("Connected! Starting live impedance monitor. Press Ctrl+C to stop.")
time.sleep(1)

try:
    while True:
        try:
            impedances_list = device.GetImpedanceEx()
            if not impedances_list or len(impedances_list) == 0:
                print("Error: Failed to receive impedance data.")
                time.sleep(1)
                continue
                
            impedances = impedances_list[0]
            channels = device.GetChannelNames()
            if len(channels) > 0 and isinstance(channels[0], (list, tuple)):
                channels = list(channels[0])
                
            # Clear terminal
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(Fore.CYAN + "=" * 65)
            print(f" g.Nautilus Live Impedance Monitor | Device: {serial} ".center(65, "="))
            print("=" * 65 + Style.RESET_ALL)
            
            cz_val = None
            for name, imp in zip(channels, impedances):
                if name.upper() == 'CZ':
                    cz_val = imp
                    break
            
            # Print warning if CZ is not connected
            if cz_val is None or cz_val < 0:
                print(Fore.RED + Style.BRIGHT + f"  [!] WARNING: Cz electrode has NO CONTACT or is INVALID ({cz_val:.2f})" + Style.RESET_ALL)
                print(Fore.YELLOW + "      g.Nautilus uses Cz and GND as references for impedance checks." + Style.RESET_ALL)
                print(Fore.YELLOW + "      Please gel and adjust Cz & GND first to see other channels." + Style.RESET_ALL)
                print("-" * 65)
            else:
                print(Fore.GREEN + Style.BRIGHT + f"  [+] Cz Reference is ACTIVE ({cz_val:.2f} kOhm)" + Style.RESET_ALL)
                print("-" * 65)
                
            # Print 4 columns of channels
            cols = 4
            rows = (len(channels) + cols - 1) // cols
            for r in range(rows):
                row_str = ""
                for c in range(cols):
                    idx = r + c * rows
                    if idx < len(channels):
                        name = channels[idx]
                        imp = impedances[idx]
                        
                        if imp < 0:
                            color = Fore.WHITE + Style.DIM
                            imp_str = f"NC ({imp:.1f})"
                        elif imp <= 30.0:
                            color = Fore.GREEN + Style.BRIGHT
                            imp_str = f"{imp:.2f}k"
                        elif imp <= 100.0:
                            color = Fore.YELLOW
                            imp_str = f"{imp:.2f}k"
                        else:
                            color = Fore.RED
                            imp_str = f"{imp:.2f}k"
                            
                        cell = f"  {name:<6}: {color}{imp_str:<10}{Style.RESET_ALL}"
                        row_str += cell
                print(row_str)
            print("-" * 65)
            print("Press Ctrl+C to exit this live monitor.")
            
        except Exception as e:
            print(f"Error reading impedances: {e}")
            
        time.sleep(1.0)
except KeyboardInterrupt:
    print("\nStopping impedance monitor...")
finally:
    try:
        device.Close()
    except:
        pass
