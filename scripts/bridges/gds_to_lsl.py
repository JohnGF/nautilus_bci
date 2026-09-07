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
import threading
import json
from datetime import datetime
from pylsl import StreamInfo, StreamOutlet, local_clock

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

try:
    from colorama import init, Fore, Style
    init()
except ImportError:
    class DummyColor:
        def __getattr__(self, name):
            return ""
    Fore = DummyColor()
    Style = DummyColor()

# Global variables for thread synchronization
stop_event = threading.Event()
gds_lock = threading.Lock()
current_battery_level = None
NON_INTERACTIVE = "--non-interactive" in sys.argv or "-y" in sys.argv

def print_header(title):
    print("\n" + "=" * 70)
    print(f" {Fore.CYAN}{title}{Style.RESET_ALL} ".center(80, "="))
    print("=" * 70)

def select_cap_type(cli_cap_type=None):
    if cli_cap_type:
        mode = cli_cap_type.lower()
        if mode in ['dry', 'sahara']:
            return 'dry'
        return 'wet'
    if NON_INTERACTIVE:
        return 'wet'
    
    print_header("Electrode Cap Selection")
    print("Choose electrode hardware type:")
    print(f"  [1] {Fore.GREEN}Wet Electrodes{Style.RESET_ALL} (g.SCARABEO / g.LADYBIRD Gel) - standard impedance (<30k), raw DC optional")
    print(f"  [2] {Fore.YELLOW}Dry Electrodes{Style.RESET_ALL} (g.SAHARA Pins) - auto hardware bandpass (prevents DC rail saturation), dry impedance (<100k)")
    choice = input("\nSelect cap type (1-2) [Default: 1]: ").strip()
    if choice == '2' or choice.lower() in ['dry', 'sahara', '2']:
        print(f"{Fore.YELLOW}[+] Selected: g.SAHARA Dry Electrode System{Style.RESET_ALL}")
        return 'dry'
    print(f"{Fore.GREEN}[+] Selected: Wet Gel Electrode System{Style.RESET_ALL}")
    return 'wet'

def battery_monitor_loop(serial):
    global current_battery_level
    log_file = "battery_log.json"
    print(f"[Battery Monitor] Background monitor started. Logging to {log_file} every 5 minutes.")
    
    # Wait for the first reading to populate
    time.sleep(5)
    
    while not stop_event.is_set():
        try:
            level = current_battery_level
            
            if level is not None:
                # Round to 1 decimal place
                level = round(level, 1)
                timestamp = datetime.now().isoformat()
                
                # Load existing log
                history = []
                if os.path.exists(log_file):
                    try:
                        with open(log_file, "r") as f:
                            history = json.load(f)
                    except:
                        pass
                
                # Append new reading
                history.append({
                    "timestamp": timestamp,
                    "battery_level": level,
                    "device_serial": serial
                })
                
                # Save log
                with open(log_file, "w") as f:
                    json.dump(history, f, indent=4)
                    
                # Calculate discharge rate and estimated time remaining
                est_str = "Calculating discharge rate..."
                if len(history) >= 2:
                    # Find readings for this specific headset
                    device_history = [h for h in history if h.get("device_serial") == serial]
                    if len(device_history) >= 2:
                        t1 = datetime.fromisoformat(device_history[0]["timestamp"])
                        t2 = datetime.fromisoformat(device_history[-1]["timestamp"])
                        b1 = device_history[0]["battery_level"]
                        b2 = device_history[-1]["battery_level"]
                        
                        duration_hours = (t2 - t1).total_seconds() / 3600.0
                        if duration_hours > 0.01:  # at least 36 seconds apart
                            pct_drop = b1 - b2
                            if pct_drop > 0:
                                rate = pct_drop / duration_hours  # % per hour
                                remaining_hours = b2 / rate
                                est_str = f"Discharge Rate: {rate:.1f}%/hr | Est. Remaining: {remaining_hours:.1f} hours"
                            elif pct_drop == 0:
                                est_str = f"Discharge Rate: 0.0%/hr (Stable) | Level: {b2}%"
                            else:
                                # Battery level went up (charging)
                                rate = -pct_drop / duration_hours
                                est_str = f"Charging Rate: {rate:.1f}%/hr | Level: {b2}%"
                                
                print(f"\n[{Fore.CYAN}BATTERY UPDATE{Style.RESET_ALL}] Level: {Fore.GREEN}{level}%{Style.RESET_ALL} | {est_str}")
            
        except Exception as e:
            # Prevent crashes in the logging loop
            pass
            
        # Sleep in 1-second chunks to check stop_event frequently
        for _ in range(300):
            if stop_event.is_set():
                break
            time.sleep(1)

def connect_headset():
    print_header("g.Nautilus Connection & Setup Manager (Dual-Transceiver Support)")
    
    # 1. Scan for devices
    print(f"{Fore.YELLOW}[*] Scanning for connected BCI USB receivers...{Style.RESET_ALL}")
    cd = pygds.ConnectedDevices()
    if len(cd) == 0:
        print(f"{Fore.RED}[-] No connected receivers found. Please plug in the USB receiver and try again.{Style.RESET_ALL}")
        sys.exit(1)
        
    print(f"\n{Fore.GREEN}[+] Detected {len(cd)} connected USB receiver(s):{Style.RESET_ALL}")
    for i, (ser, dev_type, in_use) in enumerate(cd):
        dev_label = "fNIRS Transceiver" if "NF" in ser.upper() or "NIRS" in ser.upper() else "EEG Transceiver / g.Nautilus"
        status_str = f"{Fore.RED}In Use{Style.RESET_ALL}" if in_use else f"{Fore.GREEN}Available{Style.RESET_ALL}"
        print(f"  [{i+1}] Serial: {Fore.CYAN}{ser}{Style.RESET_ALL} | Type: {dev_label} | Status: {status_str}")

    selected_idx = 0
    if len(cd) > 1 and not NON_INTERACTIVE:
        choice = input(f"\nSelect receiver to connect (1-{len(cd)}) [Default: 1]: ").strip()
        if choice:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(cd):
                    selected_idx = idx
            except ValueError:
                pass

    serial = cd[selected_idx][0]
    is_fnirs_device = "NF" in serial.upper() or "NIRS" in serial.upper()
    dev_type_name = "g.Nautilus fNIRS" if is_fnirs_device else "g.Nautilus EEG"
    print(f"\n{Fore.GREEN}[+] Selected transceiver: {serial} ({dev_type_name}){Style.RESET_ALL}")
    
    # 2. Connect loop (handles powered-off headset)
    device = None
    attempt = 1
    while True:
        try:
            print(f"{Fore.YELLOW}[*] Connecting to {dev_type_name} ({serial}) [attempt {attempt}]...{Style.RESET_ALL}")
            device = pygds.GDS(serial, open_exclusively=False)
            print(f"{Fore.GREEN}[+] Connected successfully to {dev_type_name}!{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"{Fore.RED}[-] Connection failed: {e}{Style.RESET_ALL}")
            print(f"\n{Fore.YELLOW}[!] The headset might be Powered OFF or out of wireless range.{Style.RESET_ALL}")
            print("    - Make sure the power button on the cap is turned ON (flashing LED).")
            print("    - Ensure the external battery pack is switched ON for fNIRS.")
            print("    - Ensure the headset has charge or is in range.")
            if NON_INTERACTIVE:
                if attempt >= 3:
                    print(f"{Fore.RED}[-] Non-interactive connection attempts exhausted. Exiting.{Style.RESET_ALL}")
                    sys.exit(1)
                print(f"{Fore.YELLOW}[*] Retrying in 3 seconds...{Style.RESET_ALL}")
                time.sleep(3)
                attempt += 1
                continue
            choice = input(f"\nPress {Fore.GREEN}Enter{Style.RESET_ALL} to retry connection, or type {Fore.RED}'q'{Style.RESET_ALL} to quit: ").strip().lower()
            if choice == 'q':
                print(f"{Fore.YELLOW}[*] Exiting program.{Style.RESET_ALL}")
                sys.exit(0)
            attempt += 1
            
    return device, serial, is_fnirs_device


def configure_sampling_rate(device):
    print_header("Sampling Rate Selection")
    rates = device.GetSupportedSamplingRates()[0]  # returns a dict mapping rate -> number of scans
    rates_list = sorted(list(rates.keys()))
    
    print(f"Supported Sampling Rates for this device:")
    for i, rate in enumerate(rates_list):
        print(f"  [{i+1}] {rate} Hz")
        
    default_rate = device.SamplingRate
    default_idx = rates_list.index(default_rate) + 1 if default_rate in rates_list else 1
    
    print(f"\nDefault rate configured: {Fore.GREEN}{default_rate} Hz{Style.RESET_ALL}")
    if NON_INTERACTIVE:
        print(f"{Fore.GREEN}[+] Keeping default sampling rate: {default_rate} Hz (non-interactive mode){Style.RESET_ALL}")
        return device.SamplingRate
        
    choice = input(f"Press Enter to keep default [{default_idx}], or type number to select: ").strip()
    
    if choice:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(rates_list):
                selected_rate = rates_list[idx]
                device.SamplingRate = selected_rate
                # Update NumberOfScans to match the device definition
                device.NumberOfScans = rates[selected_rate]
                print(f"{Fore.GREEN}[+] Configured sampling rate: {selected_rate} Hz (scans: {device.NumberOfScans}){Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}[-] Invalid selection. Keeping default: {default_rate} Hz{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}[-] Invalid input. Keeping default: {default_rate} Hz{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}[+] Keeping default: {default_rate} Hz{Style.RESET_ALL}")
        
    return device.SamplingRate

def run_calibration(device):
    print_header("Amplifier Internal Calibration")
    print("Note: The GDS service automatically loads the factory calibration parameters")
    print("      (scaling factors and offsets) stored in the headset's built-in EEPROM.")
    if NON_INTERACTIVE:
        print(f"{Fore.GREEN}[+] Skipping manual calibration (using built-in factory defaults in non-interactive mode).{Style.RESET_ALL}")
        return
        
    choice = input("\nWould you like to run a manual runtime calibration? (y/n) [Default: n]: ").strip().lower()
    
    if choice == 'y':
        print(f"{Fore.YELLOW}[*] Running calibration. Please keep headset still...{Style.RESET_ALL}")
        try:
            with gds_lock:
                calib_data = device.Calibrate()[0]
            # Convert _ffi_struct_wrap to Python dict
            calib_dict = calib_data._to_python()
            # Print average scaling factors and offsets
            factors = calib_dict.get('Factor', [])
            offsets = calib_dict.get('Offset', [])
            if len(factors) > 0:
                print(f"{Fore.GREEN}[+] Calibration complete!{Style.RESET_ALL}")
                print(f"    Average Scaling Factor: {np.mean(factors):.6f}")
                print(f"    Average Channel Offset: {np.mean(offsets):.2f} uV")
            else:
                print(f"{Fore.YELLOW}[!] Calibration returned no channel data.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}[-] Calibration failed: {e}{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}[+] Skipping manual calibration (using built-in factory defaults).{Style.RESET_ALL}")

def run_impedance_loop(device, cap_type="wet"):
    cap_label = "g.SAHARA Dry Pins" if cap_type == "dry" else "Wet Gel Electrodes"
    print_header(f"Impedance & Contact Quality Check ({cap_label})")
    
    if NON_INTERACTIVE:
        print(f"{Fore.GREEN}[+] Skipping electrode impedance check in non-interactive mode.{Style.RESET_ALL}")
        return

    choice = input("Would you like to run/rerun the electrode impedance check? (y/n) [Default: n]: ").strip().lower()
    if choice != 'y':
        print(f"{Fore.YELLOW}[*] Proceeding directly to EEG streaming.{Style.RESET_ALL}")
        return

    # Dry electrodes naturally have higher impedance (up to 100-200 kOhm is acceptable)
    good_thresh = 100.0 if cap_type == "dry" else 30.0
    accept_thresh = 200.0 if cap_type == "dry" else 100.0

    while True:
        print(f"{Fore.YELLOW}[*] Measuring electrode impedances...{Style.RESET_ALL}")
        try:
            with gds_lock:
                impedances_list = device.GetImpedanceEx()
            if not impedances_list or len(impedances_list) == 0:
                print(f"{Fore.RED}[-] Failed to receive impedance data.{Style.RESET_ALL}")
                if NON_INTERACTIVE:
                    break
                continue
                
            impedances = impedances_list[0]
            channels = device.GetChannelNames()
            if len(channels) > 0 and isinstance(channels[0], (list, tuple)):
                channels = list(channels[0])
                
            # Check Cz contact specifically
            cz_val = None
            for name, imp in zip(channels, impedances):
                if name.upper() == 'CZ':
                    cz_val = imp
                    break
            
            # Print a neat color-coded summary
            print("\n" + "-" * 60)
            print(f"  {'Channel':<12} | {'Impedance (kOhm)':<18} | {'Status':<15}")
            print("-" * 60)
            
            green_count = 0
            yellow_count = 0
            red_count = 0
            
            for name, imp in zip(channels, impedances):
                if imp < 0:
                    color = Fore.WHITE + Style.DIM
                    status = f"Error ({imp:.1f})"
                    red_count += 1
                elif imp <= good_thresh:
                    color = Fore.GREEN + Style.BRIGHT
                    status = "Good"
                    green_count += 1
                elif imp <= accept_thresh:
                    color = Fore.YELLOW
                    status = "Acceptable"
                    yellow_count += 1
                else:
                    color = Fore.RED
                    status = "High / Poor Contact"
                    red_count += 1
                print(f"  {Fore.CYAN}{name:<12}{Style.RESET_ALL} | {color}{imp:<18.2f}{Style.RESET_ALL} | {color}{status:<15}{Style.RESET_ALL}")
                
            print("-" * 60)
            print(f"Summary ({cap_label}): {Fore.GREEN}{green_count} Good{Style.RESET_ALL} | {Fore.YELLOW}{yellow_count} Acceptable{Style.RESET_ALL} | {Fore.RED}{red_count} Poor/NC{Style.RESET_ALL}")
            
            if cz_val is None or cz_val < 0 or cz_val > accept_thresh:
                print(f"\n{Fore.RED}[!] WARNING: Reference (Cz / Mastoid) is NOT making optimal contact ({cz_val if cz_val is not None else 'N/A'} kOhm).{Style.RESET_ALL}")
                if cap_type == "dry":
                    print(f"{Fore.YELLOW}    Detail (g.SAHARA): For dry setups, ensure the Mastoid Ground/Reference electrode has conductive paste/gel.{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}    If the reference contact floats or rails, ALL dry pins will saturate and flatline.{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}    Detail: Ensure Cz & GND are prepped, gelled, and touching the skin.{Style.RESET_ALL}")
                
            if NON_INTERACTIVE:
                print(f"{Fore.YELLOW}[*] Proceeding to streaming (non-interactive mode)...{Style.RESET_ALL}")
                break
                
            print("\nImpedance options:")
            print("  [1] Rerun contact check (e.g. after adjusting cap or gel)")
            print("  [2] Proceed to streaming anyway")
            print("  [3] Exit streamer")
            
            opt = input("Select an option (1-3) [Default: 1]: ").strip()
            if opt == '2':
                break
            elif opt == '3':
                print(f"{Fore.YELLOW}[*] Exiting program.{Style.RESET_ALL}")
                device.Close()
                sys.exit(0)
                
        except Exception as e:
            print(f"{Fore.RED}[-] Impedance check failed: {e}{Style.RESET_ALL}")
            break

def configure_input_source(device):
    print_header("Input Signal Source Selection")
    print("Choose what signal is sent to the amplifier channels:")
    print("  [1] Physical Electrodes (Live EEG) - default")
    print("  [2] Internal Test Generator (Simulated hardware sine waves, ideal for testing filters)")
    print("  [3] Shorted Inputs (Zeros for noise checks)")
    
    if NON_INTERACTIVE:
        device.InputSignal = pygds.GNAUTILUS_INPUT_SIGNAL_ELECTRODE
        print(f"{Fore.GREEN}[+] Configured: Physical electrodes (Live EEG mode in non-interactive mode){Style.RESET_ALL}")
        return
        
    choice = input("\nSelect input signal source (1-3) [Default: 1]: ").strip()
    if choice == '2':
        device.InputSignal = pygds.GNAUTILUS_INPUT_SIGNAL_TEST_SIGNAL
        print(f"{Fore.GREEN}[+] Configured: Simulated internal sine-wave test signals{Style.RESET_ALL}")
    elif choice == '3':
        device.InputSignal = pygds.GNAUTILUS_INPUT_SIGNAL_SHORTED
        print(f"{Fore.GREEN}[+] Configured: Shorted inputs{Style.RESET_ALL}")
    else:
        device.InputSignal = pygds.GNAUTILUS_INPUT_SIGNAL_ELECTRODE
        print(f"{Fore.GREEN}[+] Configured: Physical electrodes (Live EEG mode){Style.RESET_ALL}")

def configure_hardware_filters(device, cap_type="wet"):
    print_header("Hardware Filter Configuration")
    bp_idx = -1
    notch_idx = -1

    bp_filters = device.GetBandpassFilters()[0]
    bp_valid = [f for f in bp_filters if f['SamplingRate'] == device.SamplingRate]

    notch_filters = device.GetNotchFilters()[0]
    notch_valid = [f for f in notch_filters if f['SamplingRate'] == device.SamplingRate]

    if cap_type == "dry":
        print(f"{Fore.YELLOW}[i] g.SAHARA Dry Mode Active:{Style.RESET_ALL} Enabling hardware bandpass filter to eliminate DC drift and prevent ADC saturation (rail limits).")
        # Find recommended bandpass: preferably 0.5-30 Hz or 0.1-100 Hz or lowest cutoff >= 0.1 Hz
        recommended_bp = None
        for f in bp_valid:
            if 0.1 <= f['LowerCutoffFrequency'] <= 1.0 and 25.0 <= f['UpperCutoffFrequency'] <= 100.0:
                recommended_bp = f
                break
        if recommended_bp is None and len(bp_valid) > 0:
            recommended_bp = bp_valid[0]

        # Find 50/60 Hz notch
        recommended_notch = notch_valid[0] if len(notch_valid) > 0 else None

        if NON_INTERACTIVE:
            if recommended_bp:
                bp_idx = recommended_bp['BandpassFilterIndex']
                print(f"{Fore.GREEN}[+] Auto-configured Dry Hardware Bandpass: {recommended_bp['LowerCutoffFrequency']}-{recommended_bp['UpperCutoffFrequency']} Hz{Style.RESET_ALL}")
            if recommended_notch:
                notch_idx = recommended_notch['NotchFilterIndex']
                print(f"{Fore.GREEN}[+] Auto-configured Dry Hardware Notch: {recommended_notch['LowerCutoffFrequency']}-{recommended_notch['UpperCutoffFrequency']} Hz{Style.RESET_ALL}")
        else:
            print("\nRecommended Hardware Bandpass Filters for Dry Pins:")
            for i, f in enumerate(bp_valid):
                is_rec = " (Recommended)" if f == recommended_bp else ""
                print(f"  [{i+1}] {f['LowerCutoffFrequency']} - {f['UpperCutoffFrequency']} Hz{is_rec}")
            print(f"  [0] Disabled (Raw DC - Risk of Rail Saturation)")
            
            def_choice = bp_valid.index(recommended_bp) + 1 if recommended_bp in bp_valid else 1
            choice = input(f"Select bandpass filter [Default: {def_choice}]: ").strip()
            if not choice:
                choice = str(def_choice)
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(bp_valid):
                    bp_idx = bp_valid[idx]['BandpassFilterIndex']
                    print(f"{Fore.GREEN}[+] Bandpass configured: {bp_valid[idx]['LowerCutoffFrequency']}-{bp_valid[idx]['UpperCutoffFrequency']} Hz{Style.RESET_ALL}")
            except ValueError:
                pass

            if notch_valid:
                print("\nAvailable Hardware Notch Filters:")
                for i, f in enumerate(notch_valid):
                    print(f"  [{i+1}] {f['LowerCutoffFrequency']} - {f['UpperCutoffFrequency']} Hz")
                print("  [0] Disabled")
                n_choice = input(f"Select notch filter [Default: 1]: ").strip()
                if not n_choice:
                    n_choice = "1"
                try:
                    idx = int(n_choice) - 1
                    if 0 <= idx < len(notch_valid):
                        notch_idx = notch_valid[idx]['NotchFilterIndex']
                        print(f"{Fore.GREEN}[+] Notch configured: {notch_valid[idx]['LowerCutoffFrequency']}-{notch_valid[idx]['UpperCutoffFrequency']} Hz{Style.RESET_ALL}")
                except ValueError:
                    pass
    else:
        print("Wet Mode: By default, raw unfiltered EEG data (-1) is streamed for software filtering.")
        if NON_INTERACTIVE:
            print(f"{Fore.GREEN}[+] Skipping hardware filters in non-interactive wet mode (raw signal).{Style.RESET_ALL}")
        else:
            choice = input("Do you want to enable hardware-level notch/bandpass filters? (y/n) [Default: n]: ").strip().lower()
            if choice == 'y':
                if bp_valid:
                    print(f"\nAvailable Bandpass Filters (for {device.SamplingRate} Hz):")
                    print("  [0] Disabled (Default)")
                    for i, f in enumerate(bp_valid):
                        print(f"  [{i+1}] {f['LowerCutoffFrequency']} - {f['UpperCutoffFrequency']} Hz (Order: {f['Order']})")
                    bp_choice = input(f"Select bandpass filter (0-{len(bp_valid)}) [Default: 0]: ").strip()
                    if bp_choice and bp_choice != '0':
                        try:
                            idx = int(bp_choice) - 1
                            if 0 <= idx < len(bp_valid):
                                bp_idx = bp_valid[idx]['BandpassFilterIndex']
                                print(f"{Fore.GREEN}[+] Bandpass configured: {bp_valid[idx]['LowerCutoffFrequency']}-{bp_valid[idx]['UpperCutoffFrequency']} Hz{Style.RESET_ALL}")
                        except ValueError:
                            pass
                if notch_valid:
                    print(f"\nAvailable Notch Filters (for {device.SamplingRate} Hz):")
                    print("  [0] Disabled (Default)")
                    for i, f in enumerate(notch_valid):
                        print(f"  [{i+1}] {f['LowerCutoffFrequency']} - {f['UpperCutoffFrequency']} Hz (Order: {f['Order']})")
                    notch_choice = input(f"Select notch filter (0-{len(notch_valid)}) [Default: 0]: ").strip()
                    if notch_choice and notch_choice != '0':
                        try:
                            idx = int(notch_choice) - 1
                            if 0 <= idx < len(notch_valid):
                                notch_idx = notch_valid[idx]['NotchFilterIndex']
                                print(f"{Fore.GREEN}[+] Notch configured: {notch_valid[idx]['LowerCutoffFrequency']}-{notch_valid[idx]['UpperCutoffFrequency']} Hz{Style.RESET_ALL}")
                        except ValueError:
                            pass

    # Apply configured index to all channels
    for ch in device.Channels:
        ch.Acquire = 1
        ch.BandpassFilterIndex = bp_idx
        ch.NotchFilterIndex = notch_idx
        ch.BipolarChannel = -1

def main():
    global current_battery_level, NON_INTERACTIVE
    import argparse
    parser = argparse.ArgumentParser(description="g.Nautilus EEG & fNIRS to Lab Streaming Layer (LSL) Bridge")
    parser.add_argument("--cap-type", choices=["wet", "dry", "sahara"], default=None, help="Electrode cap type: 'wet' (gel) or 'dry'/'sahara' (pins)")
    parser.add_argument("--non-interactive", "-y", action="store_true", help="Run without interactive CLI prompts")
    parser.add_argument("--sampling-rate", "-fs", type=int, default=None, help="Force specific sampling rate (Hz)")
    args, _ = parser.parse_known_args()

    if args.non_interactive:
        NON_INTERACTIVE = True

    # 0. Select Cap Type (Wet vs Dry g.SAHARA)
    cap_type = select_cap_type(args.cap_type)

    # 1. Connect
    device, serial, is_fnirs_device = connect_headset()
    
    # Reset thread shutdown signal
    stop_event.clear()
    
    try:
        # Enable battery level channel configuration
        device.BatteryLevel = 1
        device.AccelerationData = 0
        device.ValidationIndicator = 0
        
        # 2. Configure sampling rate
        sampling_rate = configure_sampling_rate(device)
        
        # 3. Calibration
        run_calibration(device)
        
        # 4. Impedance Check Loop with Cap Type
        run_impedance_loop(device, cap_type=cap_type)
        
        # 5. Input Source Selection (Physical vs Test)
        configure_input_source(device)
        
        # 6. Hardware Filters (Auto bandpass in Dry mode to prevent DC railing)
        configure_hardware_filters(device, cap_type=cap_type)
        
        # 7. Apply settings to hardware
        print(f"\n{Fore.YELLOW}[*] Uploading configuration to headset hardware...{Style.RESET_ALL}")
        device.SetConfiguration()
        print(f"{Fore.GREEN}[+] Hardware configured successfully!{Style.RESET_ALL}")
        
        # Get final channel list
        channel_names = device.GetChannelNames()
        if len(channel_names) > 0 and isinstance(channel_names[0], (list, tuple)):
            channel_names = list(channel_names[0])
            
        # Determine battery column index
        idx_bat = device.IndexAfter('BatteryLevel')
        battery_col = -1
        if idx_bat > 0:
            battery_col = idx_bat - 1
            # Append "Battery" to the LSL channel listing
            channel_names.append("Battery")
            
        num_channels = len(channel_names)
        
        stream_type_str = 'fNIRS' if is_fnirs_device else 'EEG'
        stream_name_str = f'gNautilus_{stream_type_str}'

        print_header("Lab Streaming Layer (LSL) Output")
        print(f"Headset Name: {stream_name_str} (Serial: {serial})")
        print(f"Cap Hardware: {'g.SAHARA Dry Pins' if cap_type == 'dry' else 'Wet Gel Electrodes'}")
        print(f"Sampling Rate: {sampling_rate} Hz")
        print(f"Active Channels: {num_channels}")
        print(f"Channel Names: {channel_names}")
        
        # Create LSL StreamInfo
        info = StreamInfo(
            name=stream_name_str,
            type=stream_type_str,
            channel_count=num_channels,
            nominal_srate=sampling_rate,
            channel_format='float32',
            source_id=f'gNautilus_{serial}'
        )
        
        # Add channel metadata labels
        channels_meta = info.desc().append_child("channels")
        for chan_name in channel_names:
            chan = channels_meta.append_child("channel")
            chan.append_child_value("label", chan_name)
            chan.append_child_value("unit", "microvolts" if chan_name != "Battery" else "percent")
            chan.append_child_value("type", stream_type_str if chan_name != "Battery" else "AUX")
            
        outlet = StreamOutlet(info)
        print(f"\n{Fore.GREEN}[+] LSL Outlet created successfully. Stream name: '{stream_name_str}', type: '{stream_type_str}'{Style.RESET_ALL}")

        # Start background battery logger thread
        battery_thread = threading.Thread(target=battery_monitor_loop, args=(serial,))
        battery_thread.daemon = True
        battery_thread.start()
        
        # LSL Push Block size
        block_size = device.NumberOfScans
        
        # Streaming Callback with Rail Detection
        last_rail_warn = 0
        def daq_callback(data_block):
            nonlocal last_rail_warn
            global current_battery_level
            
            # Extract battery level from data block column in real-time
            if battery_col >= 0 and data_block.shape[1] > battery_col:
                current_battery_level = float(np.mean(data_block[:, battery_col]))
                
            stamp = local_clock()
            outlet.push_chunk(data_block.tolist(), stamp)
            
            # Live Rail Detection (Check if any channel hits ceiling / floor or flatlines)
            now = time.time()
            if now - last_rail_warn > 4.0:
                eeg_block = data_block[:, :min(32, data_block.shape[1])]
                max_abs = np.max(np.abs(eeg_block), axis=0)
                stds = np.std(eeg_block, axis=0)
                # Saturated: absolute voltage > 150 mV (150,000 uV) or dead flat (std == 0)
                railed_indices = [i for i, (m, s) in enumerate(zip(max_abs, stds)) if m > 150000.0 or (s < 0.01 and m > 1000.0)]
                if railed_indices:
                    bad_chans = [channel_names[i] for i in railed_indices if i < len(channel_names)]
                    print(f"[{Fore.RED}RAIL WARNING{Style.RESET_ALL}] Electrodes saturating rails: {', '.join(bad_chans)}! Check contact/ground at mastoid.")
                    last_rail_warn = now
                elif now % 3 < 0.05:
                    bat_str = f" | Battery: {current_battery_level:.1f}%" if current_battery_level is not None else ""
                    print(f"[{Fore.GREEN}LSL STREAMING{Style.RESET_ALL}] Pushed chunk size: {len(data_block)} | LSL clock: {stamp:.3f}{bat_str}")
            return True
            
        print(f"\n{Fore.GREEN}[+] LSL EEG stream is broadcasting!{Style.RESET_ALL}")
        print(f"    You can now open a visualizer in another window (e.g. 'uv run python lsl_viewer.py').")
        print(f"    * Press {Fore.RED}Ctrl+C{Style.RESET_ALL} in this window to stop streaming *")
        print("-" * 80)
        
        # Run loop
        device.GetData(block_size, daq_callback)
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}[*] Streaming interrupted by user.{Style.RESET_ALL}")
    except Exception as e:
        err_msg = str(e)
        print(f"\n{Fore.RED}[-] Error in streaming loop: {err_msg}{Style.RESET_ALL}")
        if "not creator" in err_msg or "acquisition session" in err_msg:
            print(f"\n{Fore.YELLOW}[!] DIAGNOSIS: Another g.tec software (such as 'GtecSuite.exe' or 'gRecorder') is open and holding the device acquisition lock.{Style.RESET_ALL}")
            print(f"{Fore.GREEN}[+] SOLUTION: Please close GtecSuite / gRecorder (or stop acquisition in g.tec Suite), then run 'uv run python gds_to_lsl.py' again.{Style.RESET_ALL}\n")

    finally:
        print(f"\n{Fore.YELLOW}[*] Releasing GDS connection and cleaning up...{Style.RESET_ALL}")
        # Stop background thread
        stop_event.set()
        try:
            # Revert to standard electrode settings on close
            device.InputSignal = pygds.GNAUTILUS_INPUT_SIGNAL_ELECTRODE
            device.BatteryLevel = 0
            device.SetConfiguration()
            device.Close()
            del device
            print(f"{Fore.GREEN}[+] Disconnected successfully.{Style.RESET_ALL}")
        except:
            pass

if __name__ == '__main__':
    main()
