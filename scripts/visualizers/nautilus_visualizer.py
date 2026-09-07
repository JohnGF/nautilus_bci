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
    print("\nError loading g.tec BCI drivers or pygds.")
    print("Please make sure g.tec Suite is fully installed and the g.NEEDaccess service is running.")
    print(f"Error details: {e}")
    sys.exit(1)

# Color constants for colored console output
try:
    from colorama import init, Fore, Style
    init()
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    RED = Fore.RED
    CYAN = Fore.CYAN
    RESET = Style.RESET_ALL
except ImportError:
    GREEN = ""
    YELLOW = ""
    RED = ""
    CYAN = ""
    RESET = ""

def print_header(title):
    print("\n" + "=" * 60)
    print(f" {CYAN}{title}{RESET} ".center(70, "="))
    print("=" * 60)

def scan_devices():
    print(f"\n{CYAN}[+] Scanning for connected g.tec BCI devices...{RESET}")
    try:
        cd = pygds.ConnectedDevices()
        if len(cd) == 0:
            print(f"{YELLOW}[!] No connected devices found. Please ensure the USB receiver is plugged in.{RESET}")
            return None
        
        print(f"\n{GREEN}[+] Detected {len(cd)} device(s):{RESET}")
        for i, (serial, dev_type, in_use) in enumerate(cd):
            type_str = "g.Nautilus" if dev_type == 3 else f"Type {dev_type}"
            use_str = f"{RED}In Use{RESET}" if in_use else f"{GREEN}Available{RESET}"
            print(f"  [{i}] Serial: {serial} | Type: {type_str} | Status: {use_str}")
        return cd
    except Exception as e:
        print(f"{RED}[-] Error scanning devices: {e}{RESET}")
        return None

def connect_to_headset(serial):
    print_header(f"Connecting to device: {serial}")
    print(f"{CYAN}[*] Attempting wireless connection to the headset...{RESET}")
    print(f"    Make sure the headset power button is turned ON and pairing LED is active.")
    
    device = None
    retry_count = 0
    while True:
        try:
            device = pygds.GDS(serial, open_exclusively=False)
            print(f"\n{GREEN}[+] Connected to headset successfully!{RESET}")
            return device
        except Exception as e:
            retry_count += 1
            print(f"\n{RED}[-] Connection failed: {e}{RESET}")
            print(f"\n{YELLOW}[!] The headset itself is likely powered OFF or out of range.{RESET}")
            choice = input(f"Would you like to turn it ON and try again? (y/n): ").strip().lower()
            if choice != 'y':
                print(f"{YELLOW}[!] Aborting connection.{RESET}")
                return None
            print(f"{CYAN}[*] Retrying connection (attempt {retry_count + 1})...{RESET}")

def run_impedance_check(device):
    print_header("Signal Quality & Impedance Check")
    print(f"{CYAN}[*] Initializing impedance measurement...{RESET}")
    print(f"    {YELLOW}IMPORTANT:{RESET} For g.Nautilus, Cz must be connected to GND for correct readings.")
    print("    Hold still while measuring...")
    
    try:
        # Measure impedances
        print(f"{CYAN}[*] Measuring electrode contact impedance...{RESET}")
        impedances_list = device.GetImpedanceEx()
        if not impedances_list or len(impedances_list) == 0:
            print(f"{RED}[-] Failed to receive impedance data.{RESET}")
            return
            
        impedances = impedances_list[0]
        channels = device.GetChannelNames()
        
        print("\n" + "-" * 50)
        print(f"  {'Channel':<15} | {'Impedance (kOhm)':<18} | {'Status':<12}")
        print("-" * 50)
        
        green_count = 0
        yellow_count = 0
        red_count = 0
        
        for name, imp in zip(channels, impedances):
            # Format and color code the impedance values
            if imp < 0:
                imp_str = "Error/NC"
                status = f"{RED}No Contact{RESET}"
                red_count += 1
            elif imp <= 30.0:
                imp_str = f"{imp:.2f}"
                status = f"{GREEN}Good{RESET}"
                green_count += 1
            elif imp <= 100.0:
                imp_str = f"{imp:.2f}"
                status = f"{YELLOW}Acceptable{RESET}"
                yellow_count += 1
            else:
                imp_str = f"{imp:.2f}"
                status = f"{RED}Poor{RESET}"
                red_count += 1
                
            print(f"  {CYAN}{name:<15}{RESET} | {imp_str:<18} | {status}")
            
        print("-" * 50)
        total = len(channels)
        print(f"Summary: {GREEN}{green_count} Good{RESET} | {YELLOW}{yellow_count} Acceptable{RESET} | {RED}{red_count} Poor{RESET} out of {total} channels.")
        
        if red_count > 0:
            print(f"\n{YELLOW}[!] Hint: Adjust electrodes with poor contact (Red) by applying gel or adjusting hair contact.{RESET}")
        else:
            print(f"\n{GREEN}[+] All electrodes are fully connected and measuring! Ready to record.{RESET}")
            
    except Exception as e:
        print(f"{RED}[-] Error measuring impedance: {e}{RESET}")
        print("Please check that Cz is connected to GND and the headset is turned on and placed correctly.")

def run_eeg_scope(device, test_signal_mode=False):
    title_suffix = " (Internal Test Sine-Wave Mode)" if test_signal_mode else " (Live EEG Mode)"
    print_header("Live EEG Visualization" + title_suffix)
    print(f"{CYAN}[*] Starting real-time plotting window...{RESET}")
    print("    Acquisition will run in the background.")
    print("    * To stop recording, close the Matplotlib plot window *")
    
    try:
        # Default sampling rate for g.Nautilus is 250 Hz
        device.SamplingRate = 250
        
        # Enable all channels for acquisition
        for ch in device.Channels:
            ch.Acquire = 1
            ch.BandpassFilterIndex = -1  # Disable hardware filters for raw data, or set default
            ch.NotchFilterIndex = -1     # Disable notch filter for raw data
            ch.BipolarChannel = -1       # Refer to GND
            
        if test_signal_mode:
            # Enable internal test generator
            device.InputSignal = pygds.GNAUTILUS_INPUT_SIGNAL_TEST_SIGNAL
        else:
            # Set to acquire from physical electrodes
            device.InputSignal = pygds.GNAUTILUS_INPUT_SIGNAL_ELECTRODE
            
        device.SetConfiguration()
        
        sampling_rate = device.SamplingRate
        channels = device.GetChannelNames()
        num_channels = len(channels)
        
        # Setup PyGDS Matplotlib Scope
        scope = pygds.Scope(1 / sampling_rate, ylabel='U/uV', title=f"g.Nautilus Live Signal {title_suffix}")
        
        # Get data block sizes
        block_size = 8  # Default block size of scans for 250Hz Nautilus
        
        # Define callback function to stream data to scope
        def DAQCallback(dataBlock):
            return scope(dataBlock)
            
        # Start the data acquisition loop
        print(f"\n{GREEN}[+] Running live scope window... Close the window to stop.{RESET}")
        device.GetData(block_size, DAQCallback)
        print(f"\n{YELLOW}[!] Plot window closed. Stopping acquisition...{RESET}")
        
    except Exception as e:
        print(f"{RED}[-] Error during live acquisition: {e}{RESET}")
    finally:
        # Reset back to default electrode input signal
        try:
            device.InputSignal = pygds.GNAUTILUS_INPUT_SIGNAL_ELECTRODE
            device.SetConfiguration()
        except:
            pass

def main():
    print_header("g.Nautilus BCI Python Visualizer & Diagnostics")
    print("This utility interfaces with your g.tec BCI suite and connected headsets.")
    
    devices = scan_devices()
    if not devices:
        input("\nPress Enter to exit...")
        sys.exit(0)
        
    # Use detected device serial, default to the first one found
    serial = devices[0][0]
    
    device = connect_to_headset(serial)
    if not device:
        input("\nPress Enter to exit...")
        sys.exit(0)
        
    while True:
        print_header("Diagnostic Menu")
        print(f"Connected Device: {CYAN}{serial}{RESET}")
        print("  [1] Live EEG Visualization Scope (Real-time plots)")
        print("  [2] Signal Quality & Impedance Check (Green Light Check)")
        print("  [3] Run Test Signal Check (Sine-wave hardware self-test)")
        print("  [4] Disconnect & Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            run_eeg_scope(device, test_signal_mode=False)
        elif choice == '2':
            run_impedance_check(device)
        elif choice == '3':
            run_eeg_scope(device, test_signal_mode=True)
        elif choice == '4' or not choice:
            print(f"\n{CYAN}[*] Closing device connection and cleaning up...{RESET}")
            device.Close()
            del device
            print(f"{GREEN}[+] Disconnected successfully.{RESET}")
            break
        else:
            print(f"{RED}[-] Invalid option. Please try again.{RESET}")
            
if __name__ == "__main__":
    main()
