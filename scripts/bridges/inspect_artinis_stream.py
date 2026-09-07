"""
Live Terminal fNIRS Stream Monitor
Connects to the active LSL stream (Artinis_fNIRS or any fNIRS stream) and prints
a colorized, real-time terminal dashboard with numerical values and ASCII bar meters.
"""

import sys
import time
import os
import numpy as np
from pylsl import resolve_streams, StreamInlet

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def make_bar(val, max_val=5.0, width=14):
    """Generate a dual-sided ASCII bar for positive/negative hemodynamic changes."""
    norm = np.clip(val / max_val, -1.0, 1.0)
    mid = width // 2
    chars = [" "] * width
    chars[mid] = "|"
    
    if norm > 0.005:
        pos = max(1, int(norm * (mid - 1)))
        for k in range(mid + 1, min(width, mid + 1 + pos)):
            chars[k] = "="
        col = RED
    elif norm < -0.005:
        neg = max(1, int(abs(norm) * mid))
        for k in range(max(0, mid - neg), mid):
            chars[k] = "="
        col = BLUE
    else:
        col = DIM
        
    return f"{col}[{''.join(chars)}]{RESET}"

def main():
    # Enable ANSI escape codes on Windows
    os.system("")
    
    print(f"{BOLD}{CYAN}=== Searching for fNIRS LSL Stream... ==={RESET}", flush=True)
    
    streams = resolve_streams(wait_time=3.0)
    target = None
    for s in streams:
        if any(k in s.name().upper() for k in ['ARTINIS', 'FNIRS', 'OCTAMON', 'OPTOMON']) or 'FNIRS' in s.type().upper():
            target = s
            break
            
    if not target:
        if streams:
            print(f"No fNIRS stream found, but found {len(streams)} other stream(s):", flush=True)
            for s in streams:
                print(f"  • {s.name()} ({s.type()}) @ {s.nominal_srate():.0f} Hz", flush=True)
            target = streams[0]
            print(f"\n{YELLOW}Connecting to: {target.name()}{RESET}", flush=True)
        else:
            print(f"{RED}[!] No LSL streams found on the network.{RESET}")
            print("    Please ensure 'uv run python bridges/artinis_to_lsl.py' is running.")
            return

    inlet = StreamInlet(target)
    srate = inlet.info().nominal_srate() or 50.0
    n_ch = inlet.info().channel_count()
    
    print(f"{GREEN}[+] Connected to '{target.name()}' ({n_ch} channels @ {srate:.0f} Hz){RESET}")
    print(f"{DIM}Streaming live data... (Press Ctrl+C to exit){RESET}\n")
    time.sleep(0.3)

    sample_counter = 0
    t0 = time.time()
    last_print = 0

    try:
        while True:
            samples, timestamps = inlet.pull_chunk(timeout=0.05, max_samples=32)
            if samples:
                sample_counter += len(samples)
                latest = np.array(samples[-1])
                now = time.time()
                
                # Refresh terminal table at ~10 FPS
                if now - last_print >= 0.15:
                    last_print = now
                    fps = sample_counter / (now - t0) if now > t0 else 0
                    
                    # Clear screen cleanly
                    os.system("cls")
                    
                    print(f"{BOLD}{CYAN}🧠 Artinis fNIRS Live Stream Monitor{RESET}  |  Stream: {GREEN}{target.name()}{RESET}  |  Rate: {BOLD}{fps:.1f} smp/s{RESET}")
                    print("=" * 86)
                    print(f"{BOLD}{'Channel / Pod':<22} {'Ch 1..8 (HbO/760nm)':<20} {'Meter':<18} {'Ch 9..16 (HbR/850nm)':<20} {'Meter'}{RESET}")
                    print("-" * 86)
                    
                    for i in range(min(8, len(latest))):
                        val1 = float(latest[i])
                        val2 = float(latest[8 + i]) if (8 + i) < len(latest) else 0.0
                        
                        bar1 = make_bar(val1, max_val=2.0, width=14)
                        bar2 = make_bar(val2, max_val=2.0, width=14)
                        
                        pod_side = "Left R1" if i < 4 else "Right R2"
                        label = f"Tx{i+1} ({pod_side})"
                        
                        str1 = f"{RED}{val1:+8.4f}{RESET}"
                        str2 = f"{BLUE}{val2:+8.4f}{RESET}"
                        
                        print(f"  {label:<20} {str1:<28} {bar1:<26} {str2:<28} {bar2}")
                        
                    print("-" * 86)
                    print(f"{DIM}Tip: Place your finger across an active red emitter LED and the black receiver sensor.{RESET}")
                    sys.stdout.flush()
                    
            time.sleep(0.01)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}[!] Monitor closed.{RESET}")

if __name__ == '__main__':
    main()
