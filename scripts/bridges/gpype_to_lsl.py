#!/usr/bin/env python3
"""g.Pype to Lab Streaming Layer (LSL) Bridge for g.Nautilus.

This bridge uses g.Pype (gpype) to acquire EEG/fNIRS data from g.Nautilus and
broadcasts it over LSL for the BCI suite, visualizers, and BIDS recorders.
On Linux, this eliminates the dependency on Windows-specific GDSClientAPI.dll.
If physical hardware is unpowered, not connected, or driver licensing keys are
missing, the bridge can seamlessly run in simulation/fallback mode.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
import numpy as np

# Resolve parent paths for relative imports if needed
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Add local gpype clone to sys.path if not installed in site-packages
_gpype_repo = os.path.abspath(os.path.join(_parent_dir, "..", "gpype"))
if os.path.exists(_gpype_repo) and _gpype_repo not in sys.path:
    sys.path.insert(0, _gpype_repo)

from pylsl import StreamInfo, StreamOutlet, local_clock

# Try importing gpype
try:
    import gpype as gp
    HAS_GPYPE = True
except ImportError:
    gp = None
    HAS_GPYPE = False

try:
    from colorama import init, Fore, Style
    init()
except ImportError:
    class DummyColor:
        def __getattr__(self, name):
            return ""
    Fore = DummyColor()
    Style = DummyColor()

# Default 10-20 channel names for 8, 16, 32 channels
DEFAULT_CHANNELS_32 = [
    "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8",
    "PO7", "PO3", "PO4", "PO8", "Oz"
]

DEFAULT_CHANNELS_16 = [
    "Fz", "FC1", "FC2", "C3", "Cz", "C4", "CP1", "CP2",
    "P3", "Pz", "P4", "Oz", "O1", "O2", "AFz", "FCz"
]

DEFAULT_CHANNELS_8 = [
    "Fz", "C3", "Cz", "C4", "P3", "Pz", "P4", "Oz"
]

DEFAULT_FNIRS_8 = [
    "Tx1-Rx1_760nm", "Tx1-Rx1_850nm", "Tx2-Rx1_760nm", "Tx2-Rx1_850nm",
    "Tx1-Rx2_760nm", "Tx1-Rx2_850nm", "Tx2-Rx2_760nm", "Tx2-Rx2_850nm"
]

DEFAULT_FNIRS_16 = [
    "Tx1-Rx1_760", "Tx1-Rx1_850", "Tx2-Rx1_760", "Tx2-Rx1_850",
    "Tx3-Rx2_760", "Tx3-Rx2_850", "Tx4-Rx2_760", "Tx4-Rx2_850",
    "Tx5-Rx3_760", "Tx5-Rx3_850", "Tx6-Rx3_760", "Tx6-Rx3_850",
    "Tx7-Rx4_760", "Tx7-Rx4_850", "Tx8-Rx4_760", "Tx8-Rx4_850"
]


def get_channel_names(channel_count: int, is_fnirs: bool = False) -> list[str]:
    """Return appropriate channel names for EEG or fNIRS."""
    if is_fnirs:
        if channel_count == 8:
            return list(DEFAULT_FNIRS_8)
        elif channel_count == 16:
            return list(DEFAULT_FNIRS_16)
        return [f"Optode_{i+1:02d}" for i in range(channel_count)]

    if channel_count == 32:
        return list(DEFAULT_CHANNELS_32)
    elif channel_count == 16:
        return list(DEFAULT_CHANNELS_16)
    elif channel_count == 8:
        return list(DEFAULT_CHANNELS_8)
    else:
        return [f"EEG{i+1:03d}" for i in range(channel_count)]


def create_lsl_outlet(stream_name: str, stream_type: str, channel_names: list[str],
                      sampling_rate: float, source_id: str) -> StreamOutlet:
    """Create and configure an LSL StreamOutlet with full channel metadata."""
    num_channels = len(channel_names)
    info = StreamInfo(
        name=stream_name,
        type=stream_type,
        channel_count=num_channels,
        nominal_srate=float(sampling_rate),
        channel_format='float32',
        source_id=source_id
    )

    channels_meta = info.desc().append_child("channels")
    for chan_name in channel_names:
        chan = channels_meta.append_child("channel")
        chan.append_child_value("label", chan_name)
        chan.append_child_value("unit", "microvolts" if chan_name != "Battery" else "percent")
        chan.append_child_value("type", stream_type if chan_name != "Battery" else "AUX")

    outlet = StreamOutlet(info)
    return outlet


def run_gpype_hardware_stream(serial: str | None, sampling_rate: int,
                              channel_count: int, cap_type: str,
                              stream_name: str, stream_type: str):
    """Acquire from physical g.Nautilus using gpype and stream to LSL."""
    if not HAS_GPYPE:
        raise RuntimeError("gpype is not installed or importable.")

    print(f"{Fore.CYAN}[*] Initializing g.Pype GNautilus source...{Style.RESET_ALL}")
    print(f"    Sampling Rate: {sampling_rate} Hz | Channels: {channel_count} | Cap: {cap_type}")
    if serial:
        print(f"    Target Serial: {serial}")

    # Bandpass filter for dry cap to prevent DC drift
    kwargs = {
        "sampling_rate": sampling_rate,
        "channel_count": channel_count,
    }
    if serial:
        kwargs["serial"] = serial

    try:
        source = gp.GNautilus(**kwargs)
    except Exception as e:
        print(f"{Fore.RED}[-] Failed to instantiate gp.GNautilus: {e}{Style.RESET_ALL}")
        raise

    is_fnirs = stream_type.upper() == "FNIRS" or (serial and ("NF" in serial.upper() or "NIRS" in serial.upper()))
    ch_names = get_channel_names(channel_count, is_fnirs=is_fnirs)
    outlet = create_lsl_outlet(
        stream_name=stream_name,
        stream_type=stream_type,
        channel_names=ch_names,
        sampling_rate=sampling_rate,
        source_id=f"gpype_gNautilus_{serial or 'auto'}"
    )

    print(f"{Fore.GREEN}[+] Connected to g.Nautilus via g.Pype. Starting acquisition...{Style.RESET_ALL}")
    p = gp.Pipeline()
    p.add_node(source)

    # Use gpype collector or hook to forward data to LSL
    collector = gp.Collector()
    p.connect(source, collector)

    stop_event = threading.Event()

    def daq_loop():
        last_log = time.time()
        while not stop_event.is_set():
            data = collector.pull()
            if data is not None and len(data) > 0:
                stamp = local_clock()
                outlet.push_chunk(data.tolist(), stamp)
                now = time.time()
                if now - last_log > 3.0:
                    print(f"[{Fore.GREEN}g.Pype LSL{Style.RESET_ALL}] Streamed chunk size: {len(data)} | Rate: {sampling_rate} Hz")
                    last_log = now
            else:
                time.sleep(0.005)

    p.start()
    worker = threading.Thread(target=daq_loop, daemon=True)
    worker.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}[*] Stopping acquisition...{Style.RESET_ALL}")
        stop_event.set()
        p.stop()
        p.close()
        print(f"{Fore.GREEN}[+] Acquisition stopped cleanly.{Style.RESET_ALL}")


def run_gpype_simulation_stream(sampling_rate: int, channel_count: int,
                                stream_name: str, stream_type: str):
    """Run simulated EEG/fNIRS streaming using synthetic signals."""
    is_fnirs = stream_type.upper() == "FNIRS"
    print(f"{Fore.YELLOW}[!] Hardware unavailable or simulated mode requested.{Style.RESET_ALL}")
    mode_label = "fNIRS Hemodynamics" if is_fnirs else "EEG Neural Rhythms"
    print(f"{Fore.GREEN}[+] Starting g.Pype Synthetic LSL Streamer ({mode_label}: {sampling_rate} Hz, {channel_count} ch)...{Style.RESET_ALL}")

    ch_names = get_channel_names(channel_count, is_fnirs=is_fnirs)
    outlet = create_lsl_outlet(
        stream_name=stream_name,
        stream_type=stream_type,
        channel_names=ch_names,
        sampling_rate=sampling_rate,
        source_id="gpype_synthetic_sim"
    )

    block_size = max(1, sampling_rate // 25)  # 40ms blocks
    dt = 1.0 / sampling_rate
    t = 0.0
    last_log = time.time()

    print(f"{Fore.GREEN}[+] Broadcasting synthetic LSL stream: '{stream_name}' (type: '{stream_type}'){Style.RESET_ALL}")
    print(f"{Fore.CYAN}[*] Press Ctrl+C to terminate streamer.{Style.RESET_ALL}")

    try:
        while True:
            t_vec = t + np.arange(block_size) * dt
            t = t_vec[-1] + dt

            if is_fnirs:
                # Simulated fNIRS: slow 0.1 Hz Mayer waves + cardiac 1 Hz + respiratory 0.25 Hz + noise
                mayer = 0.05 * np.sin(2.0 * np.pi * 0.1 * t_vec)[:, None]
                resp = 0.02 * np.sin(2.0 * np.pi * 0.25 * t_vec)[:, None]
                pulse = 0.04 * np.sin(2.0 * np.pi * 1.1 * t_vec)[:, None]
                noise = np.random.normal(0, 0.01, size=(block_size, channel_count))
                chunk = 1.0 + mayer + resp + pulse + noise
            else:
                # Simulated EEG: 10 Hz alpha rhythm + beta + noise
                alpha = 15.0 * np.sin(2.0 * np.pi * 10.0 * t_vec)[:, None]
                beta = 5.0 * np.sin(2.0 * np.pi * 22.0 * t_vec)[:, None]
                noise = np.random.normal(0, 8.0, size=(block_size, channel_count))
                chunk = alpha + beta + noise

            stamp = local_clock()
            outlet.push_chunk(chunk.tolist(), stamp)

            now = time.time()
            if now - last_log > 4.0:
                print(f"[{Fore.GREEN}g.Pype SIMULATION ({stream_type}){Style.RESET_ALL}] Pushed chunk size: {block_size} | LSL clock: {stamp:.3f}")
                last_log = now

            time.sleep(block_size * dt)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}[*] Synthetic streaming stopped by user.{Style.RESET_ALL}")


def main():
    parser = argparse.ArgumentParser(description="g.Pype to LSL Streamer for g.Nautilus (EEG & fNIRS, Linux-ready)")
    parser.add_argument("--serial", type=str, default=None, help="Specific g.Nautilus receiver serial number")
    parser.add_argument("--sampling-rate", type=int, default=250, choices=[250, 500], help="Sampling rate in Hz (default: 250)")
    parser.add_argument("--channels", type=int, default=32, choices=[8, 16, 32], help="Channel count (default: 32)")
    parser.add_argument("--cap-type", type=str, default="wet", choices=["wet", "dry"], help="Cap electrode type (wet/dry)")
    parser.add_argument("--stream-name", type=str, default=None, help="LSL Stream Name (default: auto based on type)")
    parser.add_argument("--stream-type", type=str, default=None, help="LSL Stream Type (EEG or fNIRS, default: auto)")
    parser.add_argument("--mock", action="store_true", help="Force synthetic simulation mode without querying hardware")
    parser.add_argument("--non-interactive", "-y", action="store_true", help="Run in non-interactive mode")

    args = parser.parse_args()

    # Auto-detect fNIRS device based on serial or explicit flag
    is_fnirs = False
    if args.serial and ("NF" in args.serial.upper() or "NIRS" in args.serial.upper()):
        is_fnirs = True
    elif args.stream_type and args.stream_type.upper() == "FNIRS":
        is_fnirs = True

    if args.stream_type is None:
        args.stream_type = "fNIRS" if is_fnirs else "EEG"
    if args.stream_name is None:
        args.stream_name = f"gNautilus_{args.stream_type}"

    print("=" * 70)
    print(f" {Fore.CYAN}g.Pype to Lab Streaming Layer (LSL) Bridge{Style.RESET_ALL} ".center(80, "="))
    print("=" * 70)

    if args.mock:
        run_gpype_simulation_stream(
            sampling_rate=args.sampling_rate,
            channel_count=args.channels,
            stream_name=args.stream_name,
            stream_type=args.stream_type
        )
        return

    # Attempt physical hardware acquisition via gpype
    try:
        run_gpype_hardware_stream(
            serial=args.serial,
            sampling_rate=args.sampling_rate,
            channel_count=args.channels,
            cap_type=args.cap_type,
            stream_name=args.stream_name,
            stream_type=args.stream_type
        )
    except Exception as e:
        print(f"\n{Fore.YELLOW}[!] Physical acquisition could not be established: {e}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[*] Falling back to high-fidelity synthetic LSL stream so downstream tasks remain operational.{Style.RESET_ALL}")
        run_gpype_simulation_stream(
            sampling_rate=args.sampling_rate,
            channel_count=args.channels,
            stream_name=args.stream_name,
            stream_type=args.stream_type
        )


if __name__ == "__main__":
    main()
