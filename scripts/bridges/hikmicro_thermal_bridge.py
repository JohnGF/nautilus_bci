"""
HIKMICRO Thermal (RoI) to Lab Streaming Layer (LSL) Bridge
==========================================================

Launches the `hikmicro-lsl` thermal streamer as a subprocess and (optionally)
records the small RoI vector to CSV while it streams. The heavy 49152-channel
grid streams stay off; the stream published for BCI use is
`HIKMICRO_Temperature_RoI` (type `TEMPERATURE_ROI`, 13 channels).

Requirements:
    uv pip install -e /home/john/git/bci_projects/hikmicro-lsl
    (the CLI entry point `hikmicro-lsl` must be on PATH, or set --python)

Streams:
    HIKMICRO_Temperature_RoI  TEMPERATURE_ROI  cf_float32  13 ch
        min_raw, max_raw, center_raw, min_c, max_c, center_c, mean_c,
        fpa_c, core_c, shutter_c, emissivity, distance_m, humidity_pct
    HIKMICRO_Meta             META             cf_string   1 ch

Usage:
    uv run python hikmicro_thermal_bridge.py
    uv run python hikmicro_thermal_bridge.py --device /dev/video0 --duration 60
    uv run python hikmicro_thermal_bridge.py --record
    uv run python hikmicro_thermal_bridge.py --discover-only
"""
import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)


import argparse
import csv
import datetime
import json
import subprocess
import time

ROI_STREAM_NAME = "HIKMICRO_Temperature_RoI"
ROI_STREAM_TYPE = "TEMPERATURE_ROI"
FULL_STREAM_TYPE = "TEMPERATURE"

ROI_LABELS = [
    "min_raw", "max_raw", "center_raw",
    "min_c", "max_c", "center_c", "mean_c",
    "fpa_c", "core_c", "shutter_c",
    "emissivity", "distance_m", "humidity_pct",
]

HIKMICRO_SRC = "/home/john/git/bci_projects/hikmicro-lsl"


def check_hikmicro_installed():
    """Return the hikmicro_lsl package if importable, else print a hint."""
    try:
        import hikmicro_lsl  # noqa: F401
    except ImportError:
        print("[-] 'hikmicro_lsl' is not importable in this environment.")
        print(f"    Install it with: uv pip install -e {HIKMICRO_SRC}")
        return None
    return hikmicro_lsl


def discover_streams(timeout=2.0):
    """Return (name, type, channels, srate) for every active LSL stream."""
    from pylsl import resolve_streams

    return [
        (s.name(), s.type(), s.channel_count(), s.nominal_srate())
        for s in resolve_streams(wait_time=timeout)
    ]


def print_discovered(streams):
    if not streams:
        print("[!] No LSL streams visible yet (the subprocess may still be starting).")
        return
    print("\n[+] Discovered LSL streams:")
    for name, stype, channels, srate in streams:
        print(f"    - '{name}' | type '{stype}' | {channels} ch | {srate:.1f} Hz")
    thermal = [s for s in streams if s[1] in (ROI_STREAM_TYPE, FULL_STREAM_TYPE)]
    if thermal:
        print(f"[+] Thermal streams present: {[s[0] for s in thermal]}")
    else:
        print(f"[!] No '{ROI_STREAM_TYPE}' / '{FULL_STREAM_TYPE}' stream yet.")


def launch_streamer(args):
    """Start `hikmicro-lsl stream --roi-only ...` as a subprocess."""
    cmd = [
        args.python, "-m", "hikmicro_lsl.cli", "stream",
        "--device", args.device,
        "--range", args.range,
        "--roi-only",
    ]
    if args.duration is not None:
        cmd += ["--duration", str(args.duration)]
    print(f"[*] Launching: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def resolve_roi_inlet(timeout):
    from pylsl import StreamInlet, resolve_byprop

    streams = resolve_byprop("type", ROI_STREAM_TYPE, timeout=timeout)
    if not streams:
        return None
    print(f"[+] Connected to '{streams[0].name()}' ({streams[0].channel_count()} ch)")
    return StreamInlet(streams[0], max_chunklen=32)


def record_roi(inlet, out_dir, duration=None, script_file=__file__):
    """Pull RoI samples into a CSV and write RUN_INFO.txt provenance beside it."""
    from utils.provenance import write_provenance

    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(out_dir, f"{stamp}_roi.csv")

    print(f"[*] Recording RoI vector to {csv_path} (Ctrl+C to stop)...")
    count = 0
    start = time.time()
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp_sec", *ROI_LABELS])
        try:
            while duration is None or (time.time() - start) < duration:
                samples, timestamps = inlet.pull_chunk(timeout=0.1)
                for sample, ts in zip(samples, timestamps):
                    writer.writerow([ts, *sample])
                    count += 1
        except KeyboardInterrupt:
            print("\n[*] Manual stop requested.")

    write_provenance(
        out_dir,
        script_file=script_file,
        params={
            "stream": ROI_STREAM_NAME,
            "stream_type": ROI_STREAM_TYPE,
            "csv": os.path.basename(csv_path),
            "samples": count,
            "labels": ROI_LABELS,
        },
    )
    print(f"[+] Wrote {count} RoI samples to {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="HIKMICRO thermal RoI stream bridge for the Nautilus BCI stack",
    )
    parser.add_argument('--device', default='/dev/video0', help="V4L2 capture node")
    parser.add_argument('--range', choices=['normal', 'high'], default='normal',
                        help="Sensor temperature range")
    parser.add_argument('--duration', type=float, default=None, metavar='S',
                        help="Stop after S seconds (default: run until Ctrl+C)")
    parser.add_argument('--record', action='store_true',
                        help="Also write the RoI vector to results/hikmicro_thermal/<ts>_roi.csv")
    parser.add_argument('--discover-only', action='store_true',
                        help="List thermal LSL streams and exit (no subprocess)")
    parser.add_argument('--python', default=sys.executable,
                        help="Interpreter used to launch the streamer subprocess")
    args = parser.parse_args()

    print("=" * 70)
    print(" HIKMICRO Thermal RoI -> LSL Bridge ".center(70, "="))
    print("=" * 70)

    if args.discover_only:
        try:
            print_discovered(discover_streams())
        except ImportError:
            print("[-] 'pylsl' is not installed in this environment.")
            sys.exit(1)
        return

    if check_hikmicro_installed() is None:
        sys.exit(1)

    proc = launch_streamer(args)
    time.sleep(2.0)
    if proc.poll() is not None:
        print(f"[-] Streamer exited early (code {proc.returncode}).")
        print("    Check that the thermal camera is connected and readable.")
        sys.exit(proc.returncode or 1)

    try:
        print_discovered(discover_streams())

        if args.record:
            inlet = resolve_roi_inlet(timeout=5.0)
            if inlet is None:
                print(f"[-] No '{ROI_STREAM_TYPE}' stream to record.")
                sys.exit(1)
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            out_dir = os.path.join(repo_root, "results", "hikmicro_thermal")
            record_roi(inlet, out_dir, duration=args.duration)
        else:
            while proc.poll() is None:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[*] Bridge stopped.")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("[*] Streamer terminated.")


if __name__ == '__main__':
    main()
