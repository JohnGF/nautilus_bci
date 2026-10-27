#!/usr/bin/env python3
"""g.Pype Data Extractor for g.Nautilus & LSL Streams.

Extracts EEG/fNIRS data from either physical g.Nautilus (via gpype), an active
LSL stream, or synthetic sources, and records directly to standard neurophysiology
formats (EDF, CSV, HDF5, MAT) or BIDS.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
import numpy as np

# Ensure root scripts directory is in sys.path
_scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

_gpype_repo = os.path.abspath(os.path.join(_scripts_dir, "..", "gpype"))
if os.path.exists(_gpype_repo) and _gpype_repo not in sys.path:
    sys.path.insert(0, _gpype_repo)

try:
    import gpype as gp
    HAS_GPYPE = True
except ImportError:
    gp = None
    HAS_GPYPE = False

from pylsl import StreamInlet, resolve_byprop


def extract_from_lsl(stream_name: str, duration_sec: float, output_path: str, format_type: str):
    """Extract EEG data from an active LSL stream and save to requested file format."""
    print(f"[*] Resolving LSL stream with name '{stream_name}'...")
    streams = resolve_byprop('name', stream_name, timeout=5.0)
    if not streams:
        print(f"[-] No LSL stream found with name '{stream_name}'.")
        print("    Ensure gpype_to_lsl.py or mock_lsl_streamer.py is running.")
        sys.exit(1)

    inlet = StreamInlet(streams[0])
    info = inlet.info()
    srate = info.nominal_srate()
    n_ch = info.channel_count()
    print(f"[+] Found stream '{stream_name}' ({n_ch} ch @ {srate} Hz)")

    # Read channel labels from LSL metadata
    ch_names = []
    ch = info.desc().child("channels").child("channel")
    for i in range(n_ch):
        if not ch.empty():
            ch_names.append(ch.child_value("label") or f"CH{i+1}")
            ch = ch.next_sibling("channel")
        else:
            ch_names.append(f"CH{i+1}")

    print(f"[*] Recording for {duration_sec:.1f} seconds to {output_path}...")
    samples = []
    timestamps = []
    start_time = time.time()

    while (time.time() - start_time) < duration_sec:
        chunk, ts = inlet.pull_chunk(timeout=0.1, max_samples=64)
        if chunk:
            samples.extend(chunk)
            timestamps.extend(ts)

    data = np.array(samples, dtype=np.float32)
    print(f"[+] Captured {len(data)} samples ({data.shape[0]/srate:.2f} s). Writing output...")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if format_type == "csv":
        header = "timestamp," + ",".join(ch_names)
        ts_col = np.array(timestamps)[:, None]
        combined = np.hstack([ts_col, data])
        np.savetxt(output_path, combined, delimiter=",", header=header, comments="", fmt="%.6f")
        print(f"[+] Saved CSV to {output_path}")

    elif format_type == "edf":
        import mne
        # LSL microvolts to Volts for MNE
        data_v = (data * 1e-6).T
        mne_info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types='eeg')
        raw = mne.io.RawArray(data_v, mne_info)
        mne.export.export_raw(output_path, raw, fmt="edf", overwrite=True)
        print(f"[+] Exported EDF to {output_path}")

    elif format_type == "mat":
        from scipy.io import savemat
        savemat(output_path, {
            "data": data,
            "timestamps": timestamps,
            "channel_names": ch_names,
            "sampling_rate": srate
        })
        print(f"[+] Saved MAT to {output_path}")

    elif format_type == "npz":
        np.savez_compressed(
            output_path,
            data=data,
            timestamps=timestamps,
            channel_names=ch_names,
            sampling_rate=srate
        )
        print(f"[+] Saved NPZ to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="g.Pype / LSL EEG Data Extractor")
    parser.add_argument("--source", choices=["lsl", "gpype"], default="lsl", help="Extraction source")
    parser.add_argument("--stream-name", type=str, default="gNautilus_EEG", help="LSL stream name if source=lsl")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds")
    parser.add_argument("--format", choices=["csv", "edf", "mat", "npz"], default="csv", help="Output format")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: results/extract_<timestamp>.<fmt>)")

    args = parser.parse_args()

    if args.output is None:
        t_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(_scripts_dir, "..", "results", f"eeg_extract_{t_str}.{args.format}")

    if args.source == "lsl":
        extract_from_lsl(
            stream_name=args.stream_name,
            duration_sec=args.duration,
            output_path=args.output,
            format_type=args.format
        )
    else:
        print("[!] Native gpype direct extraction pipeline requires gtec-gds driver.")
        print("    If running without physical dongle, please use '--source lsl' with gpype_to_lsl.py.")


if __name__ == "__main__":
    main()
