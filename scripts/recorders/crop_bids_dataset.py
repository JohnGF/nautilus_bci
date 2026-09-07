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
import argparse
import numpy as np
import mne
import mne_bids
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids

def crop_bids_session(bids_root="bids_dataset", subject_id="01", session_id="01", task_name="leftright", pad_before=2.0, pad_after=2.0, out_root=None):
    print("=" * 70)
    print(" BIDS Dataset Trimmer & Clean Tool ".center(70, "="))
    print("=" * 70)

    bids_root = os.path.abspath(bids_root)
    if not os.path.exists(bids_root):
        raise FileNotFoundError(f"BIDS root directory not found: {bids_root}")

    sub_clean = subject_id.replace("sub-", "")
    ses_clean = session_id.replace("ses-", "")

    bids_path = BIDSPath(
        subject=sub_clean,
        session=ses_clean,
        task=task_name,
        datatype="eeg",
        root=bids_root
    )

    print(f"[*] Loading BIDS session from: {bids_path.directory}")
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()

    orig_duration = raw.times[-1]
    print(f"[+] Original recording duration: {orig_duration:.2f} seconds")

    if len(raw.annotations) == 0:
        print("[!] No event annotations found in recording. Cannot crop by experiment bounds.")
        return

    # Find earliest and latest event timestamps
    onsets = raw.annotations.onset
    t_start = max(0.0, np.min(onsets) - pad_before)
    t_end = min(orig_duration, np.max(onsets) + pad_after)

    new_duration = t_end - t_start
    print(f"[*] Experiment Event Bounds: {np.min(onsets):.2f}s to {np.max(onsets):.2f}s")
    print(f"[*] Cropping raw EEG from {t_start:.2f}s to {t_end:.2f}s (Clean duration: {new_duration:.2f}s)...")

    # Crop raw dataset
    raw_cropped = raw.copy().crop(tmin=t_start, tmax=t_end)

    target_root = os.path.abspath(out_root) if out_root else bids_root
    out_bids_path = BIDSPath(
        subject=sub_clean,
        session=ses_clean,
        task=task_name,
        datatype="eeg",
        root=target_root
    )

    print(f"[*] Writing clean cropped dataset to: {target_root}...")
    write_raw_bids(
        raw_cropped,
        bids_path=out_bids_path,
        format="BrainVision",
        allow_preload=True,
        overwrite=True,
        verbose=False
    )

    trimmed_sec = orig_duration - new_duration
    print(f"[SUCCESS] Cleaned dataset saved! Trimmed {trimmed_sec:.1f} seconds of trailing/leading idle data.")

def main():
    parser = argparse.ArgumentParser(description="Crop trailing/leading idle EEG data outside experiment event markers.")
    parser.add_argument("--bids-root", type=str, default="bids_dataset", help="Path to input BIDS root")
    parser.add_argument("--sub", type=str, default="01", help="Subject ID (e.g., 01)")
    parser.add_argument("--ses", type=str, default="01", help="Session ID (e.g., 01)")
    parser.add_argument("--pad-before", type=float, default=2.0, help="Padding in seconds before first event")
    parser.add_argument("--pad-after", type=float, default=2.0, help="Padding in seconds after last event")
    parser.add_argument("--outdir", type=str, default=None, help="Output BIDS root (default: overwrites input bids-root)")
    args = parser.parse_args()

    import numpy as np
    crop_bids_session(
        bids_root=args.bids_root,
        subject_id=args.sub,
        session_id=args.ses,
        pad_before=args.pad_before,
        pad_after=args.pad_after,
        out_root=args.outdir
    )

if __name__ == "__main__":
    main()
