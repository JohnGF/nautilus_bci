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
import json
import threading
import numpy as np
import pandas as pd

from pylsl import StreamInlet, resolve_byprop, resolve_streams, local_clock
import mne
import mne_bids
from mne_bids import BIDSPath, write_raw_bids

class BIDSRecorder:
    def __init__(self, bids_root="bids_dataset", eeg_stream_name="gNautilus", marker_stream_name="MotorImageryMarkers"):
        self.bids_root = os.path.abspath(bids_root)
        self.eeg_stream_name = eeg_stream_name
        self.marker_stream_name = marker_stream_name

        self.eeg_inlet = None
        self.marker_inlet = None

        self.eeg_samples = []
        self.eeg_timestamps = []
        self.marker_events = []  # list of tuples: (timestamp, marker_str)

        self.is_recording = False
        self.recording_thread = None
        self.eeg_info = None

    def connect_streams(self, timeout=5.0):
        print(f"[*] Looking for LSL EEG stream '{self.eeg_stream_name}'...")
        streams = resolve_streams(wait_time=timeout)
        eeg_stream_info = None
        marker_stream_info = None

        for s in streams:
            if s.name() == self.eeg_stream_name or s.type().upper() == 'EEG':
                eeg_stream_info = s
            if s.name() == self.marker_stream_name or s.type().upper() == 'MARKERS':
                marker_stream_info = s

        if eeg_stream_info is None:
            raise RuntimeError(f"Could not find EEG stream named '{self.eeg_stream_name}' or type 'EEG'.")

        self.eeg_inlet = StreamInlet(eeg_stream_info, max_chunklen=32)
        print(f"[+] Connected to EEG Stream: {eeg_stream_info.name()} ({eeg_stream_info.channel_count()} ch, {eeg_stream_info.nominal_srate()} Hz)")

        if marker_stream_info is not None:
            self.marker_inlet = StreamInlet(marker_stream_info)
            print(f"[+] Connected to Marker Stream: {marker_stream_info.name()}")
        else:
            print("[!] Warning: Marker stream not found. Will attempt to reconnect or record EEG only.")

    def start_recording(self):
        if self.is_recording:
            print("[!] Already recording.")
            return

        self.eeg_samples = []
        self.eeg_timestamps = []
        self.marker_events = []
        self.is_recording = True

        self.recording_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.recording_thread.start()
        print("[START] Recording started!")

    def _record_loop(self):
        while self.is_recording:
            # Pull EEG chunks
            try:
                chunk, timestamps = self.eeg_inlet.pull_chunk(timeout=0.1)
                if timestamps:
                    self.eeg_samples.extend(chunk)
                    self.eeg_timestamps.extend(timestamps)
            except Exception as e:
                print(f"[-] Error pulling EEG data: {e}")

            # Pull Marker chunks if connected
            if self.marker_inlet:
                try:
                    markers, m_timestamps = self.marker_inlet.pull_chunk(timeout=0.0)
                    if m_timestamps:
                        for m, t in zip(markers, m_timestamps):
                            m_str = m[0] if isinstance(m, list) else str(m)
                            self.marker_events.append((t, m_str))
                except Exception as e:
                    print(f"[-] Error pulling markers: {e}")
            else:
                # Try background reconnect for markers
                try:
                    m_streams = resolve_byprop("name", self.marker_stream_name, timeout=0.1)
                    if m_streams:
                        self.marker_inlet = StreamInlet(m_streams[0])
                        print(f"[+] Late-connected to Marker Stream: {self.marker_stream_name}")
                except Exception:
                    pass

            time.sleep(0.005)

    def stop_recording_and_export_bids(self, subject_id="01", session_id="01", task_name="leftright"):
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()

        print(f"[+] Recording stopped. Captured {len(self.eeg_samples)} EEG samples and {len(self.marker_events)} marker events.")

        if len(self.eeg_samples) == 0:
            raise ValueError("No EEG samples were recorded.")

        # Clean subject and session formatting
        sub_clean = subject_id.replace("sub-", "")
        ses_clean = session_id.replace("ses-", "")

        # Extract metadata from LSL stream info
        stream_info = self.eeg_inlet.info()
        srate = stream_info.nominal_srate()
        if srate == 0.0:
            # Fallback to empirical sampling rate calculation
            diffs = np.diff(self.eeg_timestamps)
            srate = 1.0 / np.median(diffs) if len(diffs) > 0 else 250.0

        n_channels = stream_info.channel_count()
        ch_names = []
        ch_types = []

        # Parse XML channel labels if available
        ch_child = stream_info.desc().child("channels").child("channel")
        for i in range(n_channels):
            if not ch_child.empty():
                c_name = ch_child.child_value("label")
                c_type = ch_child.child_value("type").lower()
                ch_names.append(c_name if c_name else f"EEG{i+1:03d}")
                ch_types.append('eeg' if c_type in ['eeg', ''] else 'misc')
                ch_child = ch_child.next_sibling("channel")
            else:
                ch_names.append(f"EEG{i+1:03d}")
                ch_types.append('eeg')

        # Convert EEG samples to numpy array: shape (n_channels, n_samples)
        eeg_data = np.array(self.eeg_samples).T  # shape (n_channels, n_samples)
        
        # Microvolts to Volts conversion for MNE (standard EEG in MNE is in Volts)
        # Check if values are in microvolts range (e.g. > 1e-3)
        if np.max(np.abs(eeg_data)) > 1.0:
            eeg_data = eeg_data * 1e-6  # convert uV to V

        # Create MNE Info
        info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)

        # Standard 10-20 montage alignment for EEG channels
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, on_missing="ignore")

        raw = mne.io.RawArray(eeg_data, info)

        # Build Annotations from Marker Events
        onsets = []
        durations = []
        descriptions = []

        first_eeg_ts = self.eeg_timestamps[0]
        for m_ts, m_str in self.marker_events:
            onset = m_ts - first_eeg_ts
            if onset >= 0:
                onsets.append(onset)
                # Default duration of 0.0s for instantaneous markers, or custom for task durations
                durations.append(4.0 if "Task_" in m_str else (1.0 if "Cue_" in m_str else 0.0))
                descriptions.append(m_str)

        if onsets:
            total_duration = raw.times[-1] if len(raw.times) > 0 else 0.0
            valid_onsets = []
            valid_durations = []
            valid_descriptions = []
            for onset, dur, desc in zip(onsets, durations, descriptions):
                if onset < total_duration:
                    valid_onsets.append(onset)
                    valid_durations.append(min(dur, max(0.0, total_duration - onset)))
                    valid_descriptions.append(desc)

            if valid_onsets:
                annotations = mne.Annotations(onset=valid_onsets, duration=valid_durations, description=valid_descriptions)
                raw.set_annotations(annotations)

        # Auto-trim extra leading/trailing idle data outside experiment events (+/- 2.0s padding)
        if self.marker_events:
            first_m_ts = self.marker_events[0][0]
            last_m_ts = self.marker_events[-1][0]
            t_start = max(0.0, first_m_ts - first_eeg_ts - 2.0)
            t_end = min(raw.times[-1], last_m_ts - first_eeg_ts + 2.0)
            if t_end > t_start and (t_end - t_start) < raw.times[-1]:
                print(f"[*] Auto-trimming idle trailing/leading data ({t_start:.1f}s to {t_end:.1f}s)...")
                raw.crop(tmin=t_start, tmax=t_end)

        # Construct BIDSPath
        bids_path = BIDSPath(
            subject=sub_clean,
            session=ses_clean,
            task=task_name,
            datatype="eeg",
            root=self.bids_root
        )

        print(f"[*] Writing dataset to BIDS root: {self.bids_root} (Format: BrainVision)...")
        write_raw_bids(
            raw,
            bids_path=bids_path,
            format="BrainVision",
            allow_preload=True,
            overwrite=True,
            verbose=False
        )

        # Ensure dataset_description.json has appropriate BIDS fields
        desc_path = os.path.join(self.bids_root, "dataset_description.json")
        desc_data = {
            "Name": "Motor Imagery Left/Right BCI Dataset",
            "BIDSVersion": "1.8.0",
            "DatasetType": "raw",
            "Authors": ["BCI Experimenter"],
            "HowToAcknowledge": "Recorded via g.Nautilus LSL & BIDS Recorder Suite"
        }
        with open(desc_path, "w") as f:
            json.dump(desc_data, f, indent=4)

        print(f"[SUCCESS] BIDS Dataset exported successfully to: {bids_path.directory}")
        return bids_path.directory

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Record LSL EEG & Markers into BIDS dataset format.")
    parser.add_argument("--sub", type=str, default="01", help="Subject ID (e.g., 01)")
    parser.add_argument("--ses", type=str, default="01", help="Session ID (e.g., 01)")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration in seconds (0 for manual stop via Ctrl+C)")
    parser.add_argument("--outdir", type=str, default="bids_dataset", help="Output BIDS root directory")
    args = parser.parse_args()

    recorder = BIDSRecorder(bids_root=args.outdir)
    try:
        recorder.connect_streams()
        recorder.start_recording()

        if args.duration > 0:
            print(f"[*] Recording for fixed duration: {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            print("[*] Recording continuously. Press Ctrl+C to stop and save BIDS dataset...")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[*] Manual stop requested.")
    finally:
        try:
            out_path = recorder.stop_recording_and_export_bids(subject_id=args.sub, session_id=args.ses)
            print(f"[✓] Saved BIDS dataset at: {out_path}")
        except Exception as e:
            print(f"[-] Failed to export BIDS dataset: {e}")

if __name__ == "__main__":
    main()
