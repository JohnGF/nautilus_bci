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

from pylsl import StreamInlet, resolve_byprop, resolve_streams, local_clock, proc_clocksync
import mne
import mne_bids
from mne_bids import BIDSPath, write_raw_bids

class BIDSRecorder:
    def __init__(self, bids_root="bids_dataset", eeg_stream_name="gNautilus", marker_stream_name="MotorImageryMarkers", marker_source_id=None):
        self.bids_root = os.path.abspath(bids_root)
        self.eeg_stream_name = eeg_stream_name
        self.marker_stream_name = marker_stream_name
        self.marker_source_id = marker_source_id

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
            if s.name() == self.eeg_stream_name:
                eeg_stream_info = s
            elif eeg_stream_info is None and s.type().upper() == 'EEG':
                eeg_stream_info = s

            if self.marker_source_id and s.source_id() == self.marker_source_id:
                marker_stream_info = s
            elif marker_stream_info is None:
                if s.name() == self.marker_stream_name:
                    marker_stream_info = s
                elif s.type().upper() == 'MARKERS':
                    marker_stream_info = s

        if eeg_stream_info is None:
            raise RuntimeError(f"Could not find EEG stream named '{self.eeg_stream_name}' or type 'EEG'.")

        self.eeg_inlet = StreamInlet(eeg_stream_info, max_chunklen=32, processing_flags=proc_clocksync)
        try:
            self.eeg_inlet.open_stream(timeout=3.0)
        except Exception as e:
            print(f"[!] Warning opening EEG stream: {e}")
        print(f"[+] Connected to EEG Stream: {eeg_stream_info.name()} ({eeg_stream_info.channel_count()} ch, {eeg_stream_info.nominal_srate()} Hz)")

        if marker_stream_info is not None:
            self.marker_inlet = StreamInlet(marker_stream_info, processing_flags=proc_clocksync)
            try:
                self.marker_inlet.info(timeout=3.0)
                self.marker_inlet.open_stream(timeout=3.0)
            except Exception as e:
                print(f"[!] Warning handshake with Marker stream: {e}")
            print(f"[+] Connected to Marker Stream: {marker_stream_info.name()} (source_id: {marker_stream_info.source_id()})")
        else:
            print("[!] Warning: Marker stream not found. Will attempt to reconnect or record EEG only.")

        return bool(self.eeg_inlet is not None and self.marker_inlet is not None)

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
                    markers, m_timestamps = self.marker_inlet.pull_chunk(timeout=0.005)
                    if m_timestamps:
                        for m, t in zip(markers, m_timestamps):
                            m_str = m[0] if isinstance(m, list) else str(m)
                            duration = 0.1
                            if '_dur_' in m_str:
                                try:
                                    parts = m_str.split('_dur_')
                                    m_str = parts[0]
                                    duration = float(parts[1])
                                except Exception:
                                    pass
                            self.marker_events.append((t, m_str, duration))
                except Exception as e:
                    print(f"[-] Error pulling markers: {e}")
            else:
                # Try background reconnect for markers
                try:
                    m_streams = resolve_byprop("name", self.marker_stream_name, timeout=0.1)
                    if m_streams:
                        self.marker_inlet = StreamInlet(m_streams[0], processing_flags=proc_clocksync)
                        print(f"[+] Late-connected to Marker Stream: {self.marker_stream_name}")
                except Exception:
                    pass

            time.sleep(0.005)

    def stop_recording_and_export_bids(self, subject_id="01", session_id="01", task_name="leftright"):
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()

        # Drain any residual chunks remaining in the LSL stream buffers
        if self.eeg_inlet:
            try:
                chunk, timestamps = self.eeg_inlet.pull_chunk(timeout=0.05)
                if timestamps:
                    self.eeg_samples.extend(chunk)
                    self.eeg_timestamps.extend(timestamps)
            except Exception:
                pass

        if self.marker_inlet:
            try:
                markers, m_timestamps = self.marker_inlet.pull_chunk(timeout=0.05)
                if m_timestamps:
                    for m, t in zip(markers, m_timestamps):
                        m_str = m[0] if isinstance(m, list) else str(m)
                        duration = 0.1
                        if '_dur_' in m_str:
                            try:
                                parts = m_str.split('_dur_')
                                m_str = parts[0]
                                duration = float(parts[1])
                            except Exception:
                                pass
                        self.marker_events.append((t, m_str, duration))
            except Exception:
                pass

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

        # Parse XML channel labels and units if available
        ch_child = stream_info.desc().child("channels").child("channel")
        ch_units = []
        for i in range(n_channels):
            if not ch_child.empty():
                c_name = ch_child.child_value("label")
                c_type = ch_child.child_value("type").lower()
                c_unit = ch_child.child_value("unit").lower()
                ch_names.append(c_name if c_name else f"EEG{i+1:03d}")
                ch_types.append('eeg' if c_type in ['eeg', ''] else 'misc')
                ch_units.append(c_unit)
                ch_child = ch_child.next_sibling("channel")
            else:
                ch_names.append(f"EEG{i+1:03d}")
                ch_types.append('eeg')
                ch_units.append('microvolts')

        # Convert EEG samples to numpy array: shape (n_channels, n_samples)
        eeg_data = np.array(self.eeg_samples).T  # shape (n_channels, n_samples)
        
        # Unit conversion for MNE (standard EEG in MNE must be in Volts)
        # Check explicit stream units first before fallback
        has_uv_unit = any('microvolt' in u or 'uv' in u for u in ch_units if u)
        has_v_unit = any(u == 'v' or u == 'volt' or u == 'volts' for u in ch_units if u)
        
        if has_uv_unit:
            print("[*] Detected explicit microvolts (uV) in LSL metadata. Scaling to Volts.")
            eeg_data = eeg_data * 1e-6
        elif has_v_unit:
            print("[*] Detected explicit Volts (V) in LSL metadata. Keeping native scale.")
        else:
            # Fallback heuristic: typical raw EEG in uV has std > 1.0; native Volts has std < 0.01
            sample_stds = np.std(eeg_data, axis=1)
            med_std = np.median(sample_stds)
            if med_std > 0.5:
                print(f"[*] Heuristic unit detection (median std = {med_std:.2f} > 0.5): assuming microvolts (uV), converting to Volts.")
                eeg_data = eeg_data * 1e-6
            else:
                print(f"[*] Heuristic unit detection (median std = {med_std:.6f} <= 0.5): assuming Volts (V).")

        # Channel Quality Audit: detect flatlined (disconnected) and railing (saturated) pins
        # Standard EEG physiological range in Volts: std between 1 uV and 500 uV, ptp < 2000 uV
        ch_stds_uv = np.std(eeg_data, axis=1) * 1e6
        ch_ptp_uv = np.ptp(eeg_data, axis=1) * 1e6
        
        bad_channels = []
        for idx, (c_name, c_std, c_ptp) in enumerate(zip(ch_names, ch_stds_uv, ch_ptp_uv)):
            if ch_types[idx] == 'eeg':
                if c_std < 0.2:
                    bad_channels.append(c_name)
                    print(f"[!] Quality Audit: Channel {c_name} is FLATLINE (std = {c_std:.2f} uV). Marked BAD.")
                elif c_ptp > 4000.0 or c_std > 1000.0:
                    bad_channels.append(c_name)
                    print(f"[!] Quality Audit: Channel {c_name} is RAILING / SATURATED (ptp = {c_ptp:.1f} uV, std = {c_std:.1f} uV). Marked BAD.")

        # Create MNE Info and assign bad channels
        info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)
        info['bads'] = bad_channels
        if bad_channels:
            print(f"[!] Flagged {len(bad_channels)} / {n_channels} channels as BAD in BIDS metadata: {bad_channels}")

        # Standard 10-20 montage alignment for EEG channels
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, on_missing="ignore")

        raw = mne.io.RawArray(eeg_data, info)

        # Build Annotations from Marker Events
        onsets = []
        durations = []
        descriptions = []

        first_eeg_ts = self.eeg_timestamps[0]
        for item in self.marker_events:
            if len(item) == 3:
                m_ts, m_str, m_dur = item
            else:
                m_ts, m_str = item[:2]
                m_dur = 0.1
            onset = m_ts - first_eeg_ts
            if onset >= 0:
                onsets.append(onset)
                if m_dur and m_dur != 0.1:
                    durations.append(m_dur)
                else:
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

        # Retain full baseline: do NOT destructively crop raw EEG.
        # Mark pre/post experiment bounds as non-destructive annotations instead
        if self.marker_events:
            first_m_ts = self.marker_events[0][0]
            last_m_ts = self.marker_events[-1][0]
            pre_exp_dur = max(0.0, first_m_ts - first_eeg_ts)
            post_exp_start = last_m_ts - first_eeg_ts
            print(f"[*] Preserving pre-experiment baseline ({pre_exp_dur:.1f}s) and post-experiment data for calibration.")

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
