"""
Multimodal BIDS Dataset Recorder (EEG + Smartwatch IMU + Smartwatch PPG + Markers)
===================================================================================

Connects to all active LSL streams on your local network:
  - EEG Stream        (e.g., 'gNautilus' 32-channel EEG)
  - Wearable IMU      (e.g., 'IMU_L_wrist', 'IMU_R_ankle', legacy 'Smartwatch_IMU')
  - Wearable PPG      (e.g., 'HR_L_wrist', legacy 'Smartwatch_PPG')
  - Wearable touch    (e.g., 'TOUCH_L_wrist')
  - Clock sync        ('ClockSync_*' ping/echo exchanges from the wearable bridge)
  - Event Markers     (e.g., 'MotorImageryMarkers')

Saves everything into a standardized Multimodal BIDS Dataset with aligned timestamps.
Wearables are written as BIDS 1.9 motion (motion/*_tracksys-<stream>_motion.tsv) and
physio (eeg|beh/*_recording-<stream>_physio.tsv.gz) with labels, units, device and body
location in the sidecars; read them with utils.bids_wearables.read_wearable. Their raw
device clock and the clock-sync exchanges go to sourcedata/ so refit_wearable_clocks.py
can redo the clock fit offline over the whole session.

Usage:
  uv run python multimodal_bids_recorder.py --duration 30 --sub 01 --ses 01 --task leftright
"""
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
import re
import argparse
import threading
import numpy as np
import pandas as pd

from pylsl import StreamInlet, resolve_streams, local_clock, proc_clocksync
import mne
import mne_bids
from mne_bids import BIDSPath, write_raw_bids

from utils.bids_wearables import TIMESTAMP_COLUMN, write_wearable


WEARABLE_TYPES = ('IMU', 'PPG', 'TOUCH')
CLOCKSYNC_TYPE = 'CLOCKSYNC'
DEVICE_TIME_LABEL = 'DeviceTime'  # raw device clock channel appended by the wearable bridge
BIDS_MOTION_TYPES = {'accel': 'ACCEL', 'gyro': 'GYRO'}


def stream_metadata(info):
    """Channel labels/units/types and wearable identity from a full StreamInfo (inlet.info())."""
    channels = []
    ch = info.desc().child("channels").child("channel")
    while not ch.empty():
        channels.append({'label': ch.child_value("label"), 'unit': ch.child_value("unit"),
                         'type': ch.child_value("type")})
        ch = ch.next_sibling("channel")
    if len(channels) != info.channel_count() or not all(c['label'] for c in channels):
        channels = [{'label': f"ch_{i + 1}", 'unit': '', 'type': ''} for i in range(info.channel_count())]
    desc = info.desc()
    return {
        'channels': channels,
        'device_id': desc.child_value("device_id"),
        'location': desc.child_value("location"),
        'clock_sync': desc.child_value("clock_sync"),
        'source_id': info.source_id(),
    }


def is_motion(sname, stype):
    return stype == 'IMU' or 'imu' in sname.lower()


def bids_label(sname):
    return re.sub(r'[^A-Za-z0-9]', '', sname)


def motion_channel_row(label, unit, tracked_point):
    """BIDS motion channels.tsv row: Accel_X -> component x, type ACCEL; timestamps are LATENCY."""
    if label == TIMESTAMP_COLUMN:
        return {'name': label, 'component': 'n/a', 'type': 'LATENCY', 'tracked_point': 'n/a', 'units': 's'}
    prefix, _, axis = label.rpartition('_')
    return {
        'name': label,
        'component': axis.lower() if axis.upper() in ('X', 'Y', 'Z') else 'n/a',
        'type': BIDS_MOTION_TYPES.get(prefix.lower(), 'MISC'),
        'tracked_point': tracked_point or 'n/a',
        'units': unit or 'n/a',
    }


class MultimodalBIDSRecorder:
    def __init__(self, bids_root="bids_dataset_multimodal"):
        self.bids_root = os.path.abspath(bids_root)
        
        self.inlets = {}
        self.streams = {}  # every stream ever connected; inlets drop disconnected ones, export must not
        self.data_buffers = {}
        self.timestamp_buffers = {}
        self.marker_events = []

        self.is_recording = False
        self.recording_thread = None
        self.start_time_lsl = 0.0

    @staticmethod
    def _open_inlet(info):
        # proc_clocksync maps remote timestamps onto this machine's local_clock(),
        # so streams from other hosts (watch bridge, fNIRS laptop) line up with EEG.
        return StreamInlet(info, max_chunklen=64, processing_flags=proc_clocksync)

    @staticmethod
    def _full_info(inlet, resolved):
        # resolve_streams() returns infos without the <desc> block (channel labels, units,
        # device/location); only the inlet fetches the full header from the outlet
        try:
            return inlet.info(timeout=2.0)
        except Exception:
            return resolved

    def discover_and_connect_streams(self, timeout=3.0):
        print("=" * 70)
        print(" Discovering Active LSL Streams ".center(70, "="))
        print("=" * 70)
        streams = resolve_streams(wait_time=timeout)
        
        if not streams:
            print("[!] Warning: No active LSL streams detected on local network.")
            return False

        for s in streams:
            stype = s.type().upper()
            sname = s.name()
            print(f"[+] Found Stream: '{sname}' | Type: '{stype}' | Channels: {s.channel_count()} | Rate: {s.nominal_srate()} Hz")

            inlet = self._open_inlet(s)
            self.inlets[sname] = {
                'inlet': inlet,
                'info': self._full_info(inlet, s),
                'type': stype,
                'srate': s.nominal_srate(),
                'n_ch': s.channel_count()
            }
            self.streams[sname] = self.inlets[sname]
            self.data_buffers[sname] = []
            self.timestamp_buffers[sname] = []

        return True

    def start_recording(self):
        self.is_recording = True
        self.start_time_lsl = local_clock()

        # Start recording loop
        self.recording_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.recording_thread.start()

        # Start separate stream discovery loop (non-blocking, robust timeout)
        self.discovery_thread = threading.Thread(target=self._discovery_loop, daemon=True)
        self.discovery_thread.start()
        print("\n[START] Multimodal Recording actively capturing all streams (dynamic discovery enabled)...")

    def _discovery_loop(self):
        while self.is_recording:
            try:
                # Use a larger timeout (0.8s) to reliably detect LSL streams (e.g. task markers)
                streams = resolve_streams(wait_time=0.8)
                for s in streams:
                    sname = s.name()
                    if sname not in self.inlets:
                        stype = s.type().upper()
                        print(f"[+] Dynamically discovered stream: '{sname}' | Type: '{stype}'")
                        try:
                            inlet = self._open_inlet(s)
                            self.inlets[sname] = {
                                'inlet': inlet,
                                'info': self._full_info(inlet, s),
                                'type': stype,
                                'srate': s.nominal_srate(),
                                'n_ch': s.channel_count()
                            }
                            self.streams[sname] = self.inlets[sname]
                            if sname not in self.data_buffers:
                                self.data_buffers[sname] = []
                                self.timestamp_buffers[sname] = []
                        except Exception as e:
                            print(f"[-] Failed to connect to dynamic stream '{sname}': {e}")
            except Exception:
                pass
            time.sleep(1.0)

    def _record_loop(self):
        while self.is_recording:
            # Pull chunks from all active inlets
            # Create a static list of keys to prevent runtime dictionary modification issues
            active_names = list(self.inlets.keys())
            for sname in active_names:
                item = self.inlets.get(sname)
                if not item:
                    continue
                inlet = item['inlet']
                stype = item['type']
                try:
                    chunk, timestamps = inlet.pull_chunk(timeout=0.005)
                    if timestamps:
                        if stype == 'MARKERS':
                            for m, t in zip(chunk, timestamps):
                                m_str = m[0] if isinstance(m, list) else str(m)
                                duration = 0.1
                                if '_dur_' in m_str:
                                    try:
                                        parts = m_str.split('_dur_')
                                        m_str = parts[0]
                                        duration = float(parts[1])
                                    except Exception:
                                        pass
                                self.marker_events.append((t - self.start_time_lsl, m_str, duration))
                        else:
                            self.data_buffers[sname].extend(chunk)
                            self.timestamp_buffers[sname].extend([t - self.start_time_lsl for t in timestamps])
                except Exception:
                    # If stream fails/disconnects, we can remove it so we attempt reconnect later
                    print(f"[-] Stream disconnected: {sname}")
                    self.inlets.pop(sname, None)
            time.sleep(0.005)

    def stop_and_export_bids(self, subject_id="01", session_id="01", task_name="leftright"):
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()

        print("\n" + "=" * 70)
        print(" Exporting Multimodal BIDS Dataset ".center(70, "="))
        print("=" * 70)

        sub_clean = subject_id.replace("sub-", "")
        ses_clean = session_id.replace("ses-", "")

        # 1. Export EEG Stream via MNE-BIDS if present
        eeg_stream_name = None
        for sname, item in self.inlets.items():
            if item['type'] == 'EEG' or 'gNautilus' in sname:
                eeg_stream_name = sname
                break

        if eeg_stream_name and len(self.data_buffers[eeg_stream_name]) > 0:
            item = self.inlets[eeg_stream_name]
            info_xml = item['info']
            eeg_raw_data = np.array(self.data_buffers[eeg_stream_name]).T
            srate = item['srate'] if item['srate'] > 0 else 250.0

            # Parse XML channel labels
            ch_names = []
            ch_types = []
            ch_units = []
            ch_child = info_xml.desc().child("channels").child("channel")
            for i in range(item['n_ch']):
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

            # If driver gave generic/numeric channel labels, map to standard GPYPE_32 10-20 names
            GPYPE_32 = [
                "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
                "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
                "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8",
                "PO7", "PO3", "PO4", "PO8", "Oz",
            ]
            has_1020 = any(n.upper() in ["C3", "C4", "P3", "P4", "OZ"] for n in ch_names)
            if not has_1020 and len(ch_names) >= 32:
                for idx in range(32):
                    ch_names[idx] = GPYPE_32[idx]

            for idx, c_name in enumerate(ch_names):
                if c_name.lower() in ['battery', 'counter', 'sync', 'status']:
                    ch_types[idx] = 'misc'

            # Convert to Volts for MNE
            has_uv = any('microvolt' in u or 'uv' in u for u in ch_units if u)
            has_v = any(u == 'v' or u == 'volt' or u == 'volts' for u in ch_units if u)
            if has_uv:
                print("[*] Scaling LSL microvolts (uV) to Volts (V).")
                eeg_raw_data = eeg_raw_data * 1e-6
            elif has_v:
                print("[*] LSL stream natively in Volts (V). Preserving scale.")
            else:
                med_std = np.median(np.std(eeg_raw_data, axis=1))
                if med_std > 0.5:
                    print(f"[*] Heuristic unit detection (median std = {med_std:.2f} > 0.5): scaling uV to V.")
                    eeg_raw_data = eeg_raw_data * 1e-6
                else:
                    print(f"[*] Heuristic unit detection (median std = {med_std:.6f} <= 0.5): keeping V scale.")

            # Channel quality check for flatline & railing pins
            ch_stds_uv = np.std(eeg_raw_data, axis=1) * 1e6
            ch_ptp_uv = np.ptp(eeg_raw_data, axis=1) * 1e6
            bad_channels = []
            for idx, (c_name, c_std, c_ptp) in enumerate(zip(ch_names, ch_stds_uv, ch_ptp_uv)):
                if ch_types[idx] == 'eeg':
                    if c_std < 0.2:
                        bad_channels.append(c_name)
                    elif c_ptp > 4000.0 or c_std > 1000.0:
                        bad_channels.append(c_name)

            mne_info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)
            mne_info['bads'] = bad_channels
            if bad_channels:
                print(f"[!] Quality Audit: flagged {len(bad_channels)} channels as BAD: {bad_channels}")

            raw = mne.io.RawArray(eeg_raw_data, mne_info)
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage, on_missing='ignore')
            except Exception:
                pass

            # Add Marker annotations
            if self.marker_events and len(self.timestamp_buffers[eeg_stream_name]) > 0:
                first_eeg_ts = self.timestamp_buffers[eeg_stream_name][0]
                onsets = []
                durations = []
                descriptions = []
                for item in self.marker_events:
                    onset = item[0] - first_eeg_ts
                    if onset >= 0:
                        onsets.append(onset)
                        descriptions.append(item[1])
                        durations.append(item[2] if len(item) > 2 else 0.1)
                
                if onsets:
                    annot = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
                    raw.set_annotations(annot)

            bids_path = BIDSPath(
                subject=sub_clean,
                session=ses_clean,
                task=task_name,
                datatype="eeg",
                root=self.bids_root
            )
            write_raw_bids(raw, bids_path=bids_path, allow_preload=True, format='BrainVision', overwrite=True, verbose=False)
            print(f"[+] Exported EEG BIDS data: sub-{sub_clean}_ses-{ses_clean}_task-{task_name}_eeg")
        else:
            # If no EEG stream, export markers standalone so they aren't lost!
            if self.marker_events:
                bids_sub_dir = os.path.join(self.bids_root, f"sub-{sub_clean}", f"ses-{ses_clean}")
                beh_dir = os.path.join(bids_sub_dir, "beh")
                os.makedirs(beh_dir, exist_ok=True)
                
                # Markers are already relative to self.start_time_lsl (item[0])

                events_df = pd.DataFrame({
                    "onset": [max(0.0, item[0]) for item in self.marker_events],
                    "duration": [item[2] if len(item) > 2 else 0.1 for item in self.marker_events],
                    "trial_type": [item[1] for item in self.marker_events]
                })
                events_path = os.path.join(beh_dir, f"sub-{sub_clean}_ses-{ses_clean}_task-{task_name}_events.tsv")
                events_df.to_csv(events_path, sep='\t', index=False)
                print(f"[+] Exported Standalone Events BIDS data to: {events_path}")

        # 2. Export wearable streams (IMU, PPG, touch) and their clock-sync data
        self._export_wearables(sub_clean, ses_clean, task_name, eeg_stream_name)
        self._export_clock_sync(sub_clean, ses_clean, task_name)

        print("=" * 70)
        print(f" Multimodal BIDS Export Complete! Root: {self.bids_root} ".center(70, "="))
        print("=" * 70)


    def _session_dirs(self, sub, ses):
        bids_dir = os.path.join(self.bids_root, f"sub-{sub}", f"ses-{ses}")
        source_dir = os.path.join(self.bids_root, "sourcedata", f"sub-{sub}", f"ses-{ses}")
        return bids_dir, source_dir

    def _export_wearables(self, sub, ses, task, eeg_stream_name=None):
        """Wearables as strict BIDS 1.9: IMU as motion (headerless TSV, channels.tsv, motion.json),
        PPG/heart rate and touch as physio (headerless .tsv.gz next to the EEG, or in beh/).
        timestamp_sec is the first column in both (a LATENCY channel for motion), so
        utils.bids_wearables.read_wearable gives analysis code timestamp_sec + data columns.
        The raw device clock goes to sourcedata/ instead."""
        bids_dir, source_dir = self._session_dirs(sub, ses)
        eeg_stamps = self.timestamp_buffers.get(eeg_stream_name) if eeg_stream_name else None
        # Physio belongs beside the data it accompanies: the EEG run if there is one
        physio_folder = 'eeg' if eeg_stamps else 'beh'
        wearables = [sname for sname, item in self.streams.items()
                     if item['type'] in WEARABLE_TYPES or 'Smartwatch' in sname]

        for sname in wearables:
            item = self.streams[sname]
            if not self.data_buffers.get(sname):
                continue
            data = np.array(self.data_buffers[sname], dtype=np.float64)
            timestamps = np.array(self.timestamp_buffers[sname])
            meta = stream_metadata(item['info'])
            labels = [c['label'] for c in meta['channels']]
            units = {c['label']: c['unit'] or 'n/a' for c in meta['channels']}
            data_idx = [i for i, label in enumerate(labels) if label != DEVICE_TIME_LABEL]
            motion = is_motion(sname, item['type'])

            df = pd.DataFrame(data[:, data_idx], columns=[labels[i] for i in data_idx])
            df.insert(0, TIMESTAMP_COLUMN, timestamps)
            duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
            effective = (len(timestamps) - 1) / duration if duration > 0 else 0.0

            if motion:
                entity, folder, suffix, ext = 'tracksys', 'motion', 'motion', '.tsv'
            else:
                entity, folder, suffix, ext = 'recording', physio_folder, 'physio', '.tsv.gz'
            base = f"sub-{sub}_ses-{ses}_task-{task}_{entity}-{bids_label(sname)}"
            out_folder = os.path.join(bids_dir, folder)
            os.makedirs(out_folder, exist_ok=True)
            data_path = os.path.join(out_folder, f"{base}_{suffix}{ext}")
            write_wearable(data_path, df)

            sidecar = {
                "TaskName": task,
                # Wearables without a fixed rate (heart rate) report their measured rate
                "SamplingFrequency": item['srate'] if item['srate'] > 0 else round(effective, 3),
                "SamplingFrequencyEffective": round(effective, 3),
                "RecordingDuration": round(duration, 3),
            }
            if motion:
                rows = [motion_channel_row(c, units.get(c, 'n/a'), meta['location']) for c in df.columns]
                pd.DataFrame(rows).to_csv(os.path.join(out_folder, f"{base}_channels.tsv"),
                                          sep='\t', index=False)
                types = [r['type'] for r in rows]
                sidecar.update({
                    "TrackingSystemName": sname,
                    "MotionChannelCount": len(rows),
                    "ACCELChannelCount": types.count('ACCEL'),
                    "GYROChannelCount": types.count('GYRO'),
                })
            else:
                first = timestamps[0] if len(timestamps) else 0.0
                sidecar.update({
                    "Columns": list(df.columns),
                    # BIDS: seconds from the first sample of the accompanying (EEG) data
                    "StartTime": float(first - eeg_stamps[0]) if eeg_stamps else float(first),
                })
            sidecar.update({
                "Units": {c: (units.get(c, 'n/a') if c != TIMESTAMP_COLUMN else 's') for c in df.columns},
                "StreamName": sname,
                "StreamType": item['type'],
                "LSLSourceID": meta['source_id'],
                "LSLStartTime": self.start_time_lsl,
            })
            if meta['device_id']:
                sidecar["DeviceID"] = meta['device_id']
            if meta['location']:
                sidecar["TrackedPointLocation"] = meta['location']
            if meta['clock_sync']:
                sidecar["ClockSync"] = meta['clock_sync']

            if DEVICE_TIME_LABEL in labels:
                # Raw device clock beside the live LSL timestamps: what refit_wearable_clocks.py needs
                dev_folder = os.path.join(source_dir, folder)
                os.makedirs(dev_folder, exist_ok=True)
                dev_path = os.path.join(dev_folder, f"{base}_devicetime.tsv")
                pd.DataFrame({TIMESTAMP_COLUMN: timestamps,
                              'device_time_s': data[:, labels.index(DEVICE_TIME_LABEL)]}
                             ).to_csv(dev_path, sep='\t', index=False, float_format='%.9f')
                with open(dev_path.replace('.tsv', '.json'), 'w') as f:
                    json.dump({"DeviceID": meta['device_id'], "StreamName": sname,
                               "DataFile": os.path.relpath(data_path, self.bids_root),
                               "Description": "timestamp_sec as recorded live; device_time_s is the "
                                              "sample's raw device clock"}, f, indent=2)
                sidecar["DeviceTimeFile"] = os.path.relpath(dev_path, self.bids_root)

            with open(os.path.join(out_folder, f"{base}_{suffix}.json"), 'w') as f:
                json.dump(sidecar, f, indent=2)
            print(f"[+] Exported wearable {item['type']} '{sname}' to: {data_path}")

        if wearables:
            self._ensure_dataset_description()

    def _ensure_dataset_description(self):
        # MNE-BIDS writes it with the EEG; a wearables-only session needs one to be a valid dataset
        path = os.path.join(self.bids_root, "dataset_description.json")
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump({"Name": "Nautilus BCI multimodal recording", "BIDSVersion": "1.9.0",
                           "DatasetType": "raw"}, f, indent=2)

    def _export_clock_sync(self, sub, ses, task):
        """Ping/echo exchanges per wearable to sourcedata/, on the same time axis as timestamp_sec."""
        _, source_dir = self._session_dirs(sub, ses)
        for sname, item in self.streams.items():
            if item['type'] != CLOCKSYNC_TYPE or not self.data_buffers.get(sname):
                continue
            raw = np.array(self.data_buffers[sname], dtype=np.float64)
            stamps = np.array(self.timestamp_buffers[sname])
            # t1/t4 are LSL times on the bridge host; each sample is stamped t4 there, so
            # (recorded stamp - t4) moves both onto this recording's timestamp_sec axis
            shift = stamps - raw[:, 3]
            df = pd.DataFrame({'t1_ping_sent': raw[:, 0] + shift, 't2_ping_received': raw[:, 1],
                               't3_echo_sent': raw[:, 2], 't4_echo_received': raw[:, 3] + shift})
            meta = stream_metadata(item['info'])
            out_folder = os.path.join(source_dir, "clocksync")
            os.makedirs(out_folder, exist_ok=True)
            tsv_path = os.path.join(out_folder, f"sub-{sub}_ses-{ses}_task-{task}_recording-{bids_label(sname)}_clocksync.tsv")
            df.to_csv(tsv_path, sep='\t', index=False, float_format='%.9f')
            with open(tsv_path.replace('.tsv', '.json'), 'w') as f:
                json.dump({"DeviceID": meta['device_id'], "TrackedPointLocation": meta['location'],
                           "StreamName": sname,
                           "Description": "Ping/echo exchanges. t1/t4 on the recording's timestamp_sec "
                                          "axis, t2/t3 device clock (s). offset = ((t1-t2)+(t4-t3))/2, "
                                          "delay = (t4-t1)-(t3-t2)."}, f, indent=2)
            print(f"[+] Exported clock sync '{sname}' ({len(df)} exchanges) to: {tsv_path}")


def main():
    parser = argparse.ArgumentParser(description="Multimodal BIDS Dataset Recorder")
    parser.add_argument("--duration", type=float, default=15.0, help="Recording duration in seconds")
    parser.add_argument("--sub", type=str, default="01", help="Subject ID")
    parser.add_argument("--ses", type=str, default="01", help="Session ID")
    parser.add_argument("--task", type=str, default="leftright", help="Task Name")
    parser.add_argument("--root", type=str, default="bids_dataset_multimodal", help="BIDS root folder")
    args = parser.parse_args()

    recorder = MultimodalBIDSRecorder(bids_root=args.root)
    connected = recorder.discover_and_connect_streams(timeout=2.0)
    
    if not connected:
        print("\n[-] Please launch your LSL streams first!")
        print("    1. g.Nautilus EEG: uv run python mock_lsl_streamer.py (or gds_to_lsl.py)")
        print("    2. Smartwatch:     uv run python smartwatch_lsl_bridge.py --mode udp --port 5005")
        sys.exit(1)

    recorder.start_recording()
    
    print(f"[*] Recording for {args.duration} seconds... Press Ctrl+C to stop early.")
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\n[*] Manual stop requested.")

    recorder.stop_and_export_bids(subject_id=args.sub, session_id=args.ses, task_name=args.task)


if __name__ == '__main__':
    main()
