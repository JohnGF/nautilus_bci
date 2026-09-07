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
import subprocess
import threading
from pylsl import StreamInfo, StreamOutlet, local_clock

# Import BIDSRecorder directly

from recorders.bids_recorder import BIDSRecorder

def run_mock_task_markers(outlet, num_trials=4):
    print("[TEST] Sending simulated task markers...")
    outlet.push_sample(["Experiment_Start_Sub_01_Ses_01"], local_clock())
    time.sleep(0.5)

    # Block 1: No Music
    outlet.push_sample(["Block_Start_NoMusic_BlockIdx_1"], local_clock())
    time.sleep(0.5)

    directions = ["Left", "Right"] * (num_trials // 2)
    for i, direction in enumerate(directions):
        t_idx = i + 1
        outlet.push_sample([f"Trial_Start_{t_idx}_Dir_{direction}"], local_clock())
        time.sleep(0.5)

        outlet.push_sample([f"Cue_{direction}"], local_clock())
        time.sleep(1.0)

        outlet.push_sample([f"Task_{direction}"], local_clock())
        time.sleep(2.0)

        outlet.push_sample(["Rest"], local_clock())
        time.sleep(0.5)

        outlet.push_sample([f"Trial_End_{t_idx}"], local_clock())
        time.sleep(0.2)

    outlet.push_sample(["Block_End_BlockIdx_1"], local_clock())
    time.sleep(0.5)

    # Block 2: Music
    outlet.push_sample(["Block_Start_Music_BlockIdx_2"], local_clock())
    outlet.push_sample(["Audio_Started"], local_clock())
    time.sleep(0.5)

    for i, direction in enumerate(directions):
        t_idx = i + 1 + num_trials
        outlet.push_sample([f"Trial_Start_{t_idx}_Dir_{direction}"], local_clock())
        time.sleep(0.5)

        outlet.push_sample([f"Cue_{direction}"], local_clock())
        time.sleep(1.0)

        outlet.push_sample([f"Task_{direction}"], local_clock())
        time.sleep(2.0)

        outlet.push_sample(["Rest"], local_clock())
        time.sleep(0.5)

        outlet.push_sample([f"Trial_End_{t_idx}"], local_clock())
        time.sleep(0.2)

    outlet.push_sample(["Audio_Stopped"], local_clock())
    outlet.push_sample(["Block_End_BlockIdx_2"], local_clock())
    outlet.push_sample(["Experiment_End"], local_clock())
    print("[TEST] Finished sending task markers.")

def main():
    print("=" * 70)
    print(" BIDS Recording Pipeline Synthetic Test ".center(70, "="))
    print("=" * 70)

    # Step 1: Start Mock EEG Streamer process
    mock_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bridges", "mock_lsl_streamer.py"))
    print(f"[*] Launching Mock LSL Streamer: {mock_script}")
    eeg_proc = subprocess.Popen([sys.executable, mock_script])

    # Wait for LSL stream to initialize
    time.sleep(2.5)

    # Step 2: Create Mock Marker Outlet
    info = StreamInfo('MotorImageryMarkers', 'Markers', 1, 0, 'string', 'Test_Marker_001')
    marker_outlet = StreamOutlet(info)
    print("[+] Created mock marker outlet.")

    # Step 3: Initialize BIDS Recorder
    bids_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_bids_dataset")
    recorder = BIDSRecorder(bids_root=bids_root)

    try:
        recorder.connect_streams(timeout=5.0)
        recorder.start_recording()

        # Step 4: Run Task Markers in separate thread
        marker_thread = threading.Thread(target=run_mock_task_markers, args=(marker_outlet, 4))
        marker_thread.start()
        marker_thread.join()

        time.sleep(1.0)

        # Step 5: Stop recording and export BIDS
        out_dir = recorder.stop_recording_and_export_bids(subject_id="01", session_id="01", task_name="leftright")
        print(f"[+] BIDS output directory: {out_dir}")

        # Step 6: Verify required BIDS files exist
        expected_files = [
            os.path.join(bids_root, "dataset_description.json"),
            os.path.join(bids_root, "participants.tsv"),
            os.path.join(out_dir, "sub-01_ses-01_task-leftright_eeg.vhdr"),
            os.path.join(out_dir, "sub-01_ses-01_task-leftright_eeg.eeg"),
            os.path.join(out_dir, "sub-01_ses-01_task-leftright_eeg.vmrk"),
            os.path.join(out_dir, "sub-01_ses-01_task-leftright_events.tsv"),
            os.path.join(out_dir, "sub-01_ses-01_task-leftright_channels.tsv"),
            os.path.join(out_dir, "sub-01_ses-01_task-leftright_eeg.json")
        ]

        print("\n--- BIDS File Verification ---")
        missing_files = []
        for filepath in expected_files:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"[OK] Found: {os.path.relpath(filepath, bids_root)} ({size} bytes)")
            else:
                print(f"[MISSING] MISSING: {filepath}")
                missing_files.append(filepath)

        if len(missing_files) == 0:
            print("\n[SUCCESS] All BIDS compliant dataset files were verified successfully!")
        else:
            print(f"\n[FAILURE] {len(missing_files)} expected BIDS files were missing.")
            sys.exit(1)

    finally:
        print("[*] Terminating Mock EEG streamer process...")
        eeg_proc.terminate()
        eeg_proc.wait()

if __name__ == "__main__":
    main()
