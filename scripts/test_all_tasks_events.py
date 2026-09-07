import os
import sys
import time
import subprocess
import threading
import pandas as pd
from pylsl import StreamInfo, StreamOutlet, local_clock
from PySide6.QtWidgets import QApplication

base_dir = os.path.dirname(__file__)
for sub in ['recorders', 'tasks', 'bridges', 'visualizers', 'analysis', 'utils']:
    p = os.path.join(base_dir, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from recorders.bids_recorder import BIDSRecorder
from tasks.left_right_task import LeftRightTaskApp
from tasks.music_memory_task import MusicMemoryTaskApp
from tasks.video_dataset_task import VideoDatasetTaskApp

def start_mock_streamer():
    print("[*] Launching Mock LSL Streamer...")
    script_path = os.path.join(base_dir, "bridges", "mock_lsl_streamer.py")
    return subprocess.Popen([sys.executable, script_path])

def run_task_test(task_class, task_name, mock_streamer_proc):
    print(f"\n{'='*60}")
    print(f"[*] Testing Event System for: {task_class.__name__}")
    print(f"{'='*60}")

    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    # 1. Initialize the Task GUI
    task_gui = task_class()
    task_gui.sub_input.setText("99")
    task_gui.ses_input.setText("99")

    # 2. Setup Recorder
    bids_root = os.path.join(base_dir, "test_dev_bids")
    os.makedirs(bids_root, exist_ok=True)
    recorder = BIDSRecorder(bids_root=bids_root, marker_stream_name=task_gui.marker_name)

    # Give LSL time to register the new marker outlet before connecting
    time.sleep(1.0)

    # 3. Start Recording
    print(f"[*] Connecting streams for recording (looking for marker stream: {task_gui.marker_name})...")
    connected = recorder.connect_streams(timeout=3.0)
    if not connected:
        print("[!] Warning: Could not connect to both EEG and Markers. Ensure streamer is running.")
    recorder.start_recording()

    time.sleep(1.0) # Let the recording gather some baseline

    # 4. Inject test markers through the task's native mechanism
    print("[*] Injecting test markers...")
    task_gui.send_marker("TestMarker_1_Start")
    time.sleep(0.5)
    task_gui.send_marker(f"TestMarker_2_{task_name}_Active")
    time.sleep(0.5)
    task_gui.send_marker("TestMarker_3_End")
    time.sleep(0.5)

    # 5. Stop Recording & Export
    print("[*] Stopping recording and exporting to BIDS...")
    out_dir = recorder.stop_recording_and_export_bids(subject_id="99", session_id="99", task_name=task_name)

    # 6. Verify BIDS Events File
    events_file = os.path.join(out_dir, f"sub-99_ses-99_task-{task_name}_events.tsv")
    if not os.path.exists(events_file):
        raise FileNotFoundError(f"Events file was not created: {events_file}")

    df = pd.read_csv(events_file, sep='\t')
    recorded_markers = df['trial_type'].tolist()

    expected_markers = [
        "TestMarker_1_Start",
        f"TestMarker_2_{task_name}_Active",
        "TestMarker_3_End"
    ]

    missing = [m for m in expected_markers if m not in recorded_markers]
    if missing:
        raise ValueError(f"Missing expected markers in TSV for {task_name}: {missing}")

    print(f"[+] Verified {task_class.__name__} successfully saved all expected markers to _events.tsv!")
    print(f"[+] Output verified at: {events_file}")

    task_gui.deleteLater()

def main():
    print("="*60)
    print(" BCI Suite: Automated Event System Validation ")
    print("="*60)

    mock_proc = start_mock_streamer()
    time.sleep(3.0) # wait for streamer to spin up

    try:
        run_task_test(LeftRightTaskApp, "leftright", mock_proc)
        run_task_test(MusicMemoryTaskApp, "musicmemory", mock_proc)
        run_task_test(VideoDatasetTaskApp, "video", mock_proc)

        print("\n" + "="*60)
        print("[SUCCESS] All tasks successfully generated and recorded LSL event markers!")
        print("="*60)
    finally:
        print("[*] Terminating mock streamer...")
        mock_proc.terminate()
        mock_proc.wait()

if __name__ == "__main__":
    main()
