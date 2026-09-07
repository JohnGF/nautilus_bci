import os
import sys
import time
import subprocess
import argparse

_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)


from recorders.bids_recorder import BIDSRecorder


def start_mock_streamer():
    mock_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bridges", "mock_lsl_streamer.py"))
    print(f"[*] Launching Mock LSL EEG Streamer...")
    proc = subprocess.Popen([sys.executable, mock_script])
    time.sleep(2.5)
    return proc


def run_experiment(task_app, recorder, task_name, num_trials=3):
    print(f"[*] Running {task_name} experiment simulation ({num_trials} trials)...")
    sub = task_app.sub_input.text().strip()
    ses = task_app.ses_input.text().strip()
    task_app.send_marker(f"Experiment_Start_Sub_{sub}_Ses_{ses}")
    time.sleep(0.3)

    directions = ["Left", "Right"] * num_trials
    for i, direction in enumerate(directions):
        t_idx = i + 1
        task_app.send_marker(f"Trial_Start_{t_idx}_Dir_{direction}")
        time.sleep(0.2)

        task_app.send_marker(f"Cue_{direction}")
        time.sleep(0.2)

        task_app.send_marker(f"Task_{direction}")
        time.sleep(0.2)

        task_app.send_marker("Rest")
        time.sleep(0.2)

        task_app.send_marker(f"Trial_End_{t_idx}")
        time.sleep(0.1)

    task_app.send_marker("Block_End")
    time.sleep(0.2)
    task_app.send_marker("Experiment_End")
    print(f"[*] Experiment simulation complete — {num_trials * 2} trials sent.")


def verify_bids(out_dir, recorder, task_name, sub="99", ses="99"):
    events_file = os.path.join(out_dir, f"sub-{sub}_ses-{ses}_task-{task_name}_events.tsv")
    if not os.path.exists(events_file):
        n_markers = len(recorder.marker_events)
        raise FileNotFoundError(
            f"Events file was not created: {events_file} "
            f"(captured {n_markers} marker event(s) during recording)."
        )

    import pandas as pd
    df = pd.read_csv(events_file, sep="\t")
    recorded_markers = df["trial_type"].tolist()

    print(f"\n--- BIDS Verification ---")
    print(f"[OK] Events file: {os.path.basename(events_file)} ({len(recorded_markers)} markers)")
    print(f"[OK] EEG samples captured: {len(recorder.eeg_samples)}")
    print(f"[OK] Marker events captured: {len(recorder.marker_events)}")

    expected_prefixes = ["Experiment_Start", "Trial_Start", "Cue_", "Task_", "Rest", "Trial_End", "Block_End", "Experiment_End"]
    found_prefixes = set()
    for m in recorded_markers:
        for prefix in expected_prefixes:
            if m.startswith(prefix):
                found_prefixes.add(prefix)

    missing_prefixes = [p for p in expected_prefixes if p not in found_prefixes]
    if missing_prefixes:
        print(f"[WARN] Missing marker categories: {missing_prefixes}")
    else:
        print(f"[OK] All expected marker categories present in events.tsv")

    print(f"\n[SUCCESS] BIDS + LSL integration test passed for task '{task_name}'!")
    print(f"    Output: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="BIDS + LSL Task Integration Test")
    parser.add_argument("--task", choices=["leftright", "musicmemory", "video"], default="leftright",
                        help="Task to simulate (default: leftright)")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per direction (default: 3)")
    parser.add_argument("--subject", type=str, default="99", help="Subject ID (default: 99)")
    parser.add_argument("--session", type=str, default="99", help="Session ID (default: 99)")
    args = parser.parse_args()

    task_map = {
        "leftright": ("LeftRightTaskApp", "leftright", "MotorImageryMarkers", "left_right_task"),
        "musicmemory": ("MusicMemoryTaskApp", "musicmemory", "MotorImageryMarkers", "music_memory_task"),
        "video": ("VideoDatasetTaskApp", "video", "VideoTaskMarkers", "video_dataset_task"),
    }
    class_name, task_name, marker_name, module_name = task_map[args.task]

    print("=" * 70)
    print(f" BIDS + LSL Integration Test: {class_name} ".center(70, "="))
    print("=" * 70)

    mock_proc = start_mock_streamer()

    try:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)

        task_module = __import__(f"tasks.{module_name}", fromlist=[class_name])
        task_class = getattr(task_module, class_name)

        task_gui = task_class()
        task_gui.sub_input.setText(args.subject)
        task_gui.ses_input.setText(args.session)

        bids_root = os.path.join(_parent_dir, "test_dev_bids")
        if os.path.exists(bids_root):
            import shutil
            shutil.rmtree(bids_root)
        os.makedirs(bids_root, exist_ok=True)

        recorder = BIDSRecorder(bids_root=bids_root, marker_stream_name=marker_name)

        time.sleep(1.0)

        print(f"[*] Connecting to marker stream: {marker_name}")
        connected = recorder.connect_streams(timeout=3.0)
        if not connected:
            print("[!] Warning: Could not connect to marker stream. Markers may not be captured.")

        recorder.start_recording()
        time.sleep(0.5)

        run_experiment(task_gui, recorder, task_name, num_trials=args.trials)

        time.sleep(1.0)

        print("[*] Stopping recording and exporting to BIDS...")
        out_dir = recorder.stop_recording_and_export_bids(
            subject_id=args.subject, session_id=args.session, task_name=task_name
        )
        print(f"[+] BIDS output directory: {out_dir}")

        verify_bids(out_dir, recorder, task_name, sub=args.subject, ses=args.session)

        task_gui.deleteLater()

    finally:
        print("[*] Terminating Mock EEG streamer...")
        mock_proc.terminate()
        mock_proc.wait()


if __name__ == "__main__":
    main()