"""Wearable export + offline clock refit, end to end without a network:
uv run pytest recorders/test_recorder_wearables.py"""
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import pytest
from pylsl import StreamInfo

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "bridges"))

from multimodal_bids_recorder import MultimodalBIDSRecorder  # noqa: E402
from refit_wearable_clocks import refit_dataset  # noqa: E402
from utils.bids_wearables import find_wearable_file, find_wearable_files, read_wearable  # noqa: E402
from wearable_protocol import (DEVICE_TIME_CHANNEL, KNOWN_KINDS, SYNC_CHANNELS,  # noqa: E402
                               make_stream_info)

DRIFT = 40e-6          # device crystal 40 ppm fast
DEVICE_ORIGIN = 777.0  # device clock reads 777 s at recording start
BRIDGE_SHIFT = 123.4   # bridge host's LSL clock vs the recorder's
START = 5000.0         # recorder's start_time_lsl


def device_time(rec_t):
    return DEVICE_ORIGIN + rec_t * (1 + DRIFT)


def add_stream(recorder, info, rows, stamps):
    name = info.name()
    recorder.streams[name] = {'info': info, 'type': info.type().upper(), 'srate': info.nominal_srate(),
                              'n_ch': info.channel_count()}
    recorder.data_buffers[name] = [list(r) for r in rows]
    recorder.timestamp_buffers[name] = list(stamps)


@pytest.fixture
def session(tmp_path):
    rng = random.Random(7)
    recorder = MultimodalBIDSRecorder(bids_root=str(tmp_path))
    recorder.start_time_lsl = START

    # 10 min of 50 Hz IMU per wrist. Live timestamps: 8 ms off until the first burst
    # (fallback mapping), then +-1 ms, like the bridge's running fit.
    truth = {}
    for loc in ("L_wrist", "R_wrist"):
        info = make_stream_info(f"IMU_{loc}", "IMU", f"dev-{loc}_IMU", 50.0,
                                KNOWN_KINDS["IMU"][1] + [DEVICE_TIME_CHANNEL], f"dev-{loc}", loc, "double64")
        t = np.arange(0, 600, 0.02)
        live = t + np.where(t < 5, 0.008, np.array([rng.uniform(-1e-3, 1e-3) for _ in t]))
        rows = [[0.0, 0.0, 9.81, 0.0, 0.0, 0.0, device_time(ti)] for ti in t]
        add_stream(recorder, info, rows, live)
        truth[loc] = t

        # Ping bursts (40, then 5 every 10 s), 1-15 ms per leg; values in the bridge's clock
        sync = make_stream_info(f"ClockSync_{loc}", "ClockSync", f"dev-{loc}_ClockSync", 0.0,
                                SYNC_CHANNELS, f"dev-{loc}", loc, "double64")
        rows, stamps = [], []
        for burst_start in [1.0] + list(np.arange(11.0, 600, 10.0)):
            for k in range(40 if burst_start == 1.0 else 5):
                t1 = burst_start + k * 0.1
                up, down = rng.uniform(1e-3, 15e-3), rng.uniform(1e-3, 15e-3)
                t2 = device_time(t1 + up)
                t4 = t1 + up + 2e-4 + down
                rows.append([t1 + START + BRIDGE_SHIFT, t2, t2 + 2e-4, t4 + START + BRIDGE_SHIFT])
                stamps.append(t4)  # recorder's proc_clocksync already removed BRIDGE_SHIFT
        add_stream(recorder, sync, rows, stamps)

    # Legacy single-watch stream: no channel metadata, no device clock
    legacy = StreamInfo("Smartwatch_PPG", "PPG", 2, 1.0, "float32", "smartwatch_ppg_01")
    add_stream(recorder, legacy, [[70.0, 100.0]] * 10, np.arange(10.0))

    recorder.stop_and_export_bids(subject_id="01", session_id="01", task_name="walk")
    return tmp_path, truth


def session_dir(root, *parts):
    return os.path.join(root, "sub-01", "ses-01", *parts)


def test_motion_is_strict_bids_and_reads_back_for_analysis(session):
    root, _ = session
    base = session_dir(root, "motion", "sub-01_ses-01_task-walk_tracksys-IMULwrist")
    with open(base + "_motion.tsv") as f:
        assert not f.readline().startswith("timestamp_sec")  # BIDS: no header row
    df = read_wearable(base + "_motion.tsv")
    # Analysis scripts use timestamp_sec then iloc[:, 1:]: device time must not leak in
    assert list(df.columns) == ["timestamp_sec", "Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]
    channels = pd.read_csv(base + "_channels.tsv", sep="\t", keep_default_na=False)
    assert list(channels["type"]) == ["LATENCY"] + ["ACCEL"] * 3 + ["GYRO"] * 3
    assert list(channels["component"]) == ["n/a"] + ["x", "y", "z"] * 2
    assert list(channels["tracked_point"]) == ["n/a"] + ["L_wrist"] * 6
    meta = json.load(open(base + "_motion.json"))
    assert (meta["MotionChannelCount"], meta["ACCELChannelCount"], meta["GYROChannelCount"]) == (7, 3, 3)
    assert (meta["DeviceID"], meta["TrackedPointLocation"], meta["Units"]["Gyro_Z"]) == ("dev-L_wrist", "L_wrist", "rad/s")
    assert os.path.exists(os.path.join(root, "dataset_description.json"))


def test_physio_without_eeg_goes_to_beh_gzipped(session):
    root, _ = session
    base = session_dir(root, "beh", "sub-01_ses-01_task-walk_recording-SmartwatchPPG_physio")
    df = read_wearable(base + ".tsv.gz")
    assert list(df.columns) == ["timestamp_sec", "ch_1", "ch_2"]
    meta = json.load(open(base + ".json"))
    assert meta["Columns"] == ["timestamp_sec", "ch_1", "ch_2"]
    assert meta["StartTime"] == 0.0 and meta["SamplingFrequency"] == 1.0


def test_finder_picks_strict_files_and_labels(session):
    root, _ = session
    ses = session_dir(root)
    assert [os.path.basename(p) for p in find_wearable_files(ses, "01", "01", "walk", "motion")] == [
        "sub-01_ses-01_task-walk_tracksys-IMULwrist_motion.tsv", "sub-01_ses-01_task-walk_tracksys-IMURwrist_motion.tsv"]
    assert find_wearable_file(ses, "01", "01", "walk", "motion", label="IMURwrist").endswith("IMURwrist_motion.tsv")
    assert find_wearable_file(ses, "01", "01", "walk", "physio").endswith("SmartwatchPPG_physio.tsv.gz")


def test_older_headered_layouts_still_load(tmp_path):
    ses = tmp_path / "sub-01" / "ses-02"
    for folder, name, cols in [("motion", "motion", "timestamp_sec\tch_1\tch_2"),
                               ("physio", "physio", "timestamp_sec\tch_1"),
                               ("ppg", "ppg", "timestamp_sec\tch_1")]:
        (ses / folder).mkdir(parents=True)
        (ses / folder / f"sub-01_ses-02_task-video_{name}.tsv").write_text(cols + "\n" + "\t".join(["0.5"] * len(cols.split())) + "\n")
    motion = find_wearable_file(str(ses), "01", "02", "video", "motion")
    assert list(read_wearable(motion).columns) == ["timestamp_sec", "ch_1", "ch_2"]
    physio = find_wearable_files(str(ses), "01", "02", "video", "physio")
    assert [os.path.basename(os.path.dirname(p)) for p in physio] == ["physio", "ppg"]


def test_clock_sync_lands_on_recording_axis(session):
    root, _ = session
    path = os.path.join(root, "sourcedata", "sub-01", "ses-01", "clocksync",
                        "sub-01_ses-01_task-walk_recording-ClockSyncLwrist_clocksync.tsv")
    ex = pd.read_csv(path, sep="\t")
    assert ex["t1_ping_sent"].iloc[0] == pytest.approx(1.0, abs=1e-6)  # bridge host shift removed


def test_refit_beats_live_timestamps(session):
    root, truth = session
    reports = refit_dataset(str(root), apply=True)
    assert [r["status"] for r in reports] == ["applied", "applied"]
    assert reports[0]["device_drift_ppm"] == pytest.approx(40, abs=5)  # device made 40 ppm fast
    df = read_wearable(session_dir(root, "motion", "sub-01_ses-01_task-walk_tracksys-IMULwrist_motion.tsv"))
    err = np.abs(df["timestamp_sec"].to_numpy() - truth["L_wrist"])
    assert err.max() < 2e-3          # including the first seconds, where live was 8 ms off
    assert err[:250].max() < 2e-3
    sidecar = json.load(open(session_dir(root, "motion", "sub-01_ses-01_task-walk_tracksys-IMULwrist_motion.json")))
    assert sidecar["ClockRefit"]["Bursts"] == 60
    # Applying twice starts from the live timestamps again, so nothing moves
    refit_dataset(str(root), apply=True)
    again = read_wearable(session_dir(root, "motion", "sub-01_ses-01_task-walk_tracksys-IMULwrist_motion.tsv"))
    assert np.allclose(again["timestamp_sec"], df["timestamp_sec"])


def test_refit_of_short_session_reports_no_drift():
    from refit_wearable_clocks import fit_offsets
    x = np.array([0.0, 10.0])
    y = np.array([1.000, 1.002])          # 2 ms of noise, 10 s apart
    delay = np.array([0.004, 0.012])
    offsets, drift = fit_offsets(x, y, delay, np.array([0.0, 5.0, 20.0]))
    assert drift is None
    assert np.allclose(offsets, 1.000)    # fastest exchange, no extrapolated slope
