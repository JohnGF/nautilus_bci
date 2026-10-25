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
sys.path.insert(0, os.path.join(HERE, "..", "bridges"))

from multimodal_bids_recorder import MultimodalBIDSRecorder  # noqa: E402
from refit_wearable_clocks import refit_dataset  # noqa: E402
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


def test_motion_files_keep_analysis_layout_and_gain_sidecars(session):
    root, _ = session
    base = session_dir(root, "motion", "sub-01_ses-01_task-walk_tracksys-IMULwrist")
    df = pd.read_csv(base + "_motion.tsv", sep="\t")
    # Analysis scripts read timestamp_sec then iloc[:, 1:]: device time must not leak in
    assert list(df.columns) == ["timestamp_sec", "Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]
    meta = json.load(open(base + "_motion.json"))
    assert (meta["DeviceID"], meta["TrackedPointLocation"], meta["Units"]["Gyro_Z"]) == ("dev-L_wrist", "L_wrist", "rad/s")
    channels = pd.read_csv(base + "_channels.tsv", sep="\t")
    assert list(channels["type"]) == ["ACCEL"] * 3 + ["GYRO"] * 3
    assert list(channels["component"]) == ["x", "y", "z"] * 2
    assert set(channels["tracked_point"]) == {"L_wrist"}


def test_legacy_stream_exports_with_generic_columns(session):
    root, _ = session
    df = pd.read_csv(session_dir(root, "ppg", "sub-01_ses-01_task-walk_ppg.tsv"), sep="\t")
    assert list(df.columns) == ["timestamp_sec", "ch_1", "ch_2"]


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
    df = pd.read_csv(session_dir(root, "motion", "sub-01_ses-01_task-walk_tracksys-IMULwrist_motion.tsv"), sep="\t")
    err = np.abs(df["timestamp_sec"].to_numpy() - truth["L_wrist"])
    assert err.max() < 2e-3          # including the first seconds, where live was 8 ms off
    assert err[:250].max() < 2e-3
    sidecar = json.load(open(session_dir(root, "motion", "sub-01_ses-01_task-walk_tracksys-IMULwrist_motion.json")))
    assert sidecar["ClockRefit"]["Bursts"] == 60
    # Applying twice starts from the live timestamps again, so nothing moves
    refit_dataset(str(root), apply=True)
    again = pd.read_csv(session_dir(root, "motion", "sub-01_ses-01_task-walk_tracksys-IMULwrist_motion.tsv"), sep="\t")
    assert np.allclose(again["timestamp_sec"], df["timestamp_sec"])
