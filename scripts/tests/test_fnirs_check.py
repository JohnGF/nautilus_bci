"""fNIRS check tool: saved recordings to the combined report. Uses no hardware."""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import fnirs_check as check  # noqa: E402

SRATE = 50.0
LABELS = [f"R{1 if tx <= 4 else 2}_T{tx}_{wl}" for tx in range(1, 9) for wl in ("760nm", "850nm")]


def save_run(folder, name, seconds=60, watch_bpm=84.0, pulse_channels=range(4), stamp=None):
    rng = np.random.default_rng(3)
    n = int(seconds * SRATE)
    t = np.arange(n) / SRATE
    frames = rng.normal(0, 0.0005, (n, 16))
    wave = 0.004 * np.sin(2 * np.pi * (84.0 / 60.0) * t)
    for ch in pulse_channels:
        frames[:, 2 * ch] += wave
        frames[:, 2 * ch + 1] += 0.8 * wave
    frames = np.round(frames / 1e-4) * 1e-4
    heart_times = np.arange(seconds) + 0.5 if watch_bpm else np.array([])
    heart = np.full(len(heart_times), watch_bpm or 0.0)
    if len(heart):
        heart[0] = 0.0  # The watch sends 0 while it has no reading
    stamp = stamp or f"{datetime.now():%Y%m%d}_120000"
    meta = {"label": name, "started": "2026-11-06T12:00:00", "srate": SRATE, "labels": LABELS,
            "units": ["OD"] * 16, "startup_od": ["1.300"] * 16, "watch": "HR_L_wrist" if watch_bpm else None}
    path = folder / f"{stamp}_{name}.npz"
    np.savez_compressed(path, frames=frames.astype(np.float32), times=t, heart=heart.astype(np.float32),
                        heart_times=heart_times, meta=json.dumps(meta))
    return path


def test_analyse_separates_good_channels_from_bad_and_ignores_watch_zeros(tmp_path):
    run = check.analyse(save_run(tmp_path, "good"))
    assert run["contact"][:4].min() > 0.9 > run["contact"][4:].max()
    assert run["good_share"][0] == 1.0 and run["good_share"][7] == 0.0
    assert np.nanmedian(run["peaks"]) == pytest.approx(84.0, abs=6.0)
    assert run["watch_range"] == (84.0, 84.0)
    assert run["watch"][0]["agree"] == 1.0


def test_watch_comparison_finds_a_delay():
    centres = np.arange(0.0, 200.0, 5.0)
    peaks = 75.0 + 15.0 * np.sin(centres / 25.0)
    heart_times = np.arange(-20.0, 240.0)
    heart = 75.0 + 15.0 * np.sin((heart_times - 10.0) / 25.0)
    scores = check.compare_with_watch(centres, peaks, heart_times, heart)
    assert scores[10]["r"] > scores[0]["r"] and scores[10]["r"] > scores[15]["r"]


def test_short_recording_is_reported_not_crashed(tmp_path):
    run = check.analyse(save_run(tmp_path, "short", seconds=10))
    assert run["windows"] == 0
    assert any("Too short" in line for line in check.format_run(run))
    assert check.format_summary([run]) == []


def test_report_without_a_watch_or_pulse(tmp_path, capsys):
    save_run(tmp_path, "dark", watch_bpm=None, pulse_channels=[])
    assert check.report(tmp_path, None, False) == 0
    printed = capsys.readouterr().out
    assert "Watch: no readings" in printed and "none found" in printed


def test_report_combines_recordings_and_saves_a_copy(tmp_path, capsys):
    save_run(tmp_path, "one", stamp=f"{datetime.now():%Y%m%d}_100000")
    save_run(tmp_path, "two", stamp=f"{datetime.now():%Y%m%d}_110000")
    assert check.report(tmp_path, None, False) == 0
    printed = capsys.readouterr().out
    assert "2 recording(s)" in printed and "ACROSS ALL RECORDINGS" in printed and "2 of 2" in printed
    assert "0 of 2" in printed
    saved = list(tmp_path.glob("report_*.txt"))
    assert len(saved) == 1 and "ACROSS ALL RECORDINGS" in saved[0].read_text(encoding="utf-8")


def test_report_chooses_today_last_or_all(tmp_path, capsys):
    save_run(tmp_path, "old", stamp="20200101_100000")
    save_run(tmp_path, "new")
    check.report(tmp_path, None, False)
    assert "1 recording(s)" in capsys.readouterr().out
    check.report(tmp_path, None, True)
    assert "2 recording(s)" in capsys.readouterr().out
    check.report(tmp_path, 1, False)
    assert "1 recording(s)" in capsys.readouterr().out


def test_report_with_nothing_to_show(tmp_path, capsys):
    assert check.report(tmp_path, None, False) == 1
    assert "No recordings" in capsys.readouterr().out


class FakeStream:
    def __init__(self, name, stream_type):
        self._name, self._type = name, stream_type

    def name(self):
        return self._name

    def type(self):
        return self._type


def test_find_stream_keeps_searching_after_a_miss():
    answers = [[], [FakeStream("Artinis_fNIRS", "fNIRS")]]
    assert check.find_stream(lambda wait: answers.pop(0), check.is_fnirs, 30.0).name() == "Artinis_fNIRS"


def test_control_and_watch_streams_are_not_taken_for_fnirs():
    assert not check.is_fnirs(FakeStream("OptodeBaselineControl", "Control"))
    assert not check.is_fnirs(FakeStream("HR_L_wrist", "PPG"))
    assert check.is_watch(FakeStream("HR_L_wrist", "PPG"))


def test_default_folder_is_outside_the_public_repo():
    assert "nautilus_bci" not in check.DEFAULT_OUT.parts
