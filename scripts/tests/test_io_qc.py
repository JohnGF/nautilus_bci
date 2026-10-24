import mne
import numpy as np
import pytest

from nautilus_analysis import QCRefusal, apply_qc, compute_qc, discover, gate, load_raw
from nautilus_analysis.montage import GPYPE_32
from conftest import LIVE_IN_DEAD, real_dataset


@pytest.fixture(scope="module")
def loaded(synthetic_bids):
    return {r.id.rsplit("_task", 1)[0]: load_raw(r) for r in discover(synthetic_bids)}


def test_load_gives_real_names_montage_and_events(loaded):
    for key, L in loaded.items():
        raw = L.raw
        assert raw.ch_names[:32] == GPYPE_32 and raw.get_channel_types()[32] == "misc"
        assert L.channels.label == "montage assumed: gpype_32"
        assert raw.get_montage() is not None and L.line_freq == 50.0
        assert L.events_source == "events.tsv"
        assert (L.events["trial_type"] == "Task_Left").sum() >= 8  # 8 main-block + false-start trials
        assert L.units == ["µV"]
    # EEGLAB and BrainVision round-trip to the same scale (µV-level signal)
    for key in ("sub-02_ses-01", "sub-02_ses-02"):
        std = loaded[key].raw.get_data(picks="eeg").std(axis=1) * 1e6
        assert 5 < np.median(std) < 30


def test_good_recordings_pass_the_gate(loaded):
    for key, L in loaded.items():
        if key == "sub-03_ses-01":
            continue
        rep = gate(compute_qc(L.raw, key), min_live_channels=16)
        assert rep.status in ("ok", "warn") and rep.n_live == 32 and not rep.flat_channels
        assert rep.alpha_uv2_hz > 1 and not rep.probably_disconnected


def test_dead_recording_is_refused_with_clear_message(loaded):
    rep = compute_qc(loaded["sub-03_ses-01"].raw, "sub-03_ses-01")
    assert rep.live_channels == [GPYPE_32[i] for i in LIVE_IN_DEAD]
    assert len(rep.flat_channels) == 29
    with pytest.raises(QCRefusal, match=r"only 3/32 EEG channels carry signal .* need >= 8"):
        gate(rep, min_live_channels=8)
    assert rep.status == "refused"
    # The battery channel is misc, so QC never counts it as EEG.
    assert "EEG033" not in rep.channel_std_uv


def test_apply_qc_drops_flat_channels(loaded):
    raw = loaded["sub-03_ses-01"].raw
    rep = compute_qc(raw)
    clean = apply_qc(raw, rep)
    assert [c for c in clean.ch_names if c in GPYPE_32] == rep.live_channels
    assert len(raw.ch_names) == 33  # original untouched


def _raw(data_uv, sfreq=250.0):
    raw = mne.io.RawArray(np.asarray(data_uv) * 1e-6, mne.create_info(len(data_uv), sfreq, "eeg"), verbose="error")
    raw.info["line_freq"] = 50.0
    return raw


def test_drift_only_cap_is_probably_disconnected():
    # music_lrud sub-06: every channel "live" (std >> 0.5 µV) but only slow drift, alpha ~0.03 µV²/Hz
    rng = np.random.default_rng(0)
    n = 250 * 120
    drift = np.cumsum(rng.standard_normal((32, n)), axis=1) * 0.05
    rep = compute_qc(_raw(drift), "drift")
    assert rep.n_live == 32 and rep.probably_disconnected and rep.alpha_uv2_hz < 0.1
    with pytest.raises(QCRefusal, match="probably disconnected"):
        gate(rep, min_live_channels=8)
    rep = gate(compute_qc(_raw(drift)), min_live_channels=8, allow_implausible=True)
    assert rep.status == "warn"


def test_short_and_line_noise():
    t = np.arange(int(250 * 60)) / 250
    rng = np.random.default_rng(1)
    noisy = rng.standard_normal((16, t.size)) * 10 + 200 * np.sin(2 * np.pi * 50 * t)
    rep = gate(compute_qc(_raw(noisy)), min_live_channels=8, allow_implausible=True)
    assert any("50 Hz line noise" in w for w in rep.warnings)
    with pytest.raises(QCRefusal, match="only 2.0 s"):
        gate(compute_qc(_raw(rng.standard_normal((16, 500)) * 10)), min_live_channels=8)


@pytest.mark.realdata
def test_real_music_lrud_dead_sessions_are_refused():
    root = real_dataset("bids/music_lrud")
    by_id = {r.sub + r.ses: r for r in discover(root)}
    rep = compute_qc(load_raw(by_id["0402"]).raw, "sub-04_ses-02")
    assert rep.n_live == 3
    with pytest.raises(QCRefusal, match="only 3/32"):
        gate(rep, min_live_channels=8)
    rep = compute_qc(load_raw(by_id["0601"]).raw, "sub-06_ses-01")
    assert rep.n_live == 32 and rep.alpha_uv2_hz < 0.1
    with pytest.raises(QCRefusal, match="probably disconnected"):
        gate(rep, min_live_channels=8)


@pytest.mark.realdata
def test_real_good_recording_passes():
    root = real_dataset("bids_clean/bids_td_classical")
    rec = discover(root, ses="04")[0]
    rep = gate(compute_qc(load_raw(rec).raw, rec.id), min_live_channels=16)
    assert rep.n_live == 32 and rep.status in ("ok", "warn")
