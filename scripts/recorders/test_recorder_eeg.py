"""EEG export path: unit handling, GPYPE_32 naming, housekeeping channels, quality audit."""
import os
import sys

import numpy as np
import pandas as pd
from pylsl import StreamInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multimodal_bids_recorder import MultimodalBIDSRecorder  # noqa: E402


def eeg_info(n, labels, unit):
    info = StreamInfo("gNautilus_EEG", "EEG", n, 250.0, "float32", "gnautilus-test")
    chans = info.desc().append_child("channels")
    for label in labels:
        ch = chans.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", unit)
        ch.append_child_value("type", "EEG")
    return info


def test_generic_32ch_microvolt_stream_exports_with_1020_names_and_flags_flat_channel(tmp_path):
    rng = np.random.default_rng(0)
    labels = [f"{i + 1}" for i in range(32)] + ["Battery"]
    data = rng.normal(0, 10.0, size=(250 * 10, 33))  # 10 s of ~10 uV EEG
    data[:, 5] = 0.0                                  # channel 6 disconnected (flat)
    rec = MultimodalBIDSRecorder(bids_root=str(tmp_path))
    rec.start_time_lsl = 0.0
    item = {'info': eeg_info(33, labels, "microvolts"), 'type': 'EEG', 'srate': 250.0, 'n_ch': 33}
    rec.inlets["gNautilus_EEG"] = rec.streams["gNautilus_EEG"] = item
    rec.data_buffers["gNautilus_EEG"] = data.tolist()
    rec.timestamp_buffers["gNautilus_EEG"] = list(np.arange(len(data)) / 250.0)

    rec.stop_and_export_bids(subject_id="01", session_id="01", task_name="rest")

    chans = pd.read_csv(next(
        os.path.join(d, f) for d, _, fs in os.walk(tmp_path) for f in fs if f.endswith("_channels.tsv")), sep="\t")
    assert list(chans["name"][:3]) == ["Fp1", "Fp2", "AF3"]       # generic labels -> GPYPE_32
    assert chans.loc[chans["name"] == "Battery", "type"].item() == "MISC"
    assert chans.loc[chans["name"] == "F3", "status"].item() == "bad"   # 6th channel, flat
    assert (chans.loc[chans["name"] == "Cz", "status"] == "good").all()
