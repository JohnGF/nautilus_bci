"""Load one BIDS recording as an ``mne.io.Raw`` with resolved channel names.

The raw file is read directly by format with ``mne.io.read_raw``.
``mne_bids.read_raw_bids`` is not used because it rejects several of our
datasets, for example when the channel count in channels.tsv differs from the
header. Channel names and types then come from ``channels.tsv`` through
``montage.resolve_channels``, and events come from ``events.tsv``. If there is
no events.tsv, the file's own annotations are used instead.
"""

import json
import os
from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd

from .montage import BIDS_TO_MNE_TYPE, ChannelResolution, resolve_channels

DEFAULT_LINE_FREQ = 50.0


@dataclass
class LoadedRecording:
    raw: mne.io.BaseRaw
    recording: object  # discover.Recording
    channels: ChannelResolution
    events: pd.DataFrame  # columns: onset, duration, trial_type (+ any extra events.tsv columns)
    events_source: str  # "events.tsv" | "annotations" | "none"
    line_freq: float
    units: list  # distinct units listed in channels.tsv for EEG channels


def read_channels_tsv(path):
    if path is None or not os.path.exists(path):
        return None
    return pd.read_csv(path, sep="\t", dtype={"name": str}, keep_default_na=False)


def read_events_tsv(path):
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=["n/a"])
    if "trial_type" not in df.columns:
        df["trial_type"] = df["value"].astype(str) if "value" in df.columns else ""
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"])
    df["trial_type"] = df["trial_type"].fillna("").astype(str)
    return df.sort_values("onset", kind="stable").reset_index(drop=True)


def _annotations_frame(raw):
    ann = raw.annotations
    offset = raw.first_time if ann.orig_time is not None else 0.0
    return pd.DataFrame({"onset": np.asarray(ann.onset) - offset, "duration": np.asarray(ann.duration),
                         "trial_type": [str(d) for d in ann.description]})


def _line_freq(sidecar_path):
    if sidecar_path:
        try:
            with open(sidecar_path) as fh:
                val = json.load(fh).get("PowerLineFrequency")
            return float(val)
        except (OSError, ValueError, TypeError):
            pass
    return DEFAULT_LINE_FREQ


def load_raw(recording, montage="auto", preload=True):
    """Read ``recording`` (a ``discover.Recording``) and return a ``LoadedRecording``.

    ``montage`` is passed to ``resolve_channels`` (normally ``spec.montage``).
    EEG channels get the ``standard_1020`` positions. Channels marked
    ``status=bad`` in channels.tsv go into ``raw.info['bads']``.
    """
    raw = mne.io.read_raw(recording.eeg_path, preload=preload, verbose="error")
    tsv = read_channels_tsv(recording.channels_path)

    names, types, units, bads = list(raw.ch_names), None, [], []
    if tsv is not None and len(tsv) == len(raw.ch_names):
        names = tsv["name"].astype(str).tolist()
        if "type" in tsv.columns:
            types = [BIDS_TO_MNE_TYPE.get(str(t).upper(), "misc") for t in tsv["type"]]
        if "status" in tsv.columns:
            bads = [n for n, s in zip(names, tsv["status"]) if str(s).lower() == "bad"]
        if "units" in tsv.columns:
            is_eeg = types if types is not None else ["eeg"] * len(names)
            units = sorted({str(u) for u, t in zip(tsv["units"], is_eeg) if t == "eeg"})
    elif types is None:
        types = [mne.channel_type(raw.info, i) for i in range(len(raw.ch_names))]

    res = resolve_channels(names, types, montage=montage)
    raw.set_channel_types(dict(zip(raw.ch_names, res.types)), on_unit_change="ignore", verbose="error")
    raw.rename_channels(dict(zip(raw.ch_names, res.names)), verbose="error")
    raw.info["bads"] = [res.names[names.index(b)] for b in bads]
    if res.eeg_names:
        raw.set_montage("standard_1020", match_case=False, on_missing="ignore", verbose="error")

    events = read_events_tsv(recording.events_path)
    source = "events.tsv"
    if events is None:
        events = _annotations_frame(raw) if len(raw.annotations) else None
        source = "annotations" if events is not None else "none"
    if events is None:
        events = pd.DataFrame(columns=["onset", "duration", "trial_type"])
    line_freq = _line_freq(recording.sidecar_path)
    raw.info["line_freq"] = line_freq
    return LoadedRecording(raw, recording, res, events, source, line_freq, units)
