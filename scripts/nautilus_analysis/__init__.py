"""Dataset-agnostic EEG analysis library ("if it's BIDS, it runs").

Pure functions with no hard-wired paths, subjects or event names. Everything
dataset-specific comes from an ``AnalysisSpec`` (see ``spec.py``). Plan:
``docs/plans/BIDS_AGNOSTIC_PIPELINES.md``.

Typical flow::

    from nautilus_analysis import discover, load_raw, compute_qc, gate, load_spec, build_trial_table
    recs = discover("bids/music_lrud", sub="04")
    spec = load_spec("bids/music_lrud", events_frames=[load_raw(r, preload=False).events for r in recs])
    loaded = load_raw(recs[0], montage=spec.montage)
    qc = gate(compute_qc(loaded.raw, recs[0].id), min_live_channels=8)   # raises QCRefusal
    table = build_trial_table(loaded.events, spec)
"""

from .discover import Recording, discover, find_datasets
from .events import TrialTable, build_trial_table
from .io import LoadedRecording, load_raw
from .montage import GPYPE_32, ChannelResolution, resolve_channels
from .qc import QCRefusal, QCReport, QCThresholds, apply_qc, compute_qc, gate
from .spec import AnalysisSpec, SpecError, infer_spec, load_spec, read_spec

__all__ = [
    "Recording", "discover", "find_datasets", "TrialTable", "build_trial_table", "LoadedRecording",
    "load_raw", "GPYPE_32", "ChannelResolution", "resolve_channels", "QCRefusal", "QCReport",
    "QCThresholds", "apply_qc", "compute_qc", "gate", "AnalysisSpec", "SpecError", "infer_spec",
    "load_spec", "read_spec",
]
