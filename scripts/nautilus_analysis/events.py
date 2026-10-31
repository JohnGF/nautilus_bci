"""events.tsv -> trial table, using only the spec's rules.

Labels come only from within-session event markers (strict rule 1). File,
session and task-entity membership are never used. A single pass over the
events, in onset order, tracks:

- the current paradigm *mode* (``segments.mode_regex``, e.g. ``_Mode_AudioEyesClosed``);
- the current *block* (a counter that ``segments.block_regex`` increments; group 1 is the block kind);
- the most recent *baseline* event (``conditions.baseline.event_regex``), consumed by one trial;
- *label* events (``conditions.label_regex``). A trial is emitted at the label
  event, or at the next ``anchor_regex`` event within ``max_label_lag_s``.

Mixed-mode sessions (false starts in another mode) are never pooled. Only
``segments.select_mode`` is kept, or, if that is None, the mode with the most
trials, with a warning. ``main_block_only`` also keeps only that mode's
largest block.
"""

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRIAL_COLUMNS = ["label", "anchor_onset", "label_onset", "baseline_onset", "mode", "block", "block_kind"]
SHAM_COLUMNS = ["onset", "prev_label", "mode", "block", "block_kind"]
_TIE_S = 0.05  # events this close together count as simultaneous


@dataclass
class TrialTable:
    trials: pd.DataFrame
    sham: pd.DataFrame
    selected_mode: str | None
    selected_block: int | None
    modes_found: dict  # mode -> number of trials before selection
    dropped: dict = field(default_factory=dict)  # reason -> count
    warnings: list = field(default_factory=list)

    @property
    def labels(self):
        return sorted(self.trials["label"].unique())

    @property
    def class_counts(self):
        return {k: int(v) for k, v in self.trials["label"].value_counts().sort_index().items()}


def _rx(pattern):
    return re.compile(pattern) if pattern else None


def _captured(match):
    return match.group(1) if match.re.groups else match.group(0)


def build_trial_table(events, spec):
    """Turn an events DataFrame (onset, trial_type) into a ``TrialTable`` using ``spec``."""
    cond = spec.conditions
    if cond is None:
        raise ValueError("spec has no 'conditions'; cannot build trials (set conditions.label_regex)")
    label_rx, anchor_rx = _rx(cond.label_regex), _rx(cond.anchor_regex)
    base_rx = _rx(cond.baseline.event_regex)
    mode_rx, block_rx = _rx(spec.segments.mode_regex), _rx(spec.segments.block_regex)
    sham_rx = _rx(spec.sham.event_regex) if spec.sham else None

    ev = events.sort_values("onset", kind="stable")
    mode, block, block_kind = None, -1 if block_rx else 0, None
    last_base = None
    pending_label = pending_anchor = None  # (label, onset) / onset
    last_label = None
    trials, shams = [], []
    dropped = {"label_without_anchor": 0, "anchor_without_label": 0, "no_baseline_event": 0}

    def emit(label, label_onset, anchor_onset):
        nonlocal last_base, last_label
        base = anchor_onset
        if base_rx is not None:
            if last_base is None or last_base > anchor_onset + _TIE_S:
                dropped["no_baseline_event"] += 1
                return
            base, last_base = last_base, None
        trials.append(dict(label=cond.label_map.get(label, label), anchor_onset=anchor_onset,
                           label_onset=label_onset, baseline_onset=base, mode=mode, block=block,
                           block_kind=block_kind))
        last_label = trials[-1]["label"]

    for onset, tt in zip(ev["onset"].astype(float), ev["trial_type"].astype(str)):
        if mode_rx and (m := mode_rx.search(tt)):
            mode = _captured(m)
        if block_rx and (m := block_rx.search(tt)):
            block += 1
            block_kind = _captured(m)
        if base_rx and base_rx.search(tt):
            last_base = onset
        if pending_label and onset - pending_label[1] > cond.max_label_lag_s:
            dropped["label_without_anchor"] += 1
            pending_label = None
        if pending_anchor is not None and onset - pending_anchor > _TIE_S:
            dropped["anchor_without_label"] += 1
            pending_anchor = None
        if m := label_rx.search(tt):
            label = _captured(m)
            if anchor_rx is None:
                emit(label, onset, onset)
            elif pending_anchor is not None:  # anchor logged just before its label
                emit(label, onset, pending_anchor)
                pending_anchor = None
            else:
                pending_label = (label, onset)
        if anchor_rx and anchor_rx.search(tt):
            if pending_label:
                emit(pending_label[0], pending_label[1], onset)
                pending_label = None
            else:
                pending_anchor = onset
        if sham_rx and sham_rx.search(tt):
            shams.append(dict(onset=onset, prev_label=last_label, mode=mode, block=block, block_kind=block_kind))
    if pending_label:
        dropped["label_without_anchor"] += 1

    trials = pd.DataFrame(trials, columns=TRIAL_COLUMNS)
    sham = pd.DataFrame(shams, columns=SHAM_COLUMNS)
    if spec.sham is not None and spec.sham.must_end_before_regex and len(sham):
        stop_rx = _rx(spec.sham.must_end_before_regex)
        stops = np.sort(ev.loc[ev["trial_type"].astype(str).map(lambda t: bool(stop_rx.search(t))), "onset"]
                        .to_numpy(float))
        keep = []
        for onset in sham["onset"]:
            i = np.searchsorted(stops, onset, side="right")
            keep.append(i >= len(stops) or stops[i] >= onset + spec.sham.tmax)
        dropped["sham_overlaps_next_trial"] = int(len(keep) - sum(keep))
        sham = sham[np.array(keep, bool)].reset_index(drop=True)

    warnings = []
    modes_found = {str(k): int(v) for k, v in trials["mode"].value_counts(dropna=False).items()}
    selected_mode, selected_block = spec.segments.select_mode, None
    if mode_rx is not None and len(trials):
        if selected_mode is None:
            selected_mode = trials["mode"].value_counts().idxmax()
            if len(modes_found) > 1:
                warnings.append(f"mixed-mode session {modes_found}; kept only mode {selected_mode!r} "
                                "(set segments.select_mode to choose)")
        dropped["other_mode"] = int((trials["mode"] != selected_mode).sum())
        trials = trials[trials["mode"] == selected_mode]
        sham = sham[sham["mode"] == selected_mode]
        if trials.empty:
            warnings.append(f"no trials in mode {selected_mode!r}; modes present: {modes_found}")
    if spec.segments.main_block_only and len(trials):
        selected_block = int(trials["block"].value_counts().idxmax())
        dropped["other_block"] = int((trials["block"] != selected_block).sum())
        trials = trials[trials["block"] == selected_block]
        sham = sham[sham["block"] == selected_block]
    return TrialTable(trials.reset_index(drop=True), sham.reset_index(drop=True), selected_mode,
                      selected_block, modes_found, {k: v for k, v in dropped.items() if v}, warnings)
