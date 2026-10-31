"""Baseline-corrected trial epochs and sham-rest epochs (strict rules 1, 2 and 5).

``make_epochs`` cuts epochs out of the *continuous* recording, after any
band-pass. Filtering epochs one by one would add edge artifacts and would let
each epoch's filter see only its own samples. For each trial:

- window: ``anchor_onset + [window.tmin, window.tmax]``;
- baseline: ``baseline_onset + [baseline.tmin, baseline.tmax]`` (the pre-cue
  window when the spec names a baseline event, otherwise relative to the
  anchor). Its per-channel mean is subtracted from the window.

Sham epochs are un-cued ``Rest`` periods with the same window length. Their
baseline has the same length as the trial baseline and ends where the sham
window starts (e.g. Rest + [0, 0.5] s for a Rest + [0.5, 3.5] s window).
Epochs that would run past either end of the recording are dropped and counted.

``EpochSet.onsets`` keeps each window's start time. ``overlap_fraction`` is
the share of epochs whose window overlaps another epoch of the same recording
(e.g. FNF arrows every 0.4 s with a 3 s window). ``decode`` uses it to switch
to purged, time-contiguous folds, because shuffled k-fold on overlapping
epochs puts the same samples in train and test.
"""

from dataclasses import dataclass, field

import mne
import numpy as np


@dataclass
class EpochSet:
    """Epochs of one recording (or several pooled ones; see ``concat``)."""
    X: np.ndarray  # (trials, channels, samples), volts, float32
    y: np.ndarray  # class labels (str)
    groups: np.ndarray  # CV group per trial (recording id; transfer may regroup by subject)
    X_sham: np.ndarray  # (sham epochs, channels, samples)
    y_sham_prev: np.ndarray  # label of the trial before each sham epoch (None if there was none)
    groups_sham: np.ndarray
    onsets: np.ndarray  # window start (s, recording time) per trial
    recordings: np.ndarray  # recording id per trial (never regrouped; onsets are relative to it)
    ch_names: list
    sfreq: float
    tmin: float
    band: tuple  # (l_freq, h_freq) applied before epoching
    wear_time_s: float  # total recording duration
    dropped: dict = field(default_factory=dict)

    @property
    def labels(self):
        return sorted(set(self.y.tolist()))

    @property
    def class_counts(self):
        return {c: int(np.sum(self.y == c)) for c in self.labels}

    @property
    def window_s(self):
        return self.X.shape[-1] / self.sfreq if self.X.size else 0.0

    @property
    def overlap_mask(self):
        """True for epochs whose window overlaps another epoch's window in the same recording."""
        mask = np.zeros(len(self.y), bool)
        for g in np.unique(self.recordings):
            idx = np.flatnonzero(self.recordings == g)
            order = idx[np.argsort(self.onsets[idx])]
            close = np.diff(self.onsets[order]) < self.window_s - 1e-9
            mask[order[1:][close]] = True
            mask[order[:-1][close]] = True
        return mask

    @property
    def overlap_fraction(self):
        return float(self.overlap_mask.mean()) if len(self.y) else 0.0

    @property
    def active_task_s(self):
        return float(len(self.y) * self.window_s)

    def pick_channels(self, names):
        idx = [self.ch_names.index(n) for n in names]
        return self._replace(X=self.X[:, idx], X_sham=self.X_sham[:, idx], ch_names=list(names))

    def pick_labels(self, labels):
        """Keep trials of ``labels``; a sham epoch's previous label outside them becomes None."""
        keep = np.array([lab in labels for lab in self.y], bool)
        prev = np.array([p if p in labels else None for p in self.y_sham_prev], dtype=object)
        return self._replace(X=self.X[keep], y=self.y[keep], groups=self.groups[keep], onsets=self.onsets[keep],
                             recordings=self.recordings[keep],
                             y_sham_prev=prev)

    def _replace(self, **kw):
        d = dict(self.__dict__)
        d.update(kw)
        return EpochSet(**d)


def _cut(data, sfreq, onset, tmin, tmax, n_samples):
    start = int(round((onset + tmin) * sfreq))
    if start < 0 or start + n_samples > data.shape[1]:
        return None
    return data[:, start:start + n_samples]


def make_epochs(raw, table, spec, l_freq=None, h_freq=None, picks="eeg", group=None):
    """Return an ``EpochSet`` from ``raw`` and a ``TrialTable`` built with ``spec``.

    ``raw`` should already have flat channels removed (``qc.apply_qc``).
    ``l_freq``/``h_freq`` band-pass the continuous data first (None = no filter).
    ``group`` names the recording (default ``raw`` filename).
    """
    cond = spec.conditions
    sfreq = float(raw.info["sfreq"])
    ch_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude="bads")] \
        if picks == "eeg" else list(picks)
    data = raw.get_data(picks=ch_names)
    if l_freq is not None or h_freq is not None:
        data = mne.filter.filter_data(data, sfreq, l_freq, h_freq, verbose="error")
    group = group or (raw.filenames[0] if raw.filenames and raw.filenames[0] else "recording")

    n_win = int(round((cond.window.tmax - cond.window.tmin) * sfreq))
    base_len = cond.baseline.tmax - cond.baseline.tmin
    n_base = max(1, int(round(base_len * sfreq)))
    X, y, onsets, dropped = [], [], [], {"trial_out_of_bounds": 0, "sham_out_of_bounds": 0}
    for t in table.trials.itertuples(index=False):
        win = _cut(data, sfreq, t.anchor_onset, cond.window.tmin, cond.window.tmax, n_win)
        base = _cut(data, sfreq, t.baseline_onset, cond.baseline.tmin, cond.baseline.tmax, n_base)
        if win is None or base is None:
            dropped["trial_out_of_bounds"] += 1
            continue
        X.append(win - base.mean(axis=1, keepdims=True))
        y.append(t.label)
        onsets.append(t.anchor_onset + cond.window.tmin)

    Xs, ys = [], []
    if spec.sham is not None and len(table.sham):
        sh = spec.sham
        n_sham = int(round((sh.tmax - sh.tmin) * sfreq))
        if n_sham != n_win:
            raise ValueError(f"sham window ({sh.tmax - sh.tmin:g} s) must match the trial window "
                             f"({cond.window.tmax - cond.window.tmin:g} s) for a like-for-like control")
        for s in table.sham.itertuples(index=False):
            win = _cut(data, sfreq, s.onset, sh.tmin, sh.tmax, n_win)
            base = _cut(data, sfreq, s.onset, sh.tmin - base_len, sh.tmin, n_base)
            if win is None or base is None:
                dropped["sham_out_of_bounds"] += 1
                continue
            Xs.append(win - base.mean(axis=1, keepdims=True))
            ys.append(s.prev_label if isinstance(s.prev_label, str) else None)  # pandas gives NaN for "none"

    shape = (0, len(ch_names), n_win)
    return EpochSet(
        X=np.asarray(X, np.float32).reshape(-1, *shape[1:]), y=np.array(y, dtype=object),
        groups=np.array([group] * len(y), dtype=object),
        X_sham=np.asarray(Xs, np.float32).reshape(-1, *shape[1:]), y_sham_prev=np.array(ys, dtype=object),
        groups_sham=np.array([group] * len(ys), dtype=object), onsets=np.array(onsets, float),
        recordings=np.array([group] * len(y), dtype=object),
        ch_names=ch_names, sfreq=sfreq,
        tmin=cond.window.tmin, band=(l_freq, h_freq), wear_time_s=float(raw.n_times / sfreq),
        dropped={k: v for k, v in dropped.items() if v},
    )


def concat(sets):
    """Pool ``EpochSet``s on their common channels, resampled to the lowest sfreq.

    Channel order follows the first set. Raises if the sets share fewer than
    two channels or have different window lengths.
    """
    sets = [s for s in sets if len(s.y)]
    if not sets:
        raise ValueError("no epochs to pool")
    common = [c for c in sets[0].ch_names if all(c in s.ch_names for s in sets[1:])]
    if len(common) < 2:
        raise ValueError(f"recordings share only {len(common)} channel(s)")
    sfreq = min(s.sfreq for s in sets)
    out = []
    for s in sets:
        s = s.pick_channels(common)
        if s.sfreq != sfreq:
            s = s._replace(X=_resample(s.X, s.sfreq / sfreq), X_sham=_resample(s.X_sham, s.sfreq / sfreq),
                           sfreq=sfreq)
        out.append(s)
    n = {s.X.shape[-1] for s in out}
    if len(n) > 1:
        raise ValueError(f"epoch lengths differ after resampling: {sorted(n)} samples")
    cat = lambda name: np.concatenate([getattr(s, name) for s in out])
    dropped = {}
    for s in out:
        for k, v in s.dropped.items():
            dropped[k] = dropped.get(k, 0) + v
    return EpochSet(X=cat("X"), y=cat("y"), groups=cat("groups"), X_sham=cat("X_sham"),
                    y_sham_prev=cat("y_sham_prev"), groups_sham=cat("groups_sham"), onsets=cat("onsets"), recordings=cat("recordings"),
                    ch_names=common,
                    sfreq=sfreq, tmin=out[0].tmin, band=out[0].band,
                    wear_time_s=float(sum(s.wear_time_s for s in out)), dropped=dropped)


def _resample(X, down):
    n_out = int(round(X.shape[-1] / down))
    if not len(X):
        return np.empty((0, X.shape[1], n_out), np.float32)
    return mne.filter.resample(X.astype(np.float64), down=down, axis=-1, verbose="error").astype(np.float32)
