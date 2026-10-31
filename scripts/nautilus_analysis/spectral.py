"""PSD, band power and ERD% (event-related desynchronisation).

``psd`` is a whole-recording Welch spectrum per live EEG channel (µV²/Hz).
``erd`` follows Pfurtscheller's convention, with the windows taken from the
spec: band-pass the continuous recording, take the average reference, compute
Hilbert-envelope power, average the power over trials, *then* compute
``ERD% = (task - baseline) / baseline * 100``. Task is ``anchor + window``;
baseline is ``baseline_onset + conditions.baseline``. Negative values mean a
power drop (desynchronisation). Averaging per-trial ratios instead, as the
legacy scripts did, biases ERD% upwards because a noisy short baseline sits
in the denominator.

``erd_stats`` tests each (class, channel) cell against 0 with a one-sample
t-test on log(task / baseline), FDR-corrected over all cells.
"""

from dataclasses import dataclass

import mne
import numpy as np
from mne.stats import fdr_correction
from scipy.signal import hilbert, welch
from scipy.stats import ttest_1samp


def psd(raw, fmin=1.0, fmax=45.0, seg_s=2.0):
    """Return (freqs, pxx (channels, freqs) in µV²/Hz, ch_names) for the EEG channels of ``raw``."""
    picks = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude="bads")]
    sfreq = raw.info["sfreq"]
    f, pxx = welch(raw.get_data(picks=picks) * 1e6, fs=sfreq, nperseg=int(min(seg_s * sfreq, raw.n_times)))
    sel = (f >= fmin) & (f <= min(fmax, sfreq / 2))
    return f[sel], pxx[:, sel], picks


def band_power(freqs, pxx, bands):
    """{band: per-channel mean PSD in [lo, hi)} (µV²/Hz)."""
    out = {}
    for name, (lo, hi) in bands.items():
        sel = (freqs >= lo) & (freqs < hi)
        if sel.any():
            out[name] = pxx[:, sel].mean(axis=1)
    return out


@dataclass
class ERDResult:
    task_power: np.ndarray  # (trials, channels) mean band power in the task window
    base_power: np.ndarray  # (trials, channels) mean band power in the baseline window
    labels: np.ndarray
    timecourse: np.ndarray  # (trials, channels, times) band power, same units as base_power
    t_rel: np.ndarray  # seconds from the anchor
    ch_names: list
    band: tuple
    n_dropped: int

    @property
    def log_ratio(self):
        return np.log(self.task_power / self.base_power)

    def mean_erd(self, sel=None):
        """ERD% per channel from trial-averaged powers (``sel``: boolean trial mask)."""
        sel = np.ones(len(self.labels), bool) if sel is None else sel
        base = self.base_power[sel].mean(axis=0)
        return (self.task_power[sel].mean(axis=0) - base) / base * 100.0

    def mean_timecourse(self, sel=None):
        """% power change vs the trial-averaged baseline, (channels, times)."""
        sel = np.ones(len(self.labels), bool) if sel is None else sel
        base = self.base_power[sel].mean(axis=0)
        return (self.timecourse[sel].mean(axis=0) - base[:, None]) / base[:, None] * 100.0


def erd(raw, table, spec, band, tc_range=(-2.0, 1.0), reference="average"):
    """ERD% per trial for ``band`` = (lo, hi). ``raw`` should have flat channels already dropped.

    ``tc_range`` pads the time course: from ``tc_range[0]`` s before the
    anchor to ``tc_range[1]`` s after the task window ends.
    """
    cond = spec.conditions
    raw = raw.copy().pick("eeg", exclude="bads").load_data()
    raw.filter(band[0], band[1], verbose="error")
    if reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose="error")
    sf = raw.info["sfreq"]
    power = np.abs(hilbert(raw.get_data(), axis=-1)) ** 2
    t_rel = np.arange(int(round(tc_range[0] * sf)), int(round((cond.window.tmax + tc_range[1]) * sf))) / sf
    tasks, bases, tcs, labels, n_drop = [], [], [], [], 0
    n = power.shape[1]
    for t in table.trials.itertuples(index=False):
        b0, b1 = (int(round((t.baseline_onset + x) * sf)) for x in (cond.baseline.tmin, cond.baseline.tmax))
        w0, w1 = (int(round((t.anchor_onset + x) * sf)) for x in (cond.window.tmin, cond.window.tmax))
        s0 = int(round(t.anchor_onset * sf)) + int(round(tc_range[0] * sf))
        if min(b0, w0, s0) < 0 or max(b1, w1, s0 + len(t_rel)) > n or b1 <= b0:
            n_drop += 1
            continue
        bases.append(power[:, b0:b1].mean(axis=-1))
        tasks.append(power[:, w0:w1].mean(axis=-1))
        tcs.append(power[:, s0:s0 + len(t_rel)])
        labels.append(t.label)
    k = len(raw.ch_names)
    return ERDResult(np.array(tasks).reshape(-1, k), np.array(bases).reshape(-1, k),
                     np.array(labels, dtype=object), np.array(tcs).reshape(-1, k, len(t_rel)),
                     t_rel, list(raw.ch_names), tuple(band), n_drop)


def erd_stats(res, alpha=0.05):
    """Per class (and 'all'): mean ERD% per channel, t-test p, FDR-corrected significance mask."""
    groups = {"all": np.ones(len(res.labels), bool)}
    groups.update({c: res.labels == c for c in sorted(set(res.labels.tolist()))})
    means, pvals = {}, {}
    for name, sel in groups.items():
        means[name] = res.mean_erd(sel) if sel.sum() else np.full(len(res.ch_names), np.nan)
        pvals[name] = (ttest_1samp(res.log_ratio[sel], 0.0, axis=0).pvalue if sel.sum() >= 3
                       else np.ones(len(res.ch_names)))
    names = list(groups)
    flat = np.nan_to_num(np.concatenate([pvals[n] for n in names]), nan=1.0)
    reject, p_fdr = fdr_correction(flat, alpha=alpha)
    k = len(res.ch_names)
    out = {}
    for i, name in enumerate(names):
        out[name] = dict(n=int(groups[name].sum()), mean_erd=means[name], p=pvals[name],
                         p_fdr=p_fdr[i * k:(i + 1) * k], significant=reject[i * k:(i + 1) * k])
    return out
