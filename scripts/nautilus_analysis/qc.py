"""Signal QC gate. Every pipeline runs it before any analysis.

The checks were motivated by ``music_lrud``. In sub-04 ses-02, 29/32 channels
are exactly 0 µV. sub-06 is drift-only, with alpha at 0.03 µV²/Hz. Every legacy
script still produced plausible-looking numbers on both. The checks:

- flat channels (std < ``flat_std_uv``) are listed and dropped (``apply_qc``);
- fewer live EEG channels than ``min_live_channels`` -> refuse;
- implausible spectrum on the live channels (median alpha < ``min_alpha_uv2_hz``,
  or a median of more than ``max_low_freq_fraction`` of 1-45 Hz power below 4 Hz)
  -> "probably disconnected", refuse unless ``allow_implausible``;
- line noise much above its neighbouring bins -> warning only.

Statistics are medians across live channels, so a few railing channels can't
mask a dead cap. Calibration on the real datasets (2026-10-24): good recordings
have median alpha 8-4e4 µV²/Hz and a <4 Hz fraction of 0.60-0.98 (raw g.Nautilus
data is drift-heavy, so the plan's original 95% cut refused nearly everything).
Drift-only recordings have alpha 1e-8-0.035 and a fraction of 0.98-1.00.
"""

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import welch


class QCRefusal(RuntimeError):
    """Raised by ``gate`` when a recording must not be analysed."""

    def __init__(self, report):
        self.report = report
        super().__init__(f"QC refused {report.recording_id or 'recording'}: " + "; ".join(report.reasons))


@dataclass
class QCThresholds:
    flat_std_uv: float = 0.5
    min_alpha_uv2_hz: float = 0.1
    max_low_freq_fraction: float = 0.99
    line_noise_ratio_warn: float = 10.0
    min_duration_s: float = 10.0


@dataclass
class QCReport:
    recording_id: str
    duration_s: float
    sfreq: float
    n_eeg: int
    channel_std_uv: dict
    flat_channels: list
    live_channels: list
    alpha_uv2_hz: float | None
    low_freq_fraction: float | None
    line_noise_ratio: float | None
    line_freq: float
    probably_disconnected: bool
    warnings: list = field(default_factory=list)
    status: str = "unchecked"  # set by gate(): ok | warn | refused
    reasons: list = field(default_factory=list)

    @property
    def n_live(self):
        return len(self.live_channels)

    def to_dict(self):
        d = asdict(self)
        d["n_live"] = self.n_live
        return d

    def summary(self):
        alpha = "n/a" if self.alpha_uv2_hz is None else f"{self.alpha_uv2_hz:.3g}"
        lf = "n/a" if self.low_freq_fraction is None else f"{self.low_freq_fraction:.2f}"
        return (f"{self.recording_id}: {self.n_live}/{self.n_eeg} live EEG ch, {self.duration_s:.0f} s, "
                f"alpha {alpha} µV²/Hz, <4 Hz fraction {lf}, status={self.status}"
                + (f" ({'; '.join(self.reasons)})" if self.reasons else ""))


def _band_median(f, pxx, lo, hi):
    """Median over channels of each channel's mean PSD in [lo, hi)."""
    sel = (f >= lo) & (f < hi)
    return float(np.median(np.mean(pxx[:, sel], axis=1))) if sel.any() else float("nan")


def compute_qc(raw, recording_id="", thresholds=None, line_freq=None):
    """Measure ``raw`` (EEG channels only, bads included). Doesn't modify it or decide anything."""
    th = thresholds or QCThresholds()
    line_freq = float(line_freq or raw.info.get("line_freq") or 50.0)
    picks = [raw.ch_names[i] for i in range(len(raw.ch_names)) if raw.get_channel_types([i])[0] == "eeg"]
    sfreq = float(raw.info["sfreq"])
    duration = float(raw.n_times / sfreq)
    if not picks:
        return QCReport(recording_id, duration, sfreq, 0, {}, [], [], None, None, None, line_freq, True)

    data_uv = raw.get_data(picks=picks) * 1e6
    std = data_uv.std(axis=1)
    flat = [c for c, s in zip(picks, std) if not s >= th.flat_std_uv]  # NaN counts as flat
    live = [c for c in picks if c not in flat]

    alpha = lf_frac = line_ratio = None
    disconnected = not live
    if live and data_uv.shape[1] >= int(2 * sfreq):
        live_data = data_uv[[picks.index(c) for c in live]]
        f, pxx = welch(live_data, fs=sfreq, nperseg=int(2 * sfreq))
        alpha = _band_median(f, pxx, 8.0, 13.0)
        top = min(45.0, sfreq / 2 - 1)
        total = np.sum(pxx[:, (f >= 1.0) & (f < top)], axis=1)
        low = np.sum(pxx[:, (f >= 1.0) & (f < 4.0)], axis=1)
        lf_frac = float(np.median(np.where(total > 0, low / np.where(total > 0, total, 1), 1.0)))
        if line_freq < sfreq / 2 - 5:
            peak = _band_median(f, pxx, line_freq - 1, line_freq + 1.01)
            side = np.mean([_band_median(f, pxx, line_freq - 5, line_freq - 2),
                            _band_median(f, pxx, line_freq + 2, line_freq + 5)])
            line_ratio = float(peak / side) if side > 0 else None
        disconnected = bool(alpha < th.min_alpha_uv2_hz or lf_frac > th.max_low_freq_fraction)

    return QCReport(
        recording_id=recording_id, duration_s=duration, sfreq=sfreq, n_eeg=len(picks),
        channel_std_uv={c: float(s) for c, s in zip(picks, std)}, flat_channels=flat, live_channels=live,
        alpha_uv2_hz=alpha, low_freq_fraction=lf_frac, line_noise_ratio=line_ratio, line_freq=line_freq,
        probably_disconnected=disconnected,
    )


def gate(report, min_live_channels, thresholds=None, allow_implausible=False):
    """Set ``report.status``. Raises ``QCRefusal`` if the recording can't be analysed."""
    th = thresholds or QCThresholds()
    reasons, warnings = [], []
    if report.duration_s < th.min_duration_s:
        reasons.append(f"recording is only {report.duration_s:.1f} s (< {th.min_duration_s:.0f} s)")
    if report.n_live < min_live_channels:
        reasons.append(f"only {report.n_live}/{report.n_eeg} EEG channels carry signal "
                       f"(std >= {th.flat_std_uv} µV); need >= {min_live_channels}")
    if report.probably_disconnected and report.n_live:
        msg = (f"probably disconnected: alpha {report.alpha_uv2_hz:.3g} µV²/Hz "
               f"(min {th.min_alpha_uv2_hz}), {report.low_freq_fraction:.0%} of power < 4 Hz "
               f"(max {th.max_low_freq_fraction:.0%})")
        (warnings if allow_implausible else reasons).append(msg)
    if report.flat_channels and report.n_live >= min_live_channels:
        warnings.append(f"{len(report.flat_channels)} flat channel(s) dropped: {report.flat_channels}")
    if report.line_noise_ratio is not None and report.line_noise_ratio > th.line_noise_ratio_warn:
        warnings.append(f"strong {report.line_freq:g} Hz line noise (x{report.line_noise_ratio:.0f} over neighbours)")
    report.reasons = reasons
    report.warnings = warnings
    report.status = "refused" if reasons else ("warn" if warnings else "ok")
    if reasons:
        raise QCRefusal(report)
    return report


def apply_qc(raw, report):
    """A copy of ``raw`` with the report's flat channels dropped."""
    drop = [c for c in report.flat_channels if c in raw.ch_names]
    return raw.copy().drop_channels(drop) if drop else raw.copy()
