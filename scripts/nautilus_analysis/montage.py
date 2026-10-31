"""The single source of truth for g.Nautilus channel layouts.

Most recordings store only index names (``EEG001..EEG033``) in ``channels.tsv``,
so the electrode under each index has to come from a known layout. ``GPYPE_32``
is the logical-channel table in the g.Nautilus PRO manual (32-channel setup,
section 16.2, ``docs/vendor-manuals/g.Nautilus PRO Instructions for Use
1.25.02.pdf``). It matches ``bridges/gpype_to_lsl.DEFAULT_CHANNELS_32``.

``STANDARD_32_LEGACY`` is the list that was copied into many analysis scripts.
It does **not** match the cap: index 4 is C3 there but F7 on the cap, and it
names FT9/FT10/TP9/TP10/O1/O2, which this cap does not have. It is kept only so
that legacy results can be reproduced. Never use it for new work.

``resolve_channels`` keeps real 10-20 names from ``channels.tsv`` when they are
there. Otherwise it applies a layout by index, sets any trailing extra channel
(e.g. the battery level in ``EEG033``) to ``misc``, and records that the
montage was *assumed*, so reports can say so.
"""

import re
from dataclasses import dataclass, field

import mne

# g.Nautilus PRO manual, section 16.2 "32-channel setup", logical channels 1..32.
GPYPE_32 = [
    "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8",
    "PO7", "PO3", "PO4", "PO8", "Oz",
]
# From bridges/gpype_to_lsl.py. Not yet verified against a vendor table.
GPYPE_16 = [
    "Fz", "FC1", "FC2", "C3", "Cz", "C4", "CP1", "CP2",
    "P3", "Pz", "P4", "Oz", "O1", "O2", "AFz", "FCz",
]
GPYPE_8 = ["Fz", "C3", "Cz", "C4", "P3", "Pz", "P4", "Oz"]
# WRONG for this cap; kept only so legacy results can be reproduced (see module docstring).
STANDARD_32_LEGACY = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
    "Fz", "Cz", "Pz", "Oz", "FC1", "FC2", "CP1", "CP2",
    "FC5", "FC6", "CP5", "CP6", "FT9", "FT10", "TP9", "TP10",
]

LAYOUTS = {
    "gpype_32": GPYPE_32,
    "gpype_16": GPYPE_16,
    "gpype_8": GPYPE_8,
    "standard_32": STANDARD_32_LEGACY,
}
# Layout assumed for index-only files, keyed by EEG channel count.
DEFAULT_LAYOUT_BY_COUNT = {32: "gpype_32", 16: "gpype_16", 8: "gpype_8"}
MONTAGE_MODES = ("auto", "from_channels_tsv", *LAYOUTS)

# Non-EEG channels that recorders sometimes tag as EEG.
AUX_NAME_RE = re.compile(r"^(battery|counter|trigger|trig|status|validation)$", re.IGNORECASE)
# BIDS channels.tsv `type` -> MNE channel type.
BIDS_TO_MNE_TYPE = {
    "EEG": "eeg", "EOG": "eog", "HEOG": "eog", "VEOG": "eog", "ECG": "ecg", "EMG": "emg",
    "MISC": "misc", "TRIG": "stim", "STIM": "stim", "RESP": "resp", "PPG": "bio", "GSR": "gsr",
    "TEMP": "temperature",
}

_STANDARD_NAMES = {n.lower(): n for n in mne.channels.make_standard_montage("standard_1020").ch_names}


def canonical_name(name):
    """Return the standard_1020 spelling of ``name`` (e.g. 'CZ' -> 'Cz'), or None."""
    return _STANDARD_NAMES.get(str(name).strip().lower())


@dataclass
class ChannelResolution:
    """How a recording's channels were named and typed."""
    original: list
    names: list
    types: list
    source: str  # "channels_tsv" or "layout:<name>"
    assumed: bool
    extra_misc: list = field(default_factory=list)

    @property
    def eeg_names(self):
        return [n for n, t in zip(self.names, self.types) if t == "eeg"]

    @property
    def label(self):
        """Text that figure titles and reports must show (montage honesty)."""
        if self.assumed:
            return f"montage assumed: {self.source.split(':', 1)[1]}"
        return "montage from channels.tsv"


def resolve_channels(ch_names, ch_types=None, montage="auto"):
    """Map recorded channel names to electrode names.

    ``ch_types`` holds MNE types per channel (default: all 'eeg'). ``montage`` is
    one of ``MONTAGE_MODES``:

    - ``auto``: keep real names if at least 80% of the EEG channels are
      10-20 names. Otherwise apply ``DEFAULT_LAYOUT_BY_COUNT`` by index.
    - ``from_channels_tsv``: keep names, and raise if they are not real.
    - a layout name: apply that layout by index, whatever the names say.
    """
    if montage not in MONTAGE_MODES:
        raise ValueError(f"Unknown montage {montage!r}; expected one of {MONTAGE_MODES}")
    original = [str(c) for c in ch_names]
    types = list(ch_types) if ch_types is not None else ["eeg"] * len(original)
    types = ["misc" if t == "eeg" and AUX_NAME_RE.match(n) else t for n, t in zip(original, types)]
    eeg_idx = [i for i, t in enumerate(types) if t == "eeg"]
    canon = [canonical_name(original[i]) for i in eeg_idx]
    n_real = sum(c is not None for c in canon)

    if montage == "from_channels_tsv" or (montage == "auto" and eeg_idx and n_real / len(eeg_idx) >= 0.8):
        missing = [original[i] for i, c in zip(eeg_idx, canon) if c is None]
        if montage == "from_channels_tsv" and missing:
            raise ValueError(f"montage 'from_channels_tsv' but these EEG channels are not 10-20 names: {missing}")
        names = list(original)
        for i, c in zip(eeg_idx, canon):
            if c is not None:
                names[i] = c
        _check_unique(names)
        return ChannelResolution(original, names, types, "channels_tsv", assumed=False)

    if montage == "auto":
        layout_name = _layout_for_count(len(eeg_idx))
    else:
        layout_name = montage
    layout = LAYOUTS[layout_name]
    if len(eeg_idx) < len(layout):
        raise ValueError(f"Layout {layout_name} needs {len(layout)} EEG channels; recording has {len(eeg_idx)}")
    names = list(original)
    for i, name in zip(eeg_idx, layout):
        names[i] = name
    extra = eeg_idx[len(layout):]
    for i in extra:
        types[i] = "misc"
    _check_unique(names)
    return ChannelResolution(original, names, types, f"layout:{layout_name}", assumed=True,
                             extra_misc=[original[i] for i in extra])


def _layout_for_count(n_eeg):
    """A layout of n or n-1 channels; one trailing channel (e.g. a battery counter) is aux."""
    for n in (n_eeg, n_eeg - 1):
        if n in DEFAULT_LAYOUT_BY_COUNT:
            return DEFAULT_LAYOUT_BY_COUNT[n]
    raise ValueError(
        f"Cannot infer a channel layout for {n_eeg} index-named EEG channels; "
        f"set \"montage\" in the analysis spec to one of {sorted(LAYOUTS)}"
    )


def _check_unique(names):
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"Channel resolution produced duplicate names: {dupes}")
