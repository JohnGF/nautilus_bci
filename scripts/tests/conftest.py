"""Synthetic BIDS fixture: a tiny simulated dataset with a known planted effect.

Layout (task-imagery, 250 Hz, 33 index-named channels EEG001..EEG033 like the
real recorder; EEG033 is a battery counter):

- sub-01 ses-01, sub-01 ses-02, sub-02 ses-01: BrainVision
- sub-02 ses-02: EEGLAB (.set/.fdt)
- sub-03 ses-01: dead cap, with only 3/32 EEG channels carrying signal (like music_lrud sub-04 ses-02)

Every session follows the music_lrud event grammar: a 3-trial false start in
mode ``VisualOnly``, then 32 trials (8 per class) in mode ``AudioEyesClosed``
inside ``Block_Start_NoMusic``. A trial is
``Trial_Start_i_Dir_X`` + ``Fixation`` (t0), ``Cue_X`` (t0+2), ``Task_X`` (t0+3),
``Rest`` (t0+7), and the next trial starts at t0+10.

Planted effect: during ``Task_X`` (+0..+4 s) the 10 Hz mu rhythm drops to 25%
amplitude over one gpype_32 electrode per class: Left->C4, Right->C3,
Bottom->Cz, Top->Pz.
"""

import os
import warnings

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, write_raw_bids
from scipy.signal import butter, sosfiltfilt

from nautilus_analysis.montage import GPYPE_32

SFREQ = 250.0
CLASSES = ["Top", "Bottom", "Left", "Right"]
ERD_CHANNEL = {"Left": "C4", "Right": "C3", "Bottom": "Cz", "Top": "Pz"}
TRIALS_PER_CLASS = 8
FALSE_START_TRIALS = 3
TRIAL_S = 10.0
LIVE_IN_DEAD = [1, 8, 17]  # indices with signal in the dead recording (Fp2, F8, T8)
RECORDINGS = [  # (sub, ses, format, dead)
    ("01", "01", "BrainVision", False),
    ("01", "02", "BrainVision", False),
    ("02", "01", "BrainVision", False),
    ("02", "02", "EEGLAB", False),
    ("03", "01", "BrainVision", True),
]
TASK = "imagery"


def _trial_events(rng, sub, ses):
    """List of (onset, description) and the per-trial (label, task_onset) of the main block."""
    ann, main = [], []
    t = 2.0
    plan = [("VisualOnly", rng.choice(CLASSES, FALSE_START_TRIALS).tolist())]
    labels = CLASSES * TRIALS_PER_CLASS
    rng.shuffle(labels)
    plan.append(("AudioEyesClosed", labels))
    for mode, labs in plan:
        ann.append((t, f"Experiment_Start_Sub_{sub}_Ses_{ses}_Paradigm_AllInOne4Class_Mode_{mode}"))
        if mode == "AudioEyesClosed":
            ann.append((t, "Block_Start_NoMusic_Idx_1"))
        for i, lab in enumerate(labs, start=1):
            ann += [(t, f"Trial_Start_{i}_Dir_{lab}"), (t, "Fixation"), (t + 2, f"Cue_{lab}"),
                    (t + 3, f"Task_{lab}"), (t + 7, "Rest"), (t + 10, f"Trial_End_{i}")]
            if mode == "AudioEyesClosed":
                main.append((lab, t + 3))
            t += TRIAL_S
        if mode == "AudioEyesClosed":
            ann.append((t, "Block_End_Idx_1"))
    ann.append((t + 0.5, "Experiment_End"))
    return ann, main, t + 5.0


def _narrowband(rng, n, lo, hi):
    sos = butter(4, [lo, hi], btype="band", fs=SFREQ, output="sos")
    x = sosfiltfilt(sos, rng.standard_normal(n))
    return x / x.std()


def _pink(rng, n):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SFREQ)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return x / x.std()


def simulate_raw(seed, sub, ses, dead=False):
    rng = np.random.default_rng(seed)
    ann, main, duration = _trial_events(rng, sub, ses)
    n = int(duration * SFREQ)
    times = np.arange(n) / SFREQ
    data = np.zeros((33, n))
    for ch in range(32):
        bg = 8.0 * _pink(rng, n) + 2.0 * rng.standard_normal(n)
        mu = 6.0 * _narrowband(rng, n, 9.0, 11.0)
        gain = np.ones(n)
        for label, onset in main:
            if GPYPE_32[ch] == ERD_CHANNEL[label]:
                gain[(times >= onset) & (times < onset + 4.0)] = 0.25
        data[ch] = bg + mu * gain + 0.5 * np.sin(2 * np.pi * 50.0 * times)
    if dead:
        mask = np.zeros(32, bool)
        mask[LIVE_IN_DEAD] = True
        data[:32][~mask] = 0.0
    data[32] = np.repeat(np.linspace(100, 80, n // 2500 + 1).round(), 2500)[:n]  # battery %
    info = mne.create_info([f"EEG{i:03d}" for i in range(1, 34)], SFREQ, "eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose="error")
    raw.info["line_freq"] = 50.0
    raw.set_annotations(mne.Annotations([a[0] for a in ann], [0.0] * len(ann), [a[1] for a in ann]))
    return raw, main


@pytest.fixture(scope="session")
def synthetic_bids(tmp_path_factory):
    """Path to the synthetic dataset root; ``.truth`` maps 'sub-XX_ses-YY' to the main-block (label, onset) list."""
    root = tmp_path_factory.mktemp("bids") / "synthetic_lrud"
    truth = {}
    for i, (sub, ses, fmt, dead) in enumerate(RECORDINGS):
        raw, main = simulate_raw(100 + i, sub, ses, dead)
        truth[f"sub-{sub}_ses-{ses}"] = main
        bp = BIDSPath(subject=sub, session=ses, task=TASK, datatype="eeg", root=root)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_raw_bids(raw, bp, format=fmt, allow_preload=True, overwrite=True, verbose="error")
    return _Root(str(root), truth)


class _Root(str):
    """str path that also carries the simulation ground truth."""

    def __new__(cls, path, truth):
        obj = super().__new__(cls, path)
        obj.truth = truth
        return obj


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REAL_BIDS_PARENTS = [os.path.join(SCRIPTS_DIR, d) for d in ("bids", "bids_clean")]


def real_dataset(rel):
    """Absolute path to a repo dataset under scripts/, or skip the test if it is absent."""
    path = os.path.join(SCRIPTS_DIR, rel)
    if not os.path.isdir(path):
        pytest.skip(f"real dataset {rel} not present")
    return path
