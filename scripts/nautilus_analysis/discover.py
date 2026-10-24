"""BIDS discovery: dataset roots and the EEG recordings inside them.

``discover(root)`` lists only recordings that sit directly under ``root``
(``root/sub-*/[ses-*/]eeg/``), with entities parsed by
``mne_bids.get_entities_from_fname``. ``mne_bids.find_matching_paths`` is
not used. It descends into nested datasets (e.g.
``bids/bids_tower_defense/bids_tower_defense(old)/sub-01``), which silently
mixes datasets, and it rebuilds their paths relative to the outer root, so
it returns files that don't exist. Use ``find_datasets`` to enumerate nested
roots explicitly.
"""

import glob
import os
from dataclasses import dataclass

from mne_bids import get_entities_from_fname

# Priority order when one recording exists in several formats.
EEG_EXTENSIONS = (".vhdr", ".set", ".edf", ".bdf", ".fif")


@dataclass(frozen=True)
class Recording:
    """One EEG recording (one ``*_eeg.<ext>`` file) in a BIDS dataset."""
    root: str
    sub: str
    ses: str | None
    task: str | None
    run: str | None
    acq: str | None
    eeg_path: str

    @property
    def dataset(self):
        return os.path.basename(os.path.normpath(self.root))

    @property
    def id(self):
        parts = [f"sub-{self.sub}"]
        for key, val in (("ses", self.ses), ("task", self.task), ("acq", self.acq), ("run", self.run)):
            if val is not None:
                parts.append(f"{key}-{val}")
        return "_".join(parts)

    @property
    def _stem(self):
        base = os.path.basename(self.eeg_path)
        return os.path.join(os.path.dirname(self.eeg_path), base[: base.rindex("_eeg.")])

    @property
    def events_path(self):
        return self._existing(self._stem + "_events.tsv")

    @property
    def channels_path(self):
        return self._existing(self._stem + "_channels.tsv")

    @property
    def sidecar_path(self):
        return self._existing(self._stem + "_eeg.json")

    @property
    def electrodes_path(self):
        """``*_electrodes.tsv`` in the same folder (task-level or session-level)."""
        prefix = f"sub-{self.sub}" + (f"_ses-{self.ses}" if self.ses is not None else "")
        hits = sorted(glob.glob(os.path.join(glob.escape(os.path.dirname(self.eeg_path)),
                                             prefix + "*_electrodes.tsv")))
        own = [h for h in hits if os.path.basename(h).startswith(os.path.basename(self._stem))]
        return (own or hits or [None])[0]

    @property
    def size_bytes(self):
        """Size of the signal payload (``.eeg``/``.fdt`` for split formats)."""
        ext = os.path.splitext(self.eeg_path)[1]
        payload = {".vhdr": ".eeg", ".set": ".fdt"}.get(ext)
        if payload:
            data_file = os.path.splitext(self.eeg_path)[0] + payload
            if os.path.exists(data_file):
                return os.path.getsize(data_file)
        return os.path.getsize(self.eeg_path)

    @staticmethod
    def _existing(path):
        return path if os.path.exists(path) else None


def _as_list(value):
    if value is None:
        return None
    return [value] if isinstance(value, str) else list(value)


def is_dataset_root(path):
    """True if ``path`` directly contains ``sub-*`` folders."""
    return os.path.isdir(path) and any(
        e.startswith("sub-") and os.path.isdir(os.path.join(path, e)) for e in os.listdir(path)
    )


def find_datasets(search_root):
    """Every BIDS dataset root at or below ``search_root`` (sorted).

    A dataset root is a folder with ``sub-*`` children. Nothing inside a
    ``sub-*`` folder is searched, so stray ``dataset_description.json``
    copies in subject folders are ignored.
    """
    roots = []
    for dirpath, dirnames, _ in os.walk(search_root):
        if is_dataset_root(dirpath):
            roots.append(os.path.abspath(dirpath))
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(("sub-", ".")) and d not in ("derivatives", "sourcedata", "code"))
    return sorted(roots)


def discover(bids_root, sub=None, ses=None, task=None, run=None):
    """List EEG recordings directly under ``bids_root``, optionally filtered.

    Each filter takes a value or a list of values, and None means all. Raises
    ``FileNotFoundError`` when ``bids_root`` is not a dataset root, with the
    nested dataset roots listed in the message.
    """
    bids_root = os.path.abspath(bids_root)
    if not is_dataset_root(bids_root):
        nested = find_datasets(bids_root) if os.path.isdir(bids_root) else []
        hint = ("; it contains these datasets: " + ", ".join(os.path.relpath(n, bids_root) for n in nested)) if nested else ""
        raise FileNotFoundError(f"{bids_root} is not a BIDS dataset root (no sub-* folders){hint}")

    filters = {"subject": _as_list(sub), "session": _as_list(ses), "task": _as_list(task), "run": _as_list(run)}
    best = {}
    for rel_dir in (os.path.join("sub-*", "eeg"), os.path.join("sub-*", "ses-*", "eeg")):
        for rank, ext in enumerate(EEG_EXTENSIONS):
            for fpath in glob.glob(os.path.join(glob.escape(bids_root), rel_dir, f"*_eeg{ext}")):
                ents = get_entities_from_fname(os.path.basename(fpath), on_error="ignore")
                ents = {k: (None if v is None else str(v)) for k, v in ents.items()}
                if ents.get("subject") is None or not _entities_match_dirs(fpath, bids_root, ents):
                    continue
                if any(want is not None and ents.get(k) not in [str(w) for w in want] for k, want in filters.items()):
                    continue
                rec = Recording(root=bids_root, sub=ents["subject"], ses=ents.get("session"), task=ents.get("task"),
                                run=ents.get("run"), acq=ents.get("acquisition"), eeg_path=os.path.abspath(fpath))
                if rec._stem not in best or rank < best[rec._stem][0]:
                    best[rec._stem] = (rank, rec)
    return sorted((r for _, r in best.values()), key=lambda r: (r.sub, r.ses or "", r.task or "", r.acq or "", r.run or ""))


def _entities_match_dirs(fpath, bids_root, ents):
    """The file's sub-/ses- entities must agree with the folders it sits in."""
    parts = os.path.relpath(fpath, bids_root).split(os.sep)
    if parts[0] != f"sub-{ents['subject']}":
        return False
    ses_dir = parts[1] if len(parts) == 4 else None
    return ses_dir == (None if ents.get("session") is None else f"ses-{ents['session']}")
