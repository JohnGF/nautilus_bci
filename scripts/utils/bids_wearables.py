"""
Read, write and find wearable recordings in a BIDS session
==========================================================

The recorder writes strict BIDS 1.9:
  motion/*_tracksys-<label>_motion.tsv          headerless; columns from *_channels.tsv
  (eeg|beh)/*_recording-<label>_physio.tsv.gz   headerless, gzipped; columns from *_physio.json

Older sessions used headered TSVs (motion/*_motion.tsv, physio/*_physio.tsv,
ppg/*_ppg.tsv). Every reader here returns the same DataFrame for both:
'timestamp_sec' first, then the data columns, so analysis code can keep using
df['timestamp_sec'] and df.iloc[:, 1:].
"""
import glob
import gzip
import json
import os

import pandas as pd

TIMESTAMP_COLUMN = 'timestamp_sec'


def _sidecar(path):
    return path.replace('.tsv.gz', '.json').replace('.tsv', '.json')


def _channels_tsv(motion_path):
    return motion_path[:-len('_motion.tsv')] + '_channels.tsv'


def _has_header(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        return f.readline().split('\t')[0].strip() == TIMESTAMP_COLUMN


def column_names(path):
    """Column names of a headerless BIDS wearable file, from its channels.tsv or JSON sidecar."""
    if path.endswith('_motion.tsv') and os.path.exists(_channels_tsv(path)):
        return list(pd.read_csv(_channels_tsv(path), sep='\t')['name'])
    with open(_sidecar(path)) as f:
        return json.load(f)['Columns']


def read_wearable(path):
    """DataFrame with 'timestamp_sec' first, for both strict BIDS and older headered files."""
    if _has_header(path):
        return pd.read_csv(path, sep='\t')
    return pd.read_csv(path, sep='\t', header=None, names=column_names(path))


def write_wearable(path, df):
    """Write df in BIDS form: no header row, gzipped if the path ends in .gz."""
    df.to_csv(path, sep='\t', index=False, header=False, na_rep='n/a')


def _stream_type(path):
    try:
        with open(_sidecar(path)) as f:
            return json.load(f).get('StreamType', '')
    except (OSError, ValueError):
        return ''


def find_wearable_files(ses_dir, sub, ses, task, kind):
    """All wearable files of one kind ('motion' or 'physio' for PPG/heart rate) in a session,
    strict BIDS first, then the older layouts."""
    prefix = f"sub-{sub}_ses-{ses}_task-{task}"
    if kind == 'motion':
        strict = sorted(glob.glob(os.path.join(ses_dir, 'motion', f"{prefix}_tracksys-*_motion.tsv")))
        legacy = [os.path.join(ses_dir, 'motion', f"{prefix}_motion.tsv")]
    elif kind == 'physio':
        strict = sorted(p for folder in ('eeg', 'beh')
                        for p in glob.glob(os.path.join(ses_dir, folder, f"{prefix}_recording-*_physio.tsv.gz"))
                        if _stream_type(p).upper() in ('PPG', ''))
        legacy = [os.path.join(ses_dir, 'physio', f"{prefix}_physio.tsv"),
                  os.path.join(ses_dir, 'ppg', f"{prefix}_ppg.tsv")]
        legacy += sorted(glob.glob(os.path.join(ses_dir, 'ppg', f"{prefix}_recording-*_ppg.tsv")))
    else:
        raise ValueError(f"kind must be 'motion' or 'physio', not {kind!r}")
    return strict + [p for p in legacy if os.path.exists(p)]


def find_wearable_file(ses_dir, sub, ses, task, kind, label=None):
    """One wearable file, or None. With several (e.g. both wrists), pass label (the tracksys-/
    recording- value, e.g. 'IMULwrist') to choose; otherwise the first is used."""
    files = find_wearable_files(ses_dir, sub, ses, task, kind)
    if label:
        files = [p for p in files if f"-{label}_" in os.path.basename(p)]
    if len(files) > 1:
        print(f"[*] {len(files)} {kind} recordings in {ses_dir}; using {os.path.basename(files[0])} "
              f"(pass label= to choose)")
    return files[0] if files else None
