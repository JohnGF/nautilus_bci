"""
dataset.py
==========
Dataset discovery, subject & session querying, and multi-session EEG loading
for the 3 BIDS Tower Defense dataset variants:
  1. bids_tower_defense
  2. bids_tower_defense(old)
  3. bids_tower_defense_6_3_27
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd

# Add analysis directory to sys.path to reuse existing BIDS reader routines
_training_dir = Path(__file__).resolve().parent
_scripts_dir = _training_dir.parent
_analysis_dir = _scripts_dir / "analysis"
if str(_analysis_dir) not in sys.path:
    sys.path.insert(0, str(_analysis_dir))

from analyze_tower_defense_rhythm_decoding import (
    load_single_session_raw,
    preprocess_continuous_eeg,
    extract_session_epochs
)

DATASET_DESCRIPTIONS = {
    'bids_tower_defense': {
        'title': 'bids_tower_defense (Standard / Modern)',
        'description': 'Current recording setup: modern songs for sub-02 (Kiss, WhatsUp, Thunderstruck, etc.); classical with water variant for sub-01.'
    },
    'bids_tower_defense(old)': {
        'title': 'bids_tower_defense(old) (Original Classical)',
        'description': 'Earlier recording room/protocol using classical pieces (Für Elise, Bach Prelude, Spring, Waltz).'
    },
    'bids_tower_defense_6_3_27': {
        'title': 'bids_tower_defense_6_3_27 (Single Session)',
        'description': 'Specific recording session variant (ses-01).'
    }
}


def get_bids_base_dir():
    """Locates the parent bids_tower_defense folder containing the 3 subfolders."""
    cand = _scripts_dir / "bids" / "bids_tower_defense"
    if cand.exists():
        return cand
    cand2 = Path("/home/guilhermecoto/Documentos/Lasige/nautilus_bci/scripts/bids/bids_tower_defense")
    if cand2.exists():
        return cand2
    return cand


def list_available_datasets():
    """
    Returns a dict of available dataset variants and their paths:
    { 'bids_tower_defense': Path(...), 'bids_tower_defense(old)': Path(...), ... }
    """
    base = get_bids_base_dir()
    if not base.exists():
        return {}

    datasets = {}
    for entry in sorted(os.listdir(base)):
        fpath = base / entry
        if fpath.is_dir():
            # Verify if it contains sub-* folders
            subs = [d for d in os.listdir(fpath) if d.startswith("sub-") and (fpath / d).is_dir()]
            if subs:
                datasets[entry] = {
                    'path': fpath,
                    'subjects': sorted(subs),
                    'title': DATASET_DESCRIPTIONS.get(entry, {}).get('title', entry),
                    'description': DATASET_DESCRIPTIONS.get(entry, {}).get('description', 'Custom dataset variant.')
                }
    return datasets


def resolve_dataset_path(dataset_name_or_path):
    """Resolves a dataset key name or direct path to an existing Path object."""
    cand = Path(dataset_name_or_path)
    if cand.is_dir():
        return cand

    base = get_bids_base_dir()
    direct = base / dataset_name_or_path
    if direct.is_dir():
        return direct

    raise FileNotFoundError(f"Dataset folder not found: {dataset_name_or_path} (checked {base})")


def get_available_subjects(dataset_name_or_path):
    """Returns a sorted list of subject IDs e.g. ['01', '02'] for a dataset."""
    ds_path = resolve_dataset_path(dataset_name_or_path)
    subs = []
    for d in sorted(os.listdir(ds_path)):
        if d.startswith("sub-") and (ds_path / d).is_dir():
            subs.append(d.replace("sub-", ""))
    return subs


def get_available_sessions(dataset_name_or_path, sub_id):
    """Returns a sorted list of session IDs e.g. ['01', '02', ...] for a subject."""
    ds_path = resolve_dataset_path(dataset_name_or_path)
    sub_clean = sub_id.replace("sub-", "")
    sub_dir = ds_path / f"sub-{sub_clean}"
    if not sub_dir.is_dir():
        return []

    sessions = []
    for d in sorted(os.listdir(sub_dir)):
        if d.startswith("ses-") and (sub_dir / d).is_dir():
            # Check if eeg directory exists
            if (sub_dir / d / "eeg").is_dir():
                sessions.append(d.replace("ses-", ""))
    return sessions


def get_session_trial_preview(dataset_name_or_path, sub_id, ses_id):
    """
    Quickly counts trial markers in the events.tsv file for a given session.
    Returns (num_trials, trial_breakdown_dict) or (0, {}) if unreadable.
    """
    ds_path = resolve_dataset_path(dataset_name_or_path)
    sub_clean = sub_id.replace("sub-", "")
    ses_clean = ses_id.replace("ses-", "")
    eeg_dir = ds_path / f"sub-{sub_clean}" / f"ses-{ses_clean}" / "eeg"

    if not eeg_dir.is_dir():
        return 0, {}

    events_files = list(eeg_dir.glob("*events.tsv"))
    if not events_files:
        return 0, {}

    try:
        df = pd.read_csv(events_files[0], sep='\t')
        if 'trial_type' in df.columns:
            # Count elements selected
            selected = df[df['trial_type'].str.contains('selected', na=False)]
            counts = selected['trial_type'].str.replace(' selected', '').value_counts().to_dict()
            return len(selected), counts
    except Exception:
        pass
    return 0, {}


def load_dataset_sessions(
    dataset_name_or_path,
    sub_id,
    session_ids,
    sfreq=250.0,
    win_len_s=3.0,
    spatial_mode="robust_car",
    progress_callback=None
):
    """
    Loads, continuous-preprocesses, and extracts 4-class mental rhythm epochs
    pooled across all specified sessions.

    Parameters:
        dataset_name_or_path: key name or path to BIDS directory
        sub_id: '01', '02', etc.
        session_ids: list of session strings e.g. ['01', '02', '05']
        sfreq: sampling frequency (default: 250.0 Hz)
        win_len_s: epoch window duration in seconds (default: 3.0s)
        spatial_mode: 'robust_car', 'car', 'laplacian', etc.
        progress_callback: optional callable fn(message: str, fraction: float)

    Returns:
        X_pooled: np.ndarray of shape (n_total_trials, 32, n_samples)
        y_pooled: np.ndarray of shape (n_total_trials,)
        session_stats: dict mapping ses_id to {'trials': int, 'breakdown': dict}
        meta_df: pd.DataFrame with trial metadata
    """
    ds_path = resolve_dataset_path(dataset_name_or_path)
    sub_clean = sub_id.replace("sub-", "")

    all_X = []
    all_y = []
    all_meta = []
    session_stats = {}

    total_ses = len(session_ids)
    if total_ses == 0:
        raise ValueError("No sessions provided for training.")

    for idx, ses in enumerate(session_ids):
        ses_clean = ses.replace("ses-", "")
        msg = f"Loading session ses-{ses_clean} ({idx + 1}/{total_ses})..."
        if progress_callback:
            progress_callback(msg, (idx / total_ses) * 0.7)

        raw_uv, df_events, ses_sfreq, ch_names = load_single_session_raw(
            str(ds_path), sub_clean, ses_clean
        )

        clean_eeg = preprocess_continuous_eeg(
            raw_uv,
            sfreq=ses_sfreq,
            l_freq=1.0,
            h_freq=45.0,
            notch_freq=50.0,
            spatial_mode=spatial_mode,
            ch_names=ch_names
        )

        X_im, _, _, y, meta_sub, _ = extract_session_epochs(
            clean_eeg,
            df_events,
            ses_id=ses_clean,
            sfreq=ses_sfreq,
            win_len_s=win_len_s
        )

        if len(y) == 0:
            print(f"[Warning] No mental rhythm trials extracted from ses-{ses_clean}.")
            continue

        counts = dict(pd.Series(y).value_counts())
        session_stats[ses_clean] = {
            'trials': len(y),
            'breakdown': counts
        }

        all_X.append(X_im)
        all_y.append(y)
        all_meta.append(meta_sub)

    if not all_X:
        raise RuntimeError(f"No valid trials found across sessions {session_ids} for sub-{sub_clean}.")

    X_pooled = np.concatenate(all_X, axis=0)
    y_pooled = np.concatenate(all_y, axis=0)
    meta_df = pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame()

    if progress_callback:
        progress_callback(f"Successfully pooled {len(y_pooled)} total trials across {len(session_stats)} sessions.", 0.75)

    return X_pooled, y_pooled, session_stats, meta_df
