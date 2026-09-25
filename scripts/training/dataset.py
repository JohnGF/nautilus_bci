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


def get_bids_listening_dir():
    """Locates the bids_listening folder."""
    cand = _scripts_dir / "bids" / "bids_listening"
    if cand.exists():
        return cand
    cand2 = Path("/home/guilhermecoto/Documentos/Lasige/nautilus_bci/scripts/bids/bids_listening")
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


# ======================================================================
# Pure Music Listening (bids_listening) Integration
# ======================================================================

SUB01_LISTENING_TRACKS = {
    'Beethoven_Fur_Elise': {
        'display_name': 'Beethoven — Für Elise',
        'element': 'FIRE',
        'class_id': 0,
        'session': 'ses-02',
        't_start': 187.068,
        't_end': 416.464,
        'trials': 75
    },
    'Bach_Prelude': {
        'display_name': 'Bach — Prelude in C Major',
        'element': 'WATER',
        'class_id': 1,
        'session': 'ses-02',
        't_start': 48.98,
        't_end': 182.068,
        'trials': 43
    },
    'Vivaldi_Spring': {
        'display_name': 'Vivaldi — Spring (Four Seasons)',
        'element': 'WIND',
        'class_id': 2,
        'session': 'ses-02',
        't_start': 1469.64,
        't_end': 1664.745,
        'trials': 63
    },
    'Tchaikovsky_Waltz': {
        'display_name': 'Tchaikovsky — Waltz of the Flowers',
        'element': 'ELECTRICITY',
        'class_id': 3,
        'session': 'ses-02',
        't_start': 1013.272,
        't_end': 1464.632,
        'trials': 149
    }
}

SUB02_LISTENING_SESSIONS = {
    'ses-01': {
        'display_name': "ses-01 — The Weather Girls (It's Raining Men)",
        'element': 'WATER',
        'class_id': 1,
        'session': 'ses-01',
        'song': 'ItsRainingMen',
        'trials': 103
    },
    'ses-02': {
        'display_name': "ses-02 — 4 Non Blondes (What's Up)",
        'element': 'WIND',
        'class_id': 2,
        'session': 'ses-02',
        'song': 'WhatsUp',
        'trials': 98
    },
    'ses-03': {
        'display_name': 'ses-03 — AC/DC (Thunderstruck)',
        'element': 'ELECTRICITY',
        'class_id': 3,
        'session': 'ses-03',
        'song': 'Thunderstruck',
        'trials': 95
    },
    'ses-04': {
        'display_name': "ses-04 — Kiss (I Was Made for Lovin' You)",
        'element': 'FIRE',
        'class_id': 0,
        'session': 'ses-04',
        'song': 'Kiss',
        'trials': 86
    }
}


def has_listening_data(sub_id):
    """Checks if bids_listening contains data for the given subject."""
    sub_clean = sub_id.replace("sub-", "")
    lis_base = get_bids_listening_dir()
    sub_dir = lis_base / f"sub-{sub_clean}"
    return sub_dir.is_dir()


def get_available_listening_items(sub_id):
    """
    Returns a list of dicts describing available listening tracks/sessions for a subject.
    Accounts for the structural difference:
      - sub-01: Divided by event markers in a single continuous session (ses-02).
      - sub-02: Divided into individual music sessions (ses-01..ses-04).
    """
    sub_clean = sub_id.replace("sub-", "")
    lis_base = get_bids_listening_dir()
    sub_dir = lis_base / f"sub-{sub_clean}"
    if not sub_dir.is_dir():
        return []

    items = []
    if sub_clean == "01":
        for k, v in SUB01_LISTENING_TRACKS.items():
            items.append({
                'id': k,
                'display_name': v['display_name'],
                'element': v['element'],
                'class_id': v['class_id'],
                'session': v['session'],
                'mode': 'event_marker',
                'trials': v['trials']
            })
    elif sub_clean == "02":
        for k, v in SUB02_LISTENING_SESSIONS.items():
            items.append({
                'id': k,
                'display_name': v['display_name'],
                'element': v['element'],
                'class_id': v['class_id'],
                'session': v['session'],
                'mode': 'session_file',
                'trials': v['trials']
            })
    return items


def load_listening_epochs(
    sub_id,
    item_ids=None,
    sfreq=250.0,
    win_len_s=3.0,
    spatial_mode="robust_car",
    progress_callback=None
):
    """
    Loads and extracts epochs from scripts/bids/bids_listening for sub_id ('01' or '02').
    Filters by item_ids if provided (track keys for sub-01, session keys for sub-02).
    """
    sub_clean = sub_id.replace("sub-", "")
    lis_base = get_bids_listening_dir()
    sub_dir = lis_base / f"sub-{sub_clean}"
    if not sub_dir.is_dir():
        raise FileNotFoundError(f"bids_listening directory for sub-{sub_clean} not found at {sub_dir}")

    all_X = []
    all_y = []
    all_meta = []
    listening_stats = {}
    n_samples_win = int(win_len_s * sfreq)
    step_samp = int(win_len_s * sfreq)

    if sub_clean == "01":
        active_tracks = {
            k: v for k, v in SUB01_LISTENING_TRACKS.items()
            if item_ids is None or k in item_ids
        }
        if not active_tracks:
            return np.empty((0, 32, n_samples_win)), np.empty((0,), dtype=int), {}, pd.DataFrame()

        if progress_callback:
            progress_callback("Loading bids_listening sub-01 continuous ses-02...", 0.70)
        raw_uv, df_events, ses_sfreq, ch_names = load_single_session_raw(str(lis_base), "01", "02")
        clean_eeg = preprocess_continuous_eeg(
            raw_uv, sfreq=ses_sfreq, l_freq=1.0, h_freq=45.0, notch_freq=50.0,
            spatial_mode=spatial_mode, ch_names=ch_names
        )

        for t_key, spec in active_tracks.items():
            s_start = int((spec['t_start'] + 2.0) * sfreq)
            s_end = int((spec['t_end'] - 2.0) * sfreq)
            track_eps = []
            for s in range(s_start, s_end - n_samples_win, step_samp):
                if s + n_samples_win <= len(clean_eeg):
                    ep = clean_eeg[s : s + n_samples_win, :].T
                    track_eps.append(ep)
                    all_y.append(spec['class_id'])
                    all_meta.append({
                        'session': f"listening_{spec['session']}",
                        'element': spec['element'],
                        'class_id': spec['class_id'],
                        'track': t_key,
                        'source': 'bids_listening'
                    })
            if track_eps:
                all_X.append(np.array(track_eps))
                listening_stats[f"listening_{t_key}"] = {
                    'trials': len(track_eps),
                    'breakdown': {spec['class_id']: len(track_eps)}
                }

    elif sub_clean == "02":
        active_sessions = {
            k: v for k, v in SUB02_LISTENING_SESSIONS.items()
            if item_ids is None or k in item_ids or k.replace("ses-", "") in item_ids
        }
        if not active_sessions:
            return np.empty((0, 32, n_samples_win)), np.empty((0,), dtype=int), {}, pd.DataFrame()

        total_ses = len(active_sessions)
        for idx, (ses_key, spec) in enumerate(active_sessions.items()):
            ses_clean = ses_key.replace("ses-", "")
            if progress_callback:
                progress_callback(f"Loading bids_listening sub-02 {ses_key} ({idx+1}/{total_ses})...", 0.70 + 0.05 * (idx / max(1, total_ses)))

            raw_uv, df_events, ses_sfreq, ch_names = load_single_session_raw(str(lis_base), "02", ses_clean)
            clean_eeg = preprocess_continuous_eeg(
                raw_uv, sfreq=ses_sfreq, l_freq=1.0, h_freq=45.0, notch_freq=50.0,
                spatial_mode=spatial_mode, ch_names=ch_names
            )

            track_ev = df_events[df_events['trial_type'].str.contains('Track_Start', na=False)]
            if len(track_ev) > 0:
                t_start = float(track_ev.iloc[0]['onset'])
                t_dur = float(track_ev.iloc[0]['duration'])
                t_end = t_start + t_dur
            else:
                t_start = 5.0
                t_end = len(clean_eeg) / ses_sfreq

            s_start = int((t_start + 2.0) * sfreq)
            s_end = int((t_end - 2.0) * sfreq)
            ses_eps = []
            for s in range(s_start, s_end - n_samples_win, step_samp):
                if s + n_samples_win <= len(clean_eeg):
                    ep = clean_eeg[s : s + n_samples_win, :].T
                    ses_eps.append(ep)
                    all_y.append(spec['class_id'])
                    all_meta.append({
                        'session': f"listening_{ses_key}",
                        'element': spec['element'],
                        'class_id': spec['class_id'],
                        'song': spec['song'],
                        'source': 'bids_listening'
                    })
            if ses_eps:
                all_X.append(np.array(ses_eps))
                listening_stats[f"listening_{ses_key}"] = {
                    'trials': len(ses_eps),
                    'breakdown': {spec['class_id']: len(ses_eps)}
                }

    if not all_X:
        return np.empty((0, 32, n_samples_win)), np.empty((0,), dtype=int), {}, pd.DataFrame()

    X_pooled = np.concatenate(all_X, axis=0)
    y_pooled = np.array(all_y)
    meta_df = pd.DataFrame(all_meta)
    return X_pooled, y_pooled, listening_stats, meta_df


def load_dataset_sessions(
    dataset_name_or_path,
    sub_id,
    session_ids,
    include_listening=False,
    listening_item_ids=None,
    sfreq=250.0,
    win_len_s=3.0,
    spatial_mode="robust_car",
    progress_callback=None
):
    """
    Loads, continuous-preprocesses, and extracts 4-class mental rhythm epochs
    pooled across all specified sessions and optionally pure music listening data.

    Parameters:
        dataset_name_or_path: key name or path to BIDS directory
        sub_id: '01', '02', etc.
        session_ids: list of session strings e.g. ['01', '02', '05']
        include_listening: bool, whether to load and concatenate bids_listening data
        listening_item_ids: list of track IDs (sub-01) or session IDs (sub-02) to include
        sfreq: sampling frequency (default: 250.0 Hz)
        win_len_s: epoch window duration in seconds (default: 3.0s)
        spatial_mode: 'robust_car', 'car', 'laplacian', etc.
        progress_callback: optional callable fn(message: str, fraction: float)

    Returns:
        X_pooled: np.ndarray of shape (n_total_trials, 32, n_samples)
        y_pooled: np.ndarray of shape (n_total_trials,)
        session_stats: dict mapping ses_id/track to {'trials': int, 'breakdown': dict}
        meta_df: pd.DataFrame with trial metadata
    """
    sub_clean = sub_id.replace("sub-", "")

    all_X = []
    all_y = []
    all_meta = []
    session_stats = {}

    # 1. Load Tower Defense Game Sessions
    if session_ids:
        ds_path = resolve_dataset_path(dataset_name_or_path)
        total_ses = len(session_ids)
        for idx, ses in enumerate(session_ids):
            ses_clean = ses.replace("ses-", "")
            msg = f"Loading session ses-{ses_clean} ({idx + 1}/{total_ses})..."
            if progress_callback:
                progress_callback(msg, (idx / total_ses) * 0.6)

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

    # 2. Optionally Load Pure Music Listening Data
    if include_listening:
        if progress_callback:
            progress_callback(f"Loading bids_listening data for sub-{sub_clean}...", 0.68)
        X_lis, y_lis, lis_stats, meta_lis = load_listening_epochs(
            sub_clean,
            item_ids=listening_item_ids,
            sfreq=sfreq,
            win_len_s=win_len_s,
            spatial_mode=spatial_mode,
            progress_callback=progress_callback
        )
        if len(y_lis) > 0:
            all_X.append(X_lis)
            all_y.append(y_lis)
            all_meta.append(meta_lis)
            session_stats.update(lis_stats)

    if not all_X:
        raise RuntimeError(f"No valid trials found across sessions {session_ids} (listening={include_listening}) for sub-{sub_clean}.")

    X_pooled = np.concatenate(all_X, axis=0)
    y_pooled = np.concatenate(all_y, axis=0)
    meta_df = pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame()

    if progress_callback:
        progress_callback(f"Successfully pooled {len(y_pooled)} total trials across {len(session_stats)} source blocks.", 0.75)

    return X_pooled, y_pooled, session_stats, meta_df

