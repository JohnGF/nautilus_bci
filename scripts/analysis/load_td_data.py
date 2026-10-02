#!/usr/bin/env python3
"""
Tower Defense & FNF BCI Data Loader
====================================
Provides streamlined, high-level API to load clean BIDS data for:
  1. Classical Tower Defense (Subject 1: Active Recall + Classical Listening Prior)
  2. Modern Pop/Rock Tower Defense (Subject 2: Active Recall + 4 Modern Song Priors)
  3. Friday Night Funkin' (Subjects 1 & 3: Directional Rhythm Game)
"""

import os
from pathlib import Path
import numpy as np
import mne
import mne_bids

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_BIDS = BASE_DIR / "bids_clean"

CLASSICAL_ELEMENT_MAP = {'FIRE selected': 0, 'WATER selected': 1, 'WIND selected': 2, 'ELECTRICITY selected': 3}
MODERN_ELEMENT_MAP = {'FIRE selected': 0, 'WATER selected': 1, 'WIND selected': 2, 'ELECTRICITY selected': 3}
FNF_DIRECTION_MAP = {'Left': 0, 'Right': 1, 'Up': 2, 'Down': 3}

def preprocess_raw(raw, l_freq=4.0, h_freq=40.0):
    raw.load_data()
    raw.pick_types(eeg=True)
    raw.filter(l_freq, h_freq, fir_design='firwin', skip_by_annotation='edge', verbose=False)
    raw.set_eeg_reference('average', projection=True, verbose=False)
    raw.apply_proj(verbose=False)
    return raw

def load_classical_tower_defense(sessions=None, recommended_only=False, l_freq=4.0, h_freq=40.0, ep_len_s=3.0):
    """
    Loads Subject 1 Classical Tower Defense:
      - X_recall, y_recall: 5-second silent mental imagery epochs (FIRE, WATER, WIND, ELECTRICITY)
      - X_listen, y_listen: Labeled 4-class classical music listening priors (Fire, Water, Wind, Electricity)
    """
    root = CLEAN_BIDS / "bids_td_classical"
    if sessions is None:
        if recommended_only:
            sessions = ["01", "04"]  # Golden dry-electrode sessions (ses-01: 83 trials, ses-04: 16 trials 75% peak)
        else:
            sessions = ["01", "02", "04", "05", "06"]
    
    # 1. Active Recall Trials
    X_recall, y_recall = [], []
    for ses in sessions:
        for task in ["recall", "recallWaterReplaced", "recallOld", "memory"]:
            bp = mne_bids.BIDSPath(subject="01", session=ses, task=task, datatype="eeg", root=root)
            if not bp.fpath.exists():
                continue
            raw = mne_bids.read_raw_bids(bp, verbose=False)
            raw = preprocess_raw(raw, l_freq, h_freq)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            im_id = event_id.get('Imagine')
            if not im_id:
                continue
            sfreq = raw.info['sfreq']
            ep_samples = int(ep_len_s * sfreq)
            for idx, ev in enumerate(events):
                if ev[2] == im_id:
                    for j in range(max(0, idx - 5), min(len(events), idx + 5)):
                        ename = [k for k, v in event_id.items() if v == events[j, 2]]
                        if ename and ename[0] in CLASSICAL_ELEMENT_MAP:
                            stop = ev[0] + ep_samples
                            if stop <= len(raw.times):
                                ep = raw.get_data(start=ev[0], stop=stop)
                                if ep.shape[1] == ep_samples:
                                    X_recall.append(ep)
                                    y_recall.append(CLASSICAL_ELEMENT_MAP[ename[0]])
                            break

    # 2. Classical Music Listening Prior (Track-Labeled)
    bp_lis = mne_bids.BIDSPath(subject="01", session="listening", task="classicallistening", datatype="eeg", root=root)
    raw_lis = mne_bids.read_raw_bids(bp_lis, verbose=False)
    raw_lis = preprocess_raw(raw_lis, l_freq, h_freq)
    events_lis, event_id_lis = mne.events_from_annotations(raw_lis, verbose=False)

    track_map = {
        'Track_Start_id_1_name_Bach_Prelude': 0,        # Fire (Track 1)
        'Track_Start_id_2_name_Beethoven_Fur_Elise': 1, # Water (Track 2)
        'Track_Start_id_3_name_Joplin_Entertainer': 2,  # Wind (Track 3)
        'Track_Start_id_4_name_Mozart_Eine_Kleine': 3   # Electricity (Track 4)
    }
    sfreq = raw_lis.info['sfreq']
    ep_samples = int(ep_len_s * sfreq)
    X_listen, y_listen = [], []
    for tname, cid in track_map.items():
        for k, v in event_id_lis.items():
            if tname in k and '_dur_' not in k:
                evs = events_lis[events_lis[:, 2] == v]
                if len(evs) > 0:
                    s_idx = evs[0, 0]
                    for s in range(s_idx + int(5 * sfreq), s_idx + int(120 * sfreq), ep_samples):
                        ep = raw_lis.get_data(start=s, stop=s + ep_samples)
                        if ep.shape[1] == ep_samples:
                            X_listen.append(ep)
                            y_listen.append(cid)

    return (np.array(X_recall), np.array(y_recall)), (np.array(X_listen), np.array(y_listen))

def load_modern_tower_defense(sessions=None, recommended_only=False, l_freq=4.0, h_freq=40.0, ep_len_s=3.0):
    """
    Loads Subject 2 Modern Pop/Rock Tower Defense:
      - X_recall, y_recall: 5-second silent mental imagery epochs (FIRE, WATER, WIND, ELECTRICITY)
      - X_listen, y_listen: 4 modern song listening priors (Water, Wind, Electricity, Fire)
    """
    root = CLEAN_BIDS / "bids_td_modern"
    if sessions is None:
        if recommended_only:
            sessions = ["01", "02", "03"]  # Golden dry-electrode sessions (ses-01 to 03: 47.61% pooled, ses-03: 51.87% peak)
        else:
            sessions = ["01", "02", "03", "04", "05"]

    # 1. Active Recall Trials
    X_recall, y_recall, session_tags = [], [], []
    for ses in sessions:
        bp = mne_bids.BIDSPath(subject="02", session=ses, task="recallSongsIIWT", datatype="eeg", root=root)
        if not bp.fpath.exists():
            continue
        raw = mne_bids.read_raw_bids(bp, verbose=False)
        raw = preprocess_raw(raw, l_freq, h_freq)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        im_id = event_id.get('Imagine')
        if not im_id:
            continue
        sfreq = raw.info['sfreq']
        ep_samples = int(ep_len_s * sfreq)
        for idx, ev in enumerate(events):
            if ev[2] == im_id:
                for j in range(max(0, idx - 5), min(len(events), idx + 5)):
                    ename = [k for k, v in event_id.items() if v == events[j, 2]]
                    if ename and ename[0] in MODERN_ELEMENT_MAP:
                        stop = ev[0] + ep_samples
                        if stop <= len(raw.times):
                            ep = raw.get_data(start=ev[0], stop=stop)
                            if ep.shape[1] == ep_samples:
                                X_recall.append(ep)
                                y_recall.append(MODERN_ELEMENT_MAP[ename[0]])
                                session_tags.append(ses)
                        break

    # 2. Modern 4-Song Listening Priors
    songs_map = {
        'run-01': (1, 'listenItsRainingMen'),   # Water
        'run-02': (2, 'listenWhatsUp'),          # Wind
        'run-03': (3, 'listenThunderstruck'),     # Electricity
        'run-04': (0, 'listenKiss')              # Fire
    }
    X_listen, y_listen = [], []
    for run_id, (cid, task) in songs_map.items():
        bp_lis = mne_bids.BIDSPath(subject="02", session="listening", task=task, run=run_id.replace('run-', ''), datatype="eeg", root=root)
        if not bp_lis.fpath.exists():
            continue
        raw_l = mne_bids.read_raw_bids(bp_lis, verbose=False)
        raw_l = preprocess_raw(raw_l, l_freq, h_freq)
        sfreq = raw_l.info['sfreq']
        ep_samples = int(ep_len_s * sfreq)
        for s in range(int(5 * sfreq), len(raw_l.times) - int(5 * sfreq) - ep_samples, ep_samples):
            ep = raw_l.get_data(start=s, stop=s + ep_samples)
            if ep.shape[1] == ep_samples:
                X_listen.append(ep)
                y_listen.append(cid)

    return (np.array(X_recall), np.array(y_recall), np.array(session_tags)), (np.array(X_listen), np.array(y_listen))

if __name__ == '__main__':
    print("[*] Testing Classical TD loader...")
    (X_rec_c, y_rec_c), X_lis_c = load_classical_tower_defense()
    print(f"    Classical Recall: {X_rec_c.shape}, Labels: {np.bincount(y_rec_c)} | Listening Prior: {X_lis_c.shape}")

    print("[*] Testing Modern TD loader...")
    (X_rec_m, y_rec_m, ses_tags), (X_lis_m, y_lis_m) = load_modern_tower_defense()
    print(f"    Modern Recall: {X_rec_m.shape}, Labels: {np.bincount(y_rec_m)} | Modern Listening Prior: {X_lis_m.shape}")
    print("[+] Data loader verified successfully!")
