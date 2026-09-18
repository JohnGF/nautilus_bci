import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_current_dir = Path(__file__).resolve().parent
_scripts_dir = _current_dir.parent
_analysis_dir = _scripts_dir / "analysis"
if str(_analysis_dir) not in sys.path:
    sys.path.insert(0, str(_analysis_dir))

from analyze_fnf_bci import load_fnf_session, get_pipelines, DIRECTIONS

def get_bids_fnf_dir():
    cand = _scripts_dir / "bids" / "bids_fnf"
    if cand.exists():
        return cand
    return Path("scripts/bids/bids_fnf").resolve()

def get_available_subjects():
    bids_dir = get_bids_fnf_dir()
    subs = []
    if not bids_dir.exists():
        return subs
    for d in sorted(os.listdir(bids_dir)):
        if d.startswith("sub-") and (bids_dir / d).is_dir():
            subs.append(d.replace("sub-", ""))
    return subs

def get_available_sessions(sub_id):
    bids_dir = get_bids_fnf_dir()
    sub_clean = sub_id.replace("sub-", "")
    sub_dir = bids_dir / f"sub-{sub_clean}"
    if not sub_dir.is_dir():
        return []
    
    sessions = []
    for d in sorted(os.listdir(sub_dir)):
        if d.startswith("ses-") and (sub_dir / d).is_dir():
            sessions.append(d.replace("ses-", ""))
    return sessions

def load_dataset_sessions(sub_id, session_ids, progress_callback=None):
    bids_dir = get_bids_fnf_dir()
    
    all_X = []
    all_y = []
    
    total_ses = len(session_ids)
    
    direction_map = {'Left': 0, 'Right': 1, 'Up': 2, 'Down': 3, 'Rest': 4}
    
    for idx, ses in enumerate(session_ids):
        ses_clean = ses.replace("ses-", "")
        if progress_callback:
            progress_callback(f"Loading session {ses_clean} ({idx + 1}/{total_ses})...", (idx / total_ses) * 0.8)
            
        try:
            # Auto-detect task name from files
            eeg_dir = bids_dir / f"sub-{sub_id}" / f"ses-{ses_clean}" / "eeg"
            task_name = "leftright"
            if eeg_dir.exists():
                for f in os.listdir(eeg_dir):
                    if f.endswith("_eeg.vhdr"):
                        parts = f.split('_')
                        for p in parts:
                            if p.startswith('task-'):
                                task_name = p.replace('task-', '')
                                break
                        break

            # Using load_fnf_session from analyze_fnf_bci.py
            s_data = load_fnf_session(str(bids_dir), sub=sub_id, ses=ses_clean, task=task_name)
            
            # Load directional epochs
            for direction in s_data['available_directions']:
                ep_data = s_data['directional_epochs'][direction].get_data()
                all_X.append(ep_data)
                all_y.extend([direction_map[direction]] * len(ep_data))
                
        except Exception as e:
            print(f"Warning: failed to load session {ses_clean}: {e}")
            
    if not all_X:
        raise ValueError(f"No valid trials found across sessions {session_ids}")
        
    X_pooled = np.concatenate(all_X, axis=0)
    y_pooled = np.array(all_y)
    
    if progress_callback:
        progress_callback("Finished pooling data.", 0.9)
        
    return X_pooled, y_pooled
