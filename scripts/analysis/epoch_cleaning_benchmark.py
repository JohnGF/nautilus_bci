"""
BCI Trial-by-Trial Epoch Quality Pre-Screening & Benchmark
===========================================================
Implements an advanced trial-by-trial filtering function that evaluates the
electrode signal quality for each epoch, discards trials with high dropout/noise rates,
and trains a CSP + LDA BCI model on the cherry-picked high-quality trials.
"""

import sys
import os
import glob
import numpy as np
import pandas as pd

_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score


def filter_epochs_by_stability(epochs, max_bad_channels=6, min_std=1e-7, max_std=5e-3):
    """
    Evaluates signal stability channel-by-channel for each individual trial.
    Discards trials (epochs) where too many channels are flat or excessively noisy.
    """
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_samples)
    clean_indices = []
    
    for epoch_idx in range(data.shape[0]):
        epoch_data = data[epoch_idx]
        stds = np.std(epoch_data, axis=1)
        
        # Count bad channels in this specific trial
        bad_count = 0
        for ch_idx, std_val in enumerate(stds):
            if std_val < min_std or std_val > max_std:
                bad_count += 1
                
        if bad_count <= max_bad_channels:
            clean_indices.append(epoch_idx)
            
    print(f"    [Epoch Filter] Kept {len(clean_indices)} / {data.shape[0]} trials (discarded {data.shape[0] - len(clean_indices)} low-quality trials)")
    return epochs[clean_indices]


def run_clean_benchmark(bids_root="bids_musica", subject_id="01", sessions=None, max_bad_ch=6):
    print("=" * 80)
    print(" BCI Trial-by-Trial Epoch Quality Pre-Screening ".center(80, "="))
    print("=" * 80)

    bids_root = os.path.abspath(bids_root)
    sub_clean = subject_id.replace("sub-", "")
    vhdr_files = glob.glob(os.path.join(bids_root, f"sub-{sub_clean}", "ses-*", "eeg", "*_eeg.vhdr"))

    if not vhdr_files:
        print(f"[-] No EEG datasets found for sub-{sub_clean}")
        return

    # Filter sessions if specified
    if sessions is not None:
        sessions_padded = [s.zfill(2) for s in sessions]
        vhdr_files = [f for f in vhdr_files if os.path.basename(f).split('_')[1].replace("ses-", "") in sessions_padded]

    all_epochs = []
    l_freq, h_freq = 4.0, 45.0
    sfreq_target = 250.0

    for filepath in vhdr_files:
        filename = os.path.basename(filepath)
        try:
            parts = filename.split('_')
            sub_val = parts[0].replace("sub-", "")
            ses_val = parts[1].replace("ses-", "")
            task_val = parts[2].replace("task-", "")

            bids_path = BIDSPath(subject=sub_val, session=ses_val, task=task_val, datatype="eeg", root=bids_root)
            raw = read_raw_bids(bids_path=bids_path, verbose=False)
            raw.load_data()

            if 'Battery' in raw.ch_names:
                raw.set_channel_types({'Battery': 'misc'})
            
            raw_eeg = raw.copy().pick('eeg')

            # Standardize channels
            standard_32 = [
                'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 
                'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 
                'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10'
            ]
            mapping = {name: standard_32[i] for i, name in enumerate(raw_eeg.ch_names) if i < len(standard_32)}
            raw_eeg.rename_channels(mapping)

            # Drop channels only if they are flat across the ENTIRE session
            stds = np.std(raw_eeg.get_data(), axis=1)
            bad_ch = [raw_eeg.ch_names[i] for i, std in enumerate(stds) if std < 1e-8]
            if bad_ch:
                raw_eeg.drop_channels(bad_ch)

            if raw_eeg.info['sfreq'] != sfreq_target:
                raw_eeg.resample(sfreq_target, verbose=False)

            raw_filtered = raw_eeg.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
            raw_filtered.notch_filter(freqs=50.0, verbose=False)

            events, event_id = mne.events_from_annotations(raw_filtered, verbose=False)

            target_event_id = {}
            for k, v in event_id.items():
                if 'Task_Recall' in k:
                    clean_k = k.split('_dur_')[0] if '_dur_' in k else k
                    target_event_id[clean_k] = v

            if not target_event_id:
                continue

            epochs = mne.Epochs(
                raw_filtered, events, event_id=target_event_id,
                tmin=0.0, tmax=4.0, baseline=None, preload=True,
                event_repeated='drop', verbose=False
            )
            epochs.rename_channels(lambda name: name.strip())
            
            # Apply trial-by-trial quality filtering
            print(f"\n[*] Pre-screening trials for {filename}:")
            epochs_clean = filter_epochs_by_stability(epochs, max_bad_channels=max_bad_ch)
            
            if len(epochs_clean) > 0:
                all_epochs.append(epochs_clean)

        except Exception as e:
            print(f"[-] Error loading {filename}: {e}")

    if not all_epochs:
        print("[-] Error: No epochs compiled after quality filtering.")
        return

    # Align common channels & concatenate
    common_ch = list(set.intersection(*(set(ep.ch_names) for ep in all_epochs)))
    for ep in all_epochs:
        ep.pick(common_ch)
    combined_epochs = mne.concatenate_epochs(all_epochs, verbose=False)

    print(f"\n[+] Combined Clean Dataset: {len(combined_epochs)} trials, {len(common_ch)} channels.")

    X = combined_epochs.get_data()
    y_raw = combined_epochs.events[:, -1]
    unique_y = np.unique(y_raw)
    y = np.array([list(unique_y).index(val) for val in y_raw])
    n_classes = len(unique_y)

    # Train CSP + LDA
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_score = 0.0
    if len(y) >= 10:
        try:
            csp = CSP(n_components=min(4, X.shape[1]), reg='ledoit_wolf', log=True, norm_trace=False)
            lda = LinearDiscriminantAnalysis()
            scores = cross_val_score(Pipeline([('CSP', csp), ('LDA', lda)]), X, y, cv=cv)
            acc_score = np.mean(scores) * 100.0
        except Exception as e:
            print(f"[-] CSP+LDA evaluation failed: {e}")

    chance_level = 100.0 / n_classes
    print("\n" + "=" * 80)
    print(" QUALITY PRE-SCREENING BENCHMARK EXECUTIVE SUMMARY ".center(80, "="))
    print("=" * 80)
    print(f" Sessions Pooled          : {sessions if sessions else 'All stable'}")
    print(f" Aligned clean channels  : {len(common_ch)}")
    print(f" Trials Kept (Quality)    : {len(combined_epochs)}")
    print(f" Target Song Classes      : {n_classes}")
    print(f" BCI Classifier Accuracy  : {acc_score:.2f}% (Chance: {chance_level:.2f}%)")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=str, default="01")
    parser.add_argument("--sessions", type=str, default="01,03,04,05,06,07")
    parser.add_argument("--max-bad", type=int, default=6)
    args = parser.parse_args()

    sessions_list = args.sessions.split(',') if args.sessions else None
    run_clean_benchmark(subject_id=args.sub, sessions=sessions_list, max_bad_ch=args.max_bad)
