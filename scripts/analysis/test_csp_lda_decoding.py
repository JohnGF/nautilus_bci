"""
CSP + LDA Classification of Video / Imagery Conditions (Earth, Fire, Water, Wind)
===================================================================================

Evaluates whether Common Spatial Patterns (CSP) + Linear Discriminant Analysis (LDA) 
can decode the neural states corresponding to 'earth', 'fire', 'water', and 'wind' (air).
"""

import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score


def extract_ovr_csp_features(X, y, n_components=4):
    """Extract One-vs-Rest CSP log-variance features for multi-class data."""
    classes = np.unique(y)
    n_epochs, n_ch, n_times = X.shape
    feature_blocks = []

    for c in classes:
        y_binary = (y == c).astype(int)
        csp = CSP(n_components=n_components, reg='oas', log=True, norm_trace=False)
        feat = csp.fit_transform(X, y_binary)
        feature_blocks.append(feat)

    return np.hstack(feature_blocks)


def run_csp_lda_decoding(bids_root="bids_baseline", sub="01", ses="02", task="video"):
    bids_root = os.path.abspath(bids_root)
    bids_path = BIDSPath(subject=sub, session=ses, task=task, datatype="eeg", root=bids_root)
    
    print("=" * 75)
    print(" CSP + LDA Neural Decoding Studio: Earth vs Fire vs Water vs Wind ".center(75, "="))
    print("=" * 75)

    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()

    if 'Battery' in raw.ch_names:
        raw.set_channel_types({'Battery': 'misc'})

    # Filter EEG in Mu/Beta band (8 - 30 Hz)
    print("[*] Preprocessing EEG: Bandpass Filter (8 - 30 Hz) & Notch Filter (50 Hz)...")
    raw_filt = raw.copy().filter(l_freq=8.0, h_freq=30.0, verbose=False)
    raw_filt.notch_filter(freqs=50.0, verbose=False)

    # Pick EEG channels only
    raw_filt.pick_types(eeg=True)

    # Extract Annotations / Events
    events, event_id = mne.events_from_annotations(raw_filt, verbose=False)
    
    conditions = {'water': 1, 'earth': 2, 'wind': 3, 'fire': 4}
    cond_events = []
    
    for onset, duration, val in events:
        desc = [k for k, v in event_id.items() if v == val]
        if desc:
            d_str = desc[0]
            for c_name, c_code in conditions.items():
                if f"Video_Start_{c_name}" in d_str or (c_name in d_str.lower() and "video_start" in d_str.lower()):
                    cond_events.append([onset, duration, c_code])
                    break

    cond_events = np.array(cond_events)
    if len(cond_events) == 0:
        raise RuntimeError("No condition events found in dataset.")

    # Epoching (0.5s to 3.0s window after stimulus onset)
    tmin, tmax = 0.5, 3.0
    target_event_id = {k: v for k, v in conditions.items() if v in cond_events[:, 2]}
    
    epochs = mne.Epochs(
        raw_filt,
        cond_events,
        event_id=target_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    X = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    y = epochs.events[:, 2]

    print(f"[+] Loaded {len(X)} epochs ({X.shape[1]} EEG channels, {X.shape[2]} timepoints):")
    for c_name, c_code in target_event_id.items():
        print(f"    - {c_name.capitalize():6s} (Code {c_code}): {np.sum(y == c_code)} trials")

    # ---------------------------------------------------------
    # 1. Pairwise Binary Class Decoding (CSP + LDA)
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print(" 1. Pairwise Binary CSP + LDA Classification ".center(60, "-"))
    print("-" * 60)

    class_names = {v: k.capitalize() for k, v in conditions.items()}
    unique_codes = sorted(list(target_event_id.values()))

    pairwise_results = {}
    for i in range(len(unique_codes)):
        for j in range(i + 1, len(unique_codes)):
            c1, c2 = unique_codes[i], unique_codes[j]
            mask = (y == c1) | (y == c2)
            X_pair = X[mask]
            y_pair = y[mask]

            csp = CSP(n_components=4, reg='oas', log=True, norm_trace=False)
            lda = LinearDiscriminantAnalysis()
            pipeline = Pipeline([('CSP', csp), ('LDA', lda)])

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(pipeline, X_pair, y_pair, cv=cv, scoring='accuracy')

            pair_label = f"{class_names[c1]} vs. {class_names[c2]}"
            mean_score = np.mean(scores) * 100
            std_score = np.std(scores) * 100
            pairwise_results[pair_label] = (mean_score, std_score)
            print(f"  • {pair_label:22s}: {mean_score:6.2f}% ± {std_score:.2f}%  (Chance: 50.0%)")

    # ---------------------------------------------------------
    # 2. Multiclass 4-Class OVR-CSP + LDA Classification
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print(" 2. Multiclass 4-Class OVR-CSP + LDA Classification ".center(60, "-"))
    print("-" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit OVR-CSP on train
        classes = np.unique(y_train)
        csp_filters = []
        train_feats = []
        test_feats = []

        for c in classes:
            y_bin = (y_train == c).astype(int)
            csp = CSP(n_components=4, reg='oas', log=True, norm_trace=False)
            f_train = csp.fit_transform(X_train, y_bin)
            f_test = csp.transform(X_test)
            train_feats.append(f_train)
            test_feats.append(f_test)

        X_train_csp = np.hstack(train_feats)
        X_test_csp = np.hstack(test_feats)

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_csp, y_train)
        acc = lda.score(X_test_csp, y_test)
        acc_scores.append(acc)

    mean_4class_acc = np.mean(acc_scores) * 100
    std_4class_acc = np.std(acc_scores) * 100
    chance_level = 1.0 / len(unique_codes) * 100

    print(f"\n[RESULTS] 4-Class OVR-CSP + LDA Overall Accuracy: {mean_4class_acc:.2f}% +/- {std_4class_acc:.2f}% (Chance Level: {chance_level:.1f}%)")

    # ---------------------------------------------------------
    # 3. Spectral Power (PSD) Channel Feature Classification
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print(" 3. Band Power PSD Feature Classification (Alpha + Beta + Theta) ".center(60, "-"))
    print("-" * 60)

    # Compute PSD for each epoch across EEG channels in Theta(4-8Hz), Alpha(8-12Hz), Beta(13-30Hz)
    psd_epochs = epochs.compute_psd(fmin=4, fmax=30, verbose=False)
    psd_data = psd_epochs.get_data()  # shape: (n_epochs, n_channels, n_freqs)
    freqs = psd_epochs.freqs

    theta_m = (freqs >= 4) & (freqs <= 8)
    alpha_m = (freqs >= 8) & (freqs <= 12)
    beta_m = (freqs >= 13) & (freqs <= 30)

    feat_theta = np.mean(psd_data[:, :, theta_m], axis=2)
    feat_alpha = np.mean(psd_data[:, :, alpha_m], axis=2)
    feat_beta = np.mean(psd_data[:, :, beta_m], axis=2)

    X_psd = np.hstack([feat_theta, feat_alpha, feat_beta])
    # Log transform
    X_psd = np.log1p(X_psd * 1e12)

    lda_psd = LinearDiscriminantAnalysis()
    scores_psd = cross_val_score(lda_psd, X_psd, y, cv=cv, scoring='accuracy')

    print(f"[RESULTS] 4-Class PSD Channel Band Power + LDA Accuracy: {np.mean(scores_psd)*100:.2f}% +/- {np.std(scores_psd)*100:.2f}%")

    print("\n" + "=" * 75)
    print(" FINAL VERDICT & ANALYSIS SUMMARY ".center(75, "="))
    print("=" * 75)
    print(f"• 4-Class Overall CSP+LDA Accuracy : {mean_4class_acc:.1f}% (vs {chance_level:.1f}% chance)")
    print(f"• 4-Class Band Power PSD+LDA Acc   : {np.mean(scores_psd)*100:.1f}%")
    best_pair = max(pairwise_results.items(), key=lambda item: item[1][0])
    print(f"• Highest Binary Pairwise Accuracy : {best_pair[0]} at {best_pair[1][0]:.1f}%")
    print("=" * 75)

    return mean_4class_acc, pairwise_results, np.mean(scores_psd) * 100


if __name__ == '__main__':
    run_csp_lda_decoding()
