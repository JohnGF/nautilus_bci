"""
BCI Music Pairwise Decoding & ANOVA Statistics Studio
======================================================
1. Performs pairwise binary classification (all 15 track pairs) to identify
   which songs are easiest and hardest to differentiate in the EEG data.
2. Performs a one-way ANOVA across all 6 tracks on EEG frequency band powers
   to determine if different song recalls elicit statistically distinct brain states.
"""

import sys
import os
import glob
import itertools
import numpy as np
import pandas as pd
from scipy.stats import f_oneway

_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score


def run_pairwise_anova_analysis(bids_root="bids_musica", subject_id="01", sessions=None):
    print("=" * 80)
    print(" BCI Music Pairwise Decoding & ANOVA Statistics Studio ".center(80, "="))
    print("=" * 80)

    # 1. Pool and preprocess stable sessions
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
    
    print(f"[+] Pooling {len(vhdr_files)} sessions for analysis...")

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

            # Standardize channel layout
            standard_32 = [
                'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 
                'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 
                'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10'
            ]
            mapping = {name: standard_32[i] for i, name in enumerate(raw_eeg.ch_names) if i < len(standard_32)}
            raw_eeg.rename_channels(mapping)

            # Drop flatlines
            stds = np.std(raw_eeg.get_data(), axis=1)
            bad_ch = [raw_eeg.ch_names[i] for i, std in enumerate(stds) if std < 1e-7 or std > 5e-3]
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
            all_epochs.append(epochs)

        except Exception as e:
            print(f"[-] Error loading {filename}: {e}")

    if not all_epochs:
        print("[-] Error: No epochs compiled.")
        return

    # Align common channels & concatenate
    common_ch = list(set.intersection(*(set(ep.ch_names) for ep in all_epochs)))
    for ep in all_epochs:
        ep.pick(common_ch)
    combined_epochs = mne.concatenate_epochs(all_epochs, verbose=False)

    print(f"[+] Aggregated Dataset: {len(combined_epochs)} trials, {len(common_ch)} channels.")

    # 2. Pairwise Binary Classification Analysis
    # Get clean labels
    event_id_inv = {v: k for k, v in combined_epochs.event_id.items()}
    track_keys = sorted(list(combined_epochs.event_id.keys()))
    
    pairwise_results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[*] Running Pairwise Binary Classifications (15 combinations)...")
    for track_a, track_b in itertools.combinations(track_keys, 2):
        # Subset epochs for just these two tracks
        epochs_sub = combined_epochs[[track_a, track_b]]
        
        # Get data and labels (0 and 1)
        X_sub = epochs_sub.get_data()
        y_sub_raw = epochs_sub.events[:, -1]
        unique_y = np.unique(y_sub_raw)
        y_sub = np.array([list(unique_y).index(val) for val in y_sub_raw])

        # Skip if not enough samples
        if len(y_sub) < 10:
            continue

        try:
            # We use CSP + LDA for binary classification (chance level = 50.0%)
            csp = CSP(n_components=min(4, X_sub.shape[1]), reg='ledoit_wolf', log=True, norm_trace=False)
            lda = LinearDiscriminantAnalysis()
            scores = cross_val_score(Pipeline([('CSP', csp), ('LDA', lda)]), X_sub, y_sub, cv=cv)
            acc = np.mean(scores) * 100.0
            pairwise_results.append((track_a, track_b, acc))
        except Exception as e:
            print(f"    [-] Classification failed for {track_a} vs {track_b}: {e}")
            pairwise_results.append((track_a, track_b, 50.0))

    # Sort results by accuracy
    pairwise_results.sort(key=lambda x: x[2], reverse=True)

    # 3. ANOVA Analysis on Frequency Band Powers
    print("[*] Performing One-Way ANOVA across all 6 tracks...")
    # Compute power in classical frequency bands
    bands = {
        'Theta (4-8 Hz)': (4.0, 8.0),
        'Alpha (8-12 Hz)': (8.0, 12.0),
        'Beta (13-30 Hz)': (13.0, 30.0)
    }

    anova_results = {}
    
    for band_name, (fmin, fmax) in bands.items():
        # Compute PSD
        psds_obj = combined_epochs.compute_psd(fmin=fmin, fmax=fmax, verbose=False)
        psds, freqs = psds_obj.get_data(return_freqs=True)
        # Average power across frequency bins
        mean_psds = np.mean(psds, axis=2) # (n_epochs, n_channels)
        # Average across channels to get global band power per trial
        global_power = np.mean(mean_psds, axis=1) # (n_epochs,)

        # Group trial powers by track
        track_powers = []
        for track in track_keys:
            track_idx = combined_epochs.events[:, -1] == combined_epochs.event_id[track]
            track_powers.append(global_power[track_idx])
            
        # Run one-way ANOVA F-test
        try:
            f_stat, p_val = f_oneway(*track_powers)
            anova_results[band_name] = (f_stat, p_val)
        except Exception as e:
            anova_results[band_name] = (0.0, 1.0)

    # 4. Save and Print Executive Markdown Report
    report_file = os.path.abspath("analysis_results/pairwise_anova_report.md")
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# BCI Music Pairwise Classification & ANOVA Report\n")
        f.write(f"Analyzed Sessions: `{sessions if sessions else 'All stable'}` | Common Channels: `{len(common_ch)}` | Total Trials: `{len(combined_epochs)}`\n\n")
        
        f.write("## 1. Pairwise Binary Decoding Rankings (Chance: 50.0%)\n")
        f.write("Evaluates which song recalls elicit the most distinct EEG spatial patterns. Higher accuracy means the two songs are **easier to differentiate** in your brain.\n\n")
        f.write("| Rank | Song A | Song B | Binary BCI Accuracy | Classification Margin |\n")
        f.write("| :---: | :--- | :--- | :---: | :---: |\n")
        
        for idx, (t_a, t_b, acc) in enumerate(pairwise_results):
            clean_a = t_a.replace("Task_Recall_", "")
            clean_b = t_b.replace("Task_Recall_", "")
            margin = acc - 50.0
            margin_str = f"+{margin:.1f}%" if margin >= 0 else f"{margin:.1f}%"
            f.write(f"| {idx+1} | {clean_a} | {clean_b} | **{acc:.1f}%** | {margin_str} |\n")
            
        f.write("\n## 2. Multi-Class EEG Band Power ANOVA Statistics\n")
        f.write("Determines if there is a statistically significant difference in global brain rhythm power levels across the 6 different songs. A **p-value < 0.05** represents statistical significance.\n\n")
        f.write("| Brain Rhythm | F-Statistic | P-Value | Statistical Significance |\n")
        f.write("| :--- | :---: | :---: | :--- |\n")
        
        for band_name, (f_stat, p_val) in anova_results.items():
            sig = "✅ SIGNIFICANT (p < 0.05)" if p_val < 0.05 else "❌ NOT SIGNIFICANT"
            f.write(f"| {band_name} | {f_stat:.3f} | {p_val:.5f} | {sig} |\n")

    print("\n" + "=" * 80)
    print(" MUSIC PAIRWISE & ANOVA EXECUTIVE SUMMARY ".center(80, "="))
    print("=" * 80)
    print(f" Markdown Report Saved   : {report_file}")
    print("\nTop 3 Easiest Pairs to Differentiate:")
    for i in range(min(3, len(pairwise_results))):
        t_a, t_b, acc = pairwise_results[i]
        print(f"   [{i+1}] {t_a} vs {t_b}: {acc:.2f}%")
        
    print("\nTop 3 Hardest Pairs to Differentiate:")
    for i in range(1, min(4, len(pairwise_results) + 1)):
        t_a, t_b, acc = pairwise_results[-i]
        print(f"   [{i}] {t_a} vs {t_b}: {acc:.2f}%")
        
    print("\nANOVA Frequency Significance:")
    for k, v in anova_results.items():
        sig_str = "SIGNIFICANT" if v[1] < 0.05 else "NOT SIGNIFICANT"
        print(f"   [+] {k:<15}: F={v[0]:.3f}, p={v[1]:.5f} ({sig_str})")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=str, default="01")
    parser.add_argument("--sessions", type=str, default="01,06,07")
    args = parser.parse_args()

    sessions_list = args.sessions.split(',') if args.sessions else None
    run_pairwise_anova_analysis(subject_id=args.sub, sessions=sessions_list)
