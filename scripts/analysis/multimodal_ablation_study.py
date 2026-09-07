"""
Multimodal Fusion Ablation Study Tool
======================================

Performs a rigorous feature ablation study on the BIDS Baseline Dataset (`bids_baseline/sub-01/ses-02`)
to measure the exact contributions of EEG, Smartwatch PPG, and Smartwatch IMU Motion features.

Evaluates 7 Ablation Configurations:
  1. EEG Only
  2. PPG Only (Smartwatch Heart Rate & Pulse)
  3. IMU Motion Only (Smartwatch Accel/Gyro)
  4. EEG + PPG (No Motion)
  5. EEG + IMU Motion (No PPG)
  6. PPG + IMU Motion (No EEG)
  7. FULL FUSION: EEG + PPG + IMU Motion
"""

import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne_bids import BIDSPath, read_raw_bids

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score


def load_ablation_features(bids_root="bids_baseline", sub="01", ses="02", task="video"):
    bids_root = os.path.abspath(bids_root)
    bids_path = BIDSPath(subject=sub, session=ses, task=task, datatype="eeg", root=bids_root)
    
    print("=" * 75)
    print(" Multimodal Physiological Fusion Ablation Study Studio ".center(75, "="))
    print("=" * 75)

    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()
    if 'Battery' in raw.ch_names:
        raw.set_channel_types({'Battery': 'misc'})

    raw_filt = raw.copy().filter(l_freq=1.0, h_freq=40.0, verbose=False)
    raw_filt.pick_types(eeg=True)

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
    target_event_id = {k: v for k, v in conditions.items() if v in cond_events[:, 2]}

    epochs = mne.Epochs(
        raw_filt,
        cond_events,
        event_id=target_event_id,
        tmin=0.5,
        tmax=3.0,
        baseline=None,
        preload=True,
        verbose=False
    )

    X_eeg_raw = epochs.get_data()
    y = epochs.events[:, 2]

    # Extract EEG Riemannian Tangent Space + Band Power features
    covs = Covariances(estimator='oas').fit_transform(X_eeg_raw)
    eeg_ts = TangentSpace(metric='riemann').fit_transform(covs)

    psd_epochs = epochs.compute_psd(fmin=4, fmax=30, verbose=False)
    psd_data = psd_epochs.get_data()
    freqs = psd_epochs.freqs

    alpha_m = (freqs >= 8) & (freqs <= 12)
    beta_m = (freqs >= 13) & (freqs <= 30)

    eeg_alpha = np.mean(psd_data[:, :, alpha_m], axis=2)
    eeg_beta = np.mean(psd_data[:, :, beta_m], axis=2)
    
    X_eeg = np.hstack([eeg_ts, np.log1p(eeg_alpha * 1e12), np.log1p(eeg_beta * 1e12)])

    # Load Smartwatch Motion & PPG
    ses_dir = os.path.join(bids_root, f"sub-{sub}", f"ses-{ses}")
    motion_path = os.path.join(ses_dir, "motion", f"sub-{sub}_ses-{ses}_task-{task}_motion.tsv")
    physio_path = os.path.join(ses_dir, "physio", f"sub-{sub}_ses-{ses}_task-{task}_physio.tsv")

    motion_feats = []
    physio_feats = []

    has_motion = os.path.exists(motion_path)
    has_physio = os.path.exists(physio_path)

    if has_motion:
        df_m = pd.read_csv(motion_path, sep='\t')
        t_m = df_m['timestamp_sec'].values
        d_m = df_m.iloc[:, 1:].values
    if has_physio:
        df_p = pd.read_csv(physio_path, sep='\t')
        t_p = df_p['timestamp_sec'].values
        d_p = df_p.iloc[:, 1:].values

    epoch_onsets = epochs.events[:, 0] / raw.info['sfreq']

    for onset in epoch_onsets:
        # Motion features: Accel & Gyro mean, std, peak, magnitude
        if has_motion:
            idx_m = (t_m >= onset) & (t_m <= onset + 2.5)
            if np.any(idx_m):
                chunk = d_m[idx_m]
                m_mean = np.mean(chunk, axis=0)
                m_std = np.std(chunk, axis=0)
                m_max = np.max(chunk, axis=0)
                m_mag = np.sqrt(np.sum(chunk[:, :3]**2, axis=1))
                m_mag_feat = np.array([np.mean(m_mag), np.std(m_mag)])
                motion_feats.append(np.hstack([m_mean, m_std, m_max, m_mag_feat]))
            else:
                motion_feats.append(np.zeros(20))

        # Physio/PPG features: Heart rate mean, std, min, max
        if has_physio:
            idx_p = (t_p >= onset) & (t_p <= onset + 2.5)
            if np.any(idx_p):
                chunk_p = d_p[idx_p]
                p_mean = np.mean(chunk_p, axis=0)
                p_std = np.std(chunk_p, axis=0)
                p_min = np.min(chunk_p, axis=0)
                p_max = np.max(chunk_p, axis=0)
                physio_feats.append(np.hstack([p_mean, p_std, p_min, p_max]))
            else:
                physio_feats.append(np.zeros(4))

    X_motion = np.array(motion_feats) if has_motion else np.zeros((len(y), 20))
    X_physio = np.array(physio_feats) if has_physio else np.zeros((len(y), 4))

    return X_eeg, X_physio, X_motion, y


def run_ablation_study(out_dir="analysis_results"):
    os.makedirs(out_dir, exist_ok=True)
    X_eeg, X_ppg, X_imu, y = load_ablation_features()

    ablation_configs = {
        "1. EEG Only": X_eeg,
        "2. PPG Only": X_ppg,
        "3. IMU Motion Only": X_imu,
        "4. EEG + PPG": np.hstack([X_eeg, X_ppg]),
        "5. EEG + IMU Motion": np.hstack([X_eeg, X_imu]),
        "6. PPG + IMU Motion": np.hstack([X_ppg, X_imu]),
        "7. FULL FUSION (EEG + PPG + IMU)": np.hstack([X_eeg, X_ppg, X_imu])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    chance_level = 25.0  # 4 classes

    results = {}
    print("\n" + "=" * 80)
    print(" ABLATION EXPERIMENT RESULTS (5-Fold Stratified Cross-Validation) ".center(80, "="))
    print("=" * 80)
    print(f"{'Modality Combination / Config':36s} | {'GradientBoosting':20s} | {'RandomForest':18s}")
    print("-" * 80)

    for config_name, X_subset in ablation_configs.items():
        # Classifier A: HistGradientBoosting
        pipe_gb = Pipeline([('Scaler', StandardScaler()), ('GB', HistGradientBoostingClassifier(random_state=42))])
        scores_gb = cross_val_score(pipe_gb, X_subset, y, cv=cv, scoring='accuracy') * 100

        # Classifier B: RandomForest
        pipe_rf = Pipeline([('Scaler', StandardScaler()), ('RF', RandomForestClassifier(n_estimators=100, random_state=42))])
        scores_rf = cross_val_score(pipe_rf, X_subset, y, cv=cv, scoring='accuracy') * 100

        gb_str = f"{np.mean(scores_gb):5.2f}% +/- {np.std(scores_gb):4.2f}%"
        rf_str = f"{np.mean(scores_rf):5.2f}% +/- {np.std(scores_rf):4.2f}%"

        results[config_name] = {
            'GradientBoosting_mean': float(np.mean(scores_gb)),
            'GradientBoosting_std': float(np.std(scores_gb)),
            'RandomForest_mean': float(np.mean(scores_rf)),
            'RandomForest_std': float(np.std(scores_rf)),
            'n_features': int(X_subset.shape[1])
        }

        print(f"{config_name:36s} | {gb_str:20s} | {rf_str:18s}")

    print("=" * 80)

    # ---------------------------------------------------------
    # Feature Importance Ablation Analysis (Full Fusion)
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print(" Modality Feature Contribution & Importance Ranking ".center(70, "-"))
    print("-" * 70)

    X_full = ablation_configs["7. FULL FUSION (EEG + PPG + IMU)"]
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(StandardScaler().fit_transform(X_full), y)

    n_eeg_f = X_eeg.shape[1]
    n_ppg_f = X_ppg.shape[1]
    n_imu_f = X_imu.shape[1]

    importances = rf_model.feature_importances_
    eeg_imp = np.sum(importances[:n_eeg_f])
    ppg_imp = np.sum(importances[n_eeg_f:n_eeg_f + n_ppg_f])
    imu_imp = np.sum(importances[n_eeg_f + n_ppg_f:])

    total_imp = eeg_imp + ppg_imp + imu_imp
    eeg_pct = (eeg_imp / total_imp) * 100
    ppg_pct = (ppg_imp / total_imp) * 100
    imu_pct = (imu_imp / total_imp) * 100

    print(f"  • EEG Feature Contribution    : {eeg_pct:6.2f}% ({n_eeg_f} features)")
    print(f"  • Smartwatch PPG Contribution: {ppg_pct:6.2f}% ({n_ppg_f} features)")
    print(f"  • Smartwatch IMU Contribution: {imu_pct:6.2f}% ({n_imu_f} features)")

    results['modality_contributions_pct'] = {
        'EEG': float(eeg_pct),
        'PPG': float(ppg_pct),
        'IMU_Motion': float(imu_pct)
    }

    # ---------------------------------------------------------
    # Generate Ablation Comparison Chart
    # ---------------------------------------------------------
    labels = [k.replace("1. ", "").replace("2. ", "").replace("3. ", "").replace("4. ", "").replace("5. ", "").replace("6. ", "").replace("7. ", "") for k in results if 'GradientBoosting_mean' in results[k]]
    gb_means = [results[k]['GradientBoosting_mean'] for k in results if 'GradientBoosting_mean' in results[k]]
    gb_stds = [results[k]['GradientBoosting_std'] for k in results if 'GradientBoosting_mean' in results[k]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})

    bars = ax1.barh(labels, gb_means, xerr=gb_stds, color='#00ADB5', edgecolor='#2C354A', capsize=4)
    ax1.axvline(chance_level, color='#E74C3C', linestyle='--', linewidth=2, label=f'Chance Level ({chance_level}%)')
    ax1.set_xlabel("4-Class Accuracy (%)", fontweight='bold')
    ax1.set_title("Multimodal Ablation Study Accuracy Comparison", fontweight='bold', fontsize=12)
    ax1.set_xlim(0, 45)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Highlight full fusion bar
    bars[-1].set_color('#6C5CE7')

    # Pie chart of feature contribution
    contrib_labels = ['EEG Features', 'Smartwatch PPG', 'Smartwatch IMU']
    contrib_vals = [eeg_pct, ppg_pct, imu_pct]
    colors = ['#74B9FF', '#8E44AD', '#FF7675']
    ax2.pie(contrib_vals, labels=contrib_labels, autopct='%1.1f%%', colors=colors, startangle=140)
    ax2.set_title("Modality Gini Feature Contribution Split", fontweight='bold', fontsize=12)

    plt.tight_layout()
    chart_path = os.path.join(out_dir, "multimodal_ablation_study.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[+] Ablation chart saved to: {chart_path}")

    # Export JSON Summary
    json_path = os.path.join(out_dir, "multimodal_ablation_study.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[+] Ablation JSON report saved to: {json_path}")
    print("=" * 80)

    return results, chart_path, json_path


if __name__ == '__main__':
    run_ablation_study()
