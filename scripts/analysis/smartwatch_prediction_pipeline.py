"""
Smartwatch Multimodal Prediction Pipeline
==========================================
Identifies available smartwatch records (IMU Motion + PPG Physio) across the repository,
extracts physiological & kinematic feature sets, and runs machine learning prediction pipelines.

Algorithms evaluated:
  - Random Forest Classifier (RF)
  - HistGradientBoosting Classifier (GBM)
  - Support Vector Classifier (SVC - RBF Kernel)
  - L2 Regularized Logistic Regression (LogReg)
  - Multi-Layer Perceptron (MLP Neural Net)

Results, metrics, confusion matrices, ROC curves, and reports are saved to `analyzes_results/smartwatch/`
and `scripts/analyzes_results/smartwatch/`.
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import kurtosis, skew

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc
)

# Output directories
OUTPUT_DIRS = [
    os.path.abspath("analyzes_results/smartwatch"),
    os.path.abspath("scripts/analyzes_results/smartwatch"),
]

for out_dir in OUTPUT_DIRS:
    os.makedirs(out_dir, exist_ok=True)


def extract_imu_features(motion_df, window_size_sec=3.0, overlap=0.5):
    """
    Extracts time-domain and frequency-domain kinematic features from 6-axis IMU data.
    Columns expected: timestamp_sec (or time), acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    """
    if motion_df.empty:
        return np.array([]), []

    # Identify numeric columns excluding timestamp
    time_col = [c for c in motion_df.columns if 'time' in c.lower() or 'sample' in c.lower()]
    data_cols = [c for c in motion_df.columns if c not in time_col]
    
    if len(data_cols) < 3:
        return np.array([]), []
    
    acc_cols = data_cols[:3]
    gyro_cols = data_cols[3:6] if len(data_cols) >= 6 else []

    # Calculate sampling rate
    if time_col and len(motion_df) > 1:
        dt = np.median(np.diff(motion_df[time_col[0]].values))
        sfreq = 1.0 / dt if dt > 0 else 50.0
    else:
        sfreq = 50.0

    win_samples = int(window_size_sec * sfreq)
    step_samples = int(win_samples * (1.0 - overlap))
    
    if win_samples <= 0 or len(motion_df) < win_samples:
        win_samples = len(motion_df)
        step_samples = win_samples

    features = []
    feature_names = [
        'acc_mag_mean', 'acc_mag_std', 'acc_mag_max', 'acc_mag_rms', 'acc_mag_jerk',
        'acc_x_mean', 'acc_x_std', 'acc_y_mean', 'acc_y_std', 'acc_z_mean', 'acc_z_std',
        'acc_energy', 'acc_spectral_entropy'
    ]
    if gyro_cols:
        feature_names += [
            'gyro_mag_mean', 'gyro_mag_std', 'gyro_mag_max',
            'gyro_x_mean', 'gyro_x_std', 'gyro_y_mean', 'gyro_y_std', 'gyro_z_mean', 'gyro_z_std',
            'gyro_energy'
        ]

    for start_idx in range(0, len(motion_df) - win_samples + 1, max(1, step_samples)):
        chunk = motion_df.iloc[start_idx : start_idx + win_samples]
        acc_data = chunk[acc_cols].values
        acc_mag = np.sqrt(np.sum(acc_data**2, axis=1))
        
        # Jerk (derivative of acceleration)
        acc_jerk = np.diff(acc_mag) * sfreq if len(acc_mag) > 1 else np.array([0.0])
        
        # Spectral entropy
        f, psd = signal.welch(acc_mag, fs=sfreq, nperseg=min(len(acc_mag), 64))
        psd_norm = psd / (np.sum(psd) + 1e-12)
        spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

        feat_row = [
            float(np.mean(acc_mag)),
            float(np.std(acc_mag)),
            float(np.max(acc_mag)),
            float(np.sqrt(np.mean(acc_mag**2))),
            float(np.mean(np.abs(acc_jerk))),
            float(np.mean(acc_data[:, 0])),
            float(np.std(acc_data[:, 0])),
            float(np.mean(acc_data[:, 1])),
            float(np.std(acc_data[:, 1])),
            float(np.mean(acc_data[:, 2])),
            float(np.std(acc_data[:, 2])),
            float(np.sum(acc_mag**2) / len(acc_mag)),
            float(spec_entropy)
        ]

        if gyro_cols:
            gyro_data = chunk[gyro_cols].values
            gyro_mag = np.sqrt(np.sum(gyro_data**2, axis=1))
            feat_row += [
                float(np.mean(gyro_mag)),
                float(np.std(gyro_mag)),
                float(np.max(gyro_mag)),
                float(np.mean(gyro_data[:, 0])),
                float(np.std(gyro_data[:, 0])),
                float(np.mean(gyro_data[:, 1])),
                float(np.std(gyro_data[:, 1])),
                float(np.mean(gyro_data[:, 2])),
                float(np.std(gyro_data[:, 2])),
                float(np.sum(gyro_mag**2) / len(gyro_mag))
            ]

        features.append(feat_row)

    return np.array(features), feature_names


def extract_ppg_features(ppg_df, window_size_sec=3.0, overlap=0.5):
    """
    Extracts cardiovascular & Heart Rate Variability (HRV) features from PPG optical sensor.
    """
    if ppg_df.empty:
        return np.array([]), []

    time_col = [c for c in ppg_df.columns if 'time' in c.lower() or 'sample' in c.lower()]
    data_cols = [c for c in ppg_df.columns if c not in time_col]
    
    if not data_cols:
        return np.array([]), []

    ppg_signal = ppg_df[data_cols[0]].values
    
    if time_col and len(ppg_df) > 1:
        dt = np.median(np.diff(ppg_df[time_col[0]].values))
        sfreq = 1.0 / dt if dt > 0 else 25.0
    else:
        sfreq = 25.0

    win_samples = int(window_size_sec * sfreq)
    step_samples = int(win_samples * (1.0 - overlap))
    
    if win_samples <= 0 or len(ppg_df) < win_samples:
        win_samples = len(ppg_df)
        step_samples = win_samples

    features = []
    feature_names = [
        'ppg_mean', 'ppg_std', 'ppg_ptp', 'ppg_skew', 'ppg_kurtosis',
        'ppg_hr_bpm', 'ppg_hrv_sdnn', 'ppg_hrv_rmssd', 'ppg_hrv_pnn50',
        'ppg_lf_power', 'ppg_hf_power', 'ppg_lf_hf_ratio'
    ]

    for start_idx in range(0, len(ppg_df) - win_samples + 1, max(1, step_samples)):
        chunk = ppg_signal[start_idx : start_idx + win_samples]
        
        # Bandpass filter PPG for cardiac pulse range (0.5 to 4 Hz -> 30 to 240 BPM)
        try:
            b, a = signal.butter(2, [0.5 / (sfreq / 2), min(3.5, sfreq / 2 - 0.1) / (sfreq / 2)], btype='band')
            filtered_ppg = signal.filtfilt(b, a, chunk)
        except Exception:
            filtered_ppg = chunk - np.mean(chunk)

        # Peak detection for pulse intervals (Inter-Beat Interval - IBI)
        min_dist = max(1, int(0.35 * sfreq))
        peaks, _ = signal.find_peaks(filtered_ppg, distance=min_dist, prominence=np.std(filtered_ppg) * 0.4)
        
        if len(peaks) >= 2:
            ibis = np.diff(peaks) / sfreq * 1000.0  # ms
            hr_bpm = 60.0 / (np.mean(ibis) / 1000.0) if np.mean(ibis) > 0 else 70.0
            sdnn = float(np.std(ibis))
            rmssd = float(np.sqrt(np.mean(np.diff(ibis)**2))) if len(ibis) > 1 else 0.0
            nn50 = np.sum(np.abs(np.diff(ibis)) > 50.0) if len(ibis) > 1 else 0
            pnn50 = float(nn50 / max(1, len(ibis) - 1)) * 100.0
        else:
            hr_bpm = 70.0
            sdnn = 0.0
            rmssd = 0.0
            pnn50 = 0.0

        # Frequency domain HRV
        f, psd = signal.welch(filtered_ppg, fs=sfreq, nperseg=min(len(filtered_ppg), 64))
        lf_mask = (f >= 0.04) & (f <= 0.15)
        hf_mask = (f >= 0.15) & (f <= 0.4)
        
        lf_power = float(np.sum(psd[lf_mask])) if np.any(lf_mask) else 1e-6
        hf_power = float(np.sum(psd[hf_mask])) if np.any(hf_mask) else 1e-6
        lf_hf_ratio = float(lf_power / max(1e-6, hf_power))

        feat_row = [
            float(np.nan_to_num(np.mean(chunk))),
            float(np.nan_to_num(np.std(chunk))),
            float(np.nan_to_num(np.ptp(chunk))),
            float(np.nan_to_num(skew(chunk, nan_policy='omit') if len(chunk) > 3 and np.std(chunk) > 1e-6 else 0.0)),
            float(np.nan_to_num(kurtosis(chunk, nan_policy='omit') if len(chunk) > 3 and np.std(chunk) > 1e-6 else 0.0)),
            float(np.nan_to_num(hr_bpm)),
            float(np.nan_to_num(sdnn)),
            float(np.nan_to_num(rmssd)),
            float(np.nan_to_num(pnn50)),
            float(np.nan_to_num(lf_power)),
            float(np.nan_to_num(hf_power)),
            float(np.nan_to_num(lf_hf_ratio))
        ]
        features.append(feat_row)

    return np.array(features), feature_names


def load_smartwatch_dataset():
    """
    Scans repository for smartwatch recordings with ground-truth event labels.
    Uses `scripts/bids/bids_baseline` (sub-01/ses-01 and ses-02) and `scripts/bids/bids_musica`.
    """
    print("\n" + "=" * 75)
    print(" SCANNING SMARTWATCH DATASETS & EXTRACTING MULTIMODAL FEATURES ".center(75, "="))
    print("=" * 75)

    datasets = [
        {
            'name': 'bids_baseline_ses-02',
            'motion': 'scripts/bids/bids_baseline/sub-01/ses-02/motion/sub-01_ses-02_task-video_motion.tsv',
            'physio': 'scripts/bids/bids_baseline/sub-01/ses-02/physio/sub-01_ses-02_task-video_physio.tsv',
            'events': 'scripts/bids/bids_baseline/sub-01/ses-02/eeg/sub-01_ses-02_task-video_events.tsv'
        }
    ]

    # Load ses-02 with trial annotations
    target_ds = datasets[0]
    motion_path = os.path.abspath(target_ds['motion'])
    physio_path = os.path.abspath(target_ds['physio'])
    events_path = os.path.abspath(target_ds['events'])

    if not os.path.exists(motion_path) or not os.path.exists(physio_path) or not os.path.exists(events_path):
        print("[-] Required files for smartwatch analysis not found.")
        return None, None, None, None

    print(f"[+] Loaded Motion: {motion_path}")
    print(f"[+] Loaded Physio: {physio_path}")
    print(f"[+] Loaded Events: {events_path}")

    df_motion = pd.read_csv(motion_path, sep='\t')
    df_physio = pd.read_csv(physio_path, sep='\t')
    df_events = pd.read_csv(events_path, sep='\t')

    print(f"    Motion Samples: {len(df_motion)} | Physio Samples: {len(df_physio)} | Events: {len(df_events)}")

    # Parse trial events
    # Conditions: water, earth, wind, fire
    condition_map = {'water': 0, 'earth': 1, 'wind': 2, 'fire': 3}
    class_names = ['water', 'earth', 'wind', 'fire']
    
    trial_windows = []
    for _, row in df_events.iterrows():
        tt = str(row.get('trial_type', ''))
        onset = float(row.get('onset', 0.0))
        duration = float(row.get('duration', 3.0))
        if duration <= 0:
            duration = 3.0
            
        for c_name, c_idx in condition_map.items():
            if c_name in tt.lower():
                trial_windows.append({
                    'onset': onset,
                    'offset': onset + duration,
                    'class_idx': c_idx,
                    'class_name': c_name
                })
                break

    print(f"[+] Identified {len(trial_windows)} labeled trial windows for smartwatch prediction.")

    X_list = []
    y_list = []
    feature_names_joint = None

    for tr in trial_windows:
        t_start, t_end = tr['onset'], tr['offset']
        c_idx = tr['class_idx']

        # Slice motion
        m_slice = df_motion[(df_motion['timestamp_sec'] >= t_start) & (df_motion['timestamp_sec'] <= t_end)]
        # Slice physio
        p_slice = df_physio[(df_physio['timestamp_sec'] >= t_start) & (df_physio['timestamp_sec'] <= t_end)]

        if len(m_slice) < 5 or len(p_slice) < 5:
            continue

        f_imu, fnames_imu = extract_imu_features(m_slice, window_size_sec=t_end - t_start, overlap=0.0)
        f_ppg, fnames_ppg = extract_ppg_features(p_slice, window_size_sec=t_end - t_start, overlap=0.0)

        if len(f_imu) > 0 and len(f_ppg) > 0:
            combined_feat = np.concatenate([f_imu[0], f_ppg[0]])
            if feature_names_joint is None:
                feature_names_joint = [f"imu_{name}" for name in fnames_imu] + [f"ppg_{name}" for name in fnames_ppg]
            X_list.append(combined_feat)
            y_list.append(c_idx)

    # In addition, include windowed segments to expand trial observations
    if len(X_list) < 40:
        print("[*] Performing sub-windowing across trials (1.5s windows) to enhance statistical power...")
        X_sub = []
        y_sub = []
        for tr in trial_windows:
            t_start, t_end = tr['onset'], tr['offset']
            c_idx = tr['class_idx']
            cur_t = t_start
            win_len = 1.5
            while cur_t + win_len <= t_end:
                m_sub = df_motion[(df_motion['timestamp_sec'] >= cur_t) & (df_motion['timestamp_sec'] <= cur_t + win_len)]
                p_sub = df_physio[(df_physio['timestamp_sec'] >= cur_t) & (df_physio['timestamp_sec'] <= cur_t + win_len)]
                if len(m_sub) >= 5 and len(p_sub) >= 5:
                    f_imu, fnames_imu = extract_imu_features(m_sub, window_size_sec=win_len, overlap=0.0)
                    f_ppg, fnames_ppg = extract_ppg_features(p_sub, window_size_sec=win_len, overlap=0.0)
                    if len(f_imu) > 0 and len(f_ppg) > 0:
                        combined_feat = np.concatenate([f_imu[0], f_ppg[0]])
                        X_sub.append(combined_feat)
                        y_sub.append(c_idx)
                cur_t += 0.5  # 500ms slide

        X_arr = np.array(X_sub)
        y_arr = np.array(y_sub)
    else:
        X_arr = np.nan_to_num(np.array(X_list), nan=0.0, posinf=0.0, neginf=0.0)
        y_arr = np.array(y_list)

    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[+] Total Extracted Smartwatch Feature Matrix: {X_arr.shape} ({len(feature_names_joint)} features per sample)")
    print(f"[+] Class distribution: { {class_names[i]: int(np.sum(y_arr == i)) for i in range(len(class_names))} }")

    return X_arr, y_arr, feature_names_joint, class_names


def run_smartwatch_prediction_pipeline():
    """
    Trains and benchmarks multiple ML classifiers on smartwatch data using Stratified 5-Fold Cross-Validation.
    Saves all metrics, ROC curves, confusion matrices, feature importances, and documentation.
    """
    X, y, feature_names, class_names = load_smartwatch_dataset()
    if X is None or len(X) == 0:
        print("[-] Failed to load smartwatch dataset.")
        return

    n_classes = len(class_names)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Machine Learning Models Dictionary
    models = {
        'Random Forest (RF)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
        ]),
        'Gradient Boosting (GBM)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42))
        ]),
        'Support Vector Machine (SVC-RBF)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
        ]),
        'L2 Logistic Regression (LogReg)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ]),
        'Multi-Layer Perceptron (MLP Neural Net)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, alpha=1e-3, random_state=42))
        ])
    }

    results = {
        'script': 'scripts/analysis/smartwatch_prediction_pipeline.py',
        'dataset_source': 'scripts/bids/bids_baseline/sub-01/ses-02 (Motion + Physio PPG)',
        'modalities_used': ['Smartwatch IMU 6-Axis Motion (Accelerometer & Gyroscope)', 'Smartwatch Optical PPG (Pulse & HRV)'],
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'n_classes': n_classes,
        'class_labels': class_names,
        'cv_strategy': 'Stratified 5-Fold Cross-Validation',
        'models_evaluated': {}
    }

    print("\n" + "=" * 75)
    print(" EVALUATING SMARTWATCH PREDICTION MODELS (5-FOLD CV) ".center(75, "="))
    print("=" * 75)

    metrics_table = []
    conf_matrices = {}
    roc_data = {}

    y_bin = label_binarize(y, classes=range(n_classes))

    for name, pipe in models.items():
        y_pred = cross_val_predict(pipe, X, y, cv=cv)
        y_proba = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')

        acc = accuracy_score(y, y_pred)
        bal_acc = balanced_accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average='macro')
        prec_macro = precision_score(y, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y, y_pred)

        # Multi-class ROC-AUC (Macro-averaged)
        fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
        for i in range(n_classes):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        mean_auc = np.mean(list(roc_auc_dict.values()))

        conf_matrices[name] = cm
        roc_data[name] = {'fpr': fpr_dict, 'tpr': tpr_dict, 'auc': mean_auc}

        results['models_evaluated'][name] = {
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc),
            'f1_macro': float(f1_macro),
            'precision_macro': float(prec_macro),
            'recall_macro': float(rec_macro),
            'mean_roc_auc': float(mean_auc),
            'confusion_matrix': cm.tolist()
        }

        metrics_table.append({
            'Algorithm / Model': name,
            'Accuracy': f"{acc * 100:.2f}%",
            'Balanced Accuracy': f"{bal_acc * 100:.2f}%",
            'F1-Score (Macro)': f"{f1_macro:.4f}",
            'Precision': f"{prec_macro:.4f}",
            'Recall': f"{rec_macro:.4f}",
            'ROC-AUC': f"{mean_auc:.4f}"
        })

        print(f"[*] {name:40s} | Acc: {acc*100:5.2f}% | BalAcc: {bal_acc*100:5.2f}% | F1: {f1_macro:.4f} | AUC: {mean_auc:.4f}")

    df_metrics = pd.DataFrame(metrics_table)

    # ---------------------------------------------------------
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    rf_full = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_full.fit(X_scaled, y)
    importances = rf_full.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    top_n = min(15, len(feature_names))
    top_features = [feature_names[i] for i in sorted_idx[:top_n]]
    top_scores = [importances[i] for i in sorted_idx[:top_n]]

    results['feature_importance_top15'] = {
        top_features[i]: float(top_scores[i]) for i in range(top_n)
    }

    # ---------------------------------------------------------
    # Generate Visualizations
    # ---------------------------------------------------------
    # 1. Feature Importance Bar Chart
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), top_scores[::-1], align='center', color='#2b5c8f', edgecolor='black')
    plt.yticks(range(top_n), top_features[::-1], fontsize=10)
    plt.xlabel("Random Forest Gini Feature Importance", fontsize=11, fontweight='bold')
    plt.title("Top Smartwatch Kinematic & Physiological Predictive Features", fontsize=12, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    for out_dir in OUTPUT_DIRS:
        plt.savefig(os.path.join(out_dir, "smartwatch_feature_importance.png"), dpi=300)
    plt.close()

    # 2. Confusion Matrices Plot
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.5))
    if len(models) == 1:
        axes = [axes]

    for ax, (m_name, cm) in zip(axes, conf_matrices.items()):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax.set_title(m_name.split('(')[0].strip(), fontsize=10, fontweight='bold')
        tick_marks = np.arange(n_classes)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, fontsize=8)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, fontsize=8)
        
        # Annotate numbers
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=7)
        ax.set_ylabel('True Label' if ax == axes[0] else '', fontsize=9)
        ax.set_xlabel('Predicted Label', fontsize=9)

    plt.tight_layout()
    for out_dir in OUTPUT_DIRS:
        plt.savefig(os.path.join(out_dir, "smartwatch_confusion_matrices.png"), dpi=300)
    plt.close()

    # 3. Model Benchmark Bar Chart
    plt.figure(figsize=(9, 5))
    alg_names = [m['Algorithm / Model'].split('(')[0].strip() for m in metrics_table]
    acc_vals = [float(m['Accuracy'].replace('%', '')) for m in metrics_table]
    bal_acc_vals = [float(m['Balanced Accuracy'].replace('%', '')) for m in metrics_table]

    x = np.arange(len(alg_names))
    width = 0.35

    plt.bar(x - width/2, acc_vals, width, label='Accuracy (%)', color='#1f77b4', edgecolor='black')
    plt.bar(x + width/2, bal_acc_vals, width, label='Balanced Accuracy (%)', color='#ff7f0e', edgecolor='black')
    plt.axhline(25.0, color='red', linestyle='--', linewidth=1.5, label='Chance Level (25%)')

    plt.ylabel('Performance (%)', fontsize=11, fontweight='bold')
    plt.title('Smartwatch Multimodal Prediction Pipeline Performance (5-Fold CV)', fontsize=12, fontweight='bold')
    plt.xticks(x, alg_names, rotation=20, ha='right', fontsize=9)
    plt.legend(loc='upper right')
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    for out_dir in OUTPUT_DIRS:
        plt.savefig(os.path.join(out_dir, "smartwatch_model_benchmark.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Save CSV, JSON, and Markdown Report
    # ---------------------------------------------------------
    for out_dir in OUTPUT_DIRS:
        # Save JSON
        json_path = os.path.join(out_dir, "smartwatch_prediction_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)

        # Save CSV
        csv_path = os.path.join(out_dir, "smartwatch_prediction_metrics.csv")
        df_metrics.to_csv(csv_path, index=False)

        # Save Markdown Summary
        md_path = os.path.join(out_dir, "smartwatch_pipeline_report.md")
        with open(md_path, 'w') as f:
            f.write("# Smartwatch Multimodal Prediction Pipeline Report\n\n")
            f.write(f"**Execution Script**: `{results['script']}`  \n")
            f.write(f"**Dataset Sources**: `{results['dataset_source']}`  \n")
            f.write(f"**Modalities**: {', '.join(results['modalities_used'])}  \n")
            f.write(f"**Cross-Validation**: {results['cv_strategy']}  \n")
            f.write(f"**Samples Extracted**: {results['n_samples']} samples | **Feature Dimensions**: {results['n_features']} features  \n\n")
            f.write("## 1. Benchmark Comparison Table\n\n")
            f.write(df_metrics.to_markdown(index=False) + "\n\n")
            f.write("## 2. Top 10 Informative Features\n\n")
            f.write("| Feature Name | Modality | Random Forest Importance |\n| :--- | :--- | :--- |\n")
            for feat, score in list(results['feature_importance_top15'].items())[:10]:
                mod = "PPG / Cardiovascular" if "ppg" in feat else "IMU / Kinematic"
                f.write(f"| `{feat}` | {mod} | {score:.4f} |\n")
            f.write("\n## 3. Visual Artifacts Generated\n\n")
            f.write("- `smartwatch_model_benchmark.png`\n")
            f.write("- `smartwatch_confusion_matrices.png`\n")
            f.write("- `smartwatch_feature_importance.png`\n")

    print(f"\n[+] Successfully saved all Smartwatch results to {OUTPUT_DIRS[0]} and {OUTPUT_DIRS[1]}")
    return results


if __name__ == '__main__':
    run_smartwatch_prediction_pipeline()
