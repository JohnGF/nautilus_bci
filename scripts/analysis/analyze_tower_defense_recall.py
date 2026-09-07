"""
Tower Defense BCI Recall & Memory / Visual Blinking Analysis Studio
====================================================================
Analyzes the BIDS Tower Defense dataset (`scripts/bids_tower_defense`), supporting:
  - ses-01: Full Recall (auditory song imagination / element recall)
  - ses-02: Half Recall (scene_four_elements) & Half Memory (scene_reverse_four_elements)
  - ses-03: Full Memory (element imagery / memory retention)
  - ses-04: Full Memory (reverse four elements)

Phases analyzed per trial:
  1. Visual Flicker Phase: Participant looks at a blinking box (`Box start blinking`).
  2. Auditory Recall / Memory Imagery Phase: Participant imagines the selected song/element
     (`FIRE`, `WATER`, `WIND`, `ELECTRICITY`) after `Box stop blinking`.

Applies decoding and neuro-statistical algorithms from `scripts/analysis`:
  - Robust Spatial Referencing & Bad Channel Detection (`spatial_filters.py`)
  - Relative Band Power & Spectral Density Extraction (Delta, Theta, Alpha, Beta, Gamma)
  - Riemannian Geometry Decoders (Covariances + Tangent Space Logistic Regression / SVM, MDM)
  - Common Spatial Patterns (One-vs-Rest CSP + Shrinkage LDA)
  - Bandpower feature decoders (Multi-Band PSD + Random Forest / LDA)
  - Deep Learning EEGNet (PyTorch Architecture)
  - Multi-class cross-validation, confusion matrices, and publication-quality plots.
"""

import sys
import os
import glob
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure scripts directory and analysis directory are in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_script_dir, ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.decoding import CSP

# Machine Learning Imports
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
import scipy.stats as stats

# Riemannian Geometry
import pyriemann
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Internal module imports from scripts/analysis
from spatial_filters import detect_bad_channels, apply_spatial_filter


# ---------------------------------------------------------
# PyTorch EEGNet Architecture (Lawhern et al., 2018)
# ---------------------------------------------------------
class EEGNet(nn.Module):
    def __init__(self, n_channels=32, n_samples=1000, n_classes=4, F1=8, D=2, F2=16, kernel_length=32, dropout_rate=0.25):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        self._flatten_dim = None
        self._calculate_flatten_dim(n_channels, n_samples)
        self.classifier = nn.Linear(self._flatten_dim, n_classes)

    def _calculate_flatten_dim(self, n_ch, n_s):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_ch, n_s)
            x = self.drop1(self.pool1(self.act1(self.bn2(self.depthwise(self.bn1(self.conv1(x)))))))
            x = self.drop2(self.pool2(self.act2(self.bn3(self.separable(x)))))
            self._flatten_dim = x.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_eval_eegnet(X, y, n_classes=4, n_splits=5, epochs=35, lr=0.005, batch_size=16):
    """Cross Validation for EEGNet."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_epochs_count, n_channels, n_samples = X.shape
    acc_scores = []
    y_preds_all = np.zeros_like(y)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        t_X_train = torch.tensor(X_train[:, np.newaxis, :, :], dtype=torch.float32)
        t_y_train = torch.tensor(y_train, dtype=torch.long)
        t_X_test = torch.tensor(X_test[:, np.newaxis, :, :], dtype=torch.float32)
        t_y_test = torch.tensor(y_test, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(t_X_train, t_y_train), batch_size=batch_size, shuffle=True)

        model = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for ep in range(epochs):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(t_X_test.to(device)).argmax(dim=1).cpu().numpy()
            acc = accuracy_score(y_test, preds)
            acc_scores.append(acc)
            y_preds_all[test_idx] = preds

    return np.array(acc_scores), y_preds_all


# ---------------------------------------------------------
# Multi-class One-vs-Rest CSP Feature Extraction
# ---------------------------------------------------------
def extract_ovr_csp_features(X, y, n_components=4):
    """Extracts One-vs-Rest CSP features for multi-class EEG."""
    classes = np.unique(y)
    feature_blocks = []
    for c in classes:
        y_bin = (y == c).astype(int)
        actual_comp = min(n_components, X.shape[1] // 2)
        csp = CSP(n_components=actual_comp, reg='oas', log=True, norm_trace=False)
        feat = csp.fit_transform(X, y_bin)
        feature_blocks.append(feat)
    return np.hstack(feature_blocks)


# ---------------------------------------------------------
# Bandpower Feature Extraction (Welch PSD)
# ---------------------------------------------------------
def extract_bandpower_features(X, sfreq=250.0):
    """
    Computes relative power across 5 classical EEG bands:
    Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-12 Hz), Beta (13-30 Hz), Gamma (30-45 Hz)
    """
    n_epochs, n_ch, n_samples = X.shape
    bands = {
        'delta': (1.0, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 12.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 45.0)
    }
    
    n_per_seg = min(int(sfreq * 1.5), n_samples)
    from scipy.signal import welch
    freqs, psd = welch(X, fs=sfreq, nperseg=n_per_seg, axis=-1)
    
    total_power = np.sum(psd, axis=-1, keepdims=True) + 1e-12
    rel_psd = psd / total_power
    
    feats = []
    for (fmin, fmax) in bands.values():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_p = np.mean(rel_psd[:, :, idx], axis=-1)
        feats.append(band_p)
        
    return np.hstack(feats)


# ---------------------------------------------------------
# BIDS File Auto-Discovery Helper
# ---------------------------------------------------------
def normalize_ses_id(ses_str):
    """Ensures consistent two-digit zero-padded session ID (e.g. '01', '02')."""
    clean = str(ses_str).strip().replace("ses-", "")
    if clean.isdigit():
        return f"{int(clean):02d}"
    return clean


def find_bids_session_info(bids_root, sub_clean, ses_clean, task=None):
    """Finds matching BIDS header/events files in session folder."""
    ses_norm = normalize_ses_id(ses_clean)
    eeg_dir = os.path.join(bids_root, f"sub-{sub_clean}", f"ses-{ses_norm}", "eeg")
    if not os.path.exists(eeg_dir):
        raise FileNotFoundError(f"EEG session directory not found: {eeg_dir}")

    vhdr_files = glob.glob(os.path.join(eeg_dir, "*.vhdr"))
    if not vhdr_files:
        raise FileNotFoundError(f"No .vhdr EEG files found in: {eeg_dir}")

    if task:
        for f in vhdr_files:
            if f"task-{task}_" in os.path.basename(f):
                return task, f
    
    # Auto-detect task from filename
    chosen_file = vhdr_files[0]
    base = os.path.basename(chosen_file)
    task_name = "recall"
    if "task-" in base:
        task_name = base.split("task-")[1].split("_")[0]

    return task_name, chosen_file


# ---------------------------------------------------------
# Event Parsing & Condition Labeling
# ---------------------------------------------------------
def parse_tower_defense_events(events_tsv_path, sfreq, ses_clean):
    """
    Parses events TSV and tags condition ('recall' vs 'memory'):
      - ses-01: full recall
      - ses-02: scene_four_elements = recall (half), scene_reverse_four_elements = memory (half)
      - ses-03: full memory (as specified by study protocol)
      - ses-04: full memory
    """
    df_events = pd.read_csv(events_tsv_path, sep='\t')
    class_map = {'FIRE': 0, 'WATER': 1, 'WIND': 2, 'ELECTRICITY': 3}

    recall_events = []
    blink_events = []
    conditions = []
    trial_metadata = []

    current_condition = "recall"
    if ses_clean == "03" or ses_clean == "04":
        current_condition = "memory"

    events_list = df_events.to_dict('records')
    for i, ev in enumerate(events_list):
        tt = str(ev.get('trial_type', ''))
        
        # Track scene start to determine recall vs memory condition
        if 'Game_Started_scene' in tt:
            if 'reverse' in tt or ses_clean in ["03", "04"]:
                current_condition = "memory"
            else:
                current_condition = "recall"

        if tt == 'Box stop blinking':
            label_str = None
            # Search nearby for selected element
            for j in range(max(0, i - 2), min(len(events_list), i + 3)):
                cand = str(events_list[j].get('trial_type', ''))
                if 'selected' in cand:
                    if abs(events_list[j]['onset'] - ev['onset']) < 0.25:
                        label_str = cand.replace(' selected', '').strip()
                        break
            
            if label_str in class_map:
                class_id = class_map[label_str]
                sample_idx = int(ev['sample']) if 'sample' in ev and not np.isnan(ev['sample']) else int(ev['onset'] * sfreq)
                recall_events.append([sample_idx, 0, class_id])
                conditions.append(current_condition)
                trial_metadata.append({
                    'sample': sample_idx,
                    'onset': ev['onset'],
                    'class_name': label_str,
                    'class_id': class_id,
                    'condition': current_condition,
                    'session': ses_clean
                })

                # Check preceding blinking onset
                if i > 0 and events_list[i-1].get('trial_type') == 'Box start blinking':
                    b_sample = int(events_list[i-1]['sample']) if 'sample' in events_list[i-1] and not np.isnan(events_list[i-1]['sample']) else int(events_list[i-1]['onset'] * sfreq)
                    blink_events.append([b_sample, 0, class_id])

    return (
        np.array(recall_events),
        np.array(blink_events),
        np.array(conditions),
        pd.DataFrame(trial_metadata),
        class_map
    )


# ---------------------------------------------------------
# Single Session Analysis Pipeline
# ---------------------------------------------------------
def run_tower_defense_session_analysis(
    bids_root="scripts/bids_tower_defense",
    subject_id="01",
    session_id="02",
    task=None,
    condition="all",
    out_dir="scripts/analysis_results/tower_defense_recall",
    recall_tmin=0.5,
    recall_tmax=4.5,
    blink_tmin=0.5,
    blink_tmax=4.5,
    spatial_filter="robust_car"
):
    sub_clean = subject_id.replace("sub-", "")
    ses_clean = session_id.replace("ses-", "")
    session_out_dir = os.path.join(out_dir, f"sub-{sub_clean}_ses-{ses_clean}")
    if condition != "all":
        session_out_dir += f"_{condition}"
    os.makedirs(session_out_dir, exist_ok=True)

    print("=" * 80)
    print(f" BCI TOWER DEFENSE: SUB-{sub_clean} SES-{ses_clean} [{condition.upper()}] ".center(80, "="))
    print("=" * 80)

    detected_task, vhdr_path = find_bids_session_info(bids_root, sub_clean, ses_clean, task)
    print(f"[*] Session File: {vhdr_path}")
    print(f"[*] Detected BIDS Task: '{detected_task}'")

    bids_path = BIDSPath(
        subject=sub_clean,
        session=ses_clean,
        task=detected_task,
        datatype="eeg",
        root=bids_root
    )

    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()

    # Channel handling & units
    misc_chans = [ch for ch in raw.ch_names if ch.upper() in ['BATTERY', 'STATUS', 'AUX'] or ch == 'EEG033']
    if misc_chans:
        raw.set_channel_types({ch: 'misc' for ch in misc_chans})
    
    # Keep only active EEG channels
    raw.pick('eeg')
    
    sfreq = raw.info['sfreq']
    duration_s = raw.times[-1]
    ch_names = raw.ch_names
    print(f"[+] Loaded raw dataset: {sfreq} Hz | {len(ch_names)} active EEG channels | {duration_s:.1f}s duration")

    # Highpass baseline detrending for screening
    data_raw_arr = raw.get_data().T  # (n_samples, n_channels)
    stds = np.std(data_raw_arr, axis=0)
    is_volts = np.mean(stds) < 1e-3
    scale_factor = 1e6 if is_volts else 1.0
    data_uv = (data_raw_arr - np.mean(data_raw_arr, axis=0, keepdims=True)) * scale_factor

    bad_idx, bad_status = detect_bad_channels(data_uv, ch_names=ch_names)
    good_ch_names = [ch_names[i] for i in range(len(ch_names)) if i not in bad_idx]
    print(f"[+] Bad Channel Screening: {len(bad_idx)} bad channels detected: {[ch_names[i] for i in bad_idx]}")
    print(f"[+] Clean Channel Count: {len(good_ch_names)} / {len(ch_names)}")

    if spatial_filter != "none":
        print(f"[*] Applying spatial filter: '{spatial_filter}' referencing...")
        filt_data_uv = apply_spatial_filter(data_uv, ch_names, mode=spatial_filter)
        raw._data = (filt_data_uv / scale_factor).T

    # Bandpass filter (1.0 to 45.0 Hz) and 50 Hz Notch
    print("[*] Applying Bandpass Filter (1.0 - 45.0 Hz) & 50 Hz Notch filter...")
    raw_filt = raw.copy().filter(l_freq=1.0, h_freq=45.0, verbose=False)
    raw_filt.notch_filter(freqs=50.0, verbose=False)

    # Parse Events TSV
    events_tsv_path = os.path.join(bids_path.directory, f"sub-{sub_clean}_ses-{ses_clean}_task-{detected_task}_events.tsv")
    recall_events_all, blink_events_all, conditions_all, df_meta, class_map = parse_tower_defense_events(
        events_tsv_path, sfreq, ses_clean
    )
    class_names = ['FIRE', 'WATER', 'WIND', 'ELECTRICITY']
    event_id_dict = {k: v for k, v in class_map.items()}

    # Condition filtering
    if condition != "all":
        mask = (conditions_all == condition)
        recall_events = recall_events_all[mask]
        blink_events = blink_events_all[mask] if len(blink_events_all) == len(recall_events_all) else recall_events_all[mask]
        df_meta = df_meta[mask].reset_index(drop=True)
    else:
        recall_events = recall_events_all
        blink_events = blink_events_all

    print(f"\n[+] Trial Breakdown for sub-{sub_clean} ses-{ses_clean} ({condition}):")
    print(f"    - Total Identified Trials: {len(recall_events)}")
    if 'condition' in df_meta.columns:
        print(f"    - Paradigm breakdown:")
        for cond_type, cnt in df_meta['condition'].value_counts().items():
            print(f"      • {cond_type.capitalize()}: {cnt} trials")
    for c_name, c_id in class_map.items():
        cnt = np.sum(recall_events[:, 2] == c_id) if len(recall_events) > 0 else 0
        print(f"      • {c_name:12s}: {cnt} trials")

    if len(recall_events) == 0:
        print(f"[!] No trials found for condition '{condition}'. Skipping session.")
        return None

    # Epoching
    epochs_recall = mne.Epochs(
        raw_filt,
        recall_events,
        event_id=event_id_dict,
        tmin=recall_tmin,
        tmax=recall_tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    epochs_blink = mne.Epochs(
        raw_filt,
        blink_events,
        event_id=event_id_dict,
        tmin=blink_tmin,
        tmax=blink_tmax,
        baseline=None,
        preload=True,
        verbose=False
    ) if len(blink_events) > 0 else None

    X_recall_uv = epochs_recall.get_data() * 1e6
    y_recall = epochs_recall.events[:, 2]

    # Dynamic cross-validation split calculation
    class_counts = np.bincount(y_recall, minlength=4)
    min_class_count = int(np.min(class_counts))
    if min_class_count < 2:
        print(f"[!] Minimum class count is {min_class_count} (< 2). Cross-validation cannot be stratified.")
        n_splits = 2
    else:
        n_splits = min(5, min_class_count)

    print(f"\n[*] Cross-Validation Configuration: {n_splits}-Fold Stratified CV (min class size = {min_class_count})")

    # ---------------------------------------------------------
    # 1. Spectral Analysis & Power Spectral Densities
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2 if epochs_blink is not None else 1, figsize=(16 if epochs_blink is not None else 8, 6))
    if epochs_blink is None:
        axes = [axes]
    palette = {'FIRE': '#FF5722', 'WATER': '#2196F3', 'WIND': '#4CAF50', 'ELECTRICITY': '#FFC107'}

    plot_configs = []
    if epochs_blink is not None:
        plot_configs.append((epochs_blink, f"Visual Blinking Phase (sub-{sub_clean} ses-{ses_clean})", axes[0]))
        plot_configs.append((epochs_recall, f"Auditory/Memory Imagery Phase ({condition.capitalize()})", axes[1]))
    else:
        plot_configs.append((epochs_recall, f"Auditory/Memory Imagery Phase ({condition.capitalize()})", axes[0]))

    for idx, (ep_obj, title, ax) in enumerate(plot_configs):
        for c_name in class_names:
            if c_name in ep_obj.event_id:
                sub_ep = ep_obj[c_name]
                if len(sub_ep) > 0:
                    psd_obj = sub_ep.compute_psd(fmin=2.0, fmax=45.0, verbose=False)
                    psds, freqs = psd_obj.get_data(return_freqs=True)
                    mean_psd = np.mean(psds, axis=(0, 1)) * 1e12  # scale uV^2 / Hz
                    ax.plot(freqs, mean_psd, label=c_name, color=palette[c_name], linewidth=2.2)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('Power Spectral Density (µV²/Hz)', fontsize=11)
        ax.axvspan(8.0, 12.0, color='gold', alpha=0.15, label='Alpha (8-12 Hz)' if idx == 0 else "")
        ax.axvspan(13.0, 30.0, color='cyan', alpha=0.10, label='Beta (13-30 Hz)' if idx == 0 else "")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    plt.suptitle(f"Spectral PSD Comparison: Ses-{ses_clean} ({condition.capitalize()})", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    psd_plot_path = os.path.join(session_out_dir, "psd_spectral_comparison.png")
    plt.savefig(psd_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved PSD comparison plot: {psd_plot_path}")

    # ---------------------------------------------------------
    # 2. Multi-Model Decoding Suite
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f" Neural Decoding Benchmark: ses-{ses_clean} ({condition}) ".center(60, "="))
    print("=" * 60)

    models_results = {}
    confusion_matrices = {}

    if min_class_count >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Model 1: Riemannian Covariances + Tangent Space Logistic Regression
        pipe_ts_lr = Pipeline([
            ('cov', Covariances(estimator='oas')),
            ('ts', TangentSpace(metric='riemann')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, C=1.0))
        ])
        scores_ts_lr = cross_val_score(pipe_ts_lr, X_recall_uv, y_recall, cv=cv, scoring='accuracy')
        preds_ts_lr = cross_val_predict(pipe_ts_lr, X_recall_uv, y_recall, cv=cv)
        models_results['Riemannian TS + LogReg'] = {
            'accuracy_mean': float(np.mean(scores_ts_lr)),
            'accuracy_std': float(np.std(scores_ts_lr)),
            'balanced_acc': float(balanced_accuracy_score(y_recall, preds_ts_lr)),
            'f1_macro': float(f1_score(y_recall, preds_ts_lr, average='macro'))
        }
        confusion_matrices['Riemannian TS + LogReg'] = confusion_matrix(y_recall, preds_ts_lr, labels=[0,1,2,3])
        print(f"[+] [1/6] Riemannian TS (LogReg):     Acc = {np.mean(scores_ts_lr)*100:5.2f}% ± {np.std(scores_ts_lr)*100:4.2f}% | F1 = {f1_score(y_recall, preds_ts_lr, average='macro'):.3f}")

        # Model 2: Riemannian Covariances + Tangent Space SVM (RBF)
        pipe_ts_svm = Pipeline([
            ('cov', Covariances(estimator='oas')),
            ('ts', TangentSpace(metric='riemann')),
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0))
        ])
        scores_ts_svm = cross_val_score(pipe_ts_svm, X_recall_uv, y_recall, cv=cv, scoring='accuracy')
        preds_ts_svm = cross_val_predict(pipe_ts_svm, X_recall_uv, y_recall, cv=cv)
        models_results['Riemannian TS + SVM (RBF)'] = {
            'accuracy_mean': float(np.mean(scores_ts_svm)),
            'accuracy_std': float(np.std(scores_ts_svm)),
            'balanced_acc': float(balanced_accuracy_score(y_recall, preds_ts_svm)),
            'f1_macro': float(f1_score(y_recall, preds_ts_svm, average='macro'))
        }
        confusion_matrices['Riemannian TS + SVM (RBF)'] = confusion_matrix(y_recall, preds_ts_svm, labels=[0,1,2,3])
        print(f"[+] [2/6] Riemannian TS (SVM RBF):    Acc = {np.mean(scores_ts_svm)*100:5.2f}% ± {np.std(scores_ts_svm)*100:4.2f}% | F1 = {f1_score(y_recall, preds_ts_svm, average='macro'):.3f}")

        # Model 3: Riemannian Minimum Distance to Mean (MDM)
        pipe_mdm = Pipeline([
            ('cov', Covariances(estimator='oas')),
            ('clf', MDM(metric='riemann'))
        ])
        scores_mdm = cross_val_score(pipe_mdm, X_recall_uv, y_recall, cv=cv, scoring='accuracy')
        preds_mdm = cross_val_predict(pipe_mdm, X_recall_uv, y_recall, cv=cv)
        models_results['Riemannian MDM'] = {
            'accuracy_mean': float(np.mean(scores_mdm)),
            'accuracy_std': float(np.std(scores_mdm)),
            'balanced_acc': float(balanced_accuracy_score(y_recall, preds_mdm)),
            'f1_macro': float(f1_score(y_recall, preds_mdm, average='macro'))
        }
        confusion_matrices['Riemannian MDM'] = confusion_matrix(y_recall, preds_mdm, labels=[0,1,2,3])
        print(f"[+] [3/6] Riemannian MDM Classifier: Acc = {np.mean(scores_mdm)*100:5.2f}% ± {np.std(scores_mdm)*100:4.2f}% | F1 = {f1_score(y_recall, preds_mdm, average='macro'):.3f}")

        # Model 4: Multi-Class One-vs-Rest CSP + Shrinkage LDA
        try:
            raw_mubeta = raw.copy().filter(l_freq=8.0, h_freq=30.0, verbose=False)
            epochs_mubeta = mne.Epochs(raw_mubeta, recall_events, event_id=event_id_dict, tmin=recall_tmin, tmax=recall_tmax, baseline=None, preload=True, verbose=False)
            X_mubeta = epochs_mubeta.get_data() * 1e6
            X_csp_feats = extract_ovr_csp_features(X_mubeta, y_recall, n_components=min(4, max(2, min_class_count)))
            clf_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            scores_csp_lda = cross_val_score(clf_lda, X_csp_feats, y_recall, cv=cv, scoring='accuracy')
            preds_csp_lda = cross_val_predict(clf_lda, X_csp_feats, y_recall, cv=cv)
            models_results['One-vs-Rest CSP + LDA'] = {
                'accuracy_mean': float(np.mean(scores_csp_lda)),
                'accuracy_std': float(np.std(scores_csp_lda)),
                'balanced_acc': float(balanced_accuracy_score(y_recall, preds_csp_lda)),
                'f1_macro': float(f1_score(y_recall, preds_csp_lda, average='macro'))
            }
            confusion_matrices['One-vs-Rest CSP + LDA'] = confusion_matrix(y_recall, preds_csp_lda, labels=[0,1,2,3])
            print(f"[+] [4/6] OvR CSP + Shrinkage LDA:   Acc = {np.mean(scores_csp_lda)*100:5.2f}% ± {np.std(scores_csp_lda)*100:4.2f}% | F1 = {f1_score(y_recall, preds_csp_lda, average='macro'):.3f}")
        except Exception as e:
            print(f"[-] [4/6] CSP + LDA skipped: {e}")

        # Model 5: Multi-Band PSD Features + Random Forest
        X_bandpower = extract_bandpower_features(X_recall_uv, sfreq=sfreq)
        pipe_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        scores_rf = cross_val_score(pipe_rf, X_bandpower, y_recall, cv=cv, scoring='accuracy')
        preds_rf = cross_val_predict(pipe_rf, X_bandpower, y_recall, cv=cv)
        models_results['Bandpower + Random Forest'] = {
            'accuracy_mean': float(np.mean(scores_rf)),
            'accuracy_std': float(np.std(scores_rf)),
            'balanced_acc': float(balanced_accuracy_score(y_recall, preds_rf)),
            'f1_macro': float(f1_score(y_recall, preds_rf, average='macro'))
        }
        confusion_matrices['Bandpower + Random Forest'] = confusion_matrix(y_recall, preds_rf, labels=[0,1,2,3])
        print(f"[+] [5/6] Band Power + Random Forest: Acc = {np.mean(scores_rf)*100:5.2f}% ± {np.std(scores_rf)*100:4.2f}% | F1 = {f1_score(y_recall, preds_rf, average='macro'):.3f}")

        # Model 6: PyTorch Deep Learning (EEGNet)
        try:
            scores_eegnet, preds_eegnet = train_eval_eegnet(X_recall_uv, y_recall, n_classes=4, n_splits=n_splits, epochs=35)
            models_results['PyTorch EEGNet DL'] = {
                'accuracy_mean': float(np.mean(scores_eegnet)),
                'accuracy_std': float(np.std(scores_eegnet)),
                'balanced_acc': float(balanced_accuracy_score(y_recall, preds_eegnet)),
                'f1_macro': float(f1_score(y_recall, preds_eegnet, average='macro'))
            }
            confusion_matrices['PyTorch EEGNet DL'] = confusion_matrix(y_recall, preds_eegnet, labels=[0,1,2,3])
            print(f"[+] [6/6] PyTorch EEGNet DL:          Acc = {np.mean(scores_eegnet)*100:5.2f}% ± {np.std(scores_eegnet)*100:4.2f}% | F1 = {f1_score(y_recall, preds_eegnet, average='macro'):.3f}")
        except Exception as e:
            print(f"[-] [6/6] EEGNet skipped: {e}")

        # ---------------------------------------------------------
        # Benchmark Visualizations
        # ---------------------------------------------------------
        df_res = pd.DataFrame(models_results).T.reset_index()
        df_res.rename(columns={'index': 'Model'}, inplace=True)
        df_res['accuracy_pct'] = df_res['accuracy_mean'] * 100
        df_res['std_pct'] = df_res['accuracy_std'] * 100

        # Bar Chart
        plt.figure(figsize=(11, 5.5))
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c', '#e74c3c'][:len(df_res)]
        bars = plt.bar(df_res['Model'], df_res['accuracy_pct'], yerr=df_res['std_pct'], capsize=6, color=colors, alpha=0.85, edgecolor='black')
        plt.axhline(25.0, color='red', linestyle='--', linewidth=2, label='Chance Level (25.0%)')

        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., h + 1.2, f'{h:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.ylabel('Classification Accuracy (%)', fontsize=11, fontweight='bold')
        plt.title(f'4-Class Element Decoding Accuracy (Sub-{sub_clean} Ses-{ses_clean} [{condition.upper()}])', fontsize=13, fontweight='bold', pad=12)
        plt.xticks(rotation=20, ha='right', fontsize=10)
        plt.ylim(0, max(50.0, df_res['accuracy_pct'].max() + 15.0))
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()

        decoding_plot_path = os.path.join(session_out_dir, "decoding_benchmark_accuracy.png")
        plt.savefig(decoding_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Confusion Matrices
        n_m = len(confusion_matrices)
        n_cols = min(3, n_m)
        n_rows = int(np.ceil(n_m / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
        if n_m == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (m_name, cm) in enumerate(confusion_matrices.items()):
            ax = axes[idx]
            row_sums = cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
            im = ax.imshow(cm_norm, cmap='Blues', vmin=0.0, vmax=1.0)
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, fontsize=9)
            ax.set_yticklabels(class_names, fontsize=9)
            for row in range(len(class_names)):
                for col in range(len(class_names)):
                    val = cm_norm[row, col]
                    color = 'white' if val > 0.5 else 'black'
                    ax.text(col, row, f'{val:.2f}', ha='center', va='center', color=color, fontweight='bold', fontsize=10)

            ax.set_title(f"{m_name}\nAcc: {models_results[m_name]['accuracy_mean']*100:.1f}% | F1: {models_results[m_name]['f1_macro']:.2f}", fontsize=10, fontweight='bold')
            ax.set_ylabel('True Class', fontsize=9)
            ax.set_xlabel('Predicted Class', fontsize=9)

        for idx in range(n_m, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle(f"Confusion Matrices: Sub-{sub_clean} Ses-{ses_clean} ({condition.capitalize()})", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        cm_plot_path = os.path.join(session_out_dir, "confusion_matrices_grid.png")
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        csv_path = os.path.join(session_out_dir, "models_benchmark_metrics.csv")
        df_res.to_csv(csv_path, index=False)
    else:
        df_res = pd.DataFrame()

    summary_data = {
        'dataset': 'bids_tower_defense',
        'subject': sub_clean,
        'session': ses_clean,
        'condition': condition,
        'task': detected_task,
        'total_trials': len(recall_events),
        'min_class_count': min_class_count,
        'cv_splits': n_splits if min_class_count >= 2 else 0,
        'class_distribution': {c_name: int(np.sum(recall_events[:, 2] == c_id)) for c_name, c_id in class_map.items()},
        'spatial_filter_applied': spatial_filter,
        'models_benchmark': models_results
    }

    json_path = os.path.join(session_out_dir, "session_decoding_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=4)

    return summary_data


# ---------------------------------------------------------
# Multi-Session & Cross-Paradigm Aggregator
# ---------------------------------------------------------
def run_all_sessions_tower_defense_analysis(
    bids_root="scripts/bids_tower_defense",
    subject_id="01",
    sessions=("01", "02", "03", "04"),
    out_dir="scripts/analysis_results/tower_defense_recall"
):
    print("\n" + "#" * 80)
    print(" MULTI-SESSION TOWER DEFENSE RECALL & MEMORY BENCHMARK ".center(80, "#"))
    print("#" * 80 + "\n")

    os.makedirs(out_dir, exist_ok=True)
    session_summaries = {}

    for ses in sessions:
        sub_dir = os.path.join(bids_root, f"sub-{subject_id}", f"ses-{ses}")
        if not os.path.exists(sub_dir):
            continue

        print(f"\n>>> Running Analysis for Session {ses} <<<")
        s_res = run_tower_defense_session_analysis(
            bids_root=bids_root,
            subject_id=subject_id,
            session_id=ses,
            condition="all",
            out_dir=out_dir
        )
        if s_res:
            session_summaries[f"ses-{ses}_all"] = s_res

        # If ses-02 (mixed recall & memory), also run split breakdowns
        if ses == "02":
            print(f"\n>>> Running Ses-02 Recall-Only Subset <<<")
            res_rec = run_tower_defense_session_analysis(
                bids_root=bids_root,
                subject_id=subject_id,
                session_id=ses,
                condition="recall",
                out_dir=out_dir
            )
            if res_rec:
                session_summaries[f"ses-02_recall"] = res_rec

            print(f"\n>>> Running Ses-02 Memory-Only Subset <<<")
            res_mem = run_tower_defense_session_analysis(
                bids_root=bids_root,
                subject_id=subject_id,
                session_id=ses,
                condition="memory",
                out_dir=out_dir
            )
            if res_mem:
                session_summaries[f"ses-02_memory"] = res_mem

    # Build Comparative Table Across Sessions
    table_rows = []
    for s_key, s_data in session_summaries.items():
        bench = s_data.get('models_benchmark', {})
        for m_name, m_metrics in bench.items():
            table_rows.append({
                'Session_Condition': s_key,
                'Total_Trials': s_data['total_trials'],
                'Model': m_name,
                'Accuracy (%)': m_metrics['accuracy_mean'] * 100,
                'Std (%)': m_metrics['accuracy_std'] * 100,
                'Balanced_Acc (%)': m_metrics['balanced_acc'] * 100,
                'Macro_F1': m_metrics['f1_macro']
            })

    if table_rows:
        df_comparison = pd.DataFrame(table_rows)
        comp_csv = os.path.join(out_dir, "cross_session_model_comparison.csv")
        df_comparison.to_csv(comp_csv, index=False)
        print(f"\n[+] Exported cross-session comparative table: {comp_csv}")

        # Multi-session comparison plot for Riemannian TS LogReg & Random Forest
        plt.figure(figsize=(14, 6))
        top_models = ['Riemannian TS + LogReg', 'Bandpower + Random Forest', 'PyTorch EEGNet DL']
        df_top = df_comparison[df_comparison['Model'].isin(top_models)]

        if not df_top.empty:
            pivot_acc = df_top.pivot(index='Session_Condition', columns='Model', values='Accuracy (%)')
            pivot_acc.plot(kind='bar', figsize=(12, 6), edgecolor='black', alpha=0.85)
            plt.axhline(25.0, color='red', linestyle='--', linewidth=2, label='Chance (25%)')
            plt.title("4-Class Decoding Accuracy Across Tower Defense Sessions & Conditions", fontsize=13, fontweight='bold')
            plt.ylabel("Accuracy (%)", fontsize=11, fontweight='bold')
            plt.xlabel("Session / Condition", fontsize=11, fontweight='bold')
            plt.xticks(rotation=25, ha='right', fontsize=10)
            plt.ylim(0, 100)
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.legend(loc='upper right')
            plt.tight_layout()
            
            comp_plot_path = os.path.join(out_dir, "cross_session_comparison_barplot.png")
            plt.savefig(comp_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[+] Saved cross-session comparison plot: {comp_plot_path}")

    # Master summary JSON
    master_json = os.path.join(out_dir, "master_multi_session_summary.json")
    with open(master_json, 'w') as f:
        json.dump(session_summaries, f, indent=4)
    print(f"[+] Exported Master Summary: {master_json}")

    print("\n" + "=" * 80)
    print(" ALL SESSIONS ANALYSIS COMPLETE ".center(80, "="))
    print("=" * 80)


# ---------------------------------------------------------
# Combined Sessions & Cross-Session Transfer Pipeline
# ---------------------------------------------------------
def run_combined_sessions_analysis(
    bids_root="scripts/bids_tower_defense",
    subject_id="01",
    sessions=("01", "02"),
    condition="all",
    out_dir="scripts/analysis_results/tower_defense_recall",
    recall_tmin=0.5,
    recall_tmax=4.5,
    spatial_filter="robust_car"
):
    sub_clean = subject_id.replace("sub-", "")
    ses_tag = "_".join([f"ses-{normalize_ses_id(s)}" for s in sessions])
    combined_out_dir = os.path.join(out_dir, f"combined_{ses_tag}")
    if condition != "all":
        combined_out_dir += f"_{condition}"
    os.makedirs(combined_out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f" COMBINED SESSIONS ANALYSIS: sub-{sub_clean} [{ses_tag}] [{condition.upper()}] ".center(80, "="))
    print("=" * 80)

    epochs_list = []
    session_tags = []
    condition_tags = []
    class_map = {'FIRE': 0, 'WATER': 1, 'WIND': 2, 'ELECTRICITY': 3}
    class_names = ['FIRE', 'WATER', 'WIND', 'ELECTRICITY']
    event_id_dict = {k: v for k, v in class_map.items()}

    for ses in sessions:
        ses_clean = normalize_ses_id(ses)
        detected_task, vhdr_path = find_bids_session_info(bids_root, sub_clean, ses_clean)
        bids_path = BIDSPath(subject=sub_clean, session=ses_clean, task=detected_task, datatype="eeg", root=bids_root)
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        raw.load_data()

        misc_chans = [ch for ch in raw.ch_names if ch.upper() in ['BATTERY', 'STATUS', 'AUX'] or ch == 'EEG033']
        if misc_chans:
            raw.set_channel_types({ch: 'misc' for ch in misc_chans})
        raw.pick('eeg')

        sfreq = raw.info['sfreq']
        data_raw_arr = raw.get_data().T
        stds = np.std(data_raw_arr, axis=0)
        scale_factor = 1e6 if np.mean(stds) < 1e-3 else 1.0
        data_uv = (data_raw_arr - np.mean(data_raw_arr, axis=0, keepdims=True)) * scale_factor

        if spatial_filter != "none":
            filt_data_uv = apply_spatial_filter(data_uv, raw.ch_names, mode=spatial_filter)
            raw._data = (filt_data_uv / scale_factor).T

        raw_filt = raw.copy().filter(l_freq=1.0, h_freq=45.0, verbose=False)
        raw_filt.notch_filter(freqs=50.0, verbose=False)

        events_tsv_path = os.path.join(bids_path.directory, f"sub-{sub_clean}_ses-{ses_clean}_task-{detected_task}_events.tsv")
        rec_evs, _, conds, _, _ = parse_tower_defense_events(events_tsv_path, sfreq, ses_clean)

        if condition != "all":
            mask = (conds == condition)
            rec_evs = rec_evs[mask]
            conds = conds[mask]

        if len(rec_evs) == 0:
            continue

        ep = mne.Epochs(
            raw_filt,
            rec_evs,
            event_id=event_id_dict,
            tmin=recall_tmin,
            tmax=recall_tmax,
            baseline=None,
            preload=True,
            verbose=False
        )
        epochs_list.append(ep)
        session_tags.extend([f"ses-{ses_clean}"] * len(ep))
        condition_tags.extend(list(conds))

    if not epochs_list:
        print("[!] No epochs found to concatenate.")
        return None

    # Concatenate epochs across sessions
    combined_epochs = mne.concatenate_epochs(epochs_list)
    X_comb = combined_epochs.get_data() * 1e6
    y_comb = combined_epochs.events[:, 2]
    session_tags = np.array(session_tags)

    print(f"\n[+] Total Combined Trials: {len(y_comb)}")
    for s in set(session_tags):
        print(f"    • {s}: {np.sum(session_tags == s)} trials")
    for c_name, c_id in class_map.items():
        print(f"    • {c_name:12s}: {np.sum(y_comb == c_id)} trials")

    # 1. Pooled 5-Fold Stratified Cross-Validation
    print("\n" + "=" * 60)
    print(f" 1. Combined 5-Fold Cross-Validation ({ses_tag}) ".center(60, "="))
    print("=" * 60)

    class_counts = np.bincount(y_comb, minlength=4)
    min_class_count = int(np.min(class_counts))
    n_splits = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    pooled_results = {}
    pooled_cms = {}

    # Model 1: Riemannian TS + LogReg
    pipe_ts_lr = Pipeline([
        ('cov', Covariances(estimator='oas')),
        ('ts', TangentSpace(metric='riemann')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=1.0))
    ])
    scores_ts_lr = cross_val_score(pipe_ts_lr, X_comb, y_comb, cv=cv, scoring='accuracy')
    preds_ts_lr = cross_val_predict(pipe_ts_lr, X_comb, y_comb, cv=cv)
    pooled_results['Riemannian TS + LogReg'] = {
        'accuracy_mean': float(np.mean(scores_ts_lr)),
        'accuracy_std': float(np.std(scores_ts_lr)),
        'balanced_acc': float(balanced_accuracy_score(y_comb, preds_ts_lr)),
        'f1_macro': float(f1_score(y_comb, preds_ts_lr, average='macro'))
    }
    pooled_cms['Riemannian TS + LogReg'] = confusion_matrix(y_comb, preds_ts_lr, labels=[0,1,2,3])
    print(f"[+] [1/6] Riemannian TS (LogReg):     Acc = {np.mean(scores_ts_lr)*100:5.2f}% ± {np.std(scores_ts_lr)*100:4.2f}% | F1 = {f1_score(y_comb, preds_ts_lr, average='macro'):.3f}")

    # Model 2: Riemannian TS + SVM RBF
    pipe_ts_svm = Pipeline([
        ('cov', Covariances(estimator='oas')),
        ('ts', TangentSpace(metric='riemann')),
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=1.0))
    ])
    scores_ts_svm = cross_val_score(pipe_ts_svm, X_comb, y_comb, cv=cv, scoring='accuracy')
    preds_ts_svm = cross_val_predict(pipe_ts_svm, X_comb, y_comb, cv=cv)
    pooled_results['Riemannian TS + SVM (RBF)'] = {
        'accuracy_mean': float(np.mean(scores_ts_svm)),
        'accuracy_std': float(np.std(scores_ts_svm)),
        'balanced_acc': float(balanced_accuracy_score(y_comb, preds_ts_svm)),
        'f1_macro': float(f1_score(y_comb, preds_ts_svm, average='macro'))
    }
    pooled_cms['Riemannian TS + SVM (RBF)'] = confusion_matrix(y_comb, preds_ts_svm, labels=[0,1,2,3])
    print(f"[+] [2/6] Riemannian TS (SVM RBF):    Acc = {np.mean(scores_ts_svm)*100:5.2f}% ± {np.std(scores_ts_svm)*100:4.2f}% | F1 = {f1_score(y_comb, preds_ts_svm, average='macro'):.3f}")

    # Model 3: Riemannian MDM
    pipe_mdm = Pipeline([
        ('cov', Covariances(estimator='oas')),
        ('clf', MDM(metric='riemann'))
    ])
    scores_mdm = cross_val_score(pipe_mdm, X_comb, y_comb, cv=cv, scoring='accuracy')
    preds_mdm = cross_val_predict(pipe_mdm, X_comb, y_comb, cv=cv)
    pooled_results['Riemannian MDM'] = {
        'accuracy_mean': float(np.mean(scores_mdm)),
        'accuracy_std': float(np.std(scores_mdm)),
        'balanced_acc': float(balanced_accuracy_score(y_comb, preds_mdm)),
        'f1_macro': float(f1_score(y_comb, preds_mdm, average='macro'))
    }
    pooled_cms['Riemannian MDM'] = confusion_matrix(y_comb, preds_mdm, labels=[0,1,2,3])
    print(f"[+] [3/6] Riemannian MDM:             Acc = {np.mean(scores_mdm)*100:5.2f}% ± {np.std(scores_mdm)*100:4.2f}% | F1 = {f1_score(y_comb, preds_mdm, average='macro'):.3f}")

    # Model 4: Multi-Class One-vs-Rest CSP + LDA
    try:
        X_csp_feats = extract_ovr_csp_features(X_comb, y_comb, n_components=4)
        clf_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        scores_csp_lda = cross_val_score(clf_lda, X_csp_feats, y_comb, cv=cv, scoring='accuracy')
        preds_csp_lda = cross_val_predict(clf_lda, X_csp_feats, y_comb, cv=cv)
        pooled_results['One-vs-Rest CSP + LDA'] = {
            'accuracy_mean': float(np.mean(scores_csp_lda)),
            'accuracy_std': float(np.std(scores_csp_lda)),
            'balanced_acc': float(balanced_accuracy_score(y_comb, preds_csp_lda)),
            'f1_macro': float(f1_score(y_comb, preds_csp_lda, average='macro'))
        }
        pooled_cms['One-vs-Rest CSP + LDA'] = confusion_matrix(y_comb, preds_csp_lda, labels=[0,1,2,3])
        print(f"[+] [4/6] OvR CSP + Shrinkage LDA:   Acc = {np.mean(scores_csp_lda)*100:5.2f}% ± {np.std(scores_csp_lda)*100:4.2f}% | F1 = {f1_score(y_comb, preds_csp_lda, average='macro'):.3f}")
    except Exception as e:
        print(f"[-] [4/6] CSP + LDA skipped: {e}")

    # Model 5: Multi-Band PSD Features + Random Forest
    X_bandpower = extract_bandpower_features(X_comb, sfreq=250.0)
    pipe_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=120, random_state=42))
    ])
    scores_rf = cross_val_score(pipe_rf, X_bandpower, y_comb, cv=cv, scoring='accuracy')
    preds_rf = cross_val_predict(pipe_rf, X_bandpower, y_comb, cv=cv)
    pooled_results['Bandpower + Random Forest'] = {
        'accuracy_mean': float(np.mean(scores_rf)),
        'accuracy_std': float(np.std(scores_rf)),
        'balanced_acc': float(balanced_accuracy_score(y_comb, preds_rf)),
        'f1_macro': float(f1_score(y_comb, preds_rf, average='macro'))
    }
    pooled_cms['Bandpower + Random Forest'] = confusion_matrix(y_comb, preds_rf, labels=[0,1,2,3])
    print(f"[+] [5/6] Band Power + Random Forest: Acc = {np.mean(scores_rf)*100:5.2f}% ± {np.std(scores_rf)*100:4.2f}% | F1 = {f1_score(y_comb, preds_rf, average='macro'):.3f}")

    # Model 6: PyTorch Deep Learning (EEGNet)
    try:
        scores_eegnet, preds_eegnet = train_eval_eegnet(X_comb, y_comb, n_classes=4, n_splits=n_splits, epochs=35)
        pooled_results['PyTorch EEGNet DL'] = {
            'accuracy_mean': float(np.mean(scores_eegnet)),
            'accuracy_std': float(np.std(scores_eegnet)),
            'balanced_acc': float(balanced_accuracy_score(y_comb, preds_eegnet)),
            'f1_macro': float(f1_score(y_comb, preds_eegnet, average='macro'))
        }
        pooled_cms['PyTorch EEGNet DL'] = confusion_matrix(y_comb, preds_eegnet, labels=[0,1,2,3])
        print(f"[+] [6/6] PyTorch EEGNet DL:          Acc = {np.mean(scores_eegnet)*100:5.2f}% ± {np.std(scores_eegnet)*100:4.2f}% | F1 = {f1_score(y_comb, preds_eegnet, average='macro'):.3f}")
    except Exception as e:
        print(f"[-] [6/6] EEGNet skipped: {e}")

    # 2. Cross-Session Transfer: Train on Session 1 -> Test on Session 2
    transfer_results = {}
    if len(sessions) >= 2:
        ses1_tag = f"ses-{normalize_ses_id(sessions[0])}"
        ses2_tag = f"ses-{normalize_ses_id(sessions[1])}"
        mask_train = (session_tags == ses1_tag)
        mask_test = (session_tags == ses2_tag)

        print("\n" + "=" * 60)
        print(f" 2. Cross-Session Transfer: Train [{ses1_tag}] -> Test [{ses2_tag}] ".center(60, "="))
        print("=" * 60)
        print(f"[*] Train set ({ses1_tag}): {np.sum(mask_train)} trials | Test set ({ses2_tag}): {np.sum(mask_test)} trials")

        X_train, y_train = X_comb[mask_train], y_comb[mask_train]
        X_test, y_test = X_comb[mask_test], y_comb[mask_test]

        # Transfer Model 1: Riemannian TS LogReg
        pipe_ts_lr.fit(X_train, y_train)
        pred_trans_lr = pipe_ts_lr.predict(X_test)
        acc_trans_lr = accuracy_score(y_test, pred_trans_lr)
        f1_trans_lr = f1_score(y_test, pred_trans_lr, average='macro')
        transfer_results['Riemannian TS + LogReg'] = {'accuracy': acc_trans_lr, 'f1_macro': f1_trans_lr}
        print(f"[+] Transfer Riemannian TS (LogReg):     Acc = {acc_trans_lr*100:5.2f}% | F1 = {f1_trans_lr:.3f}")

        # Transfer Model 2: Riemannian MDM
        pipe_mdm.fit(X_train, y_train)
        pred_trans_mdm = pipe_mdm.predict(X_test)
        acc_trans_mdm = accuracy_score(y_test, pred_trans_mdm)
        f1_trans_mdm = f1_score(y_test, pred_trans_mdm, average='macro')
        transfer_results['Riemannian MDM'] = {'accuracy': acc_trans_mdm, 'f1_macro': f1_trans_mdm}
        print(f"[+] Transfer Riemannian MDM:             Acc = {acc_trans_mdm*100:5.2f}% | F1 = {f1_trans_mdm:.3f}")

        # Transfer Model 3: Bandpower Random Forest
        pipe_rf.fit(X_bandpower[mask_train], y_train)
        pred_trans_rf = pipe_rf.predict(X_bandpower[mask_test])
        acc_trans_rf = accuracy_score(y_test, pred_trans_rf)
        f1_trans_rf = f1_score(y_test, pred_trans_rf, average='macro')
        transfer_results['Bandpower + Random Forest'] = {'accuracy': acc_trans_rf, 'f1_macro': f1_trans_rf}
        print(f"[+] Transfer Bandpower + Random Forest:  Acc = {acc_trans_rf*100:5.2f}% | F1 = {f1_trans_rf:.3f}")

    # Visualizations
    df_pooled = pd.DataFrame(pooled_results).T.reset_index().rename(columns={'index': 'Model'})
    df_pooled['accuracy_pct'] = df_pooled['accuracy_mean'] * 100
    df_pooled['std_pct'] = df_pooled['accuracy_std'] * 100

    plt.figure(figsize=(11, 5.5))
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c', '#e74c3c'][:len(df_pooled)]
    bars = plt.bar(df_pooled['Model'], df_pooled['accuracy_pct'], yerr=df_pooled['std_pct'], capsize=6, color=colors, alpha=0.85, edgecolor='black')
    plt.axhline(25.0, color='red', linestyle='--', linewidth=2, label='Chance Level (25.0%)')

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 1.2, f'{h:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.ylabel('Classification Accuracy (%)', fontsize=11, fontweight='bold')
    plt.title(f'Combined 4-Class Element Decoding Accuracy ({ses_tag.upper()})', fontsize=13, fontweight='bold', pad=12)
    plt.xticks(rotation=20, ha='right', fontsize=10)
    plt.ylim(0, max(50.0, df_pooled['accuracy_pct'].max() + 15.0))
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    comb_plot_path = os.path.join(combined_out_dir, "combined_decoding_accuracy.png")
    plt.savefig(comb_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Confusion Matrices Grid
    n_m = len(pooled_cms)
    n_cols = min(3, n_m)
    n_rows = int(np.ceil(n_m / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
    if n_m == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (m_name, cm) in enumerate(pooled_cms.items()):
        ax = axes[idx]
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        for row in range(len(class_names)):
            for col in range(len(class_names)):
                val = cm_norm[row, col]
                color = 'white' if val > 0.5 else 'black'
                ax.text(col, row, f'{val:.2f}', ha='center', va='center', color=color, fontweight='bold', fontsize=10)

        ax.set_title(f"{m_name}\nAcc: {pooled_results[m_name]['accuracy_mean']*100:.1f}% | F1: {pooled_results[m_name]['f1_macro']:.2f}", fontsize=10, fontweight='bold')
        ax.set_ylabel('True Class', fontsize=9)
        ax.set_xlabel('Predicted Class', fontsize=9)

    for idx in range(n_m, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f"Combined Confusion Matrices ({ses_tag.upper()})", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    cm_plot_path = os.path.join(combined_out_dir, "combined_confusion_matrices.png")
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save summary data
    summary_comb = {
        'dataset': 'bids_tower_defense',
        'subject': sub_clean,
        'sessions': list(sessions),
        'condition': condition,
        'total_trials': len(y_comb),
        'session_trial_counts': {s: int(np.sum(session_tags == s)) for s in set(session_tags)},
        'class_distribution': {c_name: int(np.sum(y_comb == c_id)) for c_name, c_id in class_map.items()},
        'pooled_cv_benchmark': pooled_results,
        'cross_session_transfer': transfer_results
    }

    json_path = os.path.join(combined_out_dir, "combined_decoding_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary_comb, f, indent=4)

    csv_path = os.path.join(combined_out_dir, "combined_benchmark_metrics.csv")
    df_pooled.to_csv(csv_path, index=False)

    print(f"\n[+] Exported Combined Summary JSON: {json_path}")
    print(f"[+] Exported Combined Metrics CSV:  {csv_path}")
    return summary_comb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze BIDS Tower Defense Recall & Blinking Dataset")
    parser.add_argument("--bids-root", type=str, default="scripts/bids_tower_defense", help="Path to bids_tower_defense folder")
    parser.add_argument("--sub", type=str, default="01", help="Subject ID (e.g. 01)")
    parser.add_argument("--ses", type=str, default="all", help="Session ID ('01', '02', '03', '04', '01+02', or 'all')")
    parser.add_argument("--combine", type=str, default=None, help="Comma-separated sessions to pool together (e.g. '01,02')")
    parser.add_argument("--condition", type=str, default="all", choices=["all", "recall", "memory"], help="Condition filter ('all', 'recall', 'memory')")
    parser.add_argument("--out-dir", type=str, default="scripts/analysis_results/tower_defense_recall", help="Output directory")
    parser.add_argument("--recall-tmin", type=float, default=0.5, help="Recall epoch start relative to box stop blinking (s)")
    parser.add_argument("--recall-tmax", type=float, default=4.5, help="Recall epoch end relative to box stop blinking (s)")
    parser.add_argument("--blink-tmin", type=float, default=0.5, help="Blink epoch start relative to box start blinking (s)")
    parser.add_argument("--blink-tmax", type=float, default=4.5, help="Blink epoch end relative to box start blinking (s)")
    parser.add_argument("--spatial-filter", type=str, default="robust_car", choices=["none", "robust_car", "car", "laplacian"], help="Spatial filter mode")

    args = parser.parse_args()

    if args.combine:
        sessions_to_combine = [s.strip().replace("ses-", "") for s in args.combine.split(",")]
        run_combined_sessions_analysis(
            bids_root=args.bids_root,
            subject_id=args.sub,
            sessions=sessions_to_combine,
            condition=args.condition,
            out_dir=args.out_dir,
            recall_tmin=args.recall_tmin,
            recall_tmax=args.recall_tmax,
            spatial_filter=args.spatial_filter
        )
    elif "+" in args.ses:
        sessions_to_combine = [s.strip().replace("ses-", "") for s in args.ses.split("+")]
        run_combined_sessions_analysis(
            bids_root=args.bids_root,
            subject_id=args.sub,
            sessions=sessions_to_combine,
            condition=args.condition,
            out_dir=args.out_dir,
            recall_tmin=args.recall_tmin,
            recall_tmax=args.recall_tmax,
            spatial_filter=args.spatial_filter
        )
    elif args.ses == "all" or "," in args.ses:
        target_sessions = ["01", "02", "03", "04"] if args.ses == "all" else [s.strip().replace("ses-", "") for s in args.ses.split(",")]
        run_all_sessions_tower_defense_analysis(
            bids_root=args.bids_root,
            subject_id=args.sub,
            sessions=target_sessions,
            out_dir=args.out_dir
        )
    else:
        run_tower_defense_session_analysis(
            bids_root=args.bids_root,
            subject_id=args.sub,
            session_id=args.ses,
            condition=args.condition,
            out_dir=args.out_dir,
            recall_tmin=args.recall_tmin,
            recall_tmax=args.recall_tmax,
            blink_tmin=args.blink_tmin,
            blink_tmax=args.blink_tmax,
            spatial_filter=args.spatial_filter
        )
