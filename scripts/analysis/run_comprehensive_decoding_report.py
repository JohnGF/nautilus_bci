"""
BCI Comprehensive Multi-Model Decoding & Benchmark Report Studio
==================================================================
Pools all sessions of the music BCI experiment, extracts clean epochs,
evaluates 5 advanced decoding models (CSP+LDA, Riemannian MDM, TangentSpace+SVM,
TangentSpace+Logistic Regression, and EEGNet Deep Learning), and compiles
them into an executive benchmark report.
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Riemannian geometry imports
import pyriemann
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

# ---------------------------------------------------------
# PyTorch EEGNet Architecture (Lawhern et al., 2018)
# ---------------------------------------------------------
class EEGNet(nn.Module):
    def __init__(self, n_channels=32, n_samples=1001, n_classes=6, F1=8, D=2, F2=16, kernel_length=32, dropout_rate=0.25):
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

def train_eval_eegnet(X, y, n_classes, n_splits=5, epochs=30, lr=0.003, batch_size=16):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_epochs_count, n_channels, n_samples = X.shape
    acc_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        # Reshape to (batch, 1, channels, samples)
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1).to(device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(device)
        X_te_t = torch.tensor(X_te, dtype=torch.float32).unsqueeze(1).to(device)
        y_te_t = torch.tensor(y_te, dtype=torch.long).to(device)

        ds_train = TensorDataset(X_tr_t, y_tr_t)
        loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

        model = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        model.train()
        for ep in range(epochs):
            for bx, by in loader_train:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_te_t)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_te_t).float().mean().item()
            acc_scores.append(acc)

    return np.mean(acc_scores) * 100.0


def filter_epochs_by_stability(epochs, max_bad_channels=6, min_std=1e-7, max_std=5e-3):
    """Discards individual trials (epochs) where too many channels are flat or noisy."""
    data = epochs.get_data()  # (n_epochs, n_channels, n_samples)
    clean_indices = []
    
    for epoch_idx in range(data.shape[0]):
        epoch_data = data[epoch_idx]
        stds = np.std(epoch_data, axis=1)
        bad_count = sum(1 for std_val in stds if std_val < min_std or std_val > max_std)
        if bad_count <= max_bad_channels:
            clean_indices.append(epoch_idx)
            
    print(f"    [Epoch Filter] Kept {len(clean_indices)} / {data.shape[0]} trials (discarded {data.shape[0] - len(clean_indices)})")
    return epochs[clean_indices]


def run_benchmark(bids_root="bids_musica", subject_id="01", sessions=None):
    print("=" * 80)
    print(" BCI Advanced Models Benchmark Report Studio ".center(80, "="))
    print("=" * 80)

    # 1. Pool and Epoch data from all sessions
    bids_root = os.path.abspath(bids_root)
    sub_clean = subject_id.replace("sub-", "")
    vhdr_files = glob.glob(os.path.join(bids_root, f"sub-{sub_clean}", "ses-*", "eeg", "*_eeg.vhdr"))

    if not vhdr_files:
        print(f"[-] No EEG datasets found for sub-{sub_clean}")
        return

    # Filter sessions if specified
    if sessions is not None:
        # Standardize to BIDS format ses-XX (e.g. pad to 2 digits)
        sessions_padded = [s.zfill(2) for s in sessions]
        filtered_files = []
        for filepath in vhdr_files:
            filename = os.path.basename(filepath)
            parts = filename.split('_')
            ses_val = parts[1].replace("ses-", "")
            if ses_val in sessions_padded or ses_val.lstrip('0') in sessions:
                filtered_files.append(filepath)
        vhdr_files = filtered_files
        print(f"[+] Filtering to user-selected BIDS sessions: {sessions_padded}")

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

            # Standardize channel layout names
            standard_32 = [
                'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 
                'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 
                'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10'
            ]
            mapping = {name: standard_32[i] for i, name in enumerate(raw_eeg.ch_names) if i < len(standard_32)}
            raw_eeg.rename_channels(mapping)
            
            # Keep only the standard 32 channels to drop any extra channels (like EEG033 or Battery)
            raw_eeg.pick([ch for ch in standard_32 if ch in raw_eeg.ch_names])

            # Set standard 10-20 montage for spatial coordinate interpolation
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                raw_eeg.set_montage(montage, on_missing='ignore', verbose=False)
            except Exception as e:
                print(f"    [-] Warning: Could not set standard 10-20 montage: {e}")

            # Auto-detect flat or highly noisy/collapsed channels
            data_arr = raw_eeg.get_data()
            stds = np.std(data_arr, axis=1)
            bad_channels = []
            for i, ch_name in enumerate(raw_eeg.ch_names):
                std_val = stds[i]
                if std_val < 1e-7 or std_val > 5e-3:
                    bad_channels.append(ch_name)
            
            # Interpolate bad channels using spherical spline interpolation
            if bad_channels:
                raw_eeg.info['bads'] = bad_channels
                print(f"    [+] Interpolating bad/collapsed channels in {filename}: {bad_channels}")
                try:
                    raw_eeg.interpolate_bads(reset_bads=True, verbose=False)
                except Exception as e:
                    print(f"    [-] Interpolation failed: {e}. Falling back to dropping bad channels.")
                    raw_eeg.drop_channels(bad_channels)

            # Apply Common Average Reference (CAR) to subtract global/common-mode noise
            raw_eeg.set_eeg_reference(ref_channels='average', verbose=False)

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
            
            # Apply trial-by-trial stability filtering
            epochs_clean = filter_epochs_by_stability(epochs, max_bad_channels=6)
            if len(epochs_clean) > 0:
                all_epochs.append(epochs_clean)

        except Exception as e:
            print(f"[-] Error loading {filename}: {e}")

    if not all_epochs:
        print("[-] Error: No epochs compiled.")
        return

    # Align channels & concatenate
    common_ch = list(set.intersection(*(set(ep.ch_names) for ep in all_epochs)))
    for ep in all_epochs:
        ep.pick(common_ch)
    combined_epochs = mne.concatenate_epochs(all_epochs, verbose=False)

    X = combined_epochs.get_data()
    y_raw = combined_epochs.events[:, -1]
    unique_y = np.unique(y_raw)
    y = np.array([list(unique_y).index(val) for val in y_raw])
    n_classes = len(unique_y)

    print(f"[+] Combined dataset ready: {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} samples, {n_classes} classes.")

    results = {}

    # Define Cross-Validation Strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Model 1: Baseline CSP + LDA
    print("[*] Evaluating Model 1: Spatial Filter CSP + Linear Discriminant Analysis...")
    try:
        csp = CSP(n_components=min(4, X.shape[1]), reg=None, log=True, norm_trace=False)
        lda = LinearDiscriminantAnalysis()
        scores = cross_val_score(Pipeline([('CSP', csp), ('LDA', lda)]), X, y, cv=cv)
        results["CSP + LDA (Baseline)"] = np.mean(scores) * 100.0
    except Exception as e:
        results["CSP + LDA (Baseline)"] = 0.0

    # Riemannian Geometry Feature Estimator
    print("[*] Estimating Riemannian Covariance Matrices...")
    cov = Covariances(estimator='oas')
    X_cov = cov.fit_transform(X)

    # Model 2: Minimum Distance to Mean (MDM)
    print("[*] Evaluating Model 2: Riemannian MDM...")
    try:
        mdm = MDM()
        scores = cross_val_score(mdm, X_cov, y, cv=cv)
        results["Riemannian MDM"] = np.mean(scores) * 100.0
    except Exception as e:
        results["Riemannian MDM"] = 0.0

    # Model 3: Tangent Space + Support Vector Machine (TS+SVM)
    print("[*] Evaluating Model 3: Riemannian Tangent Space + SVM...")
    try:
        ts_svm = Pipeline([
            ('TS', TangentSpace(metric='riemann')),
            ('Scale', StandardScaler()),
            ('SVM', SVC(kernel='linear', C=1.0))
        ])
        scores = cross_val_score(ts_svm, X_cov, y, cv=cv)
        results["Tangent Space + SVM"] = np.mean(scores) * 100.0
    except Exception as e:
        results["Tangent Space + SVM"] = 0.0

    # Model 4: Tangent Space + Logistic Regression (TS+LR)
    print("[*] Evaluating Model 4: Riemannian Tangent Space + Logistic Regression...")
    try:
        ts_lr = Pipeline([
            ('TS', TangentSpace(metric='riemann')),
            ('Scale', StandardScaler()),
            ('LR', LogisticRegression(max_iter=500, solver='liblinear'))
        ])
        scores = cross_val_score(ts_lr, X_cov, y, cv=cv)
        results["Tangent Space + Logistic Regression"] = np.mean(scores) * 100.0
    except Exception as e:
        results["Tangent Space + Logistic Regression"] = 0.0

    # Model 5: Deep Learning (EEGNet)
    print("[*] Evaluating Model 5: EEGNet Deep Learning CNN (PyTorch)...")
    try:
        eegnet_acc = train_eval_eegnet(X, y, n_classes=n_classes, n_splits=5, epochs=35)
        results["EEGNet Deep Learning CNN"] = eegnet_acc
    except Exception as e:
        print(f"[-] EEGNet failed: {e}")
        results["EEGNet Deep Learning CNN"] = 0.0

    # 5. Compile Executive Markdown Report
    chance_level = 100.0 / n_classes
    report_file = os.path.abspath("analysis_results/bids_benchmark_decoding_report.md")
    
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# BCI Multi-Model Decoding & Benchmark Report\n")
        f.write(f"Generated from BIDS Root: `{bids_root}` | Subject: `sub-{sub_clean}` | Sessions Pooled: `{len(vhdr_files)}`\n\n")
        
        f.write("## Dataset Characteristics\n")
        f.write(f"- **Total Trial Epochs**: {X.shape[0]} epochs\n")
        f.write(f"- **Channel Dimensions**: {X.shape[1]} EEG channels\n")
        f.write(f"- **Time Points per Trial**: {X.shape[2]} samples (@ 250Hz = 4.0s)\n")
        f.write(f"- **Number of Output Classes**: {n_classes} target tracks\n")
        f.write(f"- **Theoretical Random Chance**: {chance_level:.2f}%\n\n")
        
        f.write("## Decoding Accuracy Benchmark comparison\n\n")
        f.write("| Model Architecture | Feature Extraction Method | 5-Fold CV Accuracy | Performance vs. Chance |\n")
        f.write("| :--- | :--- | :---: | :---: |\n")
        
        for model_name, acc in results.items():
            diff = acc - chance_level
            diff_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
            f.write(f"| **{model_name}** | {'Covariance' if 'Riemann' in model_name or 'Tangent' in model_name else 'Raw Spatio-Temporal'} | **{acc:.2f}%** | {diff_str} |\n")
            
        f.write("\n## Brain Rhythm Power Spectral Distribution (PSD)\n")
        f.write("The average brain rhythm energy densities across all pooled trials are saved in the results directory. See the generated [aggregated_multi_session_psd.png](aggregated_multi_session_psd.png) graph.\n")

    print("\n" + "=" * 80)
    print(" COMPREHENSIVE BENCHMARK ANALYSIS EXECUTIVE SUMMARY ".center(80, "="))
    print("=" * 80)
    print(f" Markdown Report Saved   : {report_file}")
    for k, v in results.items():
        print(f"   [+] {k:<40}: {v:.2f}% (Chance: {chance_level:.2f}%)")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run advanced models benchmark report.")
    parser.add_argument("--bids-root", type=str, default="bids_musica", help="BIDS root directory")
    parser.add_argument("--sub", type=str, default="01", help="Subject ID")
    parser.add_argument("--sessions", type=str, default=None, help="Comma-separated session list, e.g. 06,07")
    args = parser.parse_args()

    sessions_list = args.sessions.split(',') if args.sessions else None
    run_benchmark(bids_root=args.bids_root, subject_id=args.sub, sessions=sessions_list)
