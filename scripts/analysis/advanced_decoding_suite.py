"""
Advanced Neural Decoding Suite: EEGNet, Riemannian Geometry, & Multimodal Fusion
===================================================================================

Implements 3 state-of-the-art BCI decoding paradigms on `bids_baseline/sub-01/ses-02` dataset:
  1. 🧠 Riemannian Geometry: Covariances + TangentSpace (Logistic Regression & SVM) & MDM
  2. ⚡ Deep Learning: EEGNet Architecture (PyTorch implementation)
  3. ⌚ Multimodal Physiological Fusion: EEG + Smartwatch PPG + IMU Motion Vectors
"""

import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import json
import numpy as np
import pandas as pd

import mne
from mne_bids import BIDSPath, read_raw_bids

# Riemannian Geometry Imports
import pyriemann
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

# Machine Learning Imports
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# PyTorch Deep Learning Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------
# PyTorch EEGNet Architecture (Lawhern et al., 2018)
# ---------------------------------------------------------
class EEGNet(nn.Module):
    def __init__(self, n_channels=32, n_samples=626, n_classes=4, F1=8, D=2, F2=16, kernel_length=32, dropout_rate=0.25):
        super(EEGNet, self).__init__()
        
        # Block 1: Temporal Conv -> Depthwise Spatial Conv -> BN -> ELU -> AvgPool -> Dropout
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)
        
        # Block 2: Separable Conv -> BN -> ELU -> AvgPool -> Dropout
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)
        
        # Flatten size calculation
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
        # Input shape: (batch, 1, n_channels, n_samples)
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


def train_eval_eegnet(X, y, n_splits=5, epochs=40, lr=0.003, batch_size=16):
    """Trains and evaluates EEGNet using Stratified K-Fold Cross-Validation."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_epochs_count, n_channels, n_samples = X.shape
    unique_labels = sorted(list(np.unique(y)))
    label_map = {orig: idx for idx, orig in enumerate(unique_labels)}
    y_mapped = np.array([label_map[v] for v in y])

    acc_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_mapped)):
        X_tr, y_tr = X[train_idx], y_mapped[train_idx]
        X_te, y_te = X[test_idx], y_mapped[test_idx]

        # Reshape to (batch, 1, channels, samples)
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1).to(device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(device)
        X_te_t = torch.tensor(X_te, dtype=torch.float32).unsqueeze(1).to(device)
        y_te_t = torch.tensor(y_te, dtype=torch.long).to(device)

        ds_train = TensorDataset(X_tr_t, y_tr_t)
        loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

        model = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=len(unique_labels)).to(device)
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

    return np.mean(acc_scores) * 100, np.std(acc_scores) * 100


# ---------------------------------------------------------
# Multimodal BIDS Data Loading & Feature Extraction
# ---------------------------------------------------------
def load_multimodal_dataset(bids_root="bids_baseline", sub="01", ses="02", task="video"):
    bids_root = os.path.abspath(bids_root)
    bids_path = BIDSPath(subject=sub, session=ses, task=task, datatype="eeg", root=bids_root)
    
    print("=" * 75)
    print(" Advanced Neural & Multimodal BCI Decoding Studio ".center(75, "="))
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

    X_eeg = epochs.get_data()  # (n_epochs, n_channels, n_samples)
    y = epochs.events[:, 2]

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
        # Motion slice
        if has_motion:
            idx_m = (t_m >= onset) & (t_m <= onset + 2.5)
            if np.any(idx_m):
                chunk = d_m[idx_m]
                m_mean = np.mean(chunk, axis=0)
                m_std = np.std(chunk, axis=0)
                motion_feats.append(np.hstack([m_mean, m_std]))
            else:
                motion_feats.append(np.zeros(12))

        # Physio/PPG slice
        if has_physio:
            idx_p = (t_p >= onset) & (t_p <= onset + 2.5)
            if np.any(idx_p):
                chunk_p = d_p[idx_p]
                p_mean = np.mean(chunk_p, axis=0)
                p_std = np.std(chunk_p, axis=0)
                physio_feats.append(np.hstack([p_mean, p_std]))
            else:
                physio_feats.append(np.zeros(2))

    X_motion = np.array(motion_feats) if has_motion else None
    X_physio = np.array(physio_feats) if has_physio else None

    return X_eeg, X_motion, X_physio, y, target_event_id


def run_all_decoding_pipelines():
    X_eeg, X_motion, X_physio, y, target_event_id = load_multimodal_dataset()

    results = {}

    # ---------------------------------------------------------
    # Pipeline 1: Riemannian Geometry Classifiers
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(" 1. Riemannian Geometry Classifiers (SPD Covariances) ".center(70, "="))
    print("=" * 70)

    # A. Covariances + TangentSpace + LogisticRegression
    covs = Covariances(estimator='oas').fit_transform(X_eeg)
    
    pipe_ts_lr = Pipeline([
        ('TS', TangentSpace(metric='riemann')),
        ('Scaler', StandardScaler()),
        ('LR', LogisticRegression(max_iter=1000, C=1.0))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_ts_lr = cross_val_score(pipe_ts_lr, covs, y, cv=cv, scoring='accuracy')
    results['Riemannian_TangentSpace_LR'] = (np.mean(scores_ts_lr) * 100, np.std(scores_ts_lr) * 100)
    print(f"  [+] TangentSpace + Logistic Regression: {np.mean(scores_ts_lr)*100:6.2f}% +/- {np.std(scores_ts_lr)*100:.2f}%")

    # B. Covariances + TangentSpace + Support Vector Machine (RBF kernel)
    pipe_ts_svm = Pipeline([
        ('TS', TangentSpace(metric='riemann')),
        ('Scaler', StandardScaler()),
        ('SVM', SVC(kernel='rbf', C=1.0))
    ])
    scores_ts_svm = cross_val_score(pipe_ts_svm, covs, y, cv=cv, scoring='accuracy')
    results['Riemannian_TangentSpace_SVM'] = (np.mean(scores_ts_svm) * 100, np.std(scores_ts_svm) * 100)
    print(f"  [+] TangentSpace + RBF SVM           : {np.mean(scores_ts_svm)*100:6.2f}% +/- {np.std(scores_ts_svm)*100:.2f}%")

    # C. Minimum Distance to Mean (MDM) on Riemannian Manifold
    pipe_mdm = Pipeline([
        ('Cov', Covariances(estimator='oas')),
        ('MDM', MDM(metric='riemann'))
    ])
    scores_mdm = cross_val_score(pipe_mdm, X_eeg, y, cv=cv, scoring='accuracy')
    results['Riemannian_MDM'] = (np.mean(scores_mdm) * 100, np.std(scores_mdm) * 100)
    print(f"  [+] Minimum Distance to Mean (MDM)  : {np.mean(scores_mdm)*100:6.2f}% +/- {np.std(scores_mdm)*100:.2f}%")

    # ---------------------------------------------------------
    # Pipeline 2: PyTorch Deep Learning EEGNet
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(" 2. PyTorch Deep Learning (EEGNet ConvNet Architecture) ".center(70, "="))
    print("=" * 70)

    eegnet_mean, eegnet_std = train_eval_eegnet(X_eeg, y, n_splits=5, epochs=40, lr=0.003)
    results['PyTorch_EEGNet'] = (eegnet_mean, eegnet_std)
    print(f"  [+] EEGNet (Deep ConvNet Architecture): {eegnet_mean:6.2f}% +/- {eegnet_std:.2f}%")

    # ---------------------------------------------------------
    # Pipeline 3: Multimodal Physiological Fusion Classifier
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(" 3. Multimodal Physiological Fusion (EEG + PPG + Motion) ".center(70, "="))
    print("=" * 70)

    # Feature 1: Tangent Space EEG Covariances
    ts_eeg_feats = TangentSpace(metric='riemann').fit_transform(covs)

    # Combine EEG + Motion + Physio
    fusion_blocks = [ts_eeg_feats]
    if X_motion is not None and len(X_motion) == len(y):
        fusion_blocks.append(X_motion)
        print("  [+] Appended Smartwatch 6-DOF IMU Motion Vector features.")
    if X_physio is not None and len(X_physio) == len(y):
        fusion_blocks.append(X_physio)
        print("  [+] Appended Smartwatch PPG Heart Rate / Pulse features.")

    X_fusion = np.hstack(fusion_blocks)
    
    pipe_fusion_gb = Pipeline([
        ('Scaler', StandardScaler()),
        ('GB', HistGradientBoostingClassifier(random_state=42))
    ])
    scores_fusion_gb = cross_val_score(pipe_fusion_gb, X_fusion, y, cv=cv, scoring='accuracy')
    results['Multimodal_Fusion_GradientBoosting'] = (np.mean(scores_fusion_gb) * 100, np.std(scores_fusion_gb) * 100)
    print(f"  [+] Multimodal Fusion (HistGradientBoosting): {np.mean(scores_fusion_gb)*100:6.2f}% +/- {np.std(scores_fusion_gb)*100:.2f}%")

    pipe_fusion_rf = Pipeline([
        ('Scaler', StandardScaler()),
        ('RF', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    scores_fusion_rf = cross_val_score(pipe_fusion_rf, X_fusion, y, cv=cv, scoring='accuracy')
    results['Multimodal_Fusion_RandomForest'] = (np.mean(scores_fusion_rf) * 100, np.std(scores_fusion_rf) * 100)
    print(f"  [+] Multimodal Fusion (RandomForest)        : {np.mean(scores_fusion_rf)*100:6.2f}% +/- {np.std(scores_fusion_rf)*100:.2f}%")

    # ---------------------------------------------------------
    # Final Benchmark Comparison & Saving Results
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print(" ADVANCED NEURAL DECODING BENCHMARK RESULTS ".center(75, "="))
    print("=" * 75)
    print(f"{'Method / Pipeline':38s} | {'Accuracy (Mean +/- Std)':22s} | {'Chance':7s}")
    print("-" * 75)
    for model_name, (mean_a, std_a) in results.items():
        print(f"{model_name:38s} | {mean_a:6.2f}% +/- {std_a:5.2f}%         | 25.0%")
    print("=" * 75)

    out_file = os.path.join("analysis_results", "advanced_decoding_benchmark.json")
    os.makedirs("analysis_results", exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Saved complete benchmark results to: {out_file}")
    return results


if __name__ == '__main__':
    run_all_decoding_pipelines()
