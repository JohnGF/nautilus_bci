"""
Multimodal Adaptive Gated Deep Learning Network (PyTorch)
==========================================================

Implements an end-to-end Multimodal Neural Network with Learnable Softmax Gating:
  - Separate encoder layers for EEG, PPG, and IMU streams.
  - Automatic Modality Gating Layer: Learns dynamic softmax attention weights (w_EEG, w_PPG, w_IMU)
    that automatically scale UP informative streams and scale DOWN noisy streams (e.g. motion noise).
  - Joint Multimodal Classifier Head.

Usage:
  uv run python analysis/multimodal_gated_network.py
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# PyTorch Multimodal Adaptive Gated Architecture
# ---------------------------------------------------------
class EEGStreamEncoder(nn.Module):
    """Processes EEG 2D matrix (channels x time) into a latent representation vector."""
    def __init__(self, n_channels=32, n_samples=626, embed_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, (1, 32), padding=(0, 16), bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.depthwise = nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.act = nn.ELU()
        self.pool = nn.AvgPool2d((1, 8))
        self.drop = nn.Dropout(0.25)
        
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_samples)
            x = self.drop(self.pool(self.act(self.bn2(self.depthwise(self.bn1(self.conv1(x)))))))
            self._flat_dim = x.view(1, -1).size(1)

        self.proj = nn.Linear(self._flat_dim, embed_dim)

    def forward(self, x_eeg):
        # x_eeg: (batch, 1, channels, samples)
        x = self.conv1(x_eeg)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


class Signal1DEncoder(nn.Module):
    """Processes 1D time-series features (PPG or IMU) into a latent representation vector."""
    def __init__(self, in_features, embed_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)


class ModalityGatingMechanism(nn.Module):
    """
    Learns dynamic softmax gating weights across modalities [w_EEG, w_PPG, w_IMU].
    Automatically suppresses noisy modalities and amplifies high-SNR signals.
    """
    def __init__(self, embed_dim=64):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, h_eeg, h_ppg, h_imu):
        concat_h = torch.cat([h_eeg, h_ppg, h_imu], dim=-1)
        weights = self.gate_network(concat_h)  # shape: (batch_size, 3)
        
        w_eeg = weights[:, 0:1]
        w_ppg = weights[:, 1:2]
        w_imu = weights[:, 2:3]

        h_fused = w_eeg * h_eeg + w_ppg * h_ppg + w_imu * h_imu
        return h_fused, weights


class MultimodalAdaptiveGatedNet(nn.Module):
    """Complete End-to-End Adaptive Gated Multimodal Deep Learning Network."""
    def __init__(self, n_eeg_channels=32, n_eeg_samples=626, ppg_dim=4, imu_dim=20, embed_dim=64, n_classes=4):
        super().__init__()
        self.eeg_encoder = EEGStreamEncoder(n_channels=n_eeg_channels, n_samples=n_eeg_samples, embed_dim=embed_dim)
        self.ppg_encoder = Signal1DEncoder(in_features=ppg_dim, embed_dim=embed_dim)
        self.imu_encoder = Signal1DEncoder(in_features=imu_dim, embed_dim=embed_dim)
        
        self.gating = ModalityGatingMechanism(embed_dim=embed_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x_eeg, x_ppg, x_imu):
        h_eeg = self.eeg_encoder(x_eeg)
        h_ppg = self.ppg_encoder(x_ppg)
        h_imu = self.imu_encoder(x_imu)

        h_fused, weights = self.gating(h_eeg, h_ppg, h_imu)
        logits = self.classifier(h_fused)
        return logits, weights


# ---------------------------------------------------------
# Dataset Prep & Cross-Validation Trainer
# ---------------------------------------------------------
def load_dataset_for_gated_net(bids_root="bids_baseline", sub="01", ses="02", task="video"):
    bids_root = os.path.abspath(bids_root)
    bids_path = BIDSPath(subject=sub, session=ses, task=task, datatype="eeg", root=bids_root)
    
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
        if has_motion:
            idx_m = (t_m >= onset) & (t_m <= onset + 2.5)
            if np.any(idx_m):
                chunk = d_m[idx_m]
                m_mean = np.mean(chunk, axis=0)
                m_std = np.std(chunk, axis=0)
                m_max = np.max(chunk, axis=0)
                m_mag = np.sqrt(np.sum(chunk[:, :3]**2, axis=1))
                motion_feats.append(np.hstack([m_mean, m_std, m_max, [np.mean(m_mag), np.std(m_mag)]]))
            else:
                motion_feats.append(np.zeros(20))

        if has_physio:
            idx_p = (t_p >= onset) & (t_p <= onset + 2.5)
            if np.any(idx_p):
                chunk_p = d_p[idx_p]
                physio_feats.append(np.hstack([np.mean(chunk_p, axis=0), np.std(chunk_p, axis=0), np.min(chunk_p, axis=0), np.max(chunk_p, axis=0)]))
            else:
                physio_feats.append(np.zeros(4))

    X_ppg = np.array(physio_feats) if has_physio else np.zeros((len(y), 4))
    X_imu = np.array(motion_feats) if has_motion else np.zeros((len(y), 20))

    return X_eeg, X_ppg, X_imu, y


def train_eval_gated_network(n_splits=5, epochs=45, batch_size=16, lr=0.002):
    X_eeg, X_ppg, X_imu, y = load_dataset_for_gated_net()
    
    label_map = {orig: idx for idx, orig in enumerate(sorted(list(np.unique(y))))}
    y_mapped = np.array([label_map[v] for v in y])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print(" Training Multimodal Adaptive Gated Deep Network (PyTorch) ".center(80, "="))
    print("=" * 80)

    acc_scores = []
    all_fold_weights = []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X_eeg, y_mapped)):
        # Standardize 1D features
        scaler_ppg = StandardScaler().fit(X_ppg[tr_idx])
        X_ppg_tr = scaler_ppg.transform(X_ppg[tr_idx])
        X_ppg_te = scaler_ppg.transform(X_ppg[te_idx])

        scaler_imu = StandardScaler().fit(X_imu[tr_idx])
        X_imu_tr = scaler_imu.transform(X_imu[tr_idx])
        X_imu_te = scaler_imu.transform(X_imu[te_idx])

        # Tensors
        eeg_tr = torch.tensor(X_eeg[tr_idx], dtype=torch.float32).unsqueeze(1).to(device)
        ppg_tr = torch.tensor(X_ppg_tr, dtype=torch.float32).to(device)
        imu_tr = torch.tensor(X_imu_tr, dtype=torch.float32).to(device)
        y_tr = torch.tensor(y_mapped[tr_idx], dtype=torch.long).to(device)

        eeg_te = torch.tensor(X_eeg[te_idx], dtype=torch.float32).unsqueeze(1).to(device)
        ppg_te = torch.tensor(X_ppg_te, dtype=torch.float32).to(device)
        imu_te = torch.tensor(X_imu_te, dtype=torch.float32).to(device)
        y_te = torch.tensor(y_mapped[te_idx], dtype=torch.long).to(device)

        ds = TensorDataset(eeg_tr, ppg_tr, imu_tr, y_tr)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        model = MultimodalAdaptiveGatedNet(
            n_eeg_channels=X_eeg.shape[1],
            n_eeg_samples=X_eeg.shape[2],
            ppg_dim=X_ppg.shape[1],
            imu_dim=X_imu.shape[1],
            embed_dim=64,
            n_classes=len(label_map)
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        model.train()
        for ep in range(epochs):
            for b_eeg, b_ppg, b_imu, b_y in loader:
                optimizer.zero_grad()
                logits, _ = model(b_eeg, b_ppg, b_imu)
                loss = criterion(logits, b_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, weights = model(eeg_te, ppg_te, imu_te)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_te).float().mean().item()
            acc_scores.append(acc)
            all_fold_weights.append(weights.cpu().numpy())

        print(f" Fold {fold+1}/{n_splits} Accuracy: {acc*100:5.2f}%")

    mean_acc = np.mean(acc_scores) * 100
    std_acc = np.std(acc_scores) * 100

    avg_weights = np.mean(np.vstack(all_fold_weights), axis=0)

    print("\n" + "=" * 80)
    print(" GATED NETWORK RESULTS & AUTOMATIC MODALITY WEIGHTS ".center(80, "="))
    print("=" * 80)
    print(f"[RESULT] Multimodal Adaptive Gated Deep Net Accuracy: {mean_acc:.2f}% +/- {std_acc:.2f}% (Chance: 25.0%)")
    print("\n[WEIGHTS] Learned Automatic Modality Weights:")
    print(f"  * EEG Weight (w_EEG)       : {avg_weights[0]*100:6.2f}%")
    print(f"  * Smartwatch PPG (w_PPG)   : {avg_weights[1]*100:6.2f}%")
    print(f"  * Smartwatch IMU (w_IMU)   : {avg_weights[2]*100:6.2f}%")
    print("=" * 80)

    res = {
        'Multimodal_Gated_Net_Accuracy': float(mean_acc),
        'Multimodal_Gated_Net_Std': float(std_acc),
        'Learned_Weights': {
            'w_EEG': float(avg_weights[0]),
            'w_PPG': float(avg_weights[1]),
            'w_IMU': float(avg_weights[2])
        }
    }

    out_file = os.path.join("analysis_results", "multimodal_gated_net_results.json")
    os.makedirs("analysis_results", exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(res, f, indent=2)

    print(f"[+] Saved results to: {out_file}")
    return res


if __name__ == '__main__':
    train_eval_gated_network()
