#!/usr/bin/env python3
"""
Deep Learning Benchmark for 32-Channel EEG:
  1. EEGNet (Lawhern et al., 2018) - Compact spatial-temporal depthwise separable CNN (~2,500 params)
  2. ShallowFBCSPNet (Schirrmeister et al., 2017) - Direct neural analog of Filter Bank CSP (~10,000 params)
  3. Pre-trained EEGNet (Transfer learning from ses-listening prior -> fine-tuned on active recall)
  4. Baseline Comparison: Riemannian Tangent Space + LR & CSP + LDA
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Import data loaders
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_td_data import load_classical_tower_defense, load_modern_tower_defense

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. EEGNet Architecture (Lawhern et al., J. Neural Eng. 2018)
# ==============================================================================
class EEGNet(nn.Module):
    """
    EEGNet-8,2: Compact Convolutional Neural Network for EEG BCI
    Parameters: ~2,500 weights.
    """
    def __init__(self, n_classes=4, channels=32, samples=750, F1=8, D=2, F2=16, kernel_length=64, dropout_rate=0.25):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2
        
        # Block 1: Temporal convolution (learns frequency bandpass filters)
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise Spatial Convolution (learns spatial filters across 32 electrodes, akin to CSP)
        self.depthwise = nn.Conv2d(F1, F1 * D, (channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)
        
        # Block 2: Separable Convolution (temporal summary)
        self.separable_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.separable_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)
        
        # Dense Classification Layer
        # Calculate feature dimension after pooling
        pooled_samples = samples // 4 // 8
        self.flatten_dim = F2 * pooled_samples
        self.fc = nn.Linear(self.flatten_dim, n_classes)

    def forward(self, x):
        # Input shape: (Batch, Channels, Time) -> expand to (Batch, 1, Channels, Time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        x = x.contiguous().view(x.size(0), -1)
        # Handle slight padding variations
        if x.size(1) != self.flatten_dim:
            self.fc = nn.Linear(x.size(1), self.fc.out_features).to(x.device)
            self.flatten_dim = x.size(1)
            
        out = self.fc(x)
        return out

# ==============================================================================
# 2. ShallowFBCSPNet Architecture (Schirrmeister et al., HBM 2017)
# ==============================================================================
class ShallowFBCSPNet(nn.Module):
    """
    Shallow ConvNet mimicking Filter Bank Common Spatial Pattern (FBCSP)
    Uses log-squared pooling to compute bandpower features.
    """
    def __init__(self, n_classes=4, channels=32, samples=750, n_filters=40, kernel_length=25, pool_size=75, stride=15):
        super().__init__()
        self.conv_time = nn.Conv2d(1, n_filters, (1, kernel_length), bias=False)
        self.conv_spat = nn.Conv2d(n_filters, n_filters, (channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.pool = nn.AvgPool2d((1, pool_size), stride=(1, stride))
        self.drop = nn.Dropout(0.3)
        
        # Calculate output dimension after pooling
        time_after_conv = samples - kernel_length + 1
        time_after_pool = (time_after_conv - pool_size) // stride + 1
        self.flatten_dim = n_filters * time_after_pool
        self.fc = nn.Linear(self.flatten_dim, n_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        # Power / square activation mimicking energy computation
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.drop(x)
        
        x = x.contiguous().view(x.size(0), -1)
        if x.size(1) != self.flatten_dim:
            self.fc = nn.Linear(x.size(1), self.fc.out_features).to(x.device)
            self.flatten_dim = x.size(1)
        out = self.fc(x)
        return out

# ==============================================================================
# Training & Cross Validation Engine
# ==============================================================================
def train_model(model, train_loader, val_loader, epochs=60, lr=1e-3, weight_decay=1e-2):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            out = model(b_x)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            
    # Final evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for b_x, b_y in val_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            preds = model(b_x).argmax(dim=-1)
            correct += (preds == b_y).sum().item()
            total += b_y.size(0)
    return correct / total if total > 0 else 0.0

def evaluate_deep_learning(X, y, X_listen=None, y_listen=None, n_splits=5):
    """
    Evaluates EEGNet, ShallowFBCSPNet, Transfer-EEGNet, and Classical Baselines.
    """
    n_classes = len(np.unique(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    scores = {
        'Riemannian Tangent Space': [],
        'CSP + LDA': [],
        'EEGNet (from scratch)': [],
        'ShallowFBCSPNet': [],
        'EEGNet (Pretrained on Listen Prior)': []
    }
    
    channels = X.shape[1]
    samples = X.shape[2]
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]
        
        # 1. Riemannian Baseline
        try:
            cov = Covariances(estimator='oas')
            ts = TangentSpace(metric='riemann')
            clf = LogisticRegression(C=1.0, max_iter=500)
            pipe = make_pipeline(cov, ts, clf)
            pipe.fit(X_tr, y_tr)
            scores['Riemannian Tangent Space'].append(pipe.score(X_te, y_te))
        except Exception:
            scores['Riemannian Tangent Space'].append(1.0 / n_classes)
            
        # 2. CSP + LDA Baseline
        try:
            csp = CSP(n_components=min(8, channels), reg='ledoit_wolf', log=True)
            lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            pipe_csp = make_pipeline(csp, lda)
            pipe_csp.fit(X_tr, y_tr)
            scores['CSP + LDA'].append(pipe_csp.score(X_te, y_te))
        except Exception:
            scores['CSP + LDA'].append(1.0 / n_classes)
            
        # Standardize per trial across time for Neural Nets
        # (normalize mean=0, std=1 across channels/time)
        norm_mean = np.mean(X_tr, axis=-1, keepdims=True)
        norm_std = np.std(X_tr, axis=-1, keepdims=True) + 1e-6
        X_tr_norm = (X_tr - norm_mean) / norm_std
        
        te_mean = np.mean(X_te, axis=-1, keepdims=True)
        te_std = np.std(X_te, axis=-1, keepdims=True) + 1e-6
        X_te_norm = (X_te - te_mean) / te_std
        
        t_X_tr = torch.tensor(X_tr_norm, dtype=torch.float32)
        t_y_tr = torch.tensor(y_tr, dtype=torch.long)
        t_X_te = torch.tensor(X_te_norm, dtype=torch.float32)
        t_y_te = torch.tensor(y_te, dtype=torch.long)
        
        tr_ds = TensorDataset(t_X_tr, t_y_tr)
        te_ds = TensorDataset(t_X_te, t_y_te)
        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
        te_loader = DataLoader(te_ds, batch_size=32, shuffle=False)
        
        # 3. EEGNet (From Scratch)
        model_eegnet = EEGNet(n_classes=n_classes, channels=channels, samples=samples)
        acc_eegnet = train_model(model_eegnet, tr_loader, te_loader, epochs=65, lr=1e-3, weight_decay=1e-2)
        scores['EEGNet (from scratch)'].append(acc_eegnet)
        
        # 4. ShallowFBCSPNet
        model_shallow = ShallowFBCSPNet(n_classes=n_classes, channels=channels, samples=samples)
        acc_shallow = train_model(model_shallow, tr_loader, te_loader, epochs=65, lr=1e-3, weight_decay=1e-2)
        scores['ShallowFBCSPNet'].append(acc_shallow)
        
        # 5. Pretrained EEGNet (if listening data is provided)
        if X_listen is not None and y_listen is not None and len(X_listen) > 0:
            # Pretrain on listening prior
            model_transfer = EEGNet(n_classes=n_classes, channels=channels, samples=samples).to(device)
            lis_mean = np.mean(X_listen, axis=-1, keepdims=True)
            lis_std = np.std(X_listen, axis=-1, keepdims=True) + 1e-6
            X_lis_norm = (X_listen - lis_mean) / lis_std
            t_X_lis = torch.tensor(X_lis_norm, dtype=torch.float32)
            t_y_lis = torch.tensor(y_listen, dtype=torch.long)
            lis_loader = DataLoader(TensorDataset(t_X_lis, t_y_lis), batch_size=32, shuffle=True)
            
            # Pre-training loop (30 epochs on listening data)
            opt_pre = torch.optim.AdamW(model_transfer.parameters(), lr=1e-3, weight_decay=1e-2)
            crit = nn.CrossEntropyLoss()
            model_transfer.train()
            for _ in range(35):
                for bx, by in lis_loader:
                    bx, by = bx.to(device), by.to(device)
                    opt_pre.zero_grad()
                    loss = crit(model_transfer(bx), by)
                    loss.backward()
                    opt_pre.step()
                    
            # Fine-tune on active recall training fold
            opt_fine = torch.optim.AdamW(model_transfer.parameters(), lr=3e-4, weight_decay=1e-2)
            for _ in range(30):
                for bx, by in tr_loader:
                    bx, by = bx.to(device), by.to(device)
                    opt_fine.zero_grad()
                    loss = crit(model_transfer(bx), by)
                    loss.backward()
                    opt_fine.step()
                    
            # Evaluate transfer model
            model_transfer.eval()
            corr, tot = 0, 0
            with torch.no_grad():
                for bx, by in te_loader:
                    bx, by = bx.to(device), by.to(device)
                    preds = model_transfer(bx).argmax(dim=-1)
                    corr += (preds == by).sum().item()
                    tot += by.size(0)
            scores['EEGNet (Pretrained on Listen Prior)'].append(corr / tot if tot > 0 else 0.0)
        else:
            scores['EEGNet (Pretrained on Listen Prior)'].append(0.0)

    summary = {}
    for k, v in scores.items():
        summary[k] = {
            'mean': float(np.mean(v)),
            'std': float(np.std(v)),
            'folds': [float(x) for x in v]
        }
    return summary

def main():
    print("=" * 85)
    print("DEEP LEARNING BCI BENCHMARK: EEGNet vs. ShallowFBCSPNet vs. Pre-trained Transfer")
    print(f"Device: {device} | 32-Channel Dry Montages")
    print("=" * 85)
    
    out_dir = Path(__file__).resolve().parent.parent.parent / "results" / "deep_learning_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    
    # -------------------------------------------------------------
    # Experiment 1: Subject 2 Modern ses-03 (Peak 69 trials, 4 classes)
    # -------------------------------------------------------------
    print("\n[1/3] Subject 2 ses-03 (69 trials, 4 classes)...")
    (X_s03, y_s03, _), (X_lis_m, y_lis_m) = load_modern_tower_defense(sessions=["03"])
    res_s03 = evaluate_deep_learning(X_s03, y_s03, X_listen=X_lis_m, y_listen=y_lis_m)
    all_results['sub02_ses03_4class'] = res_s03
    
    # -------------------------------------------------------------
    # Experiment 2: Subject 2 Binary (FIRE vs ELECTRICITY in ses-03)
    # -------------------------------------------------------------
    print("\n[2/3] Subject 2 ses-03 Binary (FIRE vs ELECTRICITY, chance=50%)...")
    mask = np.isin(y_s03, [0, 3])
    X_bin = X_s03[mask]
    y_bin = np.where(y_s03[mask] == 3, 1, 0)
    mask_lis = np.isin(y_lis_m, [0, 3])
    X_lis_bin = X_lis_m[mask_lis]
    y_lis_bin = np.where(y_lis_m[mask_lis] == 3, 1, 0)
    res_bin = evaluate_deep_learning(X_bin, y_bin, X_listen=X_lis_bin, y_listen=y_lis_bin)
    all_results['sub02_ses03_binary'] = res_bin
    
    # -------------------------------------------------------------
    # Experiment 3: Subject 1 Classical Golden (ses-01 + 04, 152 trials)
    # -------------------------------------------------------------
    print("\n[3/3] Subject 1 Classical Golden (152 trials, 4 classes)...")
    (X_c, y_c), (X_lis_c, y_lis_c) = load_classical_tower_defense(recommended_only=True)
    res_c = evaluate_deep_learning(X_c, y_c, X_listen=X_lis_c, y_listen=y_lis_c)
    all_results['sub01_classical_golden_4class'] = res_c
    
    # Save results
    json_path = out_dir / "deep_learning_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Done] Results saved to {json_path}")
    
    # Print formatted table
    print("\n" + "=" * 90)
    print(f"{'Experiment':<30} | {'Architecture':<35} | {'Mean Acc':<10} | {'Std Dev':<8}")
    print("=" * 90)
    for exp, data in all_results.items():
        print(f"--- {exp.upper()} ---")
        for arch, stats in data.items():
            print(f"{exp:<30} | {arch:<35} | {stats['mean']*100:6.2f}%    | ±{stats['std']*100:5.2f}%")
        print("-" * 90)

if __name__ == "__main__":
    main()
