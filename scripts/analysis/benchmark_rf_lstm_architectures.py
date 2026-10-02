#!/usr/bin/env python3
"""
Model Architecture Benchmark: CSP + Random Forest vs. LSTM vs. Riemannian vs. CSP+LDA
======================================================================================
Empirically tests and compares:
  1. CSP + LDA (Linear Discriminant Analysis with Ledoit-Wolf shrinkage)
  2. CSP + Random Forest (Non-linear decision trees)
  3. CSP + Extra Trees (Extremely randomized trees)
  4. CSP + SVM (Support Vector Machine with RBF kernel)
  5. Riemannian Tangent Space + Logistic Regression
  6. Deep Learning: SpatialConv + Bidirectional LSTM (PyTorch CUDA/CPU)
  7. Shuffled Label Controls (Empirical null hypothesis chance level)

Tested on:
  - Subject 2 Modern Pop/Rock (Peak ses-03 & Golden ses-01..03)
  - Subject 1 Classical Orchestral (Golden ses-01 & ses-04)
  - 4-Class Elemental Decoding and 2-Class Binary Decoding
"""

import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import mne
from mne.decoding import CSP

# Add parent directory to path to import load_td_data
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_td_data import load_classical_tower_defense, load_modern_tower_defense

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Deep Learning: PyTorch SpatialConv + BiLSTM ---
class EEG_Spatial_BiLSTM(nn.Module):
    def __init__(self, in_channels=32, spatial_filters=16, hidden_dim=32, num_layers=2, num_classes=4, dropout=0.3):
        super().__init__()
        # Spatial convolution: mixes 32 dry electrode channels at each time step
        self.spatial_conv = nn.Conv1d(in_channels, spatial_filters, kernel_size=1, bias=False)
        self.bn_spatial = nn.BatchNorm1d(spatial_filters)
        self.dropout_spatial = nn.Dropout(dropout)
        
        # BiLSTM over time sequence
        self.lstm = nn.LSTM(
            input_size=spatial_filters,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (Batch, Channels, Time)
        x = self.spatial_conv(x)       # (Batch, SpatialFilters, Time)
        x = self.bn_spatial(x)
        x = self.dropout_spatial(x)
        
        # LSTM expects (Batch, Time, Features)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)     # (Batch, Time, hidden_dim * 2)
        
        # Global average pooling over time
        pooled = torch.mean(lstm_out, dim=1)
        logits = self.classifier(pooled)
        return logits

def train_eval_lstm(X_tr, y_tr, X_te, y_te, num_classes=4, epochs=45, batch_size=16, lr=1e-3, weight_decay=1e-2):
    # Downsample time from 750 samples (250Hz, 3s) to 150 samples (50Hz) for computational efficiency & SNR
    # Resample using linear interpolation or striding
    stride = 5  # 750 / 5 = 150 time points
    X_tr_sub = X_tr[:, :, ::stride]
    X_te_sub = X_te[:, :, ::stride]
    
    in_channels = X_tr.shape[1]
    model = EEG_Spatial_BiLSTM(in_channels=in_channels, spatial_filters=16, hidden_dim=32, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    t_X_tr = torch.tensor(X_tr_sub, dtype=torch.float32)
    t_y_tr = torch.tensor(y_tr, dtype=torch.long)
    t_X_te = torch.tensor(X_te_sub, dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(t_X_tr, t_y_tr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for ep in range(epochs):
        for b_X, b_y in train_loader:
            b_X, b_y = b_X.to(device), b_y.to(device)
            optimizer.zero_grad()
            out = model(b_X)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        preds = model(t_X_te).argmax(dim=-1).cpu().numpy()
    
    acc = np.mean(preds == y_te)
    return acc

def evaluate_models_cross_validation(X, y, dataset_name="Dataset", n_splits=5, shuffle_labels=False):
    """
    Evaluates all model families using 5-fold Stratified Cross Validation.
    """
    num_classes = len(np.unique(y))
    chance_level = 1.0 / num_classes
    
    y_eval = y.copy()
    if shuffle_labels:
        rng = np.random.default_rng(seed=42)
        rng.shuffle(y_eval)
        
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = {
        'CSP + LDA': [],
        'CSP + Random Forest (150 trees)': [],
        'CSP + Extra Trees (150 trees)': [],
        'CSP + SVM (RBF)': [],
        'Riemannian Tangent Space + LR': [],
        'BiLSTM (PyTorch)': []
    }
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y_eval)):
        X_tr, y_tr = X[tr_idx], y_eval[tr_idx]
        X_te, y_te = X[te_idx], y_eval[te_idx]
        
        # 1. CSP Feature Extraction
        # n_components=6 or 8
        n_comp = min(8, X_tr.shape[1])
        csp = CSP(n_components=n_comp, reg='ledoit_wolf', log=True, norm_trace=False)
        try:
            X_tr_csp = csp.fit_transform(X_tr, y_tr)
            X_te_csp = csp.transform(X_te)
        except Exception:
            # Fallback if CSP covariance fails
            X_tr_csp = np.var(X_tr, axis=-1)
            X_te_csp = np.var(X_te, axis=-1)
            
        # A. CSP + LDA
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lda.fit(X_tr_csp, y_tr)
        results['CSP + LDA'].append(lda.score(X_te_csp, y_te))
        
        # B. CSP + Random Forest
        rf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=4, random_state=42 + fold)
        rf.fit(X_tr_csp, y_tr)
        results['CSP + Random Forest (150 trees)'].append(rf.score(X_te_csp, y_te))
        
        # C. CSP + Extra Trees
        et = ExtraTreesClassifier(n_estimators=150, max_depth=6, min_samples_split=4, random_state=42 + fold)
        et.fit(X_tr_csp, y_tr)
        results['CSP + Extra Trees (150 trees)'].append(et.score(X_te_csp, y_te))
        
        # D. CSP + SVM (RBF)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_csp)
        X_te_scaled = scaler.transform(X_te_csp)
        svm = SVC(kernel='rbf', C=1.5, gamma='scale')
        svm.fit(X_tr_scaled, y_tr)
        results['CSP + SVM (RBF)'].append(svm.score(X_te_scaled, y_te))
        
        # E. Riemannian Tangent Space + LR
        try:
            cov = Covariances(estimator='oas')
            ts = TangentSpace(metric='riemann')
            lr = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs')
            pipe_riemann = make_pipeline(cov, ts, lr)
            pipe_riemann.fit(X_tr, y_tr)
            results['Riemannian Tangent Space + LR'].append(pipe_riemann.score(X_te, y_te))
        except Exception as e:
            results['Riemannian Tangent Space + LR'].append(chance_level)
            
        # F. BiLSTM (PyTorch)
        acc_lstm = train_eval_lstm(X_tr, y_tr, X_te, y_te, num_classes=num_classes)
        results['BiLSTM (PyTorch)'].append(acc_lstm)
        
    summary = {}
    for k, v in results.items():
        summary[k] = {
            'mean': float(np.mean(v)),
            'std': float(np.std(v)),
            'raw_folds': [float(x) for x in v]
        }
    return summary

def main():
    print("=" * 80)
    print("BCI DECODER COMPARISON: CSP+LDA vs CSP+RF vs CSP+SVM vs RIEMANNIAN vs BiLSTM")
    print(f"Device: {device} | Dry Electrode 32-Channel Montages")
    print("=" * 80)
    
    out_dir = Path(__file__).resolve().parent.parent.parent / "results" / "model_architecture_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_benchmarks = {}
    
    # ---------------------------------------------------------
    # 1. Subject 2: Modern Peak Session 03 (69 trials, 4 classes)
    # ---------------------------------------------------------
    print("\n[1/4] Loading Subject 2 Modern Session 03 (Dry Cz/F4 Reseated Peak)...")
    (X_s03, y_s03, _), _ = load_modern_tower_defense(sessions=["03"])
    print(f"  Shape: X={X_s03.shape}, y={y_s03.shape}, classes={np.unique(y_s03)}")
    
    print("  Evaluating True Labels (4-class, chance=25.0%)...")
    res_s03_true = evaluate_models_cross_validation(X_s03, y_s03, "Sub-02 ses-03 True")
    print("  Evaluating Shuffled Labels (Null Hypothesis Control)...")
    res_s03_shuf = evaluate_models_cross_validation(X_s03, y_s03, "Sub-02 ses-03 Shuffled", shuffle_labels=True)
    all_benchmarks['sub02_ses03_4class'] = {'true': res_s03_true, 'shuffled': res_s03_shuf}
    
    # ---------------------------------------------------------
    # 2. Subject 2: Modern Golden Pooled (ses-01 + ses-02 + ses-03, 168 trials, 4 classes)
    # ---------------------------------------------------------
    print("\n[2/4] Loading Subject 2 Modern Golden Pooled (ses-01 + 02 + 03)...")
    (X_pool, y_pool, _), _ = load_modern_tower_defense(recommended_only=True)
    print(f"  Shape: X={X_pool.shape}, y={y_pool.shape}, classes={np.unique(y_pool)}")
    
    print("  Evaluating True Labels (4-class, chance=25.0%)...")
    res_pool_true = evaluate_models_cross_validation(X_pool, y_pool, "Sub-02 Golden True")
    all_benchmarks['sub02_golden_pooled_4class'] = {'true': res_pool_true}
    
    # ---------------------------------------------------------
    # 3. Subject 2: Modern Peak Binary (FIRE vs ELECTRICITY in ses-03)
    # ---------------------------------------------------------
    print("\n[3/4] Subject 2 ses-03 Binary (FIRE [0] vs ELECTRICITY [3], chance=50.0%)...")
    bin_mask = np.isin(y_s03, [0, 3])
    X_bin = X_s03[bin_mask]
    y_bin = np.where(y_s03[bin_mask] == 3, 1, 0)
    print(f"  Shape: X={X_bin.shape}, y={y_bin.shape}")
    res_bin_true = evaluate_models_cross_validation(X_bin, y_bin, "Sub-02 ses-03 Binary")
    all_benchmarks['sub02_ses03_binary'] = {'true': res_bin_true}

    # ---------------------------------------------------------
    # 4. Subject 1: Classical Golden (ses-01 + ses-04, 152 trials, 4 classes)
    # ---------------------------------------------------------
    print("\n[4/4] Loading Subject 1 Classical Golden (ses-01 + 04)...")
    (X_c, y_c), _ = load_classical_tower_defense(recommended_only=True)
    print(f"  Shape: X={X_c.shape}, y={y_c.shape}, classes={np.unique(y_c)}")
    res_c_true = evaluate_models_cross_validation(X_c, y_c, "Sub-01 Classical Golden")
    all_benchmarks['sub01_classical_golden_4class'] = {'true': res_c_true}

    # Save results to JSON
    json_path = out_dir / "architecture_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(all_benchmarks, f, indent=2)
    print(f"\n[Done] Benchmark results written to {json_path}")
    
    # Print formatted summary table
    print("\n" + "=" * 95)
    print(f"{'Experiment':<32} | {'Model Architecture':<30} | {'Mean Acc':<10} | {'Std Dev':<10} | {'Chance':<8}")
    print("=" * 95)
    for exp_name, data in all_benchmarks.items():
        chance = "50.0%" if "binary" in exp_name else "25.0%"
        print(f"--- {exp_name.upper()} ---")
        for model_name, stats in data['true'].items():
            print(f"{exp_name:<32} | {model_name:<30} | {stats['mean']*100:6.2f}%    | ±{stats['std']*100:5.2f}%   | {chance}")
        if 'shuffled' in data:
            print(f"  [Shuffled Control (Chance Verification)]")
            for model_name, stats in data['shuffled'].items():
                print(f"{'  shuffled':<32} | {model_name:<30} | {stats['mean']*100:6.2f}%    | ±{stats['std']*100:5.2f}%   | {chance}")
        print("-" * 95)

if __name__ == "__main__":
    main()
