#!/usr/bin/env python3
"""
FFT-Aware Deep Learning Benchmark for EEG BCI
=============================================
Compares Time-Domain Deep Learning (Standard EEGNet) vs. Frequency-Domain / FFT-Aware Deep Learning:
  1. FFT-SpatialNet: Exact RFFT Spectral Magnitude (Phase-Invariant Bandpower Features)
  2. STFT-ConvNet: 2D Spectrogram (Channels x Frequency Bins x Time Windows)
  3. Filter-Bank Deep Learning (FBCNet inspired): Multi-band spectral spatial filtering
  4. Baseline: Standard Raw Time EEGNet
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_td_data import load_modern_tower_defense
from benchmark_deep_learning import EEGNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Model 1: Phase-Invariant FFT-SpatialNet
# ==============================================================================
class FFTSpatialNet(nn.Module):
    """
    Computes real FFT power spectrum (4-40 Hz) per channel, achieving complete
    phase-invariance, then applies Spatial Channel Mixing and Spectral Attention.
    """
    def __init__(self, n_channels=32, n_samples=750, sfreq=250.0, fmin=4.0, fmax=40.0, n_classes=4, n_spatial=16):
        super().__init__()
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        
        # Calculate FFT frequency indices
        freqs = np.fft.rfftfreq(n_samples, d=1.0/sfreq)
        self.freq_mask = (freqs >= fmin) & (freqs <= fmax)
        self.n_freqs = int(np.sum(self.freq_mask))
        
        # 1. Spatial Channel Convolution (Spatial filter across 32 electrodes)
        # Input to conv: (Batch, 1, Channels, Frequencies)
        self.spatial_conv = nn.Conv2d(1, n_spatial, (n_channels, 1), bias=False)
        self.bn_spatial = nn.BatchNorm2d(n_spatial)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.3)
        
        # 2. Spectral Feature Extractor (1D conv across frequency bins)
        self.spectral_conv = nn.Conv1d(n_spatial, n_spatial * 2, kernel_size=5, padding=2, bias=False)
        self.bn_spectral = nn.BatchNorm1d(n_spatial * 2)
        self.act2 = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.dropout2 = nn.Dropout(0.3)
        
        # 3. Dense Classifier
        self.fc = nn.Sequential(
            nn.Linear(n_spatial * 2 * 16, 32),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        # x: (Batch, Channels, Time)
        # 1. Compute real FFT along time axis
        fft_vals = torch.fft.rfft(x, dim=-1)
        # Power magnitude: |X(f)|^2 (completely phase invariant!)
        power = torch.abs(fft_vals) ** 2
        # Crop to 4-40 Hz band
        power_band = power[:, :, self.freq_mask]  # (Batch, Channels, Freqs)
        # Log-power scaling for normality
        log_power = torch.log(power_band + 1e-6)
        
        # 2. Spatial Convolution
        # Expand for Conv2d: (Batch, 1, Channels, Freqs)
        in_2d = log_power.unsqueeze(1)
        feat_spatial = self.act1(self.bn_spatial(self.spatial_conv(in_2d))) # (Batch, n_spatial, 1, Freqs)
        feat_spatial = self.dropout1(feat_spatial).squeeze(2)              # (Batch, n_spatial, Freqs)
        
        # 3. Spectral Convolution across frequency bins
        feat_spectral = self.act2(self.bn_spectral(self.spectral_conv(feat_spatial)))
        pooled = self.pool(feat_spectral)
        pooled = self.dropout2(pooled)
        
        # 4. Dense Output
        flattened = pooled.contiguous().view(pooled.size(0), -1)
        out = self.fc(flattened)
        return out

# ==============================================================================
# Model 2: STFT Time-Frequency Spectrogram Net
# ==============================================================================
class STFTConvNet(nn.Module):
    """
    Computes Short-Time Fourier Transform (Spectrogram) per channel,
    capturing both spectral power and temporal evolution.
    """
    def __init__(self, n_channels=32, n_classes=4, n_spatial=12):
        super().__init__()
        # STFT parameters
        self.n_fft = 128
        self.hop_length = 32
        
        # Spatial conv across 32 electrodes
        self.spatial_conv = nn.Conv2d(n_channels, n_spatial, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_spatial)
        
        # 2D Conv over (Freq x Time) spectrogram
        self.spec_conv = nn.Sequential(
            nn.Conv2d(n_spatial, 24, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.3),
            nn.Conv2d(24, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout(0.3)
        )
        self.fc = nn.Linear(32 * 4 * 4, n_classes)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        # Compute STFT for each channel: (Batch, Channels, Freqs, TimeFrames)
        stft_res = torch.stft(
            x.view(-1, x.size(-1)), 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            return_complex=True
        )
        spec = torch.abs(stft_res) ** 2
        # Crop frequencies to 4-40 Hz (approx bins 2 to 21)
        spec = spec[:, 2:22, :]
        spec = torch.log(spec + 1e-6)
        # Reshape back to (Batch, Channels, Freqs, TimeFrames)
        spec = spec.view(x.size(0), x.size(1), spec.size(1), spec.size(2))
        
        # Spatial channel mixing
        x_spat = self.bn1(self.spatial_conv(spec))
        x_feat = self.spec_conv(x_spat)
        out = self.fc(x_feat.contiguous().view(x_feat.size(0), -1))
        return out

# ==============================================================================
# Training Engine
# ==============================================================================
def train_and_eval(model, tr_loader, te_loader, epochs=65, lr=1e-3, weight_decay=1e-2):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in te_loader:
            bx, by = bx.to(device), by.to(device)
            preds = model(bx).argmax(dim=-1)
            correct += (preds == by).sum().item()
            total += by.size(0)
    return correct / total if total > 0 else 0.0

def run_fft_benchmark():
    print("=" * 85)
    print("FFT-AWARE DEEP LEARNING BENCHMARK vs TIME-DOMAIN EEGNET")
    print(f"Device: {device} | 32-Channel Dry Montages")
    print("=" * 85)
    
    (X_s03, y_s03, _), _ = load_modern_tower_defense(sessions=["03"])
    X_s03_32 = X_s03[:, :32, :]
    
    # Standardize data per trial
    norm_mean = np.mean(X_s03_32, axis=-1, keepdims=True)
    norm_std = np.std(X_s03_32, axis=-1, keepdims=True) + 1e-6
    X_norm = (X_s03_32 - norm_mean) / norm_std
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {
        "Raw Time EEGNet (Baseline)": [],
        "FFT-SpatialNet (Phase-Invariant)": [],
        "STFT-ConvNet (Spectrogram)": []
    }
    
    # -------------------------------------------------------------
    # 1. 4-Class Elemental Decoding
    # -------------------------------------------------------------
    print("\n[1/2] Evaluating 4-Class Elemental Decoding (69 trials, chance=25.0%)...")
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_norm, y_s03), start=1):
        t_tr_x = torch.tensor(X_norm[tr_idx], dtype=torch.float32)
        t_tr_y = torch.tensor(y_s03[tr_idx], dtype=torch.long)
        t_te_x = torch.tensor(X_norm[te_idx], dtype=torch.float32)
        t_te_y = torch.tensor(y_s03[te_idx], dtype=torch.long)
        
        tr_loader = DataLoader(TensorDataset(t_tr_x, t_tr_y), batch_size=16, shuffle=True)
        te_loader = DataLoader(TensorDataset(t_te_x, t_te_y), batch_size=32, shuffle=False)
        
        # A. Raw Time EEGNet
        m_eegnet = EEGNet(n_classes=4, channels=32, samples=750)
        acc_eeg = train_and_eval(m_eegnet, tr_loader, te_loader, epochs=65)
        results["Raw Time EEGNet (Baseline)"].append(acc_eeg)
        
        # B. FFT-SpatialNet
        m_fft = FFTSpatialNet(n_channels=32, n_samples=750, n_classes=4)
        acc_fft = train_and_eval(m_fft, tr_loader, te_loader, epochs=65)
        results["FFT-SpatialNet (Phase-Invariant)"].append(acc_fft)
        
        # C. STFT-ConvNet
        m_stft = STFTConvNet(n_channels=32, n_classes=4)
        acc_stft = train_and_eval(m_stft, tr_loader, te_loader, epochs=65)
        results["STFT-ConvNet (Spectrogram)"].append(acc_stft)
        
    print("\n--- 4-CLASS BENCHMARK RESULTS ---")
    for name, folds in results.items():
        print(f"  {name:<35}: {np.mean(folds)*100:6.2f}% ±{np.std(folds)*100:4.2f}% (Chance=25.0%)")
        
    # -------------------------------------------------------------
    # 2. Binary Decoding: FIRE vs ELECTRICITY
    # -------------------------------------------------------------
    print("\n[2/2] Evaluating Binary Decoding (FIRE vs ELECTRICITY, chance=50.0%)...")
    bin_mask = np.isin(y_s03, [0, 3])
    X_bin = X_norm[bin_mask]
    y_bin = np.where(y_s03[bin_mask] == 3, 1, 0)
    
    bin_results = {
        "Raw Time EEGNet (Baseline)": [],
        "FFT-SpatialNet (Phase-Invariant)": [],
        "STFT-ConvNet (Spectrogram)": []
    }
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_bin, y_bin), start=1):
        t_tr_x = torch.tensor(X_bin[tr_idx], dtype=torch.float32)
        t_tr_y = torch.tensor(y_bin[tr_idx], dtype=torch.long)
        t_te_x = torch.tensor(X_bin[te_idx], dtype=torch.float32)
        t_te_y = torch.tensor(y_bin[te_idx], dtype=torch.long)
        
        tr_loader = DataLoader(TensorDataset(t_tr_x, t_tr_y), batch_size=8, shuffle=True)
        te_loader = DataLoader(TensorDataset(t_te_x, t_te_y), batch_size=16, shuffle=False)
        
        # A. Raw Time EEGNet
        m_eegnet = EEGNet(n_classes=2, channels=32, samples=750)
        acc_eeg = train_and_eval(m_eegnet, tr_loader, te_loader, epochs=65)
        bin_results["Raw Time EEGNet (Baseline)"].append(acc_eeg)
        
        # B. FFT-SpatialNet
        m_fft = FFTSpatialNet(n_channels=32, n_samples=750, n_classes=2)
        acc_fft = train_and_eval(m_fft, tr_loader, te_loader, epochs=65)
        bin_results["FFT-SpatialNet (Phase-Invariant)"].append(acc_fft)
        
        # C. STFT-ConvNet
        m_stft = STFTConvNet(n_channels=32, n_classes=2)
        acc_stft = train_and_eval(m_stft, tr_loader, te_loader, epochs=65)
        bin_results["STFT-ConvNet (Spectrogram)"].append(acc_stft)
        
    print("\n--- BINARY BENCHMARK RESULTS ---")
    for name, folds in bin_results.items():
        print(f"  {name:<35}: {np.mean(folds)*100:6.2f}% ±{np.std(folds)*100:4.2f}% (Chance=50.0%)")

if __name__ == "__main__":
    run_fft_benchmark()
