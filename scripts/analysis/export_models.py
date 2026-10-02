#!/usr/bin/env python3
"""
Package & Export Pretrained Assistive BCI Models
================================================
Exports:
  1. models/hierarchical_riemannian_sub02.joblib (Hierarchical 2-Level Riemannian Model)
  2. models/pretrained_eegnet_sub02.pt (Pre-trained & Fine-tuned EEGNet on CUDA/CPU)
  3. models/model_metadata.json (Channel orders, classes, sampling rate, thresholds)
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR / "scripts" / "analysis"))
from load_td_data import load_modern_tower_defense
from hierarchical_assistive_decoder import HierarchicalAssistiveDecoder
from benchmark_deep_learning import EEGNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def export_hierarchical_riemannian():
    print("\n[1/2] Training & Exporting Hierarchical Riemannian Decoder...")
    # Load Subject 2 golden sessions
    (X_pool, y_pool, _), _ = load_modern_tower_defense(recommended_only=True)
    print(f"  Training on {len(X_pool)} golden pure MI trials...")
    
    decoder = HierarchicalAssistiveDecoder(confidence_threshold=0.80, max_steps=3)
    decoder.fit(X_pool, y_pool)
    
    out_path = MODELS_DIR / "hierarchical_riemannian_sub02.joblib"
    joblib.dump(decoder, out_path)
    print(f"  Successfully exported to: {out_path}")

def export_eegnet_pipeline():
    print("\n[2/2] Pretraining & Exporting EEGNet Deep Learning Pipeline...")
    (X_recall, y_recall, _), (X_listen, y_listen) = load_modern_tower_defense(recommended_only=True)
    
    channels = X_recall.shape[1]
    samples = X_recall.shape[2]
    n_classes = 4
    
    model = EEGNet(n_classes=n_classes, channels=channels, samples=samples).to(device)
    
    # 1. Pretrain on Acoustic Listening Prior (379 epochs)
    print(f"  Pre-training on {len(X_listen)} acoustic listening prior epochs...")
    lis_mean = np.mean(X_listen, axis=-1, keepdims=True)
    lis_std = np.std(X_listen, axis=-1, keepdims=True) + 1e-6
    X_lis_norm = (X_listen - lis_mean) / lis_std
    
    t_X_lis = torch.tensor(X_lis_norm, dtype=torch.float32)
    t_y_lis = torch.tensor(y_listen, dtype=torch.long)
    lis_loader = DataLoader(TensorDataset(t_X_lis, t_y_lis), batch_size=32, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for ep in range(35):
        for bx, by in lis_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    # 2. Fine-tune on Pure MI Golden Recall (168 trials)
    print(f"  Fine-tuning on {len(X_recall)} golden pure mental imagery trials...")
    rec_mean = np.mean(X_recall, axis=-1, keepdims=True)
    rec_std = np.std(X_recall, axis=-1, keepdims=True) + 1e-6
    X_rec_norm = (X_recall - rec_mean) / rec_std
    
    t_X_rec = torch.tensor(X_rec_norm, dtype=torch.float32)
    t_y_rec = torch.tensor(y_recall, dtype=torch.long)
    rec_loader = DataLoader(TensorDataset(t_X_rec, t_y_rec), batch_size=16, shuffle=True)
    
    opt_fine = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    for ep in range(35):
        for bx, by in rec_loader:
            bx, by = bx.to(device), by.to(device)
            opt_fine.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            opt_fine.step()
            
    model_path = MODELS_DIR / "pretrained_eegnet_sub02.pt"
    torch.save(model.state_dict(), model_path)
    print(f"  Successfully exported to: {model_path}")
    
    # Export Metadata
    meta = {
        "model_family": "EEGNet-8,2 & Hierarchical Riemannian",
        "channels": 32,
        "sampling_rate_hz": 250.0,
        "epoch_duration_s": 3.0,
        "epoch_samples": 750,
        "classes": {
            0: "FIRE (Kiss / Energy)",
            1: "WATER (It's Raining Men / Nature)",
            2: "WIND (What's Up / Nature)",
            3: "ELECTRICITY (Thunderstruck / Energy)"
        },
        "hierarchical_tree": {
            "root": "Energy (0, 3) vs Nature (1, 2)",
            "energy_subtree": "Fire (0) vs Electricity (3)",
            "nature_subtree": "Water (1) vs Wind (2)"
        },
        "evidence_accumulation": {
            "window_size_s": 2.0,
            "step_size_s": 0.5,
            "default_confidence_threshold": 0.80,
            "max_accumulation_steps": 3
        },
        "hardware_montage": "dry_electrodes_32ch"
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Exported metadata to: {MODELS_DIR / 'model_metadata.json'}")

if __name__ == "__main__":
    export_hierarchical_riemannian()
    export_eegnet_pipeline()
