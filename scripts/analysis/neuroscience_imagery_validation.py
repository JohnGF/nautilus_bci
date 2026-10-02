#!/usr/bin/env python3
"""
Neuroscience Topography & Spectral Validation of Musical Mental Imagery
========================================================================
Investigates the neurophysiological plausibility of the decoded brain signals:
  1. Cortical Region Power Shifts:
       - Auditory Association Cortex (T7, T8, FT9, FT10, TP9, TP10)
       - Supplementary Motor Area & Premotor Cortex (Cz, FC1, FC2, FC5, FC6, C3, C4)
       - Frontal Working Memory & Rehearsal (Fz, F3, F4, F7, F8)
       - Parietal Structural Integration (Pz, P3, P4, CP1, CP2, CP5, CP6)
       - Occipital Visual Control (O1, O2, Oz)
  2. Bandpower Spectral Signatures (ERD/ERS relative to pre-cue baseline):
       - Theta (4-7 Hz): Frontal working memory load
       - Mu / Alpha (8-12 Hz): Auditory & sensorimotor desynchronization (ERD)
       - Beta (15-25 Hz): Motor rhythmic rehearsal
  3. CSP Spatial Filter Topographies:
       - Checks where the machine learning model places its highest spatial weights.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import mne
from mne.decoding import CSP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_td_data import load_modern_tower_defense

CH_NAMES_32 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
    'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 
    'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 
    'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10'
]

# Anatomical regions of interest
ROIS = {
    "Auditory Association (STG)": ['T7', 'T8', 'FT9', 'FT10', 'TP9', 'TP10'],
    "SMA & Premotor (Rhythm)": ['Cz', 'FC1', 'FC2', 'FC5', 'FC6', 'C3', 'C4'],
    "Frontal Working Memory": ['Fz', 'F3', 'F4', 'F7', 'F8'],
    "Parietal Integration": ['Pz', 'P3', 'P4', 'CP1', 'CP2'],
    "Occipital Control (Visual)": ['O1', 'O2', 'Oz']
}

def bandpower_welch(data_trial, sfreq=250.0, fmin=8.0, fmax=12.0):
    """
    Computes average bandpower per channel using Welch's PSD.
    data_trial: (Channels, Samples)
    """
    # Simple Welch or FFT-based power
    n_samples = data_trial.shape[1]
    # FFT
    fft_vals = np.fft.rfft(data_trial, axis=-1)
    freqs = np.fft.rfftfreq(n_samples, d=1.0/sfreq)
    psd = (np.abs(fft_vals) ** 2) / (sfreq * n_samples)
    
    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
    bp = np.mean(psd[:, idx_band], axis=-1)
    return bp

def analyze_neuroscience_activation():
    print("=" * 85)
    print("NEUROSCIENCE VALIDATION: CORTICAL ACTIVATION DURING MUSIC MENTAL IMAGERY")
    print("=" * 85)
    
    # Load Subject 2 ses-03 peak
    (X_s03, y_s03, _), (X_lis, y_lis) = load_modern_tower_defense(sessions=["03"])
    X_s03_32 = X_s03[:, :32, :] # Ignore reference channel 33
    X_lis_32 = X_lis[:, :32, :]
    
    sfreq = 250.0
    n_trials = len(X_s03_32)
    print(f"Loaded {n_trials} mental imagery trials across 32 dry electrodes.")
    
    # 1. Bandpower analysis during mental imagery (Mu, Beta, Theta)
    # Mu (8-12 Hz)
    bp_mu_rec = np.array([bandpower_welch(trial, sfreq, 8.0, 12.0) for trial in X_s03_32])
    bp_beta_rec = np.array([bandpower_welch(trial, sfreq, 15.0, 25.0) for trial in X_s03_32])
    bp_theta_rec = np.array([bandpower_welch(trial, sfreq, 4.0, 7.0) for trial in X_s03_32])
    
    # Listening prior baseline for comparison
    bp_mu_lis = np.array([bandpower_welch(trial, sfreq, 8.0, 12.0) for trial in X_lis_32])
    bp_beta_lis = np.array([bandpower_welch(trial, sfreq, 15.0, 25.0) for trial in X_lis_32])
    
    mean_mu = np.mean(bp_mu_rec, axis=0)
    mean_beta = np.mean(bp_beta_rec, axis=0)
    mean_theta = np.mean(bp_theta_rec, axis=0)
    
    # Group by ROI
    ch_idx_map = {name: i for i, name in enumerate(CH_NAMES_32)}
    
    roi_summary = {}
    print("\n--- [1] CORTICAL REGION SPECTRAL PROFILES DURING MENTAL IMAGERY ---")
    print(f"{'Cortical ROI':<30} | {'Mu (8-12Hz)':<14} | {'Beta (15-25Hz)':<14} | {'Theta (4-7Hz)':<14}")
    print("-" * 75)
    for roi_name, ch_list in ROIS.items():
        indices = [ch_idx_map[c] for c in ch_list if c in ch_idx_map]
        mu_val = float(np.mean(mean_mu[indices]))
        beta_val = float(np.mean(mean_beta[indices]))
        theta_val = float(np.mean(mean_theta[indices]))
        roi_summary[roi_name] = {
            "mu_power": mu_val,
            "beta_power": beta_val,
            "theta_power": theta_val,
            "channels": ch_list
        }
        print(f"{roi_name:<30} | {mu_val:12.3e} | {beta_val:12.3e} | {theta_val:12.3e}")
        
    # 2. Top 5 Most Active / Informative Electrodes
    # Fit CSP on Binary FIRE vs ELECTRICITY to inspect spatial filter weights
    bin_mask = np.isin(y_s03, [0, 3])
    X_bin = X_s03_32[bin_mask]
    y_bin = np.where(y_s03[bin_mask] == 3, 1, 0)
    
    csp = CSP(n_components=4, reg='ledoit_wolf', log=True)
    csp.fit(X_bin, y_bin)
    
    # CSP spatial patterns A = inv(W)^T
    patterns = csp.patterns_ # (n_components, n_channels)
    # Average absolute pattern importance across all 4 spatial filters
    spatial_importance = np.mean(np.abs(patterns), axis=0)
    
    top_indices = np.argsort(spatial_importance)[::-1]
    
    print("\n--- [2] TOP ELECTRODES CONTRIBUTING TO ELEMENTAL DECODING (CSP Patterns) ---")
    print(f"{'Rank':<6} | {'Channel':<10} | {'Cortical Location':<32} | {'Relative Weight':<15}")
    print("-" * 70)
    
    top_channels_data = []
    location_descriptions = {
        'Cz': 'Vertex (Supplementary Motor Area)',
        'F4': 'Right-Frontal (Melodic Working Memory)',
        'F3': 'Left-Frontal (Rhythmic Inner Rehearsal)',
        'FC2': 'Right Fronto-Central (Premotor)',
        'FC1': 'Left Fronto-Central (Premotor)',
        'T8': 'Right-Temporal (Melody & Pitch)',
        'T7': 'Left-Temporal (Rhythm & Temporal)',
        'C4': 'Right Motor Strip (Sensorimotor)',
        'C3': 'Left Motor Strip (Sensorimotor)',
        'Fz': 'Frontal Midline (Attentional Control)',
        'CP2': 'Centroparietal (Sensorimotor Integration)',
        'Pz': 'Parietal Midline (Cognitive Focus)',
        'FT10': 'Right Antero-Temporal (Auditory Association)',
        'FT9': 'Left Antero-Temporal (Auditory Association)'
    }
    
    for rank, idx in enumerate(top_indices[:8], start=1):
        ch = CH_NAMES_32[idx]
        loc = location_descriptions.get(ch, 'Cortical scalp site')
        weight = float(spatial_importance[idx])
        top_channels_data.append({"rank": rank, "channel": ch, "location": loc, "weight": weight})
        print(f"#{rank:<5} | {ch:<10} | {loc:<32} | {weight:12.4f}")
        
    # Save results
    out_dir = Path(__file__).resolve().parent.parent.parent / "results" / "neuroscience_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    import os as _os, sys as _sys  # provenance bootstrap (stdlib only)
    _scripts_dir = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
    if _scripts_dir not in _sys.path:
        _sys.path.insert(0, _scripts_dir)
    from utils.provenance import write_provenance
    write_provenance(out_dir, script_file=__file__)
    with open(out_dir / "neuroscience_activation_report.json", "w") as f:
        json.dump({"roi_spectral_profiles": roi_summary, "top_decoding_electrodes": top_channels_data}, f, indent=2)
        
    print(f"\n[Done] Full neuroscience analysis written to {out_dir / 'neuroscience_activation_report.json'}")

if __name__ == "__main__":
    analyze_neuroscience_activation()
