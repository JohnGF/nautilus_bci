"""
Real-Time EEG Spatial Filtering & Robust Referencing Module
============================================================

Provides modular, high-performance spatial filtering and bad-channel protection:
  1. Robust CAR (Median / Trimmed CAR) - Immune to railed/outlier dry pins
  2. Standard CAR (Common Average Reference)
  3. Surface Laplacian (Hjorth Nearest-Neighbor Derivative)
  4. Dynamic Bad-Channel Masking & Detection (Flatline, Railing, Disconnected)
"""

import numpy as np
import scipy.signal as signal

# 10-20 Nearest Neighbor Mapping for 32-channel g.Nautilus layout
NEIGHBOR_MAP_1020 = {
    'C3': ['FC1', 'FC5', 'CP1', 'CP5', 'F3', 'P3', 'T7', 'Cz'],
    'C4': ['FC2', 'FC6', 'CP2', 'CP6', 'F4', 'P4', 'T8', 'Cz'],
    'Cz': ['FC1', 'FC2', 'CP1', 'CP2', 'Fz', 'Pz', 'C3', 'C4'],
    'F3': ['Fp1', 'FC1', 'FC5', 'F7', 'Fz', 'C3'],
    'F4': ['Fp2', 'FC2', 'FC6', 'F8', 'Fz', 'C4'],
    'Fz': ['Fp1', 'Fp2', 'FC1', 'FC2', 'Cz', 'F3', 'F4'],
    'P3': ['CP1', 'CP5', 'O1', 'P7', 'Pz', 'C3'],
    'P4': ['CP2', 'CP6', 'O2', 'P8', 'Pz', 'C4'],
    'Pz': ['CP1', 'CP2', 'Oz', 'Cz', 'P3', 'P4'],
    'O1': ['P3', 'P7', 'Oz', 'TP9'],
    'O2': ['P4', 'P8', 'Oz', 'TP10'],
    'Oz': ['O1', 'O2', 'Pz'],
    'FC1': ['Fz', 'Cz', 'F3', 'C3', 'FC2'],
    'FC2': ['Fz', 'Cz', 'F4', 'C4', 'FC1'],
    'CP1': ['Cz', 'Pz', 'C3', 'P3', 'CP2'],
    'CP2': ['Cz', 'Pz', 'C4', 'P4', 'CP1'],
    'FC5': ['F7', 'T7', 'F3', 'C3'],
    'FC6': ['F8', 'T8', 'F4', 'C4'],
    'CP5': ['T7', 'P7', 'C3', 'P3'],
    'CP6': ['T8', 'P8', 'C4', 'P4'],
}


def detect_bad_channels(data_block, ch_names=None, cap_type="wet"):
    """
    Identifies railed, flatlined, or extreme noise channels over a data window.
    
    Parameters:
        data_block: np.ndarray of shape (n_samples, n_channels)
        ch_names: list of channel names (optional)
        cap_type: 'wet' or 'dry'
        
    Returns:
        bad_indices: list of integer indices for bad channels
        status_dict: dict mapping index/name to status string
    """
    n_samples, n_channels = data_block.shape
    bad_indices = []
    status_dict = {}

    rail_mean = 600.0 if cap_type == "dry" else 300.0
    rail_ptp = 1000.0 if cap_type == "dry" else 500.0
    noise_std = 120.0 if cap_type == "dry" else 80.0

    stds = np.std(data_block, axis=0)
    ptps = np.ptp(data_block, axis=0)
    means = np.abs(np.mean(data_block, axis=0))

    for i in range(n_channels):
        name = ch_names[i] if ch_names and i < len(ch_names) else f"Ch_{i}"
        if name.upper() in ['BATTERY', 'STATUS', 'AUX']:
            continue
            
        std_val = stds[i]
        ptp_val = ptps[i]
        mean_val = means[i]

        if std_val < 0.5 or ptp_val < 0.5:
            bad_indices.append(i)
            status_dict[i] = "FLATLINE"
        elif mean_val > rail_mean or ptp_val > rail_ptp:
            bad_indices.append(i)
            status_dict[i] = "RAILED"
        elif std_val > noise_std:
            bad_indices.append(i)
            status_dict[i] = "HIGH_NOISE"
        else:
            status_dict[i] = "GOOD"

    return bad_indices, status_dict


def apply_spatial_filter(data_block, ch_names, mode="robust_car", cap_type="wet"):
    """
    Applies spatial referencing to multi-channel EEG data.
    
    Modes:
      - 'none' / 'raw': Detrended raw data against physical hardware reference.
      - 'robust_car': Subtracts median of valid channels (robust against railed pins).
      - 'car': Standard Common Average Reference.
      - 'laplacian': Surface Laplacian (Hjorth local neighborhood derivative).
    """
    if mode in ['none', 'raw', None]:
        return data_block - np.mean(data_block, axis=0, keepdims=True)

    n_samples, n_channels = data_block.shape
    clean_block = data_block - np.mean(data_block, axis=0, keepdims=True)
    
    # Exclude battery/aux from spatial reference calculation
    eeg_indices = [i for i, ch in enumerate(ch_names) if ch.upper() not in ['BATTERY', 'STATUS', 'AUX']]
    bad_indices, _ = detect_bad_channels(data_block[:, eeg_indices], [ch_names[i] for i in eeg_indices], cap_type=cap_type)
    good_indices = [eeg_indices[i] for i in range(len(eeg_indices)) if i not in bad_indices]

    if not good_indices:
        good_indices = eeg_indices  # Fallback if all flagged

    if mode == 'robust_car':
        # Median across good channels is resilient to single-channel outliers/railing
        ref_signal = np.median(clean_block[:, good_indices], axis=1, keepdims=True)
        out = clean_block.copy()
        out[:, eeg_indices] = clean_block[:, eeg_indices] - ref_signal
        return out

    elif mode == 'car':
        ref_signal = np.mean(clean_block[:, good_indices], axis=1, keepdims=True)
        out = clean_block.copy()
        out[:, eeg_indices] = clean_block[:, eeg_indices] - ref_signal
        return out

    elif mode == 'laplacian':
        out = clean_block.copy()
        ch_map = {name: i for i, name in enumerate(ch_names)}
        for ch, neighbors in NEIGHBOR_MAP_1020.items():
            if ch in ch_map:
                ch_idx = ch_map[ch]
                valid_neighbor_indices = [ch_map[n] for n in neighbors if n in ch_map and ch_map[n] in good_indices]
                if valid_neighbor_indices:
                    neighbor_mean = np.mean(clean_block[:, valid_neighbor_indices], axis=1)
                    out[:, ch_idx] = clean_block[:, ch_idx] - neighbor_mean
        return out

    return clean_block
