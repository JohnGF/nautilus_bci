"""
Quick Multimodal Data Analysis for BIDS Baseline Dataset
=========================================================

Analyzes EEG, Smartwatch PPG, and Smartwatch Motion data from `bids_baseline`.
Reports:
  1. Dataset & Session Structure
  2. EEG Band Powers (Alpha, Beta, Delta) & Video Condition Spectral Power
  3. Smartwatch PPG Heart Rate & PPG Signal Quality
  4. Smartwatch IMU Motion Vector Magnitude (Movement Artifacts & Rest vs Trial Motion)
"""

import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne_bids import BIDSPath, read_raw_bids


def analyze_baseline(bids_root="bids_baseline", sub="01", ses="02", task="video", out_dir="analysis_results"):
    bids_root = os.path.abspath(bids_root)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 75)
    print(f" Multimodal BIDS Baseline Quick Analysis (sub-{sub}, ses-{ses}, task-{task}) ".center(75, "="))
    print("=" * 75)

    report = {}

    # ---------------------------------------------------------
    # 1. EEG Analysis
    # ---------------------------------------------------------
    bids_path = BIDSPath(subject=sub, session=ses, task=task, datatype="eeg", root=bids_root)
    print(f"\n[*] Loading EEG Raw Data from: {bids_path.directory}")

    try:
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        raw.load_data()
        
        if 'Battery' in raw.ch_names:
            raw.set_channel_types({'Battery': 'misc'})

        sfreq = raw.info['sfreq']
        duration = raw.times[-1]
        n_channels = len(raw.ch_names)
        
        print(f"[+] Loaded EEG: {n_channels} channels | {sfreq} Hz sampling | {duration:.1f} s duration")

        # Bandpass filter 1 - 40 Hz
        raw_filt = raw.copy().filter(l_freq=1.0, h_freq=40.0, verbose=False)
        
        # Calculate PSD & Band Powers
        psd = raw_filt.compute_psd(fmin=1, fmax=40, verbose=False)
        psd_data = psd.get_data()  # shape: (n_channels, n_freqs)
        freqs = psd.freqs

        delta_mask = (freqs >= 1) & (freqs <= 4)
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        beta_mask = (freqs >= 13) & (freqs <= 30)

        p_delta = np.mean(psd_data[:, delta_mask])
        p_alpha = np.mean(psd_data[:, alpha_mask])
        p_beta = np.mean(psd_data[:, beta_mask])

        # Parse Events
        events, event_id = mne.events_from_annotations(raw_filt, verbose=False)
        
        # Categorize events by video condition
        conditions = ['water', 'earth', 'wind', 'fire']
        cond_event_id = {}
        for c in conditions:
            matches = [k for k in event_id.keys() if f"Video_Start_{c}" in k or c in k.lower()]
            if matches:
                cond_event_id[c] = event_id[matches[0]]

        epochs_by_cond = {}
        if cond_event_id:
            epochs = mne.Epochs(raw_filt, events, event_id=cond_event_id, tmin=0.0, tmax=3.0, baseline=(0, 0.5), preload=True, verbose=False)
            for c in cond_event_id.keys():
                if c in epochs.event_id:
                    epochs_by_cond[c] = epochs[c]

        report['eeg'] = {
            'sfreq': sfreq,
            'duration_sec': duration,
            'n_channels': n_channels,
            'power_delta': float(p_delta),
            'power_alpha': float(p_alpha),
            'power_beta': float(p_beta),
            'n_trials': len(epochs) if cond_event_id else 0,
            'conditions_found': list(cond_event_id.keys())
        }

    except Exception as e:
        print(f"[-] EEG analysis warning: {e}")
        report['eeg'] = {'error': str(e)}

    # ---------------------------------------------------------
    # 2. Smartwatch Motion (IMU) Analysis
    # ---------------------------------------------------------
    ses_dir = os.path.join(bids_root, f"sub-{sub}", f"ses-{ses}")
    motion_file = os.path.join(ses_dir, "motion", f"sub-{sub}_ses-{ses}_task-{task}_motion.tsv")

    print(f"\n[*] Checking Motion IMU file: {motion_file}")
    if os.path.exists(motion_file):
        df_motion = pd.read_csv(motion_file, sep='\t')
        print(f"[+] Loaded Motion Data: {len(df_motion)} samples | Columns: {list(df_motion.columns)}")

        # Calculate accelerometer magnitude if 6 channels present (Accel XYZ + Gyro XYZ)
        # Columns: timestamp_sec, ch_1, ch_2, ch_3, ch_4, ch_5, ch_6
        accel_cols = [c for c in df_motion.columns if c != 'timestamp_sec'][:3]
        if len(accel_cols) >= 3:
            accel_data = df_motion[accel_cols].values
            accel_mag = np.sqrt(np.sum(accel_data**2, axis=1))
            mean_motion_mag = np.mean(accel_mag)
            std_motion_mag = np.std(accel_mag)
            max_motion_mag = np.max(accel_mag)
        else:
            mean_motion_mag, std_motion_mag, max_motion_mag = 0, 0, 0

        report['motion'] = {
            'n_samples': len(df_motion),
            'duration_sec': float(df_motion['timestamp_sec'].iloc[-1] - df_motion['timestamp_sec'].iloc[0]) if 'timestamp_sec' in df_motion else 0,
            'mean_accel_magnitude': float(mean_motion_mag),
            'std_accel_magnitude': float(std_motion_mag),
            'max_accel_magnitude': float(max_motion_mag)
        }
    else:
        print("[-] Motion file not found.")
        report['motion'] = {'status': 'not_found'}

    # ---------------------------------------------------------
    # 3. Smartwatch Physio (PPG) Analysis
    # ---------------------------------------------------------
    physio_file = os.path.join(ses_dir, "physio", f"sub-{sub}_ses-{ses}_task-{task}_physio.tsv")
    if not os.path.exists(physio_file):
        # Check fallback ppg/ folder
        physio_file = os.path.join(ses_dir, "ppg", f"sub-{sub}_ses-{ses}_task-{task}_ppg.tsv")

    print(f"[*] Checking Physio/PPG file: {physio_file}")
    if os.path.exists(physio_file):
        df_ppg = pd.read_csv(physio_file, sep='\t')
        print(f"[+] Loaded PPG Data: {len(df_ppg)} samples | Columns: {list(df_ppg.columns)}")

        ppg_vals = df_ppg.iloc[:, 1].values  # First data column after timestamp
        mean_ppg = np.mean(ppg_vals)
        std_ppg = np.std(ppg_vals)
        min_ppg = np.min(ppg_vals)
        max_ppg = np.max(ppg_vals)

        report['physio_ppg'] = {
            'n_samples': len(df_ppg),
            'mean_value': float(mean_ppg),
            'std_value': float(std_ppg),
            'min_value': float(min_ppg),
            'max_value': float(max_ppg)
        }
    else:
        print("[-] Physio/PPG file not found.")
        report['physio_ppg'] = {'status': 'not_found'}

    # ---------------------------------------------------------
    # Generate Plots & Save Artifact
    # ---------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    fig.suptitle(f"BIDS Baseline Dataset Quick Analysis Summary (sub-{sub}_ses-{ses}_task-{task})", fontsize=14, fontweight='bold', color='#2C3E50')

    # Plot 1: EEG Spectrum
    if 'eeg' in report and 'power_alpha' in report['eeg']:
        axes[0].plot(freqs, np.mean(psd_data, axis=0), color='#2980B9', lw=2, label='Mean EEG PSD')
        axes[0].axvspan(8, 12, color='#2ECC71', alpha=0.3, label='Alpha (8-12Hz)')
        axes[0].axvspan(13, 30, color='#F39C12', alpha=0.3, label='Beta (13-30Hz)')
        axes[0].set_xlim(1, 40)
        axes[0].set_ylabel("Power Density")
        axes[0].set_title("EEG Spectrum & Rhythms", fontsize=11, fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

    # Plot 2: Motion Accel Magnitude
    if 'motion' in report and 'n_samples' in report['motion']:
        t_m = df_motion['timestamp_sec'].values
        acc_m = np.sqrt(np.sum(df_motion.iloc[:, 1:4].values**2, axis=1))
        axes[1].plot(t_m, acc_m, color='#E74C3C', lw=1.2, label='Smartwatch Accel Mag')
        axes[1].set_ylabel("Accel Mag (g)")
        axes[1].set_title("Smartwatch 6-DOF IMU Motion Vector", fontsize=11, fontweight='bold')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

    # Plot 3: PPG Signal / Heart Rate
    if 'physio_ppg' in report and 'n_samples' in report['physio_ppg']:
        t_p = df_ppg['timestamp_sec'].values
        p_val = df_ppg.iloc[:, 1].values
        axes[2].plot(t_p, p_val, color='#8E44AD', lw=1.5, label='PPG Pulse / Heart Rate')
        axes[2].set_xlabel("Time (seconds)")
        axes[2].set_ylabel("PPG Amplitude / BPM")
        axes[2].set_title("Smartwatch PPG Photoplethysmography Track", fontsize=11, fontweight='bold')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"baseline_sub-{sub}_ses-{ses}_analysis.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n[+] Analysis summary plot saved to: {plot_path}")

    # Save JSON Report
    json_path = os.path.join(out_dir, f"baseline_sub-{sub}_ses-{ses}_summary.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"[+] Analysis JSON summary saved to: {json_path}")
    print("=" * 75)

    return report, plot_path, json_path


if __name__ == '__main__':
    analyze_baseline()
