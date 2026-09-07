"""
Lateralized Cortical Activation Mapping for Left Arrow HitZone in FNF
=====================================================================
Evaluates Contralateral (Right Hemisphere) vs Ipsilateral (Left Hemisphere)
activation during Left Arrow Lock-On:
  - Motor Imagery / Mind (ses-01, ses-02, ses-03)
  - Motor Execution / Movement (ses-04)

Contralateral Motor Cortex Electrodes: C4, FC2, FC6, CP2, CP6 (Right side)
Ipsilateral Motor Cortex Electrodes:   C3, FC1, FC5, CP1, CP5 (Left side)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

import mne
from mne_bids import BIDSPath, read_raw_bids

STANDARD_32 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
    'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 
    'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 
    'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10'
]


def load_clean_epochs(bids_root, ses, task, l_freq=8.0, h_freq=30.0):
    bp = BIDSPath(subject='01', session=ses, task=task, datatype='eeg', root=bids_root)
    raw = read_raw_bids(bp, verbose=False)
    raw.load_data()
    
    mapping = {raw.ch_names[i]: STANDARD_32[i] for i in range(min(32, len(raw.ch_names)))}
    if len(raw.ch_names) > 32:
        raw.set_channel_types({raw.ch_names[32]: 'misc'})
    raw.rename_channels(mapping)
    raw.pick('eeg')
    raw.set_montage('standard_1020', match_case=False)
    
    # Filter sensorimotor Mu/Beta band (8-30 Hz) and notch
    raw.filter(l_freq, h_freq, verbose=False).notch_filter(50.0, verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    hz_code = [v for k, v in event_id.items() if 'Arrow_Left_HitZone' in k][0]
    hz_events = events[events[:, 2] == hz_code]
    
    epochs = mne.Epochs(
        raw,
        hz_events,
        tmin=-0.4,
        tmax=0.8,
        baseline=None,
        preload=True,
        verbose=False
    )
    return epochs


def generate_lateralization_maps(bids_root="scripts/bids/bids_fnf", out_dir="scripts/analysis/analysis_results_fnf"):
    os.makedirs(out_dir, exist_ok=True)
    bids_root = os.path.abspath(bids_root)
    
    print("[*] Loading epochs for Mind (ses-01..03) and Movement (ses-04)...")
    epochs_s1 = load_clean_epochs(bids_root, '01', 'leftright')
    epochs_s2 = load_clean_epochs(bids_root, '02', 'leftright')
    epochs_s3 = load_clean_epochs(bids_root, '03', 'leftright')
    epochs_s4 = load_clean_epochs(bids_root, '04', 'me')
    
    epochs_mind_pooled = mne.concatenate_epochs([epochs_s1, epochs_s2, epochs_s3])
    
    times = epochs_s3.times
    b_mask = (times >= -0.4) & (times <= -0.1)
    t_mask = (times >= 0.05) & (times <= 0.45)
    
    # Function to compute ERD% per channel
    def compute_erd_percent(epochs_obj):
        data = epochs_obj.get_data() # (trials, ch, times)
        # Compute instantaneous power via Hilbert transform envelope squared
        analytic = signal.hilbert(data, axis=-1)
        inst_power = np.abs(analytic)**2
        
        # Mean baseline power per channel
        p_base = np.mean(inst_power[:, :, b_mask], axis=(0, 2))
        p_task = np.mean(inst_power[:, :, t_mask], axis=(0, 2))
        
        erd_pct = ((p_task - p_base) / p_base) * 100.0
        
        # Time course of ERD%
        # baseline average over trials and baseline time
        base_trial_mean = np.mean(inst_power[:, :, b_mask], axis=-1, keepdims=True) # (trials, ch, 1)
        erd_timecourse = np.mean((inst_power - base_trial_mean) / (base_trial_mean + 1e-12) * 100.0, axis=0) # (ch, times)
        return erd_pct, erd_timecourse
    
    erd_pct_mind, erd_tc_mind = compute_erd_percent(epochs_mind_pooled)
    erd_pct_s3, erd_tc_s3 = compute_erd_percent(epochs_s3)
    erd_pct_move, erd_tc_move = compute_erd_percent(epochs_s4)
    
    ch_names = epochs_s3.ch_names
    c3_idx = ch_names.index('C3')
    c4_idx = ch_names.index('C4')
    cz_idx = ch_names.index('Cz')
    
    print("\n" + "="*70)
    print(" HEMISPHERIC ACTIVATION SUMMARY (LEFT ARROW HIT) ".center(70, "="))
    print("="*70)
    print(f"Movement (ses-04):  C4 (Right Hem / Contralateral): {erd_pct_move[c4_idx]:+6.2f}% ERD")
    print(f"                    C3 (Left Hem  / Ipsilateral):   {erd_pct_move[c3_idx]:+6.2f}% ERD")
    print(f"                    Difference (C4 - C3):           {erd_pct_move[c4_idx] - erd_pct_move[c3_idx]:+6.2f}% (Negative = Right Hem Stronger Activation)")
    print("-" * 70)
    print(f"Mind ses-03:        C4 (Right Hem / Contralateral): {erd_pct_s3[c4_idx]:+6.2f}% ERD")
    print(f"                    C3 (Left Hem  / Ipsilateral):   {erd_pct_s3[c3_idx]:+6.2f}% ERD")
    print(f"Mind Pooled (1..3): C4 (Right Hem / Contralateral): {erd_pct_mind[c4_idx]:+6.2f}% ERD")
    print(f"                    C3 (Left Hem  / Ipsilateral):   {erd_pct_mind[c3_idx]:+6.2f}% ERD")
    print("="*70)
    
    # -------------------------------------------------------------
    # Plot Figure: Scalp Topomaps & Hemispheric Activation Curves
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(18, 11), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.3, wspace=0.25)
    
    # Row 1: 3 Scalp Topomaps
    ax_topo1 = fig.add_subplot(gs[0, 0])
    ax_topo2 = fig.add_subplot(gs[0, 1])
    ax_topo3 = fig.add_subplot(gs[0, 2])
    
    vlim = (-40, 20)
    
    im1, _ = mne.viz.plot_topomap(
        erd_pct_s3, epochs_s3.info, axes=ax_topo1, show=False,
        cmap='RdBu_r', vlim=vlim, contours=6
    )
    ax_topo1.set_title("Mind Imagery (ses-03)\nLeft Arrow HitZone", fontsize=12, fontweight='bold', pad=10)
    
    im2, _ = mne.viz.plot_topomap(
        erd_pct_mind, epochs_s3.info, axes=ax_topo2, show=False,
        cmap='RdBu_r', vlim=vlim, contours=6
    )
    ax_topo2.set_title("Mind Imagery (Pooled ses-01..03)\nLeft Arrow HitZone", fontsize=12, fontweight='bold', pad=10)
    
    im3, _ = mne.viz.plot_topomap(
        erd_pct_move, epochs_s3.info, axes=ax_topo3, show=False,
        cmap='RdBu_r', vlim=vlim, contours=6
    )
    ax_topo3.set_title("Physical Movement (ses-04)\nLeft Arrow HitZone (Ground Truth)", fontsize=12, fontweight='bold', pad=10)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.55, 0.015, 0.35])
    cbar = fig.colorbar(im3, cax=cbar_ax)
    cbar.set_label("Event-Related Desynchronization (ERD %)\n[Blue = Cortical Activation / Power Suppression]", fontsize=10, fontweight='bold')
    
    # Row 2: Time Courses of Right (C4) vs Left (C3) Sensorimotor Activation
    ax_tc1 = fig.add_subplot(gs[1, 0])
    ax_tc2 = fig.add_subplot(gs[1, 1])
    ax_tc3 = fig.add_subplot(gs[1, 2])
    
    smooth_win = 5
    def smooth(arr):
        return np.convolve(arr, np.ones(smooth_win)/smooth_win, mode='same')
    
    # Plot ses-03 Time Course
    ax_tc1.plot(times * 1000, smooth(erd_tc_s3[c4_idx]), label='C4 (Right Hem / Contralateral)', color='#d9534f', linewidth=2.4)
    ax_tc1.plot(times * 1000, smooth(erd_tc_s3[c3_idx]), label='C3 (Left Hem / Ipsilateral)', color='#2b5c8f', linewidth=2.0, linestyle='--')
    ax_tc1.plot(times * 1000, smooth(erd_tc_s3[cz_idx]), label='Cz (Vertex)', color='#2ecc71', linewidth=1.5, alpha=0.7)
    ax_tc1.axvline(0, color='black', linestyle=':', linewidth=1.5, label='Arrow HitZone')
    ax_tc1.axvspan(50, 450, color='gray', alpha=0.12, label='Activation Window')
    ax_tc1.set_title("Mind (ses-03): C4 vs C3 Time Course", fontsize=11, fontweight='bold')
    ax_tc1.set_xlabel("Time from HitZone (ms)", fontsize=10)
    ax_tc1.set_ylabel("Mu/Beta ERD (%)", fontsize=10)
    ax_tc1.set_ylim(-60, 40)
    ax_tc1.grid(True, alpha=0.3)
    ax_tc1.legend(fontsize=8, loc='lower left')
    
    # Plot Pooled Mind Time Course
    ax_tc2.plot(times * 1000, smooth(erd_tc_mind[c4_idx]), label='C4 (Right Hem / Contralateral)', color='#d9534f', linewidth=2.4)
    ax_tc2.plot(times * 1000, smooth(erd_tc_mind[c3_idx]), label='C3 (Left Hem / Ipsilateral)', color='#2b5c8f', linewidth=2.0, linestyle='--')
    ax_tc2.plot(times * 1000, smooth(erd_tc_mind[cz_idx]), label='Cz (Vertex)', color='#2ecc71', linewidth=1.5, alpha=0.7)
    ax_tc2.axvline(0, color='black', linestyle=':', linewidth=1.5, label='Arrow HitZone')
    ax_tc2.axvspan(50, 450, color='gray', alpha=0.12)
    ax_tc2.set_title("Pooled Mind: C4 vs C3 Time Course", fontsize=11, fontweight='bold')
    ax_tc2.set_xlabel("Time from HitZone (ms)", fontsize=10)
    ax_tc2.set_ylabel("Mu/Beta ERD (%)", fontsize=10)
    ax_tc2.set_ylim(-60, 40)
    ax_tc2.grid(True, alpha=0.3)
    ax_tc2.legend(fontsize=8, loc='lower left')
    
    # Plot Movement (ses-04) Time Course
    ax_tc3.plot(times * 1000, smooth(erd_tc_move[c4_idx]), label='C4 (Right Hem / Contralateral)', color='#d9534f', linewidth=2.4)
    ax_tc3.plot(times * 1000, smooth(erd_tc_move[c3_idx]), label='C3 (Left Hem / Ipsilateral)', color='#2b5c8f', linewidth=2.0, linestyle='--')
    ax_tc3.plot(times * 1000, smooth(erd_tc_move[cz_idx]), label='Cz (Vertex)', color='#2ecc71', linewidth=1.5, alpha=0.7)
    ax_tc3.axvline(0, color='black', linestyle=':', linewidth=1.5, label='Arrow HitZone')
    ax_tc3.axvspan(50, 450, color='gray', alpha=0.12)
    ax_tc3.set_title("Movement (ses-04): C4 vs C3 Time Course", fontsize=11, fontweight='bold')
    ax_tc3.set_xlabel("Time from HitZone (ms)", fontsize=10)
    ax_tc3.set_ylabel("Mu/Beta ERD (%)", fontsize=10)
    ax_tc3.set_ylim(-60, 40)
    ax_tc3.grid(True, alpha=0.3)
    ax_tc3.legend(fontsize=8, loc='lower left')
    
    fig_path = os.path.join(out_dir, "fnf_left_arrow_right_hemisphere_lateralization_map.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"\n[+] Saved Brain Lateralization Map to: {fig_path}")
    
    # Copy to brain folder
    brain_dir = r"C:\Users\VR\.gemini\antigravity-cli\brain\b742f45d-09a7-45af-876d-8da46da6226a"
    dest_path = os.path.join(brain_dir, "fnf_left_arrow_right_hemisphere_lateralization_map.png")
    import shutil
    shutil.copyfile(fig_path, dest_path)
    print(f"[+] Copied to brain directory: {dest_path}")
    return fig_path


if __name__ == "__main__":
    generate_lateralization_maps()
