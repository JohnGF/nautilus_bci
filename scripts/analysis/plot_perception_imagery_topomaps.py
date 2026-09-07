"""
Perception vs Imagery Topographic Scalp Mapping & Spectral Entrainment Analyzer
================================================================================
Generates publication-quality 32-channel topomaps and STG spectral entrainment plots:
  1. Topographic power maps across Theta (4-8 Hz), Alpha (8-12 Hz), and Beta (13-30 Hz).
  2. Compares Pure Auditory Perception (Full Music) vs Mental Imagery (Tower Defense).
  3. Computes Superior Temporal Gyrus (T7/T8) vs Sensorimotor (C3/C4) power curves.
  4. Saves multi-panel figures to scripts/analysis_results/music_aware_tower_defense/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_script_dir, ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import mne
from analyze_music_aware_tower_defense import get_music_listening_data, get_tower_defense_session_data


def generate_perception_imagery_topomaps(
    out_dir="scripts/analysis_results/music_aware_tower_defense"
):
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 80)
    print(" GENERATING PERCEPTION VS IMAGERY TOPOGRAPHIC MAPS ".center(80, "="))
    print("=" * 80)

    # 1. Load Data
    print("[*] Loading continuous music listening data...")
    X_music, y_music, raw_music = get_music_listening_data("scripts/bids_music")

    print("[*] Loading tower defense recall data (ses-01)...")
    X_td, y_td, df_meta, raw_td = get_tower_defense_session_data("scripts/bids_tower_defense", sub="01", ses="01")

    standard_32 = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
        'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 
        'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 
        'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10'
    ]
    if raw_music.ch_names[0].startswith('EEG'):
        mapping = {name: standard_32[i] for i, name in enumerate(raw_music.ch_names) if i < len(standard_32)}
        raw_music.rename_channels(mapping)
        raw_td.rename_channels(mapping)
        
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_music.set_montage(montage, on_missing='ignore')
    raw_td.set_montage(montage, on_missing='ignore')

    # Frequency bands
    bands = {
        'Theta (4-8 Hz)': (4.0, 8.0),
        'Alpha (8-12 Hz)': (8.0, 12.0),
        'Beta (13-30 Hz)': (13.0, 30.0)
    }

    # 2. Compute Bandpower Topomap Arrays
    sfreq = raw_music.info['sfreq']
    from scipy.signal import welch

    def compute_band_powers(X, sfreq):
        freqs, psd = welch(X, fs=sfreq, nperseg=int(sfreq * 1.5), axis=-1)
        mean_psd = np.mean(psd, axis=0)  # (n_channels, n_freqs)
        bp_dict = {}
        for b_name, (f0, f1) in bands.items():
            idx_b = (freqs >= f0) & (freqs <= f1)
            bp_dict[b_name] = np.mean(mean_psd[:, idx_b], axis=-1)
        return freqs, mean_psd, bp_dict

    freqs_m, mean_psd_m, bp_music = compute_band_powers(X_music, sfreq)
    freqs_td, mean_psd_td, bp_td = compute_band_powers(X_td, sfreq)

    # 3. Create Multi-Panel Topomap Figure
    print("[*] Rendering Topographic Scalp Maps...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor='#0D1117')
    plt.rcParams['text.color'] = '#E6EDF3'
    plt.rcParams['axes.labelcolor'] = '#E6EDF3'

    band_names = list(bands.keys())

    # Row 1: Perception (Full Music Listening)
    for col_idx, b_name in enumerate(band_names):
        ax = axes[0, col_idx]
        data_p = bp_music[b_name]
        im, _ = mne.viz.plot_topomap(
            data_p, raw_music.info, axes=ax, show=False, cmap='magma',
            sphere='auto', vlim=(np.min(data_p), np.max(data_p))
        )
        ax.set_title(f"Perception (Listening)\n{b_name}", fontsize=11, fontweight='bold', color='#4DEEEA', pad=8)

    # Row 2: Imagery (Mental Recall)
    for col_idx, b_name in enumerate(band_names):
        ax = axes[1, col_idx]
        data_im = bp_td[b_name]
        im, _ = mne.viz.plot_topomap(
            data_im, raw_td.info, axes=ax, show=False, cmap='magma',
            sphere='auto', vlim=(np.min(data_im), np.max(data_im))
        )
        ax.set_title(f"Mental Recall (Imagery)\n{b_name}", fontsize=11, fontweight='bold', color='#FF7675', pad=8)

    plt.suptitle("CORTICAL TOPOGRAPHY: PERCEPTION (FULL MUSIC) VS MENTAL IMAGERY (TOWER DEFENSE)",
                 fontsize=13, fontweight='bold', color='#E6EDF3', y=0.98)
    plt.tight_layout()

    topomap_path = os.path.join(out_dir, "cortical_perception_vs_imagery_topomaps.png")
    plt.savefig(topomap_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Saved Topomap Figure: {topomap_path}")

    # 4. Spectral Entrainment Plot at Temporal (T7/T8) vs Central (C3/C4)
    print("[*] Rendering Auditory STG vs Sensorimotor Spectral Curves...")
    ch_names = raw_music.ch_names
    t_idx = [i for i, ch in enumerate(ch_names) if ch.upper() in ['T7', 'T8', 'TP9', 'TP10']]
    c_idx = [i for i, ch in enumerate(ch_names) if ch.upper() in ['C3', 'CZ', 'C4']]

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0D1117')
    for ax in (ax1, ax2):
        ax.set_facecolor('#161B22')
        ax.grid(True, linestyle=':', alpha=0.3, color='#8B949E')
        ax.tick_params(colors='#8B949E')

    # Panel 1: Temporal Auditory STG (T7/T8)
    psd_m_temp = np.mean(mean_psd_m[t_idx], axis=0) if t_idx else np.mean(mean_psd_m, axis=0)
    psd_td_temp = np.mean(mean_psd_td[t_idx], axis=0) if t_idx else np.mean(mean_psd_td, axis=0)

    f_mask = freqs_m <= 40.0
    ax1.plot(freqs_m[f_mask], psd_m_temp[f_mask], label='Perceptual Listening (Music)', color='#4DEEEA', linewidth=2.5)
    ax1.plot(freqs_td[f_mask], psd_td_temp[f_mask], label='Mental Recall (Tower Defense)', color='#FF7675', linewidth=2.5, linestyle='--')
    ax1.set_title('Auditory Temporal Channels (T7, T8, TP9, TP10)', fontsize=12, fontweight='bold', color='#E6EDF3')
    ax1.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Power Spectral Density (uV^2/Hz)', fontsize=11, fontweight='bold')
    ax1.legend(facecolor='#161B22', edgecolor='#30363D')

    # Panel 2: Sensorimotor Central (C3, Cz, C4)
    psd_m_cent = np.mean(mean_psd_m[c_idx], axis=0) if c_idx else np.mean(mean_psd_m, axis=0)
    psd_td_cent = np.mean(mean_psd_td[c_idx], axis=0) if c_idx else np.mean(mean_psd_td, axis=0)

    ax2.plot(freqs_m[f_mask], psd_m_cent[f_mask], label='Perceptual Listening (Music)', color='#4DEEEA', linewidth=2.5)
    ax2.plot(freqs_td[f_mask], psd_td_cent[f_mask], label='Mental Recall (Tower Defense)', color='#FF7675', linewidth=2.5, linestyle='--')
    ax2.set_title('Sensorimotor Central Channels (C3, Cz, C4)', fontsize=12, fontweight='bold', color='#E6EDF3')
    ax2.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Power Spectral Density (uV^2/Hz)', fontsize=11, fontweight='bold')
    ax2.legend(facecolor='#161B22', edgecolor='#30363D')

    plt.suptitle("SPECTRAL POWER ENTRAINMENT: AUDITORY STG VS SENSORIMOTOR REGIONS", fontsize=13, fontweight='bold', color='#E6EDF3', y=1.01)
    plt.tight_layout()

    spectral_path = os.path.join(out_dir, "auditory_stg_spectral_entrainment.png")
    plt.savefig(spectral_path, dpi=300, bbox_inches='tight', facecolor=fig2.get_facecolor())
    plt.close()
    print(f"[+] Saved Spectral Entrainment Figure: {spectral_path}")

    print("\n" + "=" * 80)
    print(" TOPOGRAPHIC MAPPING COMPLETE ".center(80, "="))
    print("=" * 80)


if __name__ == "__main__":
    generate_perception_imagery_topomaps()
