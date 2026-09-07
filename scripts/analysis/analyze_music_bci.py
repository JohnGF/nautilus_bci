import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import mne
import mne_bids
from mne_bids import BIDSPath, read_raw_bids
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score


def analyze_music_bci(bids_root="bids_dataset", subject_id="01", session_id="04", task_name="leftright", out_dir="analysis_results"):
    print("=" * 75)
    print(" BCI Music Memory & Auditory Imagery EEG Analysis Studio ".center(75, "="))
    print("=" * 75)

    bids_root = os.path.abspath(bids_root)
    if not os.path.exists(bids_root):
        raise FileNotFoundError(f"BIDS root directory not found: {bids_root}")

    sub_clean = subject_id.replace("sub-", "")
    ses_clean = session_id.replace("ses-", "")

    bids_path = BIDSPath(
        subject=sub_clean,
        session=ses_clean,
        task=task_name,
        datatype="eeg",
        root=bids_root
    )

    print(f"[*] Reading Music BIDS Dataset from: {bids_path.directory}")
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()

    if 'Battery' in raw.ch_names:
        raw.set_channel_types({'Battery': 'misc'})

    sfreq = raw.info['sfreq']
    n_chan = len(raw.copy().pick('eeg').ch_names)
    duration_s = raw.times[-1]

    print(f"[+] Dataset Summary: {sfreq} Hz sampling rate | {n_chan} EEG channels | {duration_s:.1f}s total duration")

    os.makedirs(out_dir, exist_ok=True)

    # 1. Bandpass Filtering (4.0 Hz - 45.0 Hz) + Notch Filter (50 Hz)
    print("[*] Applying EEG Bandpass Filter (4.0 Hz - 45.0 Hz) & 50 Hz Notch filter...")
    raw_filtered = raw.copy().filter(l_freq=4.0, h_freq=45.0, verbose=False)
    raw_filtered.notch_filter(freqs=50.0, verbose=False)

    # 2. Compute Power Spectral Density (PSD) across Frequency Bands
    print("[*] Computing Spectral Power Densities across Brain Rhythms...")
    eeg_raw = raw_filtered.copy().pick('eeg')
    
    psd_obj = eeg_raw.compute_psd(fmin=1.0, fmax=45.0, verbose=False)
    psds, freqs = psd_obj.get_data(return_freqs=True)
    
    # Scale to uV^2/Hz
    scale = 1e12 if np.mean(psds) < 1e-3 else 1.0
    psds *= scale

    # Define Brain Rhythm Frequency Bands
    bands = {
        'Delta (1-4 Hz)': (1.0, 4.0),
        'Theta (4-8 Hz)': (4.0, 8.0),
        'Alpha (8-12 Hz)': (8.0, 12.0),
        'Beta (13-30 Hz)': (13.0, 30.0),
        'Gamma (30-45 Hz)': (30.0, 45.0)
    }

    band_powers = {}
    for b_name, (fmin, fmax) in bands.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_powers[b_name] = np.mean(psds[:, idx])

    # Plot 1: Power Spectral Density Curve
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_psd_ch = np.mean(psds, axis=0)
    ax.plot(freqs, mean_psd_ch, color='#4DEEEA', linewidth=2.5, label='Mean EEG Spectrum')
    ax.fill_between(freqs, mean_psd_ch, color='#4DEEEA', alpha=0.2)

    # Highlight Alpha Band
    alpha_idx = np.logical_and(freqs >= 8.0, freqs <= 12.0)
    ax.fill_between(freqs[alpha_idx], mean_psd_ch[alpha_idx], color='#00E676', alpha=0.5, label='Alpha Sync (8-12 Hz)')

    ax.set_title(f'EEG Power Spectral Density (Subject sub-{sub_clean}, Session ses-{ses_clean})', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power Density (uV²/Hz)', fontsize=12)
    ax.set_xlim([1.0, 45.0])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)

    plot1_path = os.path.join(out_dir, "music_bci_psd_spectrum.png")
    fig.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved PSD Spectrum graph: {plot1_path}")

    # Plot 2: Brain Rhythm Power Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    b_names = list(band_powers.keys())
    b_vals = list(band_powers.values())
    colors = ['#74B9FF', '#A0A5B5', '#00E676', '#E040FB', '#FF7675']

    bars = ax.bar(b_names, b_vals, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Mean Power (uV²/Hz)', fontsize=12)
    ax.set_title('Brain Rhythm Frequency Band Power Distribution during Music Task', fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, b_vals):
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 0.05 * max(b_vals), f"{val:.2f}", ha='center', va='bottom', fontweight='bold')

    plot2_path = os.path.join(out_dir, "music_bci_band_power.png")
    fig.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved Band Power chart: {plot2_path}")

    # 3. Machine Learning Classification Analysis (CSP + LDA)
    events, event_id = mne.events_from_annotations(raw_filtered, verbose=False)
    
    # Filter event_id to only include the target 'Task_Recall' event markers (ignoring setup and start/stop markers)
    target_event_id = {}
    for k, v in event_id.items():
        if 'Task_Recall' in k:
            target_event_id[k] = v
            
    if not target_event_id:
        print("[!] Warning: No 'Task_Recall' events found in annotations. Falling back to all event IDs.")
        target_event_id = event_id

    # Epoching trial data (0s to 4s)
    tmin, tmax = 0.0, 4.0
    epochs = mne.Epochs(
        raw_filtered,
        events,
        event_id=target_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        event_repeated='drop',
        verbose=False
    )

    X = epochs.get_data()
    y = epochs.events[:, -1]

    acc_score = 0.0
    if len(y) >= 4 and len(np.unique(y)) > 1:
        n_components = min(4, X.shape[1])
        csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
        lda = LinearDiscriminantAnalysis()

        min_class_count = min(np.bincount(y - np.min(y)))
        n_splits = min(5, min_class_count) if min_class_count >= 2 else 2
        
        if len(y) >= n_splits * 2:
            try:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                X_csp = csp.fit_transform(X, y)
                scores = cross_val_score(lda, X_csp, y, cv=cv)
                acc_score = np.mean(scores) * 100.0
            except Exception as e:
                print(f"[-] CSP classification note: {e}")

    print("\n" + "=" * 75)
    print(" MUSIC BCI EEG ANALYSIS EXECUTIVE SUMMARY ".center(75, "="))
    print("=" * 75)
    print(f" Dataset Path               : {bids_path.directory}")
    print(f" Subject / Session          : sub-{sub_clean} | ses-{ses_clean}")
    print(f" Sampling Rate              : {sfreq} Hz")
    print(f" EEG Channels               : {n_chan} channels")
    print(f" Total Recording Duration   : {duration_s:.1f} seconds")
    print(f" Total Trial Epochs         : {len(epochs)} trials")
    print("-" * 75)
    print(" Brain Rhythm Power Densities:")
    for b_name, b_val in band_powers.items():
        print(f"   [+] {b_name:<24}: {b_val:.3f} uV^2/Hz")
    print("-" * 75)
    print(f" BCI Classifier Accuracy     : {acc_score:.1f}% (CSP + LDA Machine Learning)")
    print(f" Generated PSD Spectrum      : [music_bci_psd_spectrum.png](file:///{os.path.abspath(plot1_path)})")
    print(f" Generated Band Power Chart  : [music_bci_band_power.png](file:///{os.path.abspath(plot2_path)})")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze Music BCI Dataset.")
    parser.add_argument("--bids-root", type=str, default="bids_dataset", help="Path to BIDS dataset root")
    parser.add_argument("--sub", type=str, default="01", help="Subject ID (e.g., 01)")
    parser.add_argument("--ses", type=str, default="04", help="Session ID (e.g., 04)")
    parser.add_argument("--task", type=str, default="leftright", help="Task name")
    parser.add_argument("--outdir", type=str, default="analysis_results", help="Directory to save figures")
    args = parser.parse_args()

    analyze_music_bci(
        bids_root=args.bids_root, 
        subject_id=args.sub, 
        session_id=args.ses, 
        task_name=args.task, 
        out_dir=args.outdir
    )


if __name__ == "__main__":
    main()
