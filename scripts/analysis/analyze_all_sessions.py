"""
BCI Multi-Session EEG Pooling & Decoding Studio
================================================
Recursively scans the BIDS root directory, loads EEG files across all sessions,
extracts and aligns music memory recall epochs, and trains a cross-session
machine learning classifier (CSP + LDA).
"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import mne
import mne_bids
from mne_bids import BIDSPath, read_raw_bids
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score


def pool_and_analyze_all(bids_root="bids_musica", subject_id="01", out_dir="analysis_results"):
    print("=" * 80)
    print(" BCI Multi-Session EEG Pooling & Cross-Validation Studio ".center(80, "="))
    print("=" * 80)

    bids_root = os.path.abspath(bids_root)
    if not os.path.exists(bids_root):
        raise FileNotFoundError(f"BIDS root directory not found: {bids_root}")

    sub_clean = subject_id.replace("sub-", "")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Discover all EEG recording sessions for the subject
    search_path = os.path.join(bids_root, f"sub-{sub_clean}", "ses-*", "eeg", "*_eeg.vhdr")
    vhdr_files = glob.glob(search_path)

    if not vhdr_files:
        print(f"[-] No EEG datasets found for sub-{sub_clean} at: {search_path}")
        return

    print(f"[+] Found {len(vhdr_files)} BIDS session files to aggregate.")

    all_epochs = []
    
    # Common frequency parameters
    l_freq, h_freq = 4.0, 45.0
    sfreq_target = 250.0

    for idx, filepath in enumerate(vhdr_files):
        filename = os.path.basename(filepath)
        print(f"\n[*] Processing Session {idx+1}/{len(vhdr_files)}: {filename}")
        
        # Parse subject, session, task from filename to build standard BIDSPath
        try:
            parts = filename.split('_')
            sub_val = parts[0].replace("sub-", "")
            ses_val = parts[1].replace("ses-", "")
            task_val = parts[2].replace("task-", "")

            bids_path = BIDSPath(
                subject=sub_val,
                session=ses_val,
                task=task_val,
                datatype="eeg",
                root=bids_root
            )

            # Load raw data
            raw = read_raw_bids(bids_path=bids_path, verbose=False)
            raw.load_data()

            # Clean channel list (use only EEG)
            if 'Battery' in raw.ch_names:
                raw.set_channel_types({'Battery': 'misc'})
            
            raw_eeg = raw.copy().pick('eeg')

            # Standardize channel names across sessions to ensure consistency
            standard_32 = [
                'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 
                'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 
                'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10'
            ]
            mapping = {}
            for i, name in enumerate(raw_eeg.ch_names):
                if i < len(standard_32):
                    mapping[name] = standard_32[i]
            raw_eeg.rename_channels(mapping)

            # Resample if sampling rates mismatch (standardize to 250Hz)
            if raw_eeg.info['sfreq'] != sfreq_target:
                print(f"    Resampling from {raw_eeg.info['sfreq']} Hz to {sfreq_target} Hz...")
                raw_eeg.resample(sfreq_target, verbose=False)

            # Bandpass Filter & 50Hz Notch Filter
            raw_filtered = raw_eeg.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
            raw_filtered.notch_filter(freqs=50.0, verbose=False)

            # Extract events
            events, event_id = mne.events_from_annotations(raw_filtered, verbose=False)

            # Filter for Task_Recall events
            target_event_id = {}
            for k, v in event_id.items():
                if 'Task_Recall' in k:
                    # Simplify event names (e.g. strip trial info if present)
                    clean_k = k.split('_dur_')[0] if '_dur_' in k else k
                    target_event_id[clean_k] = v

            if not target_event_id:
                print(f"    [-] Skipping: No 'Task_Recall' events found in session.")
                continue

            # Epoch session data (0.0s to 4.0s)
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
            
            # Standardize channels names across sessions to ensure consistency
            epochs.rename_channels(lambda name: name.strip())
            all_epochs.append(epochs)
            print(f"    [+] Successfully loaded {len(epochs)} trials.")

        except Exception as e:
            print(f"    [-] Error loading session {filename}: {e}")

    if not all_epochs:
        print("\n[-] Error: Could not extract any valid epochs across sessions.")
        return

    # 2. Concatenate all session trials
    print("\n" + "-" * 80)
    print("[*] Concatenating and aligning all session epochs...")
    
    # Ensure all epochs have the same channels
    common_ch = list(set.intersection(*(set(ep.ch_names) for ep in all_epochs)))
    print(f"[+] Aligning to {len(common_ch)} common EEG channels.")
    
    for i in range(len(all_epochs)):
        all_epochs[i].pick(common_ch)

    combined_epochs = mne.concatenate_epochs(all_epochs, verbose=False)
    print(f"[+] Total Aggregated Trials: {len(combined_epochs)}")

    # Group classes: mapping to clean labels
    # We want to group by track category name (e.g. Task_Recall_Track_1, Task_Recall_Track_4)
    # Get the data and target vector
    X = combined_epochs.get_data()
    y_raw = combined_epochs.events[:, -1]
    
    # Re-map events values to sequential 0-indexed targets for classification
    unique_y = np.unique(y_raw)
    y = np.array([list(unique_y).index(val) for val in y_raw])

    # Show class distribution
    print("\n[+] Grouped Trial Class Distribution:")
    event_id_inv = {v: k for k, v in combined_epochs.event_id.items()}
    for val in unique_y:
        name = event_id_inv.get(val, f"Code_{val}")
        count = np.sum(y_raw == val)
        print(f"   - {name:<30}: {count} trials")

    # 3. Train Machine Learning Model (CSP + LDA)
    acc_score = 0.0
    if len(y) >= 4 and len(np.unique(y)) > 1:
        n_components = min(4, X.shape[1])
        csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
        lda = LinearDiscriminantAnalysis()

        min_class_count = min(np.bincount(y))
        n_splits = min(5, min_class_count) if min_class_count >= 2 else 2
        
        if len(y) >= n_splits * 2:
            try:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                X_csp = csp.fit_transform(X, y)
                scores = cross_val_score(lda, X_csp, y, cv=cv)
                acc_score = np.mean(scores) * 100.0
            except Exception as e:
                print(f"[-] CSP Machine Learning error: {e}")

    # 4. Generate & Save Overall PSD plots
    print("\n[*] Plotting multi-session spectral distributions...")
    try:
        psd_obj = combined_epochs.compute_psd(fmin=1.0, fmax=45.0, verbose=False)
        psds, freqs = psd_obj.get_data(return_freqs=True)
        psds_mean = np.mean(psds, axis=0) # Mean across epochs
        
        # Scale
        scale = 1e12 if np.mean(psds_mean) < 1e-3 else 1.0
        psds_mean *= scale

        fig, ax = plt.subplots(figsize=(10, 5))
        mean_curve = np.mean(psds_mean, axis=0)
        ax.plot(freqs, mean_curve, color='#9b59b6', linewidth=2.5, label='Aggregated Mean EEG')
        ax.fill_between(freqs, mean_curve, color='#9b59b6', alpha=0.2)
        ax.set_title('Aggregated Power Spectral Density (All Sessions)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (uV²/Hz)')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plot_path = os.path.join(out_dir, "aggregated_multi_session_psd.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] Saved aggregated PSD graph: {plot_path}")
    except Exception as e:
        print(f"[-] Failed to generate plots: {e}")

    print("\n" + "=" * 80)
    print(" AGGREGATED BCI EXPERIMENT DECODING SUMMARY ".center(80, "="))
    print("=" * 80)
    print(f" BIDS Root Folder             : {bids_root}")
    print(f" Subject                      : sub-{sub_clean}")
    print(f" Total EEG Sessions Pooled    : {len(vhdr_files)}")
    print(f" Common Channels              : {len(common_ch)} channels")
    print(f" Total Trial Epochs           : {len(combined_epochs)} trials")
    print(f" Distinct Track Classes       : {len(unique_y)} classes")
    print("-" * 80)
    print(f" BCI Machine Learning Acc     : {acc_score:.2f}% (CSP + LDA 5-Fold Cross-Val)")
    print(f" Output Results Plot          : {os.path.abspath(out_dir)}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Pool and decode all BIDS sessions.")
    parser.add_argument("--bids-root", type=str, default="bids_musica", help="BIDS root directory")
    parser.add_argument("--sub", type=str, default="01", help="Subject ID")
    parser.add_argument("--outdir", type=str, default="analysis_results", help="Output directory")
    args = parser.parse_args()

    pool_and_analyze_all(bids_root=args.bids_root, subject_id=args.sub, out_dir=args.outdir)
