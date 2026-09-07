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

def analyze_dataset(bids_root="bids_dataset", subject_id="01", session_id="01", task_name="leftright", out_dir="analysis_results"):
    print("=" * 70)
    print(" BCI Motor Imagery & Music Condition Analysis ".center(70, "="))
    print("=" * 70)

    bids_root = os.path.abspath(bids_root)
    if not os.path.exists(bids_root):
        # Fallback to test_bids_dataset if bids_dataset is missing
        test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_bids_dataset"))
        if os.path.exists(test_dir):
            print(f"[!] BIDS root '{bids_root}' not found. Falling back to '{test_dir}'")
            bids_root = test_dir
        else:
            raise FileNotFoundError(f"BIDS root directory not found: {bids_root}")

    # Clean ID formatting
    sub_clean = subject_id.replace("sub-", "")
    ses_clean = session_id.replace("ses-", "")

    bids_path = BIDSPath(
        subject=sub_clean,
        session=ses_clean,
        task=task_name,
        datatype="eeg",
        root=bids_root
    )

    print(f"[*] Reading BIDS dataset from: {bids_path.directory}")
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()

    # Explicitly set Battery channel to misc if present
    if 'Battery' in raw.ch_names:
        raw.set_channel_types({'Battery': 'misc'})

    print(f"[+] Loaded raw dataset: {raw.info['sfreq']} Hz | {len(raw.ch_names)} channels | {raw.times[-1]:.1f} s duration")

    # Filter data for Motor Imagery (Mu 8-12 Hz & Beta 13-30 Hz)
    print("[*] Applying Bandpass Filter (8 - 30 Hz) & 50 Hz Notch filter...")
    raw_filtered = raw.copy().filter(l_freq=8.0, h_freq=30.0, verbose=False)
    raw_filtered.notch_filter(freqs=50.0, verbose=False)

    # Extract Events from Annotations
    events, event_id = mne.events_from_annotations(raw_filtered, verbose=False)

    # Identify Task Event IDs dynamically (Top, Bottom, Left, Right)
    selected_event_id = {}
    for target in ['Top', 'Bottom', 'Left', 'Right']:
        matching_keys = [k for k in event_id.keys() if f"Task_{target}" in k or k == target]
        if matching_keys:
            selected_event_id[target] = event_id[matching_keys[0]]

    if len(selected_event_id) < 2:
        # Fallback search
        for target in ['Top', 'Bottom', 'Left', 'Right']:
            matching_keys = [k for k in event_id.keys() if target in k]
            if matching_keys and target not in selected_event_id:
                selected_event_id[target] = event_id[matching_keys[0]]

    if not selected_event_id:
        raise ValueError(f"Could not find task events in dataset annotations. Found event IDs: {list(event_id.keys())}")

    # Epoching: 0s to 4s task execution window
    tmin, tmax = 0.0, 4.0
    epochs = mne.Epochs(
        raw_filtered,
        events,
        event_id=selected_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    print(f"[+] Created Epochs: {len(epochs)} trials total ({len(epochs['Left'])} Left, {len(epochs['Right'])} Right)")

    # Prepare Output Results Folder
    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------------------------------
    # Identify Music vs No Music Blocks from Annotations
    # ----------------------------------------------------
    nomusic_intervals = []
    music_intervals = []

    # Parse raw annotations for block boundaries
    for annot in raw.annotations:
        desc = annot['description']
        onset = annot['onset']
        if 'NoMusic' in desc:
            nomusic_intervals.append((onset, onset + 120.0))  # default block window
        elif 'Music' in desc and 'NoMusic' not in desc:
            music_intervals.append((onset, onset + 120.0))

    # Partition Epochs into No Music vs Music
    nomusic_mask = []
    music_mask = []

    for ep_time in epochs.events[:, 0] / raw.info['sfreq']:
        in_music = any(start <= ep_time <= end for start, end in music_intervals)
        music_mask.append(in_music)
        nomusic_mask.append(not in_music)

    nomusic_mask = np.array(nomusic_mask)
    music_mask = np.array(music_mask)

    epochs_nomusic = epochs[nomusic_mask] if np.any(nomusic_mask) else epochs
    epochs_music = epochs[music_mask] if np.any(music_mask) else None

    # ----------------------------------------------------
    # Analysis 1: C3 vs C4 Power Lateralization (ERD/ERS)
    # ----------------------------------------------------
    eeg_ch_names = raw.copy().pick('eeg').ch_names
    c3_ch = 'C3' if 'C3' in eeg_ch_names else eeg_ch_names[0]
    c4_ch = 'C4' if 'C4' in eeg_ch_names else eeg_ch_names[min(1, len(eeg_ch_names)-1)]
    c3_idx = raw_filtered.ch_names.index(c3_ch)
    c4_idx = raw_filtered.ch_names.index(c4_ch)

    # Calculate overall PSD
    psd_left = epochs['Left'].compute_psd(fmin=8, fmax=30, verbose=False).get_data()
    psd_right = epochs['Right'].compute_psd(fmin=8, fmax=30, verbose=False).get_data()

    scale_factor = 1e12 if np.mean(psd_left) < 1e-3 else 1.0

    power_left_c3 = np.mean(psd_left[:, c3_idx, :]) * scale_factor
    power_left_c4 = np.mean(psd_left[:, c4_idx, :]) * scale_factor
    power_right_c3 = np.mean(psd_right[:, c3_idx, :]) * scale_factor
    power_right_c4 = np.mean(psd_right[:, c4_idx, :]) * scale_factor

    # Plot 1: Bar Chart of C3 vs C4 Power
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.35

    ax.bar(x - width/2, [power_left_c3, power_right_c3], width, label=f'Left Motor Cortex ({c3_ch})', color='#4DEEEA')
    ax.bar(x + width/2, [power_left_c4, power_right_c4], width, label=f'Right Motor Cortex ({c4_ch})', color='#E040FB')

    ax.set_ylabel('Mean Mu/Beta Power (uV²/Hz)')
    ax.set_title('Motor Imagery ERD/ERS Power Lateralization (C3 vs C4)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Think LEFT', 'Think RIGHT'])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plot1_path = os.path.join(out_dir, "erd_c3_c4_lateralization.png")
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close()

    # ----------------------------------------------------
    # Analysis 2: Music vs No Music Condition Comparison
    # ----------------------------------------------------
    p_nomusic_left = np.mean(epochs_nomusic['Left'].compute_psd(fmin=8, fmax=30, verbose=False).get_data()) * scale_factor if len(epochs_nomusic['Left']) > 0 else 0
    p_nomusic_right = np.mean(epochs_nomusic['Right'].compute_psd(fmin=8, fmax=30, verbose=False).get_data()) * scale_factor if len(epochs_nomusic['Right']) > 0 else 0

    if epochs_music is not None and len(epochs_music) > 0:
        p_music_left = np.mean(epochs_music['Left'].compute_psd(fmin=8, fmax=30, verbose=False).get_data()) * scale_factor if len(epochs_music['Left']) > 0 else 0
        p_music_right = np.mean(epochs_music['Right'].compute_psd(fmin=8, fmax=30, verbose=False).get_data()) * scale_factor if len(epochs_music['Right']) > 0 else 0
    else:
        p_music_left, p_music_right = 0, 0

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.35

    ax.bar(x - width/2, [p_nomusic_left, p_nomusic_right], width, label='No Music (Silent)', color='#74B9FF')
    ax.bar(x + width/2, [p_music_left, p_music_right], width, label='Music Condition', color='#FF7675')

    ax.set_ylabel('Mean Brain Power (uV²/Hz)')
    ax.set_title('Condition Comparison: No Music vs Music')
    ax.set_xticks(x)
    ax.set_xticklabels(['Think LEFT', 'Think RIGHT'])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plot2_path = os.path.join(out_dir, "music_vs_nomusic_comparison.png")
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved Music vs No Music condition plot to: {plot2_path}")

    # ----------------------------------------------------
    # Analysis 3: BCI Machine Learning Classification (CSP + LDA)
    # ----------------------------------------------------
    epochs_eeg = epochs.copy().pick('eeg')
    X = epochs_eeg.get_data()
    y = epochs_eeg.events[:, -1]

    acc_score = 0.0
    if len(y) >= 4 and len(np.unique(y)) > 1:
        from sklearn.pipeline import Pipeline
        n_components = min(4, X.shape[1])
        csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
        lda = LinearDiscriminantAnalysis()
        clf_pipeline = Pipeline([('csp', csp), ('lda', lda)])

        n_splits = min(5, min(np.bincount(y - np.min(y))))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            # Strict leakage-free Cross Validation (CSP fit ONLY on training fold per iteration)
            scores = cross_val_score(clf_pipeline, X, y, cv=cv)
            acc_score = np.mean(scores) * 100.0

            try:
                csp.fit(X, y)
                fig = csp.plot_patterns(epochs_eeg.info, ch_type='eeg', show=False)
                plot3_path = os.path.join(out_dir, "csp_patterns.png")
                fig.savefig(plot3_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
            except Exception:
                pass

    print("\n" + "=" * 70)
    print(" MUSIC vs. NO MUSIC CONDITION COMPARISON REPORT ".center(70, "="))
    print("=" * 70)
    print(f"Dataset Path              : {bids_path.directory}")
    print(f"Total Trials              : {len(epochs)} ({len(epochs_nomusic)} No-Music, {len(epochs_music) if epochs_music else 0} Music)")
    print("-" * 70)
    print(f"No Music (Silent) Power   : Left Task={p_nomusic_left:.2f} uV^2/Hz | Right Task={p_nomusic_right:.2f} uV^2/Hz")
    print(f"Music Condition Power     : Left Task={p_music_left:.2f} uV^2/Hz | Right Task={p_music_right:.2f} uV^2/Hz")
    print("-" * 70)
    print(f"BCI Classifier Accuracy   : {acc_score:.1f}% (CSP + LDA)")
    print(f"Saved Comparison Graph    : {plot2_path}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Analyze Left/Right BCI Dataset with Music vs No Music comparison.")
    parser.add_argument("--bids-root", type=str, default="bids_dataset", help="Path to BIDS dataset root")
    parser.add_argument("--sub", type=str, default="01", help="Subject ID (e.g., 01)")
    parser.add_argument("--ses", type=str, default="01", help="Session ID (e.g., 01)")
    parser.add_argument("--outdir", type=str, default="analysis_results", help="Directory to save figures")
    args = parser.parse_args()

    analyze_dataset(bids_root=args.bids_root, subject_id=args.sub, session_id=args.ses, out_dir=args.outdir)

if __name__ == "__main__":
    main()
