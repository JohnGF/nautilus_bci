#!/usr/bin/env python3
"""
BIDS Dataset Consolidation Engine
=================================
Consolidates all fragmented BIDS recordings in scripts/bids/ into clean,
canonical, standard-compliant BIDS datasets per game/paradigm:
  1. scripts/bids_clean/bids_tower_defense
  2. scripts/bids_clean/bids_fnf
  3. scripts/bids_clean/bids_music
  4. scripts/bids_clean/bids_baseline

This script operates non-destructively: original files remain untouched.
"""

import os
import shutil
import json
import glob
import re
import mne
import mne_bids

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_BIDS_DIR = os.path.join(BASE_DIR, "bids")
CLEAN_BIDS_DIR = os.path.join(BASE_DIR, "bids_clean")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def copy_or_link_file(src, dst):
    """Copies or hardlinks/symlinks files safely."""
    ensure_dir(os.path.dirname(dst))
    if os.path.exists(dst):
        os.remove(dst)
    # Use symlink or copy
    try:
        os.symlink(os.path.abspath(src), dst)
    except Exception:
        shutil.copy2(src, dst)

def copy_and_fix_vhdr(src_vhdr, dst_vhdr, new_prefix):
    """
    Copies BrainVision .vhdr file and updates internal DataFile= and MarkerFile=
    pointers to match new_prefix.
    """
    ensure_dir(os.path.dirname(dst_vhdr))
    with open(src_vhdr, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    fixed_lines = []
    for line in lines:
        if line.strip().startswith('DataFile='):
            fixed_lines.append(f"DataFile={new_prefix}.eeg\n")
        elif line.strip().startswith('MarkerFile='):
            fixed_lines.append(f"MarkerFile={new_prefix}.vmrk\n")
        else:
            fixed_lines.append(line)

    with open(dst_vhdr, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)

def copy_and_fix_vmrk(src_vmrk, dst_vmrk, new_prefix):
    """
    Copies BrainVision .vmrk file and updates internal DataFile= pointer.
    """
    ensure_dir(os.path.dirname(dst_vmrk))
    with open(src_vmrk, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    fixed_lines = []
    for line in lines:
        if line.strip().startswith('DataFile='):
            fixed_lines.append(f"DataFile={new_prefix}.eeg\n")
        else:
            fixed_lines.append(line)

    with open(dst_vmrk, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)

def import_session(src_ses_dir, dst_root, target_sub, target_ses, target_task=None):
    """
    Imports a session directory into dst_root/sub-<target_sub>/ses-<target_ses>/.
    Handles eeg, motion, physio, ppg modalities and updates BrainVision references.
    """
    dst_sub_dir = os.path.join(dst_root, f"sub-{target_sub}")
    dst_ses_dir = os.path.join(dst_sub_dir, f"ses-{target_ses}")
    ensure_dir(dst_ses_dir)

    # Scans file
    src_scans = glob.glob(os.path.join(src_ses_dir, "*_scans.tsv"))
    if src_scans:
        dst_scans = os.path.join(dst_ses_dir, f"sub-{target_sub}_ses-{target_ses}_scans.tsv")
        copy_or_link_file(src_scans[0], dst_scans)

    # Process all subdirectories (eeg, motion, physio, ppg)
    for mod in ['eeg', 'motion', 'physio', 'ppg']:
        src_mod_dir = os.path.join(src_ses_dir, mod)
        if not os.path.exists(src_mod_dir):
            continue

        dst_mod_dir = os.path.join(dst_ses_dir, mod)
        ensure_dir(dst_mod_dir)

        # Look for files
        files = os.listdir(src_mod_dir)
        for fname in files:
            src_file = os.path.join(src_mod_dir, fname)

            # Determine task from filename or override
            # Example filename: sub-01_ses-01_task-recall_eeg.vhdr
            match = re.search(r"task-([a-zA-Z0-9]+)", fname)
            curr_task = target_task if target_task else (match.group(1) if match else "task")

            # Determine suffix and extension
            # Remove original sub-XX_ses-YY_task-ZZZ_
            rest = re.sub(r"^sub-[^_]+_ses-[^_]+_(task-[^_]+_)?", "", fname)
            dst_fname = f"sub-{target_sub}_ses-{target_ses}_task-{curr_task}_{rest}" if not rest.startswith(f"sub-{target_sub}") else fname

            # Special case for coordinates/electrodes
            if "space-" in fname:
                dst_fname = re.sub(r"^sub-[^_]+_ses-[^_]+_", f"sub-{target_sub}_ses-{target_ses}_", fname)

            dst_file = os.path.join(dst_mod_dir, dst_fname)
            prefix = f"sub-{target_sub}_ses-{target_ses}_task-{curr_task}_{mod}"

            if fname.endswith(".vhdr"):
                copy_and_fix_vhdr(src_file, dst_file, prefix)
            elif fname.endswith(".vmrk"):
                copy_and_fix_vmrk(src_file, dst_file, prefix)
            else:
                copy_or_link_file(src_file, dst_file)

def create_dataset_description(dst_root, name):
    desc = {
        "Name": name,
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "License": "CC0",
        "Authors": ["Nautilus BCI Research Team"]
    }
    with open(os.path.join(dst_root, "dataset_description.json"), 'w') as f:
        json.dump(desc, f, indent=4)

def create_participants_tsv(dst_root, subjects):
    rows = ["participant_id\tage\tsex\thand\tweight\theight"]
    for s in sorted(subjects):
        rows.append(f"sub-{s}\tn/a\tn/a\tn/a\tn/a\tn/a")
    with open(os.path.join(dst_root, "participants.tsv"), 'w') as f:
        f.write("\n".join(rows) + "\n")

# ==============================================================================
# 1. Tower Defense Consolidation
# ==============================================================================
def consolidate_tower_defense():
    print("\n--- Consolidating Tower Defense BIDS ---")
    dst = os.path.join(CLEAN_BIDS_DIR, "bids_tower_defense")
    ensure_dir(dst)

    # Sub-01 sessions
    td_sub1 = [
        # ses-01: 6_3_27 primary recall
        (os.path.join(SRC_BIDS_DIR, "bids_tower_defense/bids_tower_defense_6_3_27/sub-01/ses-01"), "01", "01", "recall"),
        # ses-02: recallWaterReplaced
        (os.path.join(SRC_BIDS_DIR, "bids_tower_defense/bids_tower_defense/sub-01/ses-02"), "01", "02", "recallWaterReplaced"),
        # ses-03: recallSongsIIWT
        (os.path.join(SRC_BIDS_DIR, "bids_tower_defense/bids_tower_defense/sub-01/ses-03"), "01", "03", "recallSongsIIWT"),
        # ses-04: recall original
        (os.path.join(SRC_BIDS_DIR, "bids_tower_defense/bids_tower_defense/sub-01/ses-01"), "01", "04", "recall"),
        # ses-05: old prototype recall
        (os.path.join(SRC_BIDS_DIR, "bids_tower_defense/bids_tower_defense(old)/sub-01/ses-01"), "01", "05", "recallOld"),
        # ses-06: old prototype memory
        (os.path.join(SRC_BIDS_DIR, "bids_tower_defense/bids_tower_defense(old)/sub-01/ses-04"), "01", "06", "memory")
    ]

    for src_dir, sub, ses, task in td_sub1:
        if os.path.exists(src_dir):
            import_session(src_dir, dst, sub, ses, task)
            print(f"[+] TD: Imported {src_dir} -> sub-{sub}/ses-{ses} ({task})")

    # Sub-02 sessions (ses-01..05)
    for i in range(1, 6):
        ses_str = f"0{i}"
        src_dir = os.path.join(SRC_BIDS_DIR, f"bids_tower_defense/bids_tower_defense/sub-02/ses-{ses_str}")
        if os.path.exists(src_dir):
            import_session(src_dir, dst, "02", ses_str, "recallSongsIIWT")
            print(f"[+] TD: Imported {src_dir} -> sub-02/ses-{ses_str} (recallSongsIIWT)")

    create_dataset_description(dst, "Tower Defense BCI Game Dataset")
    create_participants_tsv(dst, ["01", "02"])

# ==============================================================================
# 2. Friday Night Funkin' Consolidation
# ==============================================================================
def consolidate_fnf():
    print("\n--- Consolidating Friday Night Funkin' (FNF) BIDS ---")
    dst = os.path.join(CLEAN_BIDS_DIR, "bids_fnf")
    ensure_dir(dst)

    # Sub-01 sessions (ses-01..07)
    for i in range(1, 8):
        ses_str = f"0{i}"
        src_dir = os.path.join(SRC_BIDS_DIR, f"bids_fnf/sub-01/ses-{ses_str}")
        if os.path.exists(src_dir):
            import_session(src_dir, dst, "01", ses_str)
            print(f"[+] FNF: Imported {src_dir} -> sub-01/ses-{ses_str}")

    # Sub-03 sessions (ses-01..03)
    for i in range(1, 4):
        ses_str = f"0{i}"
        src_dir = os.path.join(SRC_BIDS_DIR, f"bids_fnf/sub-03/ses-{ses_str}")
        if os.path.exists(src_dir):
            import_session(src_dir, dst, "03", ses_str)
            print(f"[+] FNF: Imported {src_dir} -> sub-03/ses-{ses_str}")

    create_dataset_description(dst, "Friday Night Funkin BCI Rhythm Game Dataset")
    create_participants_tsv(dst, ["01", "03"])

# ==============================================================================
# 3. Music BCI & Auditory Priming Consolidation
# ==============================================================================
def consolidate_music():
    print("\n--- Consolidating Music BCI & Auditory Priming BIDS ---")
    dst = os.path.join(CLEAN_BIDS_DIR, "bids_music")
    ensure_dir(dst)

    # Sub-01 active recall sessions (from bids_musica ses-01..07)
    for i in range(1, 8):
        ses_str = f"0{i}"
        src_dir = os.path.join(SRC_BIDS_DIR, f"bids_musica/sub-01/ses-{ses_str}")
        if os.path.exists(src_dir):
            import_session(src_dir, dst, "01", ses_str)
            print(f"[+] Music: Imported {src_dir} -> sub-01/ses-{ses_str} (Active Recall)")

    # Sub-01 continuous passive listening session (from bids_listening/sub-01/ses-02 -> ses-08)
    src_listen_sub1 = os.path.join(SRC_BIDS_DIR, "bids_listening/sub-01/ses-02")
    if os.path.exists(src_listen_sub1):
        import_session(src_listen_sub1, dst, "01", "08", "musiclistening")
        print(f"[+] Music: Imported {src_listen_sub1} -> sub-01/ses-08 (Passive Listening Classical)")

    # Sub-02 4 modern song listening sessions (from bids_listening/sub-02/ses-01..04)
    tasks_sub2 = {
        "01": "listenItsRainingMen",
        "02": "listenWhatsUp",
        "03": "listenThunderstruck",
        "04": "listenKiss"
    }
    for ses_str, task_name in tasks_sub2.items():
        src_dir = os.path.join(SRC_BIDS_DIR, f"bids_listening/sub-02/ses-{ses_str}")
        if os.path.exists(src_dir):
            import_session(src_dir, dst, "02", ses_str, task_name)
            print(f"[+] Music: Imported {src_dir} -> sub-02/ses-{ses_str} ({task_name})")

    create_dataset_description(dst, "Music BCI & Auditory Priming Dataset")
    create_participants_tsv(dst, ["01", "02"])

# ==============================================================================
# 4. Baseline & Calibration Consolidation
# ==============================================================================
def consolidate_baseline():
    print("\n--- Consolidating Baseline & Calibration BIDS ---")
    dst = os.path.join(CLEAN_BIDS_DIR, "bids_baseline")
    ensure_dir(dst)

    # Sub-01 Video / Audio-Visual Baseline (bids_baseline ses-01, ses-02)
    for i in range(1, 3):
        ses_str = f"0{i}"
        src_dir = os.path.join(SRC_BIDS_DIR, f"bids_baseline/sub-01/ses-{ses_str}")
        if os.path.exists(src_dir):
            import_session(src_dir, dst, "01", ses_str, "video")
            print(f"[+] Baseline: Imported {src_dir} -> sub-01/ses-{ses_str} (video)")

    # Sub-01 Motor Imagery Calibration (bids_dataset ses-01..04 -> ses-03..06)
    for i in range(1, 5):
        ses_str = f"0{i}"
        tgt_ses = f"0{i+2}"
        src_dir = os.path.join(SRC_BIDS_DIR, f"bids_dataset/sub-01/ses-{ses_str}")
        if os.path.exists(src_dir):
            import_session(src_dir, dst, "01", tgt_ses, "motorimagery")
            print(f"[+] Baseline: Imported {src_dir} -> sub-01/ses-{tgt_ses} (motorimagery)")

    create_dataset_description(dst, "Baseline & Motor Imagery Calibration Dataset")
    create_participants_tsv(dst, ["01"])

# ==============================================================================
# 5. Integrity Verification via MNE-BIDS
# ==============================================================================
def verify_all_clean_bids():
    print("\n" + "=" * 80)
    print(" VERIFYING CULMINATED BIDS DATASETS WITH MNE-BIDS ".center(80, "="))
    print("=" * 80)
    all_ok = True
    for ds_name in ["bids_tower_defense", "bids_fnf", "bids_music", "bids_baseline"]:
        root = os.path.join(CLEAN_BIDS_DIR, ds_name)
        if not os.path.exists(root):
            print(f"[-] Missing {ds_name}")
            all_ok = False
            continue
        try:
            bps = mne_bids.find_matching_paths(root, datatypes='eeg', extensions='.vhdr')
            print(f"[✓] {ds_name:25s}: Found {len(bps)} valid EEG runs.")
            for bp in bps[:3]:
                raw = mne_bids.read_raw_bids(bids_path=bp, verbose=False)
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                print(f"    - Sample: sub-{bp.subject} ses-{bp.session} task-{bp.task} | channels={len(raw.ch_names)} sfreq={raw.info['sfreq']}Hz events={len(events)}")
        except Exception as e:
            print(f"[X] Error verifying {ds_name}: {e}")
            all_ok = False
    return all_ok

if __name__ == '__main__':
    ensure_dir(CLEAN_BIDS_DIR)
    consolidate_tower_defense()
    consolidate_fnf()
    consolidate_music()
    consolidate_baseline()
    verify_all_clean_bids()
