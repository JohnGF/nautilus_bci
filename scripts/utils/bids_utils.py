"""
BIDS Dataset Utility Functions
==============================

Helper utilities for scanning BIDS dataset directory structures and determining
automatic session IDs based on existing sessions.
"""

import os
import re


def find_next_available_session(bids_root: str, subject_id: str) -> int:
    """
    Finds the lowest missing positive integer session ID starting from 1 for a given BIDS root and subject ID.

    Examples:
      - If existing sessions are [1, 3, 4], returns 2.
      - If existing sessions are [1, 2, 3], returns 4.
      - If existing sessions are [2, 3], returns 1.
      - If no sessions exist or folder missing, returns 1.
    """
    if not bids_root or not os.path.exists(bids_root):
        return 1

    clean_sub = str(subject_id).strip()
    if clean_sub.lower().startswith("sub-"):
        clean_sub = clean_sub[4:]

    if not clean_sub:
        return 1

    possible_sub_dirs = [
        os.path.join(bids_root, f"sub-{clean_sub}"),
    ]
    if clean_sub.isdigit():
        possible_sub_dirs.append(os.path.join(bids_root, f"sub-{int(clean_sub):02d}"))
        possible_sub_dirs.append(os.path.join(bids_root, f"sub-{int(clean_sub)}"))

    existing_sessions = set()
    target_sub_dirs = []

    for d in possible_sub_dirs:
        if os.path.exists(d) and os.path.isdir(d) and d not in target_sub_dirs:
            target_sub_dirs.append(d)

    # Scan bids_root for sub-XX folders matching numeric or clean_sub
    try:
        for entry in os.listdir(bids_root):
            entry_path = os.path.join(bids_root, entry)
            if os.path.isdir(entry_path) and entry.lower().startswith("sub-"):
                s_id = entry[4:]
                if s_id == clean_sub or (clean_sub.isdigit() and s_id.isdigit() and int(s_id) == int(clean_sub)):
                    if entry_path not in target_sub_dirs:
                        target_sub_dirs.append(entry_path)
    except Exception:
        pass

    # If no subject folders exist yet, check files directly in bids_root
    if not target_sub_dirs:
        try:
            for entry in os.listdir(bids_root):
                if clean_sub in entry:
                    matches = re.findall(r'ses-(\d+)', entry, re.IGNORECASE)
                    for m in matches:
                        existing_sessions.add(int(m))
        except Exception:
            pass

    # Scan subject directories for ses-XX folders or filenames containing ses-XX
    for sub_path in target_sub_dirs:
        try:
            for root_dir, dirs, files in os.walk(sub_path):
                for item in dirs + files:
                    matches = re.findall(r'ses-(\d+)', item, re.IGNORECASE)
                    for m in matches:
                        existing_sessions.add(int(m))
        except Exception:
            pass

    # Find the lowest missing positive integer starting from 1
    next_ses = 1
    while next_ses in existing_sessions:
        next_ses += 1

    return next_ses


def get_formatted_next_session(bids_root: str, subject_id: str, current_ses_text: str = "01") -> str:
    """
    Returns the next session formatted as a string, preserving any prefix (e.g. 'ses-') from current_ses_text.
    """
    next_num = find_next_available_session(bids_root, subject_id)
    has_prefix = current_ses_text.strip().lower().startswith("ses-")
    formatted_num = f"{next_num:02d}"
    if has_prefix:
        return f"ses-{formatted_num}"
    return formatted_num
