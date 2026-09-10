#!/usr/bin/env python3
"""
Comprehensive Analysis of Pure Music Listening (sub-02, bids_listening)
and Its Impact on 4-Class BCI Rhythm Decoding in Tower Defense (sub-02).
========================================================================
Sessions in bids_listening/sub-02:
  - ses-01: It's Raining Men -> WATER (Class 1)
  - ses-02: What's Up -> WIND (Class 2)
  - ses-03: Thunderstruck -> ELECTRICITY (Class 3)
  - ses-04: I Was Made for Lovin' You -> FIRE (Class 0)
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann

# Configuration
SFREQ = 250.0
CLASS_MAP = {'FIRE': 0, 'WATER': 1, 'WIND': 2, 'ELECTRICITY': 3}
CLASS_NAMES = ['FIRE', 'WATER', 'WIND', 'ELECTRICITY']
COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

def invsqrtm(C):
    """Computes C^(-1/2) for symmetric positive definite matrix."""
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-10)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

def preprocess_eeg(raw_32, sfreq=250.0, l_freq=1.0, h_freq=45.0, notch_freq=50.0):
    """Butterworth bandpass, 50Hz notch, and robust median CAR referencing."""
    nyq = sfreq / 2.0
    b_band, a_band = signal.butter(4, [l_freq / nyq, h_freq / nyq], btype='band')
    filt = signal.filtfilt(b_band, a_band, raw_32, axis=0)
    
    b_notch, a_notch = signal.iirnotch(notch_freq, 30.0, sfreq)
    filt = signal.filtfilt(b_notch, a_notch, filt, axis=0)
    
    # Identify active channels (exclude flatline/dead channels)
    stds = np.std(filt, axis=0)
    active_idx = np.where((stds > 0.05) & (stds < 200.0))[0]
    if len(active_idx) == 0:
        active_idx = np.arange(filt.shape[1])
    med_ref = np.median(filt[:, active_idx], axis=1, keepdims=True)
    clean = filt - med_ref
    return clean

def load_listening_sub02(bids_root="scripts/bids/bids_listening", sub_id="02", win_len_s=3.0, step_s=1.5, discard_start_s=5.0):
    """
    Loads pure listening continuous sessions for sub-02.
    Starts at discard_start_s (music starts ~5s after recording began).
    """
    session_info = [
        ("ses-01", 1, "WATER", "ItsRainingMen"),
        ("ses-02", 2, "WIND", "WhatsUp"),
        ("ses-03", 3, "ELECTRICITY", "Thunderstruck"),
        ("ses-04", 0, "FIRE", "Kiss")
    ]
    
    epochs_list = []
    labels_list = []
    meta_list = []
    
    n_win = int(win_len_s * SFREQ)
    n_step = int(step_s * SFREQ)
    n_discard = int(discard_start_s * SFREQ)
    
    for ses, cid, elem, song in session_info:
        eeg_dir = os.path.join(bids_root, f"sub-{sub_id}", ses, "eeg")
        eeg_files = glob.glob(os.path.join(eeg_dir, "*.eeg"))
        if not eeg_files:
            raise FileNotFoundError(f"Missing eeg file in {eeg_dir}")
        raw = np.fromfile(eeg_files[0], dtype=np.float32).reshape(-1, 33)[:, :32]
        mean_std = np.mean(np.std(raw, axis=0))
        raw_uv = raw / 1000.0 if mean_std > 500.0 else raw
        clean = preprocess_eeg(raw_uv, sfreq=SFREQ)
        
        s_start = n_discard
        s_end = clean.shape[0] - int(2.0 * SFREQ)
        
        cnt = 0
        for s in range(s_start, s_end - n_win, n_step):
            ep = clean[s : s + n_win, :].T  # shape (32, n_samples)
            epochs_list.append(ep)
            labels_list.append(cid)
            meta_list.append({
                'session': ses,
                'element': elem,
                'song': song,
                'class_id': cid,
                'sample_start': s
            })
            cnt += 1
        print(f"[+] Loaded bids_listening sub-{sub_id} {ses} ({elem} - {song}): {cnt} epochs (length={win_len_s}s)")
        
    X_listen = np.array(epochs_list)
    y_listen = np.array(labels_list)
    df_meta = pd.DataFrame(meta_list)
    return X_listen, y_listen, df_meta

def load_tower_defense_sub02(bids_root="scripts/bids/bids_tower_defense", sub_id="02", win_len_s=3.0):
    """
    Loads all Tower Defense game sessions for sub-02 (ses-01..ses-05).
    Extracts both Imagine and Listen epochs.
    """
    ses_dirs = sorted(glob.glob(os.path.join(bids_root, f"sub-{sub_id}", "ses-*")))
    all_im, all_lis, all_y, all_ses_group = [], [], [], []
    
    n_win = int(win_len_s * SFREQ)
    class_map = {'FIRE': 0, 'WATER': 1, 'WIND': 2, 'ELECTRICITY': 3}
    
    for ses_dir in ses_dirs:
        ses_name = os.path.basename(ses_dir)
        eeg_dir = os.path.join(ses_dir, "eeg")
        eeg_files = glob.glob(os.path.join(eeg_dir, "*.eeg"))
        tsv_files = glob.glob(os.path.join(eeg_dir, "*events.tsv"))
        if not eeg_files or not tsv_files:
            continue
            
        raw = np.fromfile(eeg_files[0], dtype=np.float32).reshape(-1, 33)[:, :32]
        mean_std = np.mean(np.std(raw, axis=0))
        raw_uv = raw / 1000.0 if mean_std > 500.0 else raw
        clean = preprocess_eeg(raw_uv, sfreq=SFREQ)
        
        df_ev = pd.read_csv(tsv_files[0], sep='\t')
        df_ev = df_ev.drop_duplicates(subset=['trial_type', 'sample'], keep='first')
        events_list = df_ev.to_dict('records')
        
        last_im_sample = -100000
        for i, ev in enumerate(events_list):
            tt = str(ev.get('trial_type', ''))
            if 'selected' in tt:
                elem = tt.replace(' selected', '').strip()
                if elem in class_map:
                    cid = class_map[elem]
                    cur_sample = int(ev['sample'])
                    if cur_sample - last_im_sample < int(2.0 * SFREQ):
                        continue
                    last_im_sample = cur_sample
                    
                    # Look back for Start Listen
                    listen_s = None
                    for j in range(i - 1, max(-1, i - 16), -1):
                        cand_tt = events_list[j].get('trial_type', '')
                        if cand_tt == 'Start Listen':
                            listen_s = int(events_list[j]['sample'])
                            break
                            
                    im_s = cur_sample
                    if im_s + n_win <= clean.shape[0]:
                        ep_im = clean[im_s : im_s + n_win, :].T
                        if listen_s is not None and listen_s + n_win <= clean.shape[0]:
                            ep_lis = clean[listen_s : listen_s + n_win, :].T
                        else:
                            ep_lis = ep_im
                            
                        all_im.append(ep_im)
                        all_lis.append(ep_lis)
                        all_y.append(cid)
                        all_ses_group.append(ses_name)
                        
        print(f"[+] Loaded Tower Defense sub-{sub_id} {ses_name}: {sum(1 for g in all_ses_group if g == ses_name)} trials")
        
    X_im = np.array(all_im)
    X_lis = np.array(all_lis)
    y_td = np.array(all_y)
    ses_groups = np.array(all_ses_group)
    return X_im, X_lis, y_td, ses_groups

def evaluate_models_cv(X, y, cv, title="CV Benchmark"):
    """Evaluates multiple decoders with Stratified CV."""
    cov_est = Covariances(estimator='oas')
    C = cov_est.fit_transform(X)
    ts = TangentSpace(metric='riemann')
    feat_ts = ts.fit_transform(C)
    
    # Welch PSD bandpowers
    n_epochs, n_ch, n_samples = X.shape
    freqs, psd = signal.welch(X, fs=SFREQ, nperseg=min(n_samples, 250), axis=-1)
    bands = [(1, 4), (4, 8), (8, 13), (14, 30), (30, 45)]
    feat_psd = []
    for ep in psd:
        ep_feats = []
        for l_f, h_f in bands:
            m = (freqs >= l_f) & (freqs <= h_f)
            ep_feats.extend(np.mean(ep[:, m], axis=-1))
        feat_psd.append(ep_feats)
    feat_psd = np.array(feat_psd)
    
    models = {
        'Riemannian TS + LogReg': (feat_ts, LogisticRegression(max_iter=1000, C=1.0)),
        'Riemannian TS + Ridge': (feat_ts, RidgeClassifier(alpha=1.0)),
        'Riemannian TS + Linear SVM': (feat_ts, SVC(kernel='linear', C=1.0)),
        'Welch PSD + Shrinkage LDA': (feat_psd, LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
        'Welch PSD + Random Forest': (feat_psd, RandomForestClassifier(n_estimators=100, random_state=42))
    }
    
    results = {}
    for name, (feats, clf) in models.items():
        accs, baccs, f1s = [], [], []
        oof_preds = np.zeros_like(y)
        for tr, te in cv.split(feats, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(feats[tr])
            X_te = scaler.transform(feats[te])
            clf.fit(X_tr, y[tr])
            pred = clf.predict(X_te)
            oof_preds[te] = pred
            accs.append(accuracy_score(y[te], pred))
            baccs.append(balanced_accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, average='macro'))
            
        results[name] = {
            'acc_mean': float(np.mean(accs)),
            'acc_std': float(np.std(accs)),
            'balanced_acc': float(np.mean(baccs)),
            'f1_macro': float(np.mean(f1s)),
            'confusion_matrix': confusion_matrix(y, oof_preds).tolist(),
            'predictions': oof_preds.tolist()
        }
        print(f"  {name:30s}: Acc = {np.mean(accs)*100:5.2f}% ± {np.std(accs)*100:4.2f}% | BalAcc = {np.mean(baccs)*100:5.2f}% | F1 = {np.mean(f1s):.3f}")
    return results

def evaluate_pairwise(X, y, class_names=CLASS_NAMES):
    """Computes pairwise classification for all 6 pairs."""
    cov_est = Covariances(estimator='oas')
    ts = TangentSpace(metric='riemann')
    pairs = [
        (0, 1, 'FIRE-WATER'),
        (0, 2, 'FIRE-WIND'),
        (0, 3, 'FIRE-ELECTRICITY'),
        (1, 2, 'WATER-WIND'),
        (1, 3, 'WATER-ELECTRICITY'),
        (2, 3, 'WIND-ELECTRICITY')
    ]
    
    pw_results = {}
    for c1, c2, pair_name in pairs:
        mask = (y == c1) | (y == c2)
        X_sub = X[mask]
        y_sub = (y[mask] == c2).astype(int)
        
        C_sub = cov_est.fit_transform(X_sub)
        F_sub = ts.fit_transform(C_sub)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        for tr, te in cv.split(F_sub, y_sub):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(F_sub[tr])
            X_te = scaler.transform(F_sub[te])
            clf = LogisticRegression(max_iter=500, C=1.0)
            clf.fit(X_tr, y_sub[tr])
            accs.append(accuracy_score(y_sub[te], clf.predict(X_te)))
        pw_results[pair_name] = {
            'accuracy': float(np.mean(accs)),
            'std': float(np.std(accs))
        }
        print(f"    Pair {pair_name:18s}: {np.mean(accs)*100:5.2f}% ± {np.std(accs)*100:4.2f}%")
    return pw_results

def main():
    out_dir = "scripts/analysis_results/listening_sub02_impact"
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 80)
    print(" SUB-02: PURE MUSIC LISTENING ANALYSIS & TOWER DEFENSE TRANSFER STUDIO ".center(80, "="))
    print("=" * 80)
    
    # 1. Load Pure Listening Data
    print("\n--- 1. Loading Pure Listening Data (bids_listening/sub-02) ---")
    X_listen, y_listen, df_listen_meta = load_listening_sub02(win_len_s=3.0, step_s=1.5, discard_start_s=5.0)
    print(f"Total Listening Epochs: {X_listen.shape[0]} | Shape: {X_listen.shape}")
    print(f"Class counts: { {c: int(np.sum(y_listen == i)) for i, c in enumerate(CLASS_NAMES)} }")
    
    # 2. Load Tower Defense Data
    print("\n--- 2. Loading Tower Defense Data (bids_tower_defense/sub-02) ---")
    X_td_im, X_td_lis, y_td, td_groups = load_tower_defense_sub02(win_len_s=3.0)
    print(f"Total Tower Defense Trials: {X_td_im.shape[0]} across 5 sessions")
    print(f"Class counts: { {c: int(np.sum(y_td == i)) for i, c in enumerate(CLASS_NAMES)} }")
    
    cv_5fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # -----------------------------------------------------------------
    # EXPERIMENT 1: Within-Domain Pure Listening 4-Class Decoding
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" EXPERIMENT 1: PURE MUSIC LISTENING 4-CLASS DECODING (CV) ".center(80, "="))
    print("=" * 80)
    res_listen_models = evaluate_models_cv(X_listen, y_listen, cv_5fold, title="Pure Listening")
    print("\n  Pairwise Pure Listening Binary Accuracies:")
    pw_listen = evaluate_pairwise(X_listen, y_listen)
    
    # -----------------------------------------------------------------
    # EXPERIMENT 2: Baseline Tower Defense Game Imagine Decoding
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" EXPERIMENT 2: BASELINE TOWER DEFENSE IMAGINE DECODING (CV) ".center(80, "="))
    print("=" * 80)
    res_td_base = evaluate_models_cv(X_td_im, y_td, cv_5fold, title="TD Baseline Imagine")
    print("\n  Pairwise TD Baseline Imagine Accuracies:")
    pw_td_base = evaluate_pairwise(X_td_im, y_td)
    
    # -----------------------------------------------------------------
    # EXPERIMENT 3: Zero-Shot Cross-Paradigm Transfer
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" EXPERIMENT 3: ZERO-SHOT TRANSFER (Pure Listening -> TD Game) ".center(80, "="))
    print("=" * 80)
    cov_est = Covariances(estimator='oas')
    C_listen = cov_est.fit_transform(X_listen)
    C_td_im = cov_est.fit_transform(X_td_im)
    C_td_lis = cov_est.fit_transform(X_td_lis)
    
    # Shared tangent space anchored at geometric mean of listening data
    C_ref_listen = mean_riemann(C_listen)
    C_ref_invsqrt = invsqrtm(C_ref_listen)
    
    # Whiten all covariances using listening reference
    C_listen_w = np.array([C_ref_invsqrt @ c @ C_ref_invsqrt for c in C_listen])
    C_td_im_w = np.array([C_ref_invsqrt @ c @ C_ref_invsqrt for c in C_td_im])
    C_td_lis_w = np.array([C_ref_invsqrt @ c @ C_ref_invsqrt for c in C_td_lis])
    
    ts = TangentSpace(metric='riemann')
    F_listen_w = ts.fit_transform(C_listen_w)
    F_td_im_w = ts.transform(C_td_im_w)
    F_td_lis_w = ts.transform(C_td_lis_w)
    
    scaler = StandardScaler()
    F_listen_scaled = scaler.fit_transform(F_listen_w)
    F_td_im_scaled = scaler.transform(F_td_im_w)
    F_td_lis_scaled = scaler.transform(F_td_lis_w)
    
    clf_transfer = LogisticRegression(max_iter=1000, C=1.0)
    clf_transfer.fit(F_listen_scaled, y_listen)
    
    preds_zero_lis = clf_transfer.predict(F_td_lis_scaled)
    acc_zero_lis = float(accuracy_score(y_td, preds_zero_lis))
    bacc_zero_lis = float(balanced_accuracy_score(y_td, preds_zero_lis))
    
    preds_zero_im = clf_transfer.predict(F_td_im_scaled)
    acc_zero_im = float(accuracy_score(y_td, preds_zero_im))
    bacc_zero_im = float(balanced_accuracy_score(y_td, preds_zero_im))
    
    print(f"[*] Zero-Shot Transfer: Pure Listening -> TD Listen: Acc = {acc_zero_lis*100:5.2f}% | BalAcc = {bacc_zero_lis*100:5.2f}% (Chance=25%)")
    print(f"[*] Zero-Shot Transfer: Pure Listening -> TD Imagine: Acc = {acc_zero_im*100:5.2f}% | BalAcc = {bacc_zero_im*100:5.2f}% (Chance=25%)")
    
    # -----------------------------------------------------------------
    # EXPERIMENT 4: Data Augmentation / Co-Training (Listening + TD Imagine)
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" EXPERIMENT 4: DATA AUGMENTATION / CO-TRAINING ".center(80, "="))
    print(" (Augment each training fold with pure listening prototypes) ".center(80, "="))
    print("=" * 80)
    
    # In each fold of TD Imagine, add X_listen to training data
    accs_aug, baccs_aug, f1s_aug = [], [], []
    preds_aug = np.zeros_like(y_td)
    
    for tr, te in cv_5fold.split(X_td_im, y_td):
        # Pool TD training fold + all Pure Listening epochs
        X_tr_pooled = np.concatenate([X_td_im[tr], X_listen], axis=0)
        y_tr_pooled = np.concatenate([y_td[tr], y_listen], axis=0)
        
        # Estimate covariances
        C_tr = cov_est.fit_transform(X_tr_pooled)
        C_te = cov_est.fit_transform(X_td_im[te])
        
        # Fit Tangent Space on training manifold
        ts_fold = TangentSpace(metric='riemann')
        F_tr = ts_fold.fit_transform(C_tr)
        F_te = ts_fold.transform(C_te)
        
        sc_fold = StandardScaler()
        F_tr = sc_fold.fit_transform(F_tr)
        F_te = sc_fold.transform(F_te)
        
        clf_aug = LogisticRegression(max_iter=1000, C=1.0)
        clf_aug.fit(F_tr, y_tr_pooled)
        pred_te = clf_aug.predict(F_te)
        preds_aug[te] = pred_te
        
        accs_aug.append(accuracy_score(y_td[te], pred_te))
        baccs_aug.append(balanced_accuracy_score(y_td[te], pred_te))
        f1s_aug.append(f1_score(y_td[te], pred_te, average='macro'))
        
    mean_acc_aug = float(np.mean(accs_aug))
    std_acc_aug = float(np.std(accs_aug))
    mean_bacc_aug = float(np.mean(baccs_aug))
    print(f"[*] TD Imagine + Pure Listening Augmentation:")
    print(f"    Acc = {mean_acc_aug*100:5.2f}% ± {std_acc_aug*100:4.2f}% | BalAcc = {mean_bacc_aug*100:5.2f}%")
    base_best = res_td_base['Riemannian TS + LogReg']['acc_mean']
    gain = (mean_acc_aug - base_best) * 100
    print(f"    Gain over TD Baseline Riemannian TS ({base_best*100:.2f}%): {gain:+5.2f}%")
    
    # -----------------------------------------------------------------
    # EXPERIMENT 5: Pairwise Augmentation Impact
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" EXPERIMENT 5: PAIRWISE AUGMENTATION IMPACT ".center(80, "="))
    print("=" * 80)
    pairs = [
        (0, 1, 'FIRE-WATER'),
        (0, 2, 'FIRE-WIND'),
        (0, 3, 'FIRE-ELECTRICITY'),
        (1, 2, 'WATER-WIND'),
        (1, 3, 'WATER-ELECTRICITY'),
        (2, 3, 'WIND-ELECTRICITY')
    ]
    pw_augmented = {}
    for c1, c2, pair_name in pairs:
        mask_td = (y_td == c1) | (y_td == c2)
        X_td_pair = X_td_im[mask_td]
        y_td_pair = (y_td[mask_td] == c2).astype(int)
        
        mask_lis = (y_listen == c1) | (y_listen == c2)
        X_lis_pair = X_listen[mask_lis]
        y_lis_pair = (y_listen[mask_lis] == c2).astype(int)
        
        accs_p = []
        cv_p = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, te in cv_p.split(X_td_pair, y_td_pair):
            X_tr = np.concatenate([X_td_pair[tr], X_lis_pair], axis=0)
            y_tr = np.concatenate([y_td_pair[tr], y_lis_pair], axis=0)
            
            C_tr = cov_est.fit_transform(X_tr)
            C_te = cov_est.fit_transform(X_td_pair[te])
            
            ts_p = TangentSpace(metric='riemann')
            F_tr = ts_p.fit_transform(C_tr)
            F_te = ts_p.transform(C_te)
            
            sc_p = StandardScaler()
            clf_p = LogisticRegression(max_iter=500, C=1.0)
            clf_p.fit(sc_p.fit_transform(F_tr), y_tr)
            accs_p.append(accuracy_score(y_td_pair[te], clf_p.predict(sc_p.transform(F_te))))
            
        acc_p_mean = float(np.mean(accs_p))
        std_p = float(np.std(accs_p))
        base_pw = pw_td_base[pair_name]['accuracy']
        pw_augmented[pair_name] = {
            'baseline_accuracy': base_pw,
            'augmented_accuracy': acc_p_mean,
            'delta': acc_p_mean - base_pw
        }
        print(f"    Pair {pair_name:18s}: Base={base_pw*100:5.2f}% -> Augmented={acc_p_mean*100:5.2f}% (Δ = {(acc_p_mean - base_pw)*100:+5.2f}%)")
        
    # -----------------------------------------------------------------
    # Generate Publication-Grade Visualizations
    # -----------------------------------------------------------------
    print("\n[*] Generating Comprehensive Visualization Suite...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    # Panel 1: Pure Listening vs TD Imagine Model Benchmark
    ax1 = axes[0, 0]
    model_names = list(res_listen_models.keys())
    short_names = ['Riemann TS\n(LogReg)', 'Riemann TS\n(Ridge)', 'Riemann TS\n(SVM)', 'Welch PSD\n(LDA)', 'Welch PSD\n(RF)']
    x = np.arange(len(model_names))
    width = 0.35
    
    lis_accs = [res_listen_models[m]['acc_mean'] * 100 for m in model_names]
    td_accs = [res_td_base[m]['acc_mean'] * 100 for m in model_names]
    
    r1 = ax1.bar(x - width/2, lis_accs, width, label='Pure Listening (sub-02, 4 Songs)', color='#3498db', edgecolor='black')
    r2 = ax1.bar(x + width/2, td_accs, width, label='TD Game Imagine (sub-02, 5 Sessions)', color='#e74c3c', edgecolor='black')
    ax1.axhline(25.0, color='gray', linestyle='--', linewidth=1.5, label='Chance Level (25%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, fontweight='bold', fontsize=9)
    ax1.set_ylabel('4-Class Accuracy (%)', fontweight='bold', fontsize=11)
    ax1.set_title('A. 4-Class Decoding: Pure Listening vs. Mental Imagery', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, 110)
    for rect in r1 + r2:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords='offset points', ha='center', va='bottom', fontsize=8, fontweight='bold')
                     
    # Panel 2: Confusion Matrix of Pure Listening
    ax2 = axes[0, 1]
    cm_listen = np.array(res_listen_models['Riemannian TS + LogReg']['confusion_matrix'])
    cm_listen_pct = cm_listen.astype('float') / cm_listen.sum(axis=1)[:, np.newaxis] * 100
    im2 = ax2.imshow(cm_listen_pct, cmap='Blues', vmin=0, vmax=100)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(CLASS_NAMES, fontweight='bold', fontsize=10)
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(CLASS_NAMES, fontweight='bold', fontsize=10)
    for r in range(4):
        for c in range(4):
            v = cm_listen_pct[r, c]
            ax2.text(c, r, f"{v:.1f}%", ha='center', va='center',
                     color='white' if v > 50 else 'black', fontweight='bold', fontsize=11)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title(f'B. Pure Listening Confusion Matrix (Riemannian TS: {res_listen_models["Riemannian TS + LogReg"]["acc_mean"]*100:.1f}%)', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Predicted Element', fontweight='bold', fontsize=11)
    ax2.set_ylabel('True Element (Song)', fontweight='bold', fontsize=11)
    
    # Panel 3: Pairwise Binary Separability: Baseline vs. Augmented
    ax3 = axes[1, 0]
    p_names = list(pw_augmented.keys())
    p_short = ['FIRE\nvs WATER', 'FIRE\nvs WIND', 'FIRE\nvs ELEC', 'WATER\nvs WIND', 'WATER\nvs ELEC', 'WIND\nvs ELEC']
    xp = np.arange(len(p_names))
    base_pw_vals = [pw_augmented[p]['baseline_accuracy'] * 100 for p in p_names]
    aug_pw_vals = [pw_augmented[p]['augmented_accuracy'] * 100 for p in p_names]
    
    r3 = ax3.bar(xp - width/2, base_pw_vals, width, label='TD Imagine Baseline', color='#95a5a6', edgecolor='black')
    r4 = ax3.bar(xp + width/2, aug_pw_vals, width, label='TD Imagine + Pure Listening Aug', color='#2ecc71', edgecolor='black')
    ax3.axhline(50.0, color='red', linestyle='--', linewidth=1.5, label='Chance (50%)')
    ax3.set_xticks(xp)
    ax3.set_xticklabels(p_short, fontweight='bold', fontsize=9)
    ax3.set_ylabel('Binary Accuracy (%)', fontweight='bold', fontsize=11)
    ax3.set_title('C. Pairwise Separability: TD Imagine Baseline vs. Augmented', fontweight='bold', fontsize=12)
    ax3.legend(loc='lower left', fontsize=9)
    ax3.set_ylim(0, 105)
    for rect in r3 + r4:
        h = rect.get_height()
        ax3.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords='offset points', ha='center', va='bottom', fontsize=8, fontweight='bold')
                     
    # Panel 4: Confusion Matrix of TD Imagine Augmented
    ax4 = axes[1, 1]
    cm_aug = confusion_matrix(y_td, preds_aug)
    cm_aug_pct = cm_aug.astype('float') / cm_aug.sum(axis=1)[:, np.newaxis] * 100
    im4 = ax4.imshow(cm_aug_pct, cmap='Greens', vmin=0, vmax=100)
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(CLASS_NAMES, fontweight='bold', fontsize=10)
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(CLASS_NAMES, fontweight='bold', fontsize=10)
    for r in range(4):
        for c in range(4):
            v = cm_aug_pct[r, c]
            ax4.text(c, r, f"{v:.1f}%", ha='center', va='center',
                     color='white' if v > 50 else 'black', fontweight='bold', fontsize=11)
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.set_title(f'D. TD Imagine Augmented Confusion Matrix ({mean_acc_aug*100:.1f}%)', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Predicted Element', fontweight='bold', fontsize=11)
    ax4.set_ylabel('True Element', fontweight='bold', fontsize=11)
    
    fig_path = os.path.join(out_dir, "sub02_listening_impact_analysis.png")
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Exported High-Res Figure: {fig_path}")
    
    # Save master results JSON
    master_dict = {
        'subject': 'sub-02',
        'listening_sessions': {
            'ses-01': 'WATER (Its Raining Men)',
            'ses-02': 'WIND (Whats Up)',
            'ses-03': 'ELECTRICITY (Thunderstruck)',
            'ses-04': 'FIRE (I Was Made For Lovin You)'
        },
        'listening_4class_models': res_listen_models,
        'listening_pairwise': pw_listen,
        'td_baseline_imagine_models': res_td_base,
        'td_baseline_pairwise': pw_td_base,
        'zero_shot_transfer': {
            'listening_to_td_listen_acc': acc_zero_lis,
            'listening_to_td_listen_bacc': bacc_zero_lis,
            'listening_to_td_imagine_acc': acc_zero_im,
            'listening_to_td_imagine_bacc': bacc_zero_im
        },
        'co_training_augmented': {
            'acc_mean': mean_acc_aug,
            'acc_std': std_acc_aug,
            'balanced_acc': mean_bacc_aug,
            'gain_over_baseline': gain,
            'confusion_matrix': cm_aug.tolist()
        },
        'pairwise_augmented': pw_augmented
    }
    
    json_path = os.path.join(out_dir, "sub02_listening_impact_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(master_dict, f, indent=2)
    print(f"[+] Exported Summary JSON: {json_path}")
    print("=" * 80)
    print(" PIPELINE FINISHED SUCCESSFULLY! ".center(80, "="))
    print("=" * 80)

if __name__ == "__main__":
    main()
