"""
Tower Defense 4-Class Rhythm Decoding & Neuro-Statistical Analysis Studio
==========================================================================
Analyzes the BIDS Tower Defense dataset (`scripts/bids/bids_tower_defense`),
evaluating whether neural activity (EEG) can decode the 4 mental rhythms/elements:
  - FIRE
  - WATER
  - WIND
  - ELECTRICITY

Supports:
  - Single session analysis (`--ses 01`)
  - Multi-session discovery & pooling (`--ses all` or `--ses 01,02,03...`)
  - Pairwise 2-Class Binary Rhythm Decoding (FIRE-WATER, FIRE-WIND, etc.)
  - Leave-One-Session-Out Cross-Validation (LOSO-CV) across recording blocks
  - Perception-to-Imagery Zero-Shot Transfer Learning (`Listen` -> `Imagine`)
  - Representational Similarity Analysis (RSA) and sliding-window temporal trajectory
"""

import os
import sys
import glob
import json
import itertools
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.linalg import eigh
import scipy.stats as stats

# Ensure paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
_ws_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
if _ws_dir not in sys.path:
    sys.path.insert(0, _ws_dir)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Machine Learning Imports
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score

# Spatial Filters
try:
    from spatial_filters import detect_bad_channels, apply_spatial_filter
except ImportError:
    from scripts.analysis.spatial_filters import detect_bad_channels, apply_spatial_filter


# ----------------------------------------------------------------------
# 1. BIDS Multi-Session Loading & Robust Preprocessing
# ----------------------------------------------------------------------
def find_available_sessions(bids_root, sub_id="01"):
    """Finds all available session folders for a given subject."""
    sub_clean = sub_id.replace("sub-", "")
    sub_dir = os.path.join(bids_root, f"sub-{sub_clean}")
    if not os.path.exists(sub_dir):
        raise FileNotFoundError(f"Subject directory not found: {sub_dir}")
        
    ses_dirs = sorted(glob.glob(os.path.join(sub_dir, "ses-*")))
    ses_ids = [os.path.basename(d).replace("ses-", "") for d in ses_dirs if os.path.isdir(d)]
    return ses_ids


def load_single_session_raw(bids_root, sub_clean, ses_clean):
    """Loads raw binary EEG and events TSV for a single session."""
    eeg_dir = os.path.join(bids_root, f"sub-{sub_clean}", f"ses-{ses_clean}", "eeg")
    if not os.path.exists(eeg_dir):
        raise FileNotFoundError(f"EEG directory not found: {eeg_dir}")
        
    eeg_files = glob.glob(os.path.join(eeg_dir, "*.eeg"))
    events_files = glob.glob(os.path.join(eeg_dir, "*events.tsv"))
    
    if not eeg_files or not events_files:
        raise FileNotFoundError(f"Missing .eeg or .tsv in {eeg_dir}")
        
    eeg_path = eeg_files[0]
    events_path = events_files[0]
    
    raw_all = np.fromfile(eeg_path, dtype=np.float32).reshape(-1, 33)
    sfreq = 250.0
    raw_eeg = raw_all[:, :32]
    
    mean_std = np.mean(np.std(raw_eeg, axis=0))
    raw_eeg_uv = raw_eeg / 1000.0 if mean_std > 500.0 else raw_eeg
    df_events = pd.read_csv(events_path, sep='\t')
    ch_names = [f"EEG{i+1:03d}" for i in range(32)]
    
    return raw_eeg_uv, df_events, sfreq, ch_names


def preprocess_continuous_eeg(eeg_data, sfreq=250.0, l_freq=1.0, h_freq=45.0, notch_freq=50.0, spatial_mode="robust_car", ch_names=None):
    """Zero-phase Butterworth bandpass filter, notch filter, and Robust CAR referencing."""
    nyq = sfreq / 2.0
    b_band, a_band = signal.butter(4, [l_freq / nyq, h_freq / nyq], btype='band')
    filt_eeg = signal.filtfilt(b_band, a_band, eeg_data, axis=0)
    
    b_notch, a_notch = signal.iirnotch(notch_freq, 30.0, sfreq)
    filt_eeg = signal.filtfilt(b_notch, a_notch, filt_eeg, axis=0)
    
    if spatial_mode == "robust_car":
        stds = np.std(filt_eeg, axis=0)
        good_mask = (stds > 2.0) & (stds < 250.0)
        good_indices = np.where(good_mask)[0]
        if len(good_indices) == 0:
            good_indices = np.arange(filt_eeg.shape[1])
        median_ref = np.median(filt_eeg[:, good_indices], axis=1, keepdims=True)
        clean_eeg = filt_eeg - median_ref
    elif spatial_mode == "laplacian":
        clean_eeg = apply_spatial_filter(filt_eeg, ch_names, mode="laplacian")
    elif spatial_mode == "car":
        clean_eeg = filt_eeg - np.mean(filt_eeg, axis=1, keepdims=True)
    else:
        clean_eeg = filt_eeg - np.mean(filt_eeg, axis=0, keepdims=True)
        
    return clean_eeg


def extract_session_epochs(clean_eeg, df_events, ses_id, sfreq=250.0, win_len_s=3.0):
    """Extracts Imagine, Listen, and Blinking epochs from a preprocessed session."""
    class_map = {'FIRE': 0, 'WATER': 1, 'WIND': 2, 'ELECTRICITY': 3}
    class_names = ['FIRE', 'WATER', 'WIND', 'ELECTRICITY']
    # Deduplicate event records if any repeated rows exist at same trial_type & sample
    df_events_clean = df_events.drop_duplicates(subset=['trial_type', 'sample'], keep='first')
    events_list = df_events_clean.to_dict('records')
    n_samples_win = int(win_len_s * sfreq)
    
    epochs_im, epochs_lis, epochs_blk, labels, meta = [], [], [], [], []
    
    # 1. Check if this is a continuous music listening session (e.g., bids_listening)
    is_music_session = any('Track_Start_id_' in str(ev.get('trial_type', '')) for ev in events_list)
    if is_music_session:
        track_map = {
            'Beethoven_Fur_Elise': ('FIRE', 0, 187.068, 416.464),
            'Bach_Prelude': ('WATER', 1, 48.98, 182.068),
            'Vivaldi_Spring': ('WIND', 2, 1469.64, 1664.745),
            'Tchaikovsky_Waltz': ('ELECTRICITY', 3, 1013.272, 1464.632)
        }
        for track_key, (el_name, cls_id, t_start, t_end) in track_map.items():
            s_start = int((t_start + 2.0) * sfreq)
            s_end = int((t_end - 2.0) * sfreq)
            step_samp = int(3.0 * sfreq)
            for s in range(s_start, s_end - n_samples_win, step_samp):
                if s + n_samples_win <= len(clean_eeg):
                    ep = clean_eeg[s : s + n_samples_win, :].T
                    epochs_im.append(ep)
                    epochs_lis.append(ep)
                    epochs_blk.append(ep)
                    labels.append(cls_id)
                    meta.append({
                        'session': str(ses_id),
                        'element': el_name,
                        'class_id': cls_id,
                        'imagine_sample': s,
                        'listen_sample': s
                    })
        return np.array(epochs_im), np.array(epochs_lis), np.array(epochs_blk), np.array(labels), pd.DataFrame(meta), class_names

    # 2. Standard or Legacy Tower Defense session with trials
    has_listen = any(ev.get('trial_type') == 'Start Listen' for ev in events_list)
    
    last_im_sample = -100000
    for i, ev in enumerate(events_list):
        tt = str(ev.get('trial_type', ''))
        if 'selected' in tt:
            element = tt.replace(' selected', '').strip()
            if element in class_map:
                cls_id = class_map[element]
                cur_sample = int(ev['sample'])
                if cur_sample - last_im_sample < int(2.0 * sfreq):
                    continue
                last_im_sample = cur_sample
                
                listen_s = None
                blink_s = None
                stop_blink_s = None
                for j in range(i - 1, max(-1, i - 16), -1):
                    cand_tt = events_list[j].get('trial_type', '')
                    cand_samp = int(events_list[j]['sample'])
                    if cand_tt == 'Start Listen' and listen_s is None:
                        listen_s = cand_samp
                    elif cand_tt == 'Box start blinking' and blink_s is None:
                        blink_s = cand_samp
                    elif cand_tt == 'Box stop blinking' and stop_blink_s is None:
                        stop_blink_s = cand_samp
                    if 'selected' in cand_tt:
                        break
                        
                # Determine imagine onset
                # In sessions without listening (e.g. bids_tower_defense(old)),
                # the mental imagery period starts ~0.5s after Box stop blinking (recall window 3-4s).
                if not has_listen and stop_blink_s is not None:
                    im_start = int(stop_blink_s + 0.50 * sfreq)
                    imagine_s = stop_blink_s
                else:
                    imagine_s = int(ev['sample'])
                    im_start = int(imagine_s + 0.25 * sfreq)
                    
                blk_start = int(blink_s + 0.25 * sfreq) if blink_s is not None else 0
                lis_start = int(listen_s + 0.50 * sfreq) if listen_s is not None else None
                
                # Check valid bounds
                if (im_start + n_samples_win <= len(clean_eeg) and 
                    blk_start + n_samples_win <= len(clean_eeg) and
                    (lis_start is None or lis_start + n_samples_win <= len(clean_eeg))):
                    
                    epochs_im.append(clean_eeg[im_start : im_start + n_samples_win, :].T)
                    epochs_blk.append(clean_eeg[blk_start : blk_start + n_samples_win, :].T)
                    if lis_start is not None:
                        epochs_lis.append(clean_eeg[lis_start : lis_start + n_samples_win, :].T)
                    labels.append(cls_id)
                    meta.append({
                        'session': str(ses_id),
                        'element': element,
                        'class_id': cls_id,
                        'imagine_sample': imagine_s,
                        'listen_sample': listen_s
                    })
                    
    epochs_im_arr = np.array(epochs_im) if epochs_im else np.empty((0, clean_eeg.shape[1], n_samples_win))
    epochs_lis_arr = np.array(epochs_lis) if len(epochs_lis) == len(epochs_im) and len(epochs_lis) > 0 else np.empty((0, clean_eeg.shape[1], n_samples_win))
    epochs_blk_arr = np.array(epochs_blk) if epochs_blk else np.empty((0, clean_eeg.shape[1], n_samples_win))
    
    return epochs_im_arr, epochs_lis_arr, epochs_blk_arr, np.array(labels), pd.DataFrame(meta), class_names


# ----------------------------------------------------------------------
# 2. Algorithmic Feature Extractors
# ----------------------------------------------------------------------
def compute_ovr_csp(X, y, n_components=4):
    """Common Spatial Pattern (CSP) spatial filters (Binary or One-vs-Rest)."""
    n_epochs, n_ch, _ = X.shape
    classes = np.unique(y)
    covs = [np.cov(X[i]) / (np.trace(np.cov(X[i])) + 1e-12) for i in range(n_epochs)]
    
    if len(classes) == 2:
        c1, c2 = classes[0], classes[1]
        cov1 = np.mean([covs[k] for k in range(len(covs)) if y[k] == c1], axis=0) + 1e-5 * np.eye(n_ch)
        cov2 = np.mean([covs[k] for k in range(len(covs)) if y[k] == c2], axis=0) + 1e-5 * np.eye(n_ch)
        vals, vecs = eigh(cov1, cov1 + cov2)
        half = max(1, min(n_components // 2, n_ch // 2))
        return np.hstack([vecs[:, -half:], vecs[:, :half]])
        
    filters = []
    for c_id in classes:
        mask = (y == c_id)
        cov_target = np.mean([covs[k] for k in range(len(covs)) if mask[k]], axis=0) + 1e-5 * np.eye(n_ch)
        cov_rest = np.mean([covs[k] for k in range(len(covs)) if not mask[k]], axis=0) + 1e-5 * np.eye(n_ch)
        vals, vecs = eigh(cov_target, cov_target + cov_rest)
        half = max(1, min(n_components // 2, n_ch // 2))
        filters.append(np.hstack([vecs[:, -half:], vecs[:, :half]]))
        
    return np.hstack(filters)


def project_csp_features(X, W):
    """Projects epochs through CSP spatial filters to log-variance features."""
    return np.array([np.log(np.var(np.dot(W.T, X[i]), axis=1) + 1e-12) for i in range(len(X))])


def extract_filter_bank_csp(X_train, y_train, X_test, sfreq=250.0, n_components=4):
    """Multi-Band Filter Bank CSP (Theta, Alpha, Low-Beta, High-Beta, Gamma)."""
    bands = [
        ('Theta', 4.0, 8.0),
        ('Alpha', 8.0, 12.0),
        ('Low-Beta', 12.0, 20.0),
        ('High-Beta', 20.0, 32.0),
        ('Gamma', 32.0, 45.0)
    ]
    nyq = sfreq / 2.0
    tr_blocks, te_blocks = [], []
    for _, fmin, fmax in bands:
        b, a = signal.butter(4, [fmin / nyq, fmax / nyq], btype='band')
        X_tr_f = np.array([signal.filtfilt(b, a, X_train[i], axis=-1) for i in range(len(X_train))])
        X_te_f = np.array([signal.filtfilt(b, a, X_test[i], axis=-1) for i in range(len(X_test))])
        
        W_band = compute_ovr_csp(X_tr_f, y_train, n_components=n_components)
        tr_blocks.append(project_csp_features(X_tr_f, W_band))
        te_blocks.append(project_csp_features(X_te_f, W_band))
        
    return np.hstack(tr_blocks), np.hstack(te_blocks)


def compute_covariance_matrices(X):
    """Regularized covariance matrices."""
    n_epochs, n_ch, _ = X.shape
    covs = np.zeros((n_epochs, n_ch, n_ch))
    for i in range(n_epochs):
        c = np.cov(X[i])
        c = c / (np.trace(c) + 1e-12)
        c += 1e-5 * np.eye(n_ch)
        covs[i] = c
    return covs


def compute_riemannian_mean(covmats, max_iter=25, tol=1e-6):
    """Fréchet geometric mean on SPD manifold."""
    C_mean = np.mean(covmats, axis=0)
    for _ in range(max_iter):
        vals, vecs = eigh(C_mean)
        vals = np.maximum(vals, 1e-8)
        sqrt_C = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        inv_sqrt_C = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        
        tangents = []
        for i in range(len(covmats)):
            m = inv_sqrt_C @ covmats[i] @ inv_sqrt_C
            v, w = eigh(m)
            v = np.maximum(v, 1e-8)
            log_m = w @ np.diag(np.log(v)) @ w.T
            tangents.append(log_m)
            
        mean_t = np.mean(tangents, axis=0)
        if np.linalg.norm(mean_t, ord='fro') < tol:
            break
        v, w = eigh(mean_t)
        exp_t = w @ np.diag(np.exp(v)) @ w.T
        C_mean = sqrt_C @ exp_t @ sqrt_C
        
    return C_mean


def project_to_riemannian_tangent_space(covmats, C_ref=None):
    """Projects covariances onto Euclidean Tangent Space."""
    if C_ref is None:
        C_ref = compute_riemannian_mean(covmats)
        
    vals, vecs = eigh(C_ref)
    vals = np.maximum(vals, 1e-8)
    inv_sqrt_C = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    
    n_epochs, n_ch, _ = covmats.shape
    triu_idx = np.triu_indices(n_ch)
    diag_mask = (triu_idx[0] == triu_idx[1])
    
    ts_vectors = []
    for i in range(n_epochs):
        m = inv_sqrt_C @ covmats[i] @ inv_sqrt_C
        v, w = eigh(m)
        v = np.maximum(v, 1e-8)
        log_m = w @ np.diag(np.log(v)) @ w.T
        vec = log_m[triu_idx].copy()
        vec[~diag_mask] *= np.sqrt(2.0)
        ts_vectors.append(vec)
        
    return np.array(ts_vectors), C_ref


def extract_welch_bandpower_features(X, sfreq=250.0):
    """Relative Welch Power Spectral Density across 5 classical EEG bands."""
    bands = {'delta': (1.0, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 12.0), 'beta': (13.0, 30.0), 'gamma': (30.0, 45.0)}
    freqs, psd = signal.welch(X, fs=sfreq, nperseg=min(int(sfreq * 1.5), X.shape[-1]), axis=-1)
    tot = np.sum(psd, axis=-1, keepdims=True) + 1e-12
    rel_psd = psd / tot
    
    band_feats = []
    for fmin, fmax in bands.values():
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_feats.append(np.mean(rel_psd[:, :, mask], axis=-1))
    return np.hstack(band_feats)


# ----------------------------------------------------------------------
# 3. Model Benchmark & Leave-One-Session-Out Cross-Validation
# ----------------------------------------------------------------------
def evaluate_phase_decoding(X_data, y, groups=None, sfreq=250.0, n_splits=5):
    """Evaluates multi-model decoding under Stratified K-Fold CV."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = {
        'CSP_ShrinkageLDA': [],
        'CSP_SVM_RBF': [],
        'FilterBank_CSP_LogReg': [],
        'Riemannian_TangentSpace_LogReg': [],
        'Riemannian_TangentSpace_SVM_Linear': [],
        'Riemannian_TangentSpace_Ridge': [],
        'Riemannian_TangentSpace_SVM': [],
        'Welch_PSD_RandomForest': [],
        'Welch_PSD_ShrinkageLDA': [],
        'Ensemble_Voting': []
    }
    preds_record = {m: np.zeros_like(y) for m in models}
    
    for tr_idx, te_idx in cv.split(X_data, y):
        X_tr, X_te = X_data[tr_idx], X_data[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        # 1. CSP + LDA
        W_csp = compute_ovr_csp(X_tr, y_tr, n_components=4)
        f_tr_csp = project_csp_features(X_tr, W_csp)
        f_te_csp = project_csp_features(X_te, W_csp)
        clf_csp_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf_csp_lda.fit(f_tr_csp, y_tr)
        p_csp_lda = clf_csp_lda.predict(f_te_csp)
        prob_csp_lda = clf_csp_lda.predict_proba(f_te_csp)
        models['CSP_ShrinkageLDA'].append(accuracy_score(y_te, p_csp_lda))
        preds_record['CSP_ShrinkageLDA'][te_idx] = p_csp_lda
        
        # 2. CSP + SVM
        scaler_csp = StandardScaler()
        clf_csp_svm = SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
        clf_csp_svm.fit(scaler_csp.fit_transform(f_tr_csp), y_tr)
        p_csp_svm = clf_csp_svm.predict(scaler_csp.transform(f_te_csp))
        models['CSP_SVM_RBF'].append(accuracy_score(y_te, p_csp_svm))
        preds_record['CSP_SVM_RBF'][te_idx] = p_csp_svm
        
        # 3. FBCSP + LogReg
        f_tr_fb, f_te_fb = extract_filter_bank_csp(X_tr, y_tr, X_te, sfreq=sfreq, n_components=4)
        scaler_fb = StandardScaler()
        clf_fb_lr = LogisticRegression(C=0.5, max_iter=500, random_state=42)
        clf_fb_lr.fit(scaler_fb.fit_transform(f_tr_fb), y_tr)
        p_fb_lr = clf_fb_lr.predict(scaler_fb.transform(f_te_fb))
        prob_fb_lr = clf_fb_lr.predict_proba(scaler_fb.transform(f_te_fb))
        models['FilterBank_CSP_LogReg'].append(accuracy_score(y_te, p_fb_lr))
        preds_record['FilterBank_CSP_LogReg'][te_idx] = p_fb_lr
        
        # 4. Riemannian Tangent Space
        cov_tr = compute_covariance_matrices(X_tr)
        cov_te = compute_covariance_matrices(X_te)
        ts_tr, C_ref = project_to_riemannian_tangent_space(cov_tr)
        ts_te, _ = project_to_riemannian_tangent_space(cov_te, C_ref=C_ref)
        scaler_ts = StandardScaler()
        ts_tr_s = scaler_ts.fit_transform(ts_tr)
        ts_te_s = scaler_ts.transform(ts_te)
        
        clf_ts_lr = LogisticRegression(C=0.1, max_iter=500, random_state=42)
        clf_ts_lr.fit(ts_tr_s, y_tr)
        p_ts_lr = clf_ts_lr.predict(ts_te_s)
        prob_ts_lr = clf_ts_lr.predict_proba(ts_te_s)
        models['Riemannian_TangentSpace_LogReg'].append(accuracy_score(y_te, p_ts_lr))
        preds_record['Riemannian_TangentSpace_LogReg'][te_idx] = p_ts_lr
        
        clf_ts_svm_lin = SVC(C=0.1, kernel='linear', random_state=42)
        clf_ts_svm_lin.fit(ts_tr_s, y_tr)
        p_ts_svm_lin = clf_ts_svm_lin.predict(ts_te_s)
        models['Riemannian_TangentSpace_SVM_Linear'].append(accuracy_score(y_te, p_ts_svm_lin))
        preds_record['Riemannian_TangentSpace_SVM_Linear'][te_idx] = p_ts_svm_lin
        
        clf_ts_ridge = RidgeClassifier(alpha=10.0, random_state=42)
        clf_ts_ridge.fit(ts_tr_s, y_tr)
        p_ts_ridge = clf_ts_ridge.predict(ts_te_s)
        models['Riemannian_TangentSpace_Ridge'].append(accuracy_score(y_te, p_ts_ridge))
        preds_record['Riemannian_TangentSpace_Ridge'][te_idx] = p_ts_ridge
        
        clf_ts_svm = SVC(C=1.0, kernel='rbf', random_state=42)
        clf_ts_svm.fit(ts_tr_s, y_tr)
        p_ts_svm = clf_ts_svm.predict(ts_te_s)
        models['Riemannian_TangentSpace_SVM'].append(accuracy_score(y_te, p_ts_svm))
        preds_record['Riemannian_TangentSpace_SVM'][te_idx] = p_ts_svm
        
        # 5. Welch PSD
        psd_tr = extract_welch_bandpower_features(X_tr, sfreq=sfreq)
        psd_te = extract_welch_bandpower_features(X_te, sfreq=sfreq)
        clf_rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
        clf_rf.fit(psd_tr, y_tr)
        p_rf = clf_rf.predict(psd_te)
        models['Welch_PSD_RandomForest'].append(accuracy_score(y_te, p_rf))
        preds_record['Welch_PSD_RandomForest'][te_idx] = p_rf
        
        clf_psd_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf_psd_lda.fit(psd_tr, y_tr)
        p_psd_lda = clf_psd_lda.predict(psd_te)
        models['Welch_PSD_ShrinkageLDA'].append(accuracy_score(y_te, p_psd_lda))
        preds_record['Welch_PSD_ShrinkageLDA'][te_idx] = p_psd_lda
        
        # 6. Ensemble Voting
        prob_ens = (prob_csp_lda + prob_fb_lr + prob_ts_lr) / 3.0
        p_ens = np.argmax(prob_ens, axis=1)
        models['Ensemble_Voting'].append(accuracy_score(y_te, p_ens))
        preds_record['Ensemble_Voting'][te_idx] = p_ens
        
    summary = {}
    for m, accs in models.items():
        summary[m] = {
            'mean_accuracy': float(np.mean(accs)),
            'std_accuracy': float(np.std(accs)),
            'balanced_accuracy': float(balanced_accuracy_score(y, preds_record[m])),
            'macro_f1': float(f1_score(y, preds_record[m], average='macro')),
            'cohen_kappa': float(cohen_kappa_score(y, preds_record[m])),
            'fold_accuracies': [float(a) for a in accs],
            'predictions': preds_record[m].tolist()
        }
    return summary, preds_record


def evaluate_loso_cross_validation(X_data, y, session_labels, sfreq=250.0):
    """Leave-One-Session-Out (LOSO) Cross-Validation across distinct recording sessions."""
    unique_sessions = np.unique(session_labels)
    if len(unique_sessions) < 2:
        return None
        
    loso = LeaveOneGroupOut()
    loso_scores = {'CSP_ShrinkageLDA': [], 'Riemannian_TangentSpace': [], 'FBCSP_LogReg': []}
    
    for tr_idx, te_idx in loso.split(X_data, y, groups=session_labels):
        test_ses = session_labels[te_idx[0]]
        X_tr, X_te = X_data[tr_idx], X_data[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        # 1. CSP
        W = compute_ovr_csp(X_tr, y_tr, n_components=4)
        clf_csp = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf_csp.fit(project_csp_features(X_tr, W), y_tr)
        acc_csp = accuracy_score(y_te, clf_csp.predict(project_csp_features(X_te, W)))
        loso_scores['CSP_ShrinkageLDA'].append({'test_session': test_ses, 'accuracy': float(acc_csp)})
        
        # 2. Riemannian TS
        cov_tr = compute_covariance_matrices(X_tr)
        cov_te = compute_covariance_matrices(X_te)
        ts_tr, C_ref = project_to_riemannian_tangent_space(cov_tr)
        ts_te, _ = project_to_riemannian_tangent_space(cov_te, C_ref=C_ref)
        sc = StandardScaler()
        clf_ts = LogisticRegression(C=0.1, max_iter=500)
        clf_ts.fit(sc.fit_transform(ts_tr), y_tr)
        acc_ts = accuracy_score(y_te, clf_ts.predict(sc.transform(ts_te)))
        loso_scores['Riemannian_TangentSpace'].append({'test_session': test_ses, 'accuracy': float(acc_ts)})
        
        # 3. FBCSP
        f_tr_fb, f_te_fb = extract_filter_bank_csp(X_tr, y_tr, X_te, sfreq=sfreq, n_components=4)
        sc_fb = StandardScaler()
        clf_fb = LogisticRegression(C=0.5, max_iter=500)
        clf_fb.fit(sc_fb.fit_transform(f_tr_fb), y_tr)
        acc_fb = accuracy_score(y_te, clf_fb.predict(sc_fb.transform(f_te_fb)))
        loso_scores['FBCSP_LogReg'].append({'test_session': test_ses, 'accuracy': float(acc_fb)})
        
    return loso_scores


def evaluate_cross_condition_transfer(X_listen, X_imagine, y, sfreq=250.0):
    """Train on Auditory Perception (Listen) -> Zero-Shot Predict Mental Imagery (Imagine)."""
    # CSP + LDA
    W_lis = compute_ovr_csp(X_listen, y, n_components=4)
    clf_csp_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_csp_lda.fit(project_csp_features(X_listen, W_lis), y)
    p_im = clf_csp_lda.predict(project_csp_features(X_imagine, W_lis))
    
    # FBCSP + LogReg
    f_lis_fb, f_im_fb = extract_filter_bank_csp(X_listen, y, X_imagine, sfreq=sfreq, n_components=4)
    sc_fb = StandardScaler()
    clf_fb = LogisticRegression(C=0.5, max_iter=500, random_state=42)
    clf_fb.fit(sc_fb.fit_transform(f_lis_fb), y)
    p_im_fb = clf_fb.predict(sc_fb.transform(f_im_fb))
    
    # Riemannian TS
    cov_lis = compute_covariance_matrices(X_listen)
    cov_im = compute_covariance_matrices(X_imagine)
    ts_lis, C_ref = project_to_riemannian_tangent_space(cov_lis)
    ts_im, _ = project_to_riemannian_tangent_space(cov_im, C_ref=C_ref)
    sc_ts = StandardScaler()
    clf_ts = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    clf_ts.fit(sc_ts.fit_transform(ts_lis), y)
    p_im_ts = clf_ts.predict(sc_ts.transform(ts_im))
    
    return {
        'CSP_ShrinkageLDA_ListenToImagine': {
            'accuracy': float(accuracy_score(y, p_im)),
            'f1': float(f1_score(y, p_im, average='macro')),
            'balanced_accuracy': float(balanced_accuracy_score(y, p_im)),
            'kappa': float(cohen_kappa_score(y, p_im)),
            'predictions': p_im.tolist(),
            'confusion_matrix': confusion_matrix(y, p_im).tolist()
        },
        'FBCSP_LogReg_ListenToImagine': {
            'accuracy': float(accuracy_score(y, p_im_fb)),
            'f1': float(f1_score(y, p_im_fb, average='macro')),
            'predictions': p_im_fb.tolist()
        },
        'Riemannian_TangentSpace_ListenToImagine': {
            'accuracy': float(accuracy_score(y, p_im_ts)),
            'f1': float(f1_score(y, p_im_ts, average='macro')),
            'predictions': p_im_ts.tolist()
        }
    }


def compute_representational_similarity(X_listen, X_imagine, y, class_names):
    """Representational Dissimilarity Matrices (RDMs) for Perception vs Imagery."""
    n_classes = len(class_names)
    cov_lis = compute_covariance_matrices(X_listen)
    cov_im = compute_covariance_matrices(X_imagine)
    ts_lis, C_ref = project_to_riemannian_tangent_space(cov_lis)
    ts_im, _ = project_to_riemannian_tangent_space(cov_im, C_ref=C_ref)
    
    means_lis = np.array([np.mean(ts_lis[y == c], axis=0) for c in range(n_classes)])
    means_im = np.array([np.mean(ts_im[y == c], axis=0) for c in range(n_classes)])
    
    rdm_lis = np.zeros((n_classes, n_classes))
    rdm_im = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            r_l, _ = stats.pearsonr(means_lis[i], means_lis[j])
            r_m, _ = stats.pearsonr(means_im[i], means_im[j])
            rdm_lis[i, j] = 1.0 - r_l
            rdm_im[i, j] = 1.0 - r_m
            
    triu_idx = np.triu_indices(n_classes, k=1)
    spearman_rho, spearman_p = stats.spearmanr(rdm_lis[triu_idx], rdm_im[triu_idx])
    return rdm_lis, rdm_im, float(spearman_rho), float(spearman_p)


def evaluate_pairwise_decoding(
    X_data,
    y,
    class_names=None,
    sfreq=250.0,
    n_splits=5,
    models=None,
    verbose=True,
    phase_name="Mental Imagery"
):
    """
    Evaluates 2-class pairwise decoding across all class combinations.
    For the 4 Tower Defense rhythm classes (FIRE, WATER, WIND, ELECTRICITY),
    benchmarks all 6 binary combinations:
      - (FIRE vs. WATER)
      - (FIRE vs. WIND)
      - (FIRE vs. ELECTRICITY)
      - (WATER vs. WIND)
      - (WATER vs. ELECTRICITY)
      - (WIND vs. ELECTRICITY)
      
    For each pair, evaluates:
      - CSP + Shrinkage LDA (Standard BCI benchmark, chance = 50.0%)
      - Riemannian Tangent Space + Logistic Regression
      - Multi-Band Filter Bank CSP (FBCSP) + Logistic Regression
      - Welch PSD Bandpower + Shrinkage LDA
      
    Parameters
    ----------
    X_data : np.ndarray, shape (n_epochs, n_channels, n_samples)
        Preprocessed EEG epochs.
    y : np.ndarray, shape (n_epochs,)
        Class labels (integers 0..3 or strings).
    class_names : list, tuple, or dict, optional
        Human-readable class names matching class IDs.
        Defaults to ['FIRE', 'WATER', 'WIND', 'ELECTRICITY'].
    sfreq : float, default=250.0
        Sampling frequency in Hz.
    n_splits : int, default=5
        Number of Stratified K-Fold CV splits.
    models : list of str, optional
        Models to benchmark for each pair.
        Defaults to ['CSP_ShrinkageLDA', 'Riemannian_TangentSpace_LogReg',
                     'FilterBank_CSP_LogReg', 'Welch_PSD_ShrinkageLDA'].
    verbose : bool, default=True
        Whether to print a formatted pairwise decoding table.
    phase_name : str, default="Mental Imagery"
        Condition/phase name for display.
        
    Returns
    -------
    pairwise_results : dict
        Detailed results for all pairs, ranking table, summary statistics,
        and pairwise separability matrix.
    """
    # Map classes
    unique_classes = sorted(list(np.unique(y)))
    if class_names is None:
        if all(isinstance(c, (str, np.str_)) for c in unique_classes):
            class_map = {c: str(c) for c in unique_classes}
        else:
            default_names = ['FIRE', 'WATER', 'WIND', 'ELECTRICITY']
            class_map = {c: default_names[c] if (isinstance(c, (int, np.integer)) and 0 <= c < len(default_names)) else f"Class_{c}" for c in unique_classes}
    elif isinstance(class_names, dict):
        class_map = {c: str(class_names.get(c, f"Class_{c}")) for c in unique_classes}
    elif isinstance(class_names, (list, tuple)):
        class_map = {}
        for c in unique_classes:
            if isinstance(c, (int, np.integer)) and 0 <= c < len(class_names):
                class_map[c] = str(class_names[c])
            else:
                class_map[c] = str(c)
    else:
        class_map = {c: str(c) for c in unique_classes}
        
    names_list = [class_map[c] for c in unique_classes]
    
    # Models to benchmark
    available_models = [
        'CSP_ShrinkageLDA',
        'Riemannian_TangentSpace_SVM_Linear',
        'Riemannian_TangentSpace_Ridge',
        'Riemannian_TangentSpace_LogReg',
        'FilterBank_CSP_LogReg',
        'Welch_PSD_ShrinkageLDA'
    ]
    if models is None:
        models_to_run = available_models
    elif isinstance(models, str):
        models_to_run = [models]
    else:
        models_to_run = [m for m in models if m in available_models]
        if not models_to_run:
            models_to_run = available_models

    pairs_dict = {}
    rankings = []
    
    # Matrix of pairwise accuracies (using CSP+LDA)
    n_cls = len(unique_classes)
    acc_matrix = np.full((n_cls, n_cls), 0.50, dtype=float)
    np.fill_diagonal(acc_matrix, 1.0)
    
    for c_a, c_b in itertools.combinations(unique_classes, 2):
        name_a = class_map[c_a]
        name_b = class_map[c_b]
        pair_key = f"{name_a}-{name_b}"
        pair_label = f"{name_a} vs. {name_b}"
        
        mask = (y == c_a) | (y == c_b)
        X_pair = X_data[mask]
        y_pair = y[mask]
        
        n_a = int(np.sum(y_pair == c_a))
        n_b = int(np.sum(y_pair == c_b))
        n_pair = len(y_pair)
        
        if min(n_a, n_b) < 2:
            continue
            
        y_bin = (y_pair == c_b).astype(int)
        k_folds = max(2, min(n_splits, n_a, n_b))
        cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        model_scores = {m: [] for m in models_to_run}
        preds_record = {m: np.zeros(n_pair, dtype=int) for m in models_to_run}
        
        for tr_idx, te_idx in cv.split(X_pair, y_bin):
            X_tr, X_te = X_pair[tr_idx], X_pair[te_idx]
            y_tr, y_te = y_bin[tr_idx], y_bin[te_idx]
            
            # 1. Binary CSP + Shrinkage LDA
            if 'CSP_ShrinkageLDA' in models_to_run:
                try:
                    W_csp = compute_ovr_csp(X_tr, y_tr, n_components=4)
                    f_tr_csp = project_csp_features(X_tr, W_csp)
                    f_te_csp = project_csp_features(X_te, W_csp)
                    clf_csp = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                    clf_csp.fit(f_tr_csp, y_tr)
                    p_csp = clf_csp.predict(f_te_csp)
                    model_scores['CSP_ShrinkageLDA'].append(accuracy_score(y_te, p_csp))
                    preds_record['CSP_ShrinkageLDA'][te_idx] = p_csp
                except Exception:
                    pass

            # 2. Riemannian Tangent Space + Logistic Regression
            if 'Riemannian_TangentSpace_LogReg' in models_to_run:
                try:
                    cov_tr = compute_covariance_matrices(X_tr)
                    cov_te = compute_covariance_matrices(X_te)
                    ts_tr, C_ref = project_to_riemannian_tangent_space(cov_tr)
                    ts_te, _ = project_to_riemannian_tangent_space(cov_te, C_ref=C_ref)
                    sc_ts = StandardScaler()
                    clf_ts = LogisticRegression(C=0.1, max_iter=500, random_state=42)
                    clf_ts.fit(sc_ts.fit_transform(ts_tr), y_tr)
                    p_ts = clf_ts.predict(sc_ts.transform(ts_te))
                    model_scores['Riemannian_TangentSpace_LogReg'].append(accuracy_score(y_te, p_ts))
                    preds_record['Riemannian_TangentSpace_LogReg'][te_idx] = p_ts
                except Exception:
                    pass

            # 2b. Riemannian TS + Linear SVM
            if 'Riemannian_TangentSpace_SVM_Linear' in models_to_run:
                try:
                    cov_tr = compute_covariance_matrices(X_tr)
                    cov_te = compute_covariance_matrices(X_te)
                    ts_tr, C_ref = project_to_riemannian_tangent_space(cov_tr)
                    ts_te, _ = project_to_riemannian_tangent_space(cov_te, C_ref=C_ref)
                    sc_ts = StandardScaler()
                    clf_svm_lin = SVC(C=0.1, kernel='linear', random_state=42)
                    clf_svm_lin.fit(sc_ts.fit_transform(ts_tr), y_tr)
                    p_svm_lin = clf_svm_lin.predict(sc_ts.transform(ts_te))
                    model_scores['Riemannian_TangentSpace_SVM_Linear'].append(accuracy_score(y_te, p_svm_lin))
                    preds_record['Riemannian_TangentSpace_SVM_Linear'][te_idx] = p_svm_lin
                except Exception:
                    pass

            # 2c. Riemannian TS + Ridge Classifier
            if 'Riemannian_TangentSpace_Ridge' in models_to_run:
                try:
                    cov_tr = compute_covariance_matrices(X_tr)
                    cov_te = compute_covariance_matrices(X_te)
                    ts_tr, C_ref = project_to_riemannian_tangent_space(cov_tr)
                    ts_te, _ = project_to_riemannian_tangent_space(cov_te, C_ref=C_ref)
                    sc_ts = StandardScaler()
                    clf_ridge = RidgeClassifier(alpha=10.0, random_state=42)
                    clf_ridge.fit(sc_ts.fit_transform(ts_tr), y_tr)
                    p_ridge = clf_ridge.predict(sc_ts.transform(ts_te))
                    model_scores['Riemannian_TangentSpace_Ridge'].append(accuracy_score(y_te, p_ridge))
                    preds_record['Riemannian_TangentSpace_Ridge'][te_idx] = p_ridge
                except Exception:
                    pass

            # 3. Filter Bank CSP (FBCSP) + Logistic Regression
            if 'FilterBank_CSP_LogReg' in models_to_run:
                try:
                    f_tr_fb, f_te_fb = extract_filter_bank_csp(X_tr, y_tr, X_te, sfreq=sfreq, n_components=4)
                    sc_fb = StandardScaler()
                    clf_fb = LogisticRegression(C=0.5, max_iter=500, random_state=42)
                    clf_fb.fit(sc_fb.fit_transform(f_tr_fb), y_tr)
                    p_fb = clf_fb.predict(sc_fb.transform(f_te_fb))
                    model_scores['FilterBank_CSP_LogReg'].append(accuracy_score(y_te, p_fb))
                    preds_record['FilterBank_CSP_LogReg'][te_idx] = p_fb
                except Exception:
                    pass

            # 4. Welch PSD + Shrinkage LDA
            if 'Welch_PSD_ShrinkageLDA' in models_to_run:
                try:
                    psd_tr = extract_welch_bandpower_features(X_tr, sfreq=sfreq)
                    psd_te = extract_welch_bandpower_features(X_te, sfreq=sfreq)
                    clf_psd = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                    clf_psd.fit(psd_tr, y_tr)
                    p_psd = clf_psd.predict(psd_te)
                    model_scores['Welch_PSD_ShrinkageLDA'].append(accuracy_score(y_te, p_psd))
                    preds_record['Welch_PSD_ShrinkageLDA'][te_idx] = p_psd
                except Exception:
                    pass
                    
        # Compute metrics per model
        pair_models = {}
        for m in models_to_run:
            accs = model_scores.get(m, [])
            if accs:
                cm = confusion_matrix(y_bin, preds_record[m])
                pair_models[m] = {
                    'mean_accuracy': float(np.mean(accs)),
                    'std_accuracy': float(np.std(accs)),
                    'balanced_accuracy': float(balanced_accuracy_score(y_bin, preds_record[m])),
                    'macro_f1': float(f1_score(y_bin, preds_record[m], average='macro')),
                    'cohen_kappa': float(cohen_kappa_score(y_bin, preds_record[m])),
                    'fold_accuracies': [float(a) for a in accs],
                    'predictions': [int(p) for p in preds_record[m]],
                    'confusion_matrix': cm.tolist()
                }
                
        # Find best model for this pair
        if pair_models:
            best_m = max(pair_models.keys(), key=lambda k: pair_models[k]['mean_accuracy'])
            best_acc = pair_models[best_m]['mean_accuracy']
        else:
            best_m = 'None'
            best_acc = 0.50
            
        csp_acc = pair_models.get('CSP_ShrinkageLDA', {}).get('mean_accuracy', 0.50)
        
        pair_info = {
            'pair_key': pair_key,
            'pair_label': pair_label,
            'class_a': name_a,
            'class_b': name_b,
            'class_a_id': int(c_a) if isinstance(c_a, (int, np.integer)) else str(c_a),
            'class_b_id': int(c_b) if isinstance(c_b, (int, np.integer)) else str(c_b),
            'n_samples': int(n_pair),
            'n_samples_a': int(n_a),
            'n_samples_b': int(n_b),
            'best_model': best_m,
            'best_accuracy': float(best_acc),
            'csp_lda_accuracy': float(csp_acc),
            'models': pair_models
        }
        pairs_dict[pair_key] = pair_info
        
        rankings.append({
            'pair': pair_key,
            'pair_label': pair_label,
            'n_samples': int(n_pair),
            'best_model': best_m,
            'best_accuracy': float(best_acc),
            'csp_lda_accuracy': float(csp_acc),
            'riemannian_accuracy': float(pair_models.get('Riemannian_TangentSpace_LogReg', {}).get('mean_accuracy', 0.50)),
            'riemannian_svm_lin_accuracy': float(pair_models.get('Riemannian_TangentSpace_SVM_Linear', {}).get('mean_accuracy', 0.50)),
            'riemannian_ridge_accuracy': float(pair_models.get('Riemannian_TangentSpace_Ridge', {}).get('mean_accuracy', 0.50)),
            'fbcsp_accuracy': float(pair_models.get('FilterBank_CSP_LogReg', {}).get('mean_accuracy', 0.50)),
            'welch_psd_accuracy': float(pair_models.get('Welch_PSD_ShrinkageLDA', {}).get('mean_accuracy', 0.50)),
            'csp_lda_std': float(pair_models.get('CSP_ShrinkageLDA', {}).get('std_accuracy', 0.0))
        })
        
        # Populate matrix
        i_a = unique_classes.index(c_a)
        i_b = unique_classes.index(c_b)
        acc_matrix[i_a, i_b] = csp_acc
        acc_matrix[i_b, i_a] = csp_acc

    # Sort rankings descending by best accuracy
    rankings.sort(key=lambda r: (r['best_accuracy'], r['csp_lda_accuracy']), reverse=True)
    for idx, r in enumerate(rankings):
        r['rank'] = idx + 1
        
    # Summary DataFrame
    df_rows = []
    for r in rankings:
        df_rows.append({
            'Rank': r['rank'],
            'Pair': r['pair_label'],
            'N': r['n_samples'],
            'CSP+LDA (%)': f"{r['csp_lda_accuracy']*100:.1f} ± {r['csp_lda_std']*100:.1f}",
            'Riemannian TS (%)': f"{r['riemannian_accuracy']*100:.1f}",
            'Riemannian SVM (%)': f"{r['riemannian_svm_lin_accuracy']*100:.1f}",
            'FBCSP (%)': f"{r['fbcsp_accuracy']*100:.1f}",
            'Welch PSD (%)': f"{r['welch_psd_accuracy']*100:.1f}",
            'Best Pipeline': f"{r['best_model']} ({r['best_accuracy']*100:.1f}%)"
        })
    df_summary = pd.DataFrame(df_rows)
    
    mean_csp = float(np.mean([r['csp_lda_accuracy'] for r in rankings])) if rankings else 0.50
    mean_best = float(np.mean([r['best_accuracy'] for r in rankings])) if rankings else 0.50
    
    # Formatted terminal printing
    if verbose and rankings:
        title_str = f" PAIRWISE 2-CLASS RHYTHM DECODING: {phase_name.upper()} (CHANCE: 50.0%) "
        print("\n" + "=" * 95)
        print(title_str.center(95, "="))
        print("=" * 95)
        print(f"{'Pair':<24} {'N':<5} {'CSP+LDA':<16} {'Riemannian TS':<15} {'FBCSP':<10} {'Welch PSD':<12} {'Best Pipeline'}")
        print("-" * 95)
        for r in rankings:
            csp_str = f"{r['csp_lda_accuracy']*100:.1f}% ± {r['csp_lda_std']*100:.1f}%"
            riem_str = f"{r['riemannian_accuracy']*100:.1f}%"
            fb_str = f"{r['fbcsp_accuracy']*100:.1f}%"
            psd_str = f"{r['welch_psd_accuracy']*100:.1f}%"
            best_str = f"{r['best_model']} ({r['best_accuracy']*100:.1f}%)"
            print(f"{r['pair_label']:<24} {r['n_samples']:<5} {csp_str:<16} {riem_str:<15} {fb_str:<10} {psd_str:<12} {best_str}")
        print("-" * 95)
        print(f" • Overall Mean Pairwise Accuracy (CSP+LDA) : {mean_csp*100:.2f}% (Chance: 50.0%)")
        print(f" • Overall Mean Pairwise Accuracy (Best)    : {mean_best*100:.2f}%")
        if rankings:
            print(f" • Top Separable Pair   : {rankings[0]['pair_label']} ({rankings[0]['best_accuracy']*100:.1f}% via {rankings[0]['best_model']})")
            print(f" • Lowest Separable Pair: {rankings[-1]['pair_label']} ({rankings[-1]['best_accuracy']*100:.1f}% via {rankings[-1]['best_model']})")
        print("=" * 95)
        
    # Serializable dict for JSON export
    serializable_dict = {
        'phase': phase_name,
        'chance_accuracy': 0.50,
        'mean_csp_lda_accuracy': mean_csp,
        'mean_best_accuracy': mean_best,
        'best_pair': rankings[0] if rankings else None,
        'rankings': rankings,
        'pairs': {k: {
            'pair_key': v['pair_key'],
            'pair_label': v['pair_label'],
            'class_a': v['class_a'],
            'class_b': v['class_b'],
            'n_samples': v['n_samples'],
            'best_model': v['best_model'],
            'best_accuracy': v['best_accuracy'],
            'csp_lda_accuracy': v['csp_lda_accuracy'],
            'models': {
                m_name: {
                    'mean_accuracy': m_info['mean_accuracy'],
                    'std_accuracy': m_info['std_accuracy'],
                    'balanced_accuracy': m_info['balanced_accuracy'],
                    'macro_f1': m_info['macro_f1'],
                    'cohen_kappa': m_info['cohen_kappa'],
                    'fold_accuracies': m_info['fold_accuracies']
                } for m_name, m_info in v['models'].items()
            }
        } for k, v in pairs_dict.items()}
    }

    pairwise_results = {
        'pairs': pairs_dict,
        'rankings': rankings,
        'summary_table': df_summary,
        'accuracy_matrix': acc_matrix,
        'matrix_classes': names_list,
        'mean_pairwise_accuracy': mean_best,
        'mean_csp_lda_accuracy': mean_csp,
        'best_pair': rankings[0] if rankings else None,
        'worst_pair': rankings[-1] if rankings else None,
        'chance_accuracy': 0.50,
        'phase_name': phase_name,
        'serializable': serializable_dict
    }
    
    # Provide direct item access e.g. pairwise_results['FIRE-WATER'] or ('FIRE', 'WATER')
    for k, v in pairs_dict.items():
        pairwise_results[k] = v
        pairwise_results[(v['class_a'], v['class_b'])] = v
        
    return pairwise_results


# Aliases for convenience
pairwise_comparison = evaluate_pairwise_decoding
evaluate_pairwise_comparison = evaluate_pairwise_decoding


def plot_pairwise_decoding(pairwise_results, out_path=None, title_suffix=""):
    """
    Plots publication-grade figures for pairwise rhythm decoding:
      - Panel A: Horizontal bar chart comparing all pairs against 50% chance level
      - Panel B: Pairwise separability matrix heatmap
    """
    fig, (ax_bar, ax_mat) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Left: Bar chart of pairs
    rankings = pairwise_results.get('rankings', [])
    if rankings:
        labels = [r['pair_label'] for r in rankings]
        csp_accs = [r.get('csp_lda_accuracy', 0.0) * 100 for r in rankings]
        best_accs = [r.get('best_accuracy', 0.0) * 100 for r in rankings]
        
        y_pos = np.arange(len(labels))
        height = 0.35
        
        rects1 = ax_bar.barh(y_pos - height/2, csp_accs, height, label='CSP + Shrinkage LDA', color='#3498db', edgecolor='black', alpha=0.9)
        rects2 = ax_bar.barh(y_pos + height/2, best_accs, height, label='Best Pipeline', color='#e67e22', edgecolor='black', alpha=0.9)
        
        ax_bar.axvline(50.0, color='#e74c3c', linestyle='--', linewidth=2.0, label='Chance Level (50.0%)')
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(labels, fontsize=10, fontweight='bold')
        ax_bar.invert_yaxis()  # top-down ranking
        ax_bar.set_xlabel('Decoding Accuracy (%)', fontsize=11, fontweight='bold')
        ax_bar.set_xlim(0, 100)
        ax_bar.set_title(f'Pairwise Decoding Rankings {title_suffix}', fontsize=12, fontweight='bold', pad=12)
        ax_bar.legend(loc='lower right', frameon=True, fontsize=9)
        
        for rect in rects1:
            w = rect.get_width()
            ax_bar.annotate(f'{w:.1f}%', xy=(w, rect.get_y() + rect.get_height()/2),
                            xytext=(5, 0), textcoords='offset points', ha='left', va='center', fontsize=8, fontweight='bold')
                            
    # Right: Separability heatmap
    acc_mat = pairwise_results.get('accuracy_matrix')
    class_names = pairwise_results.get('matrix_classes', ['FIRE', 'WATER', 'WIND', 'ELECTRICITY'])
    if acc_mat is not None:
        mat_pct = acc_mat * 100.0
        v_max = max(75.0, float(np.max(mat_pct[np.triu_indices(len(class_names), k=1)])) + 5.0) if len(class_names) > 1 else 100.0
        im = ax_mat.imshow(mat_pct, cmap='Blues', vmin=45.0, vmax=min(100.0, v_max))
        ax_mat.set_xticks(range(len(class_names)))
        ax_mat.set_xticklabels(class_names, fontsize=9, fontweight='bold', rotation=25)
        ax_mat.set_yticks(range(len(class_names)))
        ax_mat.set_yticklabels(class_names, fontsize=9, fontweight='bold')
        ax_mat.set_title(f'Pairwise Separability Matrix (CSP+LDA) {title_suffix}', fontsize=12, fontweight='bold', pad=12)
        
        for r in range(len(class_names)):
            for c in range(len(class_names)):
                if r == c:
                    ax_mat.text(c, r, "-", ha='center', va='center', color='gray', fontsize=12, fontweight='bold')
                else:
                    val = mat_pct[r, c]
                    ax_mat.text(c, r, f"{val:.1f}%", ha='center', va='center',
                                color='white' if val > 65.0 else 'black', fontsize=9, fontweight='bold')
                                
        fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04, label='Accuracy (%)')
        
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
    return fig


# ----------------------------------------------------------------------
# 4. Master Multi-Session Pipeline Runner
# ----------------------------------------------------------------------
def run_tower_defense_rhythm_analysis(
    bids_root="scripts/bids/bids_tower_defense,scripts/bids/bids_tower_defense_6_3_27",
    sub_id="01",
    ses_id="all",
    out_dir="scripts/analysis_results/tower_defense_recall",
    win_len_s=3.0,
    n_splits=5
):
    print("=" * 80)
    print(" BCI TOWER DEFENSE: 4-CLASS RHYTHM DECODING STUDIO (MULTI-SESSION) ".center(80, "="))
    print("=" * 80)
    
    sub_clean = sub_id.replace("sub-", "")
    
    # 1. Discover sessions (support multiple BIDS roots separated by comma)
    bids_root_list = [p.strip() for p in bids_root.split(",") if p.strip()]
    
    roots_with_sub = [b for b in bids_root_list if os.path.exists(os.path.join(b, f"sub-{sub_clean}"))]
    session_tuples = []  # (bids_dir, ses_clean, display_name)
    for b_root in bids_root_list:
        sub_dir = os.path.join(b_root, f"sub-{sub_clean}")
        if not os.path.exists(sub_dir):
            continue
        discovered = find_available_sessions(b_root, sub_clean)
        for s in discovered:
            if ses_id == "all" or s in [x.strip().replace("ses-", "") for x in ses_id.split(",")]:
                b_name = os.path.basename(os.path.normpath(b_root))
                disp = f"{b_name}_ses-{s}" if len(roots_with_sub) > 1 else f"ses-{s}"
                session_tuples.append((b_root, s, disp))
                
    if not session_tuples:
        raise FileNotFoundError(f"No matching sessions found in {bids_root} for sub-{sub_clean}")
        
    print(f"[*] Subject: sub-{sub_clean}")
    print(f"[*] BIDS Roots ({len(bids_root_list)}): {bids_root_list}")
    print(f"[*] Target Sessions for Analysis ({len(session_tuples)}): {[t[2] for t in session_tuples]}")
    
    # Determine output folder
    if len(session_tuples) == 1:
        session_out_dir = os.path.join(out_dir, f"sub-{sub_clean}_{session_tuples[0][2]}")
    else:
        session_out_dir = os.path.join(out_dir, f"sub-{sub_clean}_pooled_{len(session_tuples)}sessions")
    os.makedirs(session_out_dir, exist_ok=True)
    
    # 2. Load & Pool Data across all target sessions
    all_X_im, all_X_lis, all_X_blk, all_y, all_session_ids = [], [], [], [], []
    class_names = ['FIRE', 'WATER', 'WIND', 'ELECTRICITY']
    sfreq = 250.0
    
    for b_root, ses, disp in session_tuples:
        print(f"\n---> Loading & Preprocessing Session {disp} ({b_root})...")
        raw_uv, df_events, sfreq, ch_names = load_single_session_raw(b_root, sub_clean, ses)
        clean_eeg = preprocess_continuous_eeg(raw_uv, sfreq=sfreq, l_freq=1.0, h_freq=45.0, notch_freq=50.0, spatial_mode="robust_car", ch_names=ch_names)
        X_im, X_lis, X_blk, y, df_meta, _ = extract_session_epochs(clean_eeg, df_events, ses_id=disp, sfreq=sfreq, win_len_s=win_len_s)
        
        print(f"     [+] {disp}: Extracted {len(y)} trials {dict(pd.Series(y).value_counts())}")
        all_X_im.append(X_im)
        all_X_lis.append(X_lis)
        all_X_blk.append(X_blk)
        all_y.append(y)
        all_session_ids.extend([disp] * len(y))
        
    X_im_pooled = np.concatenate(all_X_im, axis=0)
    has_listening_data = (len(all_X_lis) > 0 and all(len(x) > 0 for x in all_X_lis) and sum(len(x) for x in all_X_lis) == len(X_im_pooled))
    X_lis_pooled = np.concatenate(all_X_lis, axis=0) if has_listening_data else None
    has_blinking_data = (len(all_X_blk) > 0 and all(len(x) > 0 for x in all_X_blk) and sum(len(x) for x in all_X_blk) == len(X_im_pooled))
    X_blk_pooled = np.concatenate(all_X_blk, axis=0) if has_blinking_data else None
    y_pooled = np.concatenate(all_y, axis=0)
    session_ids_arr = np.array(all_session_ids)
    
    print("\n" + "=" * 80)
    print(f" TOTAL POOLED DATASET SUMMARY: {len(y_pooled)} TRIALS ACROSS {len(session_tuples)} SESSION(S) ".center(80, "="))
    print(f" Class Breakdown: {dict(pd.Series(y_pooled).value_counts())}")
    print(f" Epoch Shape: {X_im_pooled.shape}")
    print(f" Listening Available: {has_listening_data} | Blinking Available: {has_blinking_data}")
    print("=" * 80)
    
    # 3. Stratified K-Fold CV Benchmark on Pooled Dataset
    print("\n[*] Benchmarking Mental Imagery (Imagine Phase) Decoders...")
    summary_im, preds_im = evaluate_phase_decoding(X_im_pooled, y_pooled, sfreq=sfreq, n_splits=n_splits)
    
    if has_listening_data:
        print("[*] Benchmarking Auditory Perception (Listen Phase) Decoders...")
        summary_lis, preds_lis = evaluate_phase_decoding(X_lis_pooled, y_pooled, sfreq=sfreq, n_splits=n_splits)
    else:
        print("[*] Auditory Perception (Listen Phase): Skipped (not present in this dataset)")
        summary_lis, preds_lis = None, None
    
    if has_blinking_data:
        print("[*] Benchmarking Visual Flicker (Blinking Phase) Decoders...")
        summary_blk, preds_blk = evaluate_phase_decoding(X_blk_pooled, y_pooled, sfreq=sfreq, n_splits=n_splits)
    else:
        summary_blk, preds_blk = None, None
    
    # 3b. Pairwise 2-Class Rhythm Decoding (Mental Imagery & Auditory Perception)
    print("\n[*] Benchmarking Pairwise 2-Class Rhythm Decoding (Mental Imagery)...")
    pairwise_im = evaluate_pairwise_decoding(
        X_im_pooled, y_pooled, class_names=class_names, sfreq=sfreq, n_splits=n_splits, phase_name="Mental Imagery (Imagine)"
    )
    
    if has_listening_data:
        print("\n[*] Benchmarking Pairwise 2-Class Rhythm Decoding (Auditory Perception)...")
        pairwise_lis = evaluate_pairwise_decoding(
            X_lis_pooled, y_pooled, class_names=class_names, sfreq=sfreq, n_splits=n_splits, phase_name="Auditory Perception (Listen)"
        )
    else:
        pairwise_lis = None
    
    # 4. Leave-One-Session-Out Cross-Validation (if > 1 session)
    loso_results = None
    if len(session_tuples) > 1:
        print("\n[*] Computing Leave-One-Session-Out (LOSO) Cross-Validation...")
        loso_results = evaluate_loso_cross_validation(X_im_pooled, y_pooled, session_ids_arr, sfreq=sfreq)
        if loso_results:
            for mod, sc_list in loso_results.items():
                mean_loso = np.mean([s['accuracy'] for s in sc_list])
                print(f"    - LOSO {mod:25s}: Mean Acc = {mean_loso*100:.2f}% | Folds: {[round(s['accuracy']*100,1) for s in sc_list]}")
                
    # 5. Cross-Condition Transfer
    if has_listening_data:
        print("\n[*] Computing Cross-Condition Transfer (Train: Listen -> Test: Imagine)...")
        transfer_summary = evaluate_cross_condition_transfer(X_lis_pooled, X_im_pooled, y_pooled, sfreq=sfreq)
        transfer_acc = transfer_summary['CSP_ShrinkageLDA_ListenToImagine']['accuracy'] * 100.0
        print(f"    [+] Zero-Shot Transfer Accuracy (Listen -> Imagine): {transfer_acc:.2f}% (Chance: 25.0%)")
    else:
        transfer_summary = None
        transfer_acc = 0.0
    
    # 6. RSA Alignment
    if has_listening_data:
        print("[*] Computing Representational Similarity Analysis (RSA)...")
        rdm_lis, rdm_im, rsa_rho, rsa_p = compute_representational_similarity(X_lis_pooled, X_im_pooled, y_pooled, class_names)
        print(f"    [+] RSA Spearman Correlation: rho = {rsa_rho:.3f} (p = {rsa_p:.4f})")
        
        # Plot and save RDM figure
        fig_rsa, axes_rsa = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)
        im0 = axes_rsa[0].imshow(rdm_lis, cmap='viridis', interpolation='nearest')
        axes_rsa[0].set_title("Auditory Perception (Listen RDM)", fontsize=11, fontweight='bold')
        axes_rsa[0].set_xticks(range(len(class_names)))
        axes_rsa[0].set_xticklabels(class_names, rotation=25, fontweight='bold')
        axes_rsa[0].set_yticks(range(len(class_names)))
        axes_rsa[0].set_yticklabels(class_names, fontweight='bold')
        for r in range(len(class_names)):
            for c in range(len(class_names)):
                axes_rsa[0].text(c, r, f"{rdm_lis[r, c]:.2f}", ha="center", va="center", color="white" if rdm_lis[r, c] < 0.5 else "black", fontsize=9, fontweight='bold')
        fig_rsa.colorbar(im0, ax=axes_rsa[0], fraction=0.046, pad=0.04)

        im1 = axes_rsa[1].imshow(rdm_im, cmap='viridis', interpolation='nearest')
        axes_rsa[1].set_title(f"Mental Imagery (Imagine RDM)\nSpearman rho = {rsa_rho:.3f} (p = {rsa_p:.4f})", fontsize=11, fontweight='bold')
        axes_rsa[1].set_xticks(range(len(class_names)))
        axes_rsa[1].set_xticklabels(class_names, rotation=25, fontweight='bold')
        axes_rsa[1].set_yticks(range(len(class_names)))
        axes_rsa[1].set_yticklabels(class_names, fontweight='bold')
        for r in range(len(class_names)):
            for c in range(len(class_names)):
                axes_rsa[1].text(c, r, f"{rdm_im[r, c]:.2f}", ha="center", va="center", color="white" if rdm_im[r, c] < 0.5 else "black", fontsize=9, fontweight='bold')
        fig_rsa.colorbar(im1, ax=axes_rsa[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig_rsa.savefig(os.path.join(session_out_dir, "rsa_perception_imagery_rdm.png"), bbox_inches='tight')
        plt.close(fig_rsa)
    else:
        rsa_rho, rsa_p = None, None
    
    # 7. Generate Figures
    print("\n[*] Exporting Publication-Grade Figures & Visualizations...")
    
    # Benchmark Plot
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    models_plot = [
        ('CSP + Shrinkage LDA', 'CSP_ShrinkageLDA'),
        ('Filter Bank CSP (FBCSP)', 'FilterBank_CSP_LogReg'),
        ('Riemannian TS (Linear SVM)', 'Riemannian_TangentSpace_SVM_Linear'),
        ('Riemannian TS (LogReg)', 'Riemannian_TangentSpace_LogReg'),
        ('Welch PSD + LDA', 'Welch_PSD_ShrinkageLDA'),
        ('Ensemble Soft Voting', 'Ensemble_Voting')
    ]
    x = np.arange(len(models_plot))
    width = 0.22 if (has_listening_data and has_blinking_data) else 0.35
    im_means = [summary_im[k]['mean_accuracy'] * 100 for _, k in models_plot]
    im_stds = [summary_im[k]['std_accuracy'] * 100 for _, k in models_plot]
    
    if has_listening_data and has_blinking_data:
        lis_means = [summary_lis[k]['mean_accuracy'] * 100 for _, k in models_plot]
        lis_stds = [summary_lis[k]['std_accuracy'] * 100 for _, k in models_plot]
        blk_means = [summary_blk[k]['mean_accuracy'] * 100 for _, k in models_plot]
        blk_stds = [summary_blk[k]['std_accuracy'] * 100 for _, k in models_plot]
        
        rects1 = ax.bar(x - width, im_means, width, yerr=im_stds, label='Mental Imagery (Imagine)', color='#e74c3c', alpha=0.9, capsize=4, edgecolor='black', linewidth=0.8)
        rects2 = ax.bar(x, lis_means, width, yerr=lis_stds, label='Auditory Perception (Listen)', color='#3498db', alpha=0.9, capsize=4, edgecolor='black', linewidth=0.8)
        rects3 = ax.bar(x + width, blk_means, width, yerr=blk_stds, label='Visual Flicker (Blinking)', color='#2ecc71', alpha=0.9, capsize=4, edgecolor='black', linewidth=0.8)
        all_rects = rects1 + rects2 + rects3
    elif has_blinking_data:
        blk_means = [summary_blk[k]['mean_accuracy'] * 100 for _, k in models_plot]
        blk_stds = [summary_blk[k]['std_accuracy'] * 100 for _, k in models_plot]
        rects1 = ax.bar(x - width/2, im_means, width, yerr=im_stds, label='Mental Imagery (Imagine)', color='#e74c3c', alpha=0.9, capsize=4, edgecolor='black', linewidth=0.8)
        rects2 = ax.bar(x + width/2, blk_means, width, yerr=blk_stds, label='Visual Flicker (Blinking)', color='#2ecc71', alpha=0.9, capsize=4, edgecolor='black', linewidth=0.8)
        all_rects = rects1 + rects2
    else:
        rects1 = ax.bar(x, im_means, width, yerr=im_stds, label='Mental Imagery (Imagine)', color='#e74c3c', alpha=0.9, capsize=4, edgecolor='black', linewidth=0.8)
        all_rects = rects1
        
    ax.axhline(25.0, color='#e67e22', linestyle='--', linewidth=2.0, label='Chance Level (25.0%)')
    if transfer_acc > 0:
        ax.axhline(transfer_acc, color='#9b59b6', linestyle=':', linewidth=2.0, label=f'Transfer (Listen->Imagine): {transfer_acc:.1f}%')
    ax.set_ylabel('Decoding Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'BCI Tower Defense 4-Class Rhythm Decoding Performance\n(sub-{sub_clean} | {len(y_pooled)} Pooled Trials across {len(session_tuples)} Session(s))', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in models_plot], fontsize=9, fontweight='bold', rotation=15, ha='right')
    ax.set_ylim(0, 75)
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    for rect in all_rects:
        h = rect.get_height()
        if h > 5:
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(session_out_dir, "rhythm_decoding_benchmark.png"))
    plt.close(fig)
    
    # Confusion Matrices
    if has_listening_data and transfer_summary:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300)
        cms = [
            ('Mental Imagery (Imagine)', confusion_matrix(y_pooled, preds_im['CSP_ShrinkageLDA'])),
            ('Auditory Perception (Listen)', confusion_matrix(y_pooled, preds_lis['CSP_ShrinkageLDA'])),
            ('Transfer (Train Listen -> Test Imagine)', np.array(transfer_summary['CSP_ShrinkageLDA_ListenToImagine']['confusion_matrix']))
        ]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6.5, 5.2), dpi=300)
        axes = [axes]
        cms = [
            ('Mental Imagery (Imagine)', confusion_matrix(y_pooled, preds_im['CSP_ShrinkageLDA']))
        ]
        
    for ax, (title, cm) in zip(axes, cms):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=60)
        acc = np.trace(cm) / np.sum(cm) * 100.0
        ax.set_title(f'{title}\nAcc: {acc:.2f}% (Chance: 25%)', fontsize=11, fontweight='bold', pad=10)
        ax.set_xticks(range(4))
        ax.set_xticklabels(class_names, fontsize=9, fontweight='bold', rotation=25)
        ax.set_yticks(range(4))
        ax.set_yticklabels(class_names, fontsize=9, fontweight='bold')
        ax.set_ylabel('True Element', fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted Element', fontsize=10, fontweight='bold')
        for r in range(4):
            for c in range(4):
                ax.text(c, r, f"{cm[r, c]}\n({cm_norm[r, c]:.1f}%)", ha="center", va="center", color="white" if cm_norm[r, c] > 30.0 else "black", fontsize=9, fontweight='bold')
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label='Class Accuracy (%)')
    fig.savefig(os.path.join(session_out_dir, "confusion_matrices_all_phases.png"), bbox_inches='tight')
    plt.close(fig)
    
    # Pairwise Decoding Visualizations & CSV Reports
    plot_pairwise_decoding(
        pairwise_im,
        out_path=os.path.join(session_out_dir, "pairwise_rhythm_decoding_imagery.png"),
        title_suffix="(Mental Imagery)"
    )
    pairwise_im['summary_table'].to_csv(os.path.join(session_out_dir, "pairwise_decoding_imagery.csv"), index=False)
    
    if pairwise_lis is not None:
        plot_pairwise_decoding(
            pairwise_lis,
            out_path=os.path.join(session_out_dir, "pairwise_rhythm_decoding_perception.png"),
            title_suffix="(Auditory Perception)"
        )
        pairwise_lis['summary_table'].to_csv(os.path.join(session_out_dir, "pairwise_decoding_perception.csv"), index=False)
    
    # Master JSON Summary Export
    target_names = [t[2] for t in session_tuples]
    master_summary = {
        'subject': sub_clean,
        'sessions_analyzed': target_names,
        'n_sessions': len(session_tuples),
        'n_total_trials': len(y_pooled),
        'class_names': class_names,
        'chance_accuracy': 0.25,
        'imagine_decoding': summary_im,
        'listen_decoding': summary_lis,
        'blinking_decoding': summary_blk,
        'pairwise_decoding': {
            'imagine': pairwise_im['serializable'],
            'listen': pairwise_lis['serializable'] if pairwise_lis else None
        },
        'transfer_decoding': transfer_summary,
        'loso_decoding': loso_results,
        'rsa_alignment': {'spearman_rho': rsa_rho, 'spearman_p': rsa_p} if rsa_rho is not None else None
    }
    
    json_path = os.path.join(session_out_dir, "rhythm_decoding_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(master_summary, f, indent=2)
    print(f"[+] Exported Master JSON Summary: {json_path}")
    
    # Export Benchmark Metrics CSV
    metrics_rows = []
    for m_name, m_stats in summary_im.items():
        metrics_rows.append({
            'Model': m_name,
            'Phase': 'Mental Imagery',
            'accuracy_mean': m_stats.get('mean_accuracy', 0.0),
            'accuracy_std': m_stats.get('std_accuracy', 0.0),
            'balanced_acc': m_stats.get('balanced_accuracy', 0.0),
            'f1_macro': m_stats.get('macro_f1', 0.0),
            'cohen_kappa': m_stats.get('cohen_kappa', 0.0),
            'accuracy_pct': m_stats.get('mean_accuracy', 0.0) * 100.0,
            'std_pct': m_stats.get('std_accuracy', 0.0) * 100.0
        })
    if summary_lis is not None:
        for m_name, m_stats in summary_lis.items():
            metrics_rows.append({
                'Model': m_name,
                'Phase': 'Auditory Perception',
                'accuracy_mean': m_stats.get('mean_accuracy', 0.0),
                'accuracy_std': m_stats.get('std_accuracy', 0.0),
                'balanced_acc': m_stats.get('balanced_accuracy', 0.0),
                'f1_macro': m_stats.get('macro_f1', 0.0),
                'cohen_kappa': m_stats.get('cohen_kappa', 0.0),
                'accuracy_pct': m_stats.get('mean_accuracy', 0.0) * 100.0,
                'std_pct': m_stats.get('std_accuracy', 0.0) * 100.0
            })
    if summary_blk is not None:
        for m_name, m_stats in summary_blk.items():
            metrics_rows.append({
                'Model': m_name,
                'Phase': 'Visual Blinking',
                'accuracy_mean': m_stats.get('mean_accuracy', 0.0),
                'accuracy_std': m_stats.get('std_accuracy', 0.0),
                'balanced_acc': m_stats.get('balanced_accuracy', 0.0),
                'f1_macro': m_stats.get('macro_f1', 0.0),
                'cohen_kappa': m_stats.get('cohen_kappa', 0.0),
                'accuracy_pct': m_stats.get('mean_accuracy', 0.0) * 100.0,
                'std_pct': m_stats.get('std_accuracy', 0.0) * 100.0
            })
    pd.DataFrame(metrics_rows).to_csv(os.path.join(session_out_dir, "models_benchmark_metrics.csv"), index=False)
    print(f"[+] Exported Benchmark Metrics CSV: {os.path.join(session_out_dir, 'models_benchmark_metrics.csv')}")
    
    print("\n" + "=" * 80)
    print(" ANALYSIS COMPLETED SUCCESSFULLY! ".center(80, "="))
    print(f" Output directory: {session_out_dir} ".center(80, " "))
    print("=" * 80)
    return master_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tower Defense 4-Class Rhythm BCI Decoding Analysis Studio")
    parser.add_argument("--bids-root", type=str, default="scripts/bids/bids_tower_defense,scripts/bids/bids_tower_defense_6_3_27", help="Path to BIDS dataset")
    parser.add_argument("--sub", type=str, default="01", help="Subject ID (e.g., '01')")
    parser.add_argument("--ses", type=str, default="all", help="Session ID ('all', '01', '01,02,03'...)")
    parser.add_argument("--out-dir", type=str, default="scripts/analysis_results/tower_defense_recall", help="Output directory")
    parser.add_argument("--win-len", type=float, default=3.0, help="Epoch duration in seconds")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    
    args = parser.parse_args()
    run_tower_defense_rhythm_analysis(
        bids_root=args.bids_root,
        sub_id=args.sub,
        ses_id=args.ses,
        out_dir=args.out_dir,
        win_len_s=args.win_len,
        n_splits=args.n_splits
    )
