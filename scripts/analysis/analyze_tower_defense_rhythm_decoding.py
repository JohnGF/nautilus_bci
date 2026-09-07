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
  - Leave-One-Session-Out Cross-Validation (LOSO-CV) across recording blocks
  - Perception-to-Imagery Zero-Shot Transfer Learning (`Listen` -> `Imagine`)
  - Representational Similarity Analysis (RSA) and sliding-window temporal trajectory
"""

import os
import sys
import glob
import json
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
from sklearn.linear_model import LogisticRegression
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
    events_list = df_events.to_dict('records')
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

    # 2. Standard Tower Defense session with trials
    for i, ev in enumerate(events_list):
        tt = str(ev.get('trial_type', ''))
        if 'selected' in tt:
            element = tt.replace(' selected', '').strip()
            if element in class_map:
                cls_id = class_map[element]
                
                listen_s = None
                blink_s = None
                for j in range(max(0, i - 6), i):
                    prev_tt = events_list[j].get('trial_type', '')
                    if prev_tt == 'Start Listen':
                        listen_s = int(events_list[j]['sample'])
                    elif prev_tt == 'Box start blinking':
                        blink_s = int(events_list[j]['sample'])
                        
                imagine_s = int(ev['sample'])
                im_start = int(imagine_s + 0.25 * sfreq)
                lis_start = int(listen_s + 0.50 * sfreq) if listen_s else 0
                blk_start = int(blink_s + 0.25 * sfreq) if blink_s else 0
                
                if (im_start + n_samples_win <= len(clean_eeg) and 
                    lis_start + n_samples_win <= len(clean_eeg) and 
                    blk_start + n_samples_win <= len(clean_eeg)):
                    
                    epochs_im.append(clean_eeg[im_start : im_start + n_samples_win, :].T)
                    epochs_lis.append(clean_eeg[lis_start : lis_start + n_samples_win, :].T)
                    epochs_blk.append(clean_eeg[blk_start : blk_start + n_samples_win, :].T)
                    labels.append(cls_id)
                    meta.append({
                        'session': str(ses_id),
                        'element': element,
                        'class_id': cls_id,
                        'imagine_sample': imagine_s,
                        'listen_sample': listen_s
                    })
                    
    return np.array(epochs_im), np.array(epochs_lis), np.array(epochs_blk), np.array(labels), pd.DataFrame(meta), class_names


# ----------------------------------------------------------------------
# 2. Algorithmic Feature Extractors
# ----------------------------------------------------------------------
def compute_ovr_csp(X, y, n_components=4):
    """One-vs-Rest CSP spatial filters."""
    n_epochs, n_ch, _ = X.shape
    classes = np.unique(y)
    covs = [np.cov(X[i]) / (np.trace(np.cov(X[i])) + 1e-12) for i in range(n_epochs)]
    
    filters = []
    for c_id in classes:
        mask = (y == c_id)
        cov_target = np.mean([covs[k] for k in range(len(covs)) if mask[k]], axis=0) + 1e-5 * np.eye(n_ch)
        cov_rest = np.mean([covs[k] for k in range(len(covs)) if not mask[k]], axis=0) + 1e-5 * np.eye(n_ch)
        vals, vecs = eigh(cov_target, cov_target + cov_rest)
        half = max(1, n_components // 2)
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


# ----------------------------------------------------------------------
# 4. Master Multi-Session Pipeline Runner
# ----------------------------------------------------------------------
def run_tower_defense_rhythm_analysis(
    bids_root="scripts/bids/bids_tower_defense",
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
    
    session_tuples = []  # (bids_dir, ses_clean, display_name)
    for b_root in bids_root_list:
        sub_dir = os.path.join(b_root, f"sub-{sub_clean}")
        if not os.path.exists(sub_dir):
            continue
        discovered = find_available_sessions(b_root, sub_clean)
        for s in discovered:
            if ses_id == "all" or s in [x.strip().replace("ses-", "") for x in ses_id.split(",")]:
                b_name = os.path.basename(os.path.normpath(b_root))
                disp = f"{b_name}_ses-{s}" if len(bids_root_list) > 1 else f"ses-{s}"
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
    X_lis_pooled = np.concatenate(all_X_lis, axis=0)
    X_blk_pooled = np.concatenate(all_X_blk, axis=0)
    y_pooled = np.concatenate(all_y, axis=0)
    session_ids_arr = np.array(all_session_ids)
    
    print("\n" + "=" * 80)
    print(f" TOTAL POOLED DATASET SUMMARY: {len(y_pooled)} TRIALS ACROSS {len(session_tuples)} SESSION(S) ".center(80, "="))
    print(f" Class Breakdown: {dict(pd.Series(y_pooled).value_counts())}")
    print(f" Epoch Shape: {X_im_pooled.shape}")
    print("=" * 80)
    
    # 3. Stratified K-Fold CV Benchmark on Pooled Dataset
    print("\n[*] Benchmarking Mental Imagery (Imagine Phase) Decoders...")
    summary_im, preds_im = evaluate_phase_decoding(X_im_pooled, y_pooled, sfreq=sfreq, n_splits=n_splits)
    
    print("[*] Benchmarking Auditory Perception (Listen Phase) Decoders...")
    summary_lis, preds_lis = evaluate_phase_decoding(X_lis_pooled, y_pooled, sfreq=sfreq, n_splits=n_splits)
    
    print("[*] Benchmarking Visual Flicker (Blinking Phase) Decoders...")
    summary_blk, preds_blk = evaluate_phase_decoding(X_blk_pooled, y_pooled, sfreq=sfreq, n_splits=n_splits)
    
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
    print("\n[*] Computing Cross-Condition Transfer (Train: Listen -> Test: Imagine)...")
    transfer_summary = evaluate_cross_condition_transfer(X_lis_pooled, X_im_pooled, y_pooled, sfreq=sfreq)
    transfer_acc = transfer_summary['CSP_ShrinkageLDA_ListenToImagine']['accuracy'] * 100.0
    print(f"    [+] Zero-Shot Transfer Accuracy (Listen -> Imagine): {transfer_acc:.2f}% (Chance: 25.0%)")
    
    # 6. RSA Alignment
    print("[*] Computing Representational Similarity Analysis (RSA)...")
    rdm_lis, rdm_im, rsa_rho, rsa_p = compute_representational_similarity(X_lis_pooled, X_im_pooled, y_pooled, class_names)
    print(f"    [+] RSA Spearman Correlation: rho = {rsa_rho:.3f} (p = {rsa_p:.4f})")
    
    # 7. Generate Figures
    print("\n[*] Exporting Publication-Grade Figures & Visualizations...")
    
    # Benchmark Plot
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    models_plot = [
        ('CSP + Shrinkage LDA', 'CSP_ShrinkageLDA'),
        ('Filter Bank CSP (FBCSP)', 'FilterBank_CSP_LogReg'),
        ('Riemannian Tangent Space', 'Riemannian_TangentSpace_LogReg'),
        ('Welch PSD + Random Forest', 'Welch_PSD_RandomForest'),
        ('Ensemble Soft Voting', 'Ensemble_Voting')
    ]
    x = np.arange(len(models_plot))
    width = 0.22
    im_means = [summary_im[k]['mean_accuracy'] * 100 for _, k in models_plot]
    im_stds = [summary_im[k]['std_accuracy'] * 100 for _, k in models_plot]
    lis_means = [summary_lis[k]['mean_accuracy'] * 100 for _, k in models_plot]
    lis_stds = [summary_lis[k]['std_accuracy'] * 100 for _, k in models_plot]
    blk_means = [summary_blk[k]['mean_accuracy'] * 100 for _, k in models_plot]
    blk_stds = [summary_blk[k]['std_accuracy'] * 100 for _, k in models_plot]
    
    rects1 = ax.bar(x - width, im_means, width, yerr=im_stds, label='Mental Imagery (Imagine)', color='#e74c3c', alpha=0.9, capsize=4, edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x, lis_means, width, yerr=lis_stds, label='Auditory Perception (Listen)', color='#3498db', alpha=0.9, capsize=4, edgecolor='black', linewidth=0.8)
    rects3 = ax.bar(x + width, blk_means, width, yerr=blk_stds, label='Visual Flicker (Blinking)', color='#2ecc71', alpha=0.9, capsize=4, edgecolor='black', linewidth=0.8)
    ax.axhline(25.0, color='#e67e22', linestyle='--', linewidth=2.0, label='Chance Level (25.0%)')
    ax.axhline(transfer_acc, color='#9b59b6', linestyle=':', linewidth=2.0, label=f'Transfer (Listen->Imagine): {transfer_acc:.1f}%')
    ax.set_ylabel('Decoding Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'BCI Tower Defense 4-Class Rhythm Decoding Performance\n(sub-{sub_clean} | {len(y_pooled)} Pooled Trials across {len(session_tuples)} Session(s))', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in models_plot], fontsize=10, fontweight='bold', rotation=15, ha='right')
    ax.set_ylim(0, 65)
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    for rect in rects1 + rects2 + rects3:
        h = rect.get_height()
        if h > 5:
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(session_out_dir, "rhythm_decoding_benchmark.png"))
    plt.close(fig)
    
    # Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300)
    cms = [
        ('Mental Imagery (Imagine)', confusion_matrix(y_pooled, preds_im['CSP_ShrinkageLDA'])),
        ('Auditory Perception (Listen)', confusion_matrix(y_pooled, preds_lis['CSP_ShrinkageLDA'])),
        ('Transfer (Train Listen -> Test Imagine)', np.array(transfer_summary['CSP_ShrinkageLDA_ListenToImagine']['confusion_matrix']))
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
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04, label='Class Accuracy (%)')
    fig.savefig(os.path.join(session_out_dir, "confusion_matrices_all_phases.png"), bbox_inches='tight')
    plt.close(fig)
    
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
        'transfer_decoding': transfer_summary,
        'loso_decoding': loso_results,
        'rsa_alignment': {'spearman_rho': rsa_rho, 'spearman_p': rsa_p}
    }
    
    json_path = os.path.join(session_out_dir, "rhythm_decoding_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(master_summary, f, indent=2)
    print(f"[+] Exported Master JSON Summary: {json_path}")
    
    print("\n" + "=" * 80)
    print(" ANALYSIS COMPLETED SUCCESSFULLY! ".center(80, "="))
    print(f" Output directory: {session_out_dir} ".center(80, " "))
    print("=" * 80)
    return master_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tower Defense 4-Class Rhythm BCI Decoding Analysis Studio")
    parser.add_argument("--bids-root", type=str, default="scripts/bids/bids_tower_defense", help="Path to BIDS dataset")
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
