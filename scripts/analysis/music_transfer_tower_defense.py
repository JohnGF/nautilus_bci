"""
Music-Guided Transfer Learning & Domain Adaptation for Tower Defense BCI
========================================================================
Integrates full-length music listening EEG (scripts/bids_music/sub-01/ses-02) 
with Tower Defense recall EEG (scripts/bids_tower_defense) using:
  1. Riemannian Reference Whitening (C_ref from listening)
  2. Regularized Perception-Guided CSP (RCSP: blending imagery & listening covariances)
  3. Pretrained Deep Learning Transfer (EEGNet pre-trained on listening -> fine-tuned on recall)
  4. Manifold Feature Concatenation & Auditory Filterbanks
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_script_dir, ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.decoding import CSP

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

import pyriemann
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm, sqrtm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from spatial_filters import detect_bad_channels, apply_spatial_filter


def load_and_preprocess_raw(bids_root, sub, ses, task):
    """Loads BIDS EEG raw, strips misc channels, filters properly with CAR."""
    bp = BIDSPath(subject=sub, session=ses, task=task, datatype="eeg", root=bids_root)
    raw = read_raw_bids(bp, verbose=False)
    raw.load_data()
    
    misc = [ch for ch in raw.ch_names if ch.upper() in ['BATTERY', 'STATUS', 'AUX'] or ch == 'EEG033']
    if misc:
        raw.set_channel_types({ch: 'misc' for ch in misc})
    raw.pick('eeg')
    
    # 1. Bandpass filter 1-45 Hz & Notch 50 Hz first (removes DC drift)
    raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)
    raw.notch_filter(freqs=50.0, verbose=False)
    
    # 2. Bad channel detection on filtered data
    data_arr = raw.get_data().T
    stds = np.std(data_arr, axis=0)
    scale = 1e6 if np.mean(stds) < 1e-3 else 1.0
    data_uv = data_arr * scale
    
    bad_idx, bad_dict = detect_bad_channels(data_uv, ch_names=raw.ch_names)
    # Apply robust CAR
    filt_uv = apply_spatial_filter(data_uv, raw.ch_names, mode="robust_car")
    raw._data = (filt_uv / scale).T
    
    return raw, bp


def extract_music_listening_segments(music_bids_root="scripts/bids_music", sub="01", ses="02", win_len_s=3.0, step_s=1.0):
    """
    Extracts sliding window epochs from continuous music listening session.
    Track IDs:
      0: Bach Prelude (Element: FIRE / 0)
      1: Beethoven Fur Elise (Element: WATER / 1)
      2: Joplin Entertainer (Element: WIND / 2)
      3: Mozart Eine Kleine (Element: ELECTRICITY / 3)
    """
    raw_music, bp = load_and_preprocess_raw(music_bids_root, sub, ses, "musiclistening")
    sfreq = raw_music.info['sfreq']
    
    events_tsv = os.path.join(bp.directory, f"sub-{sub}_ses-{ses}_task-musiclistening_events.tsv")
    df_events = pd.read_csv(events_tsv, sep='\t')
    
    # Identify track start/ends
    track_ranges = {
        0: (48.98, 182.068),    # Bach Prelude -> Class 0 (FIRE)
        1: (187.068, 416.464),  # Beethoven Fur Elise -> Class 1 (WATER)
        2: (421.48, 664.784),   # Joplin Entertainer -> Class 2 (WIND)
        3: (669.78, 1008.256)   # Mozart Eine Kleine -> Class 3 (ELECTRICITY)
    }
    
    music_epochs = []
    music_labels = []
    raw_data = raw_music.get_data() * 1e6  # (n_ch, n_samples) in uV
    
    win_samples = int(win_len_s * sfreq)
    step_samples = int(step_s * sfreq)
    
    for class_id, (t_start, t_end) in track_ranges.items():
        s_start = int((t_start + 2.0) * sfreq)  # skip first 2s transition
        s_end = int((t_end - 2.0) * sfreq)      # skip last 2s
        for s in range(s_start, s_end - win_samples, step_samples):
            chunk = raw_data[:, s:s + win_samples]
            music_epochs.append(chunk)
            music_labels.append(class_id)
            
    X_music = np.array(music_epochs)  # (N, n_ch, win_samples)
    y_music = np.array(music_labels)
    print(f"[+] Extracted Music Listening Segments: {X_music.shape[0]} windows across 4 tracks (Shape: {X_music.shape})")
    return X_music, y_music, raw_music


def parse_td_recall_events(events_tsv, sfreq):
    df = pd.read_csv(events_tsv, sep='\t')
    class_map = {'FIRE selected': 0, 'WATER selected': 1, 'WIND selected': 2, 'ELECTRICITY selected': 3}
    events_list = df.to_dict('records')
    
    recall_events = []
    for ev in events_list:
        label = str(ev.get('trial_type', '')).strip()
        if label in class_map:
            cid = class_map[label]
            sample = int(ev['sample']) if 'sample' in ev and not np.isnan(ev['sample']) else int(ev['onset'] * sfreq)
            recall_events.append([sample, 0, cid])
            
    return np.array(recall_events), class_map


class EEGNetTransfer(nn.Module):
    def __init__(self, n_channels=32, n_samples=750, n_classes=4, F1=8, D=2, F2=16, kernel_length=32, dropout_rate=0.25):
        super(EEGNetTransfer, self).__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        # Compute flatten dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            out = self.drop1(self.pool1(self.act1(self.bn2(self.depthwise(self.bn1(self.conv1(dummy)))))))
            out = self.drop2(self.pool2(self.act2(self.bn3(self.separable(out)))))
            self.flatten_dim = out.view(1, -1).size(1)

        self.classifier = nn.Linear(self.flatten_dim, n_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.view(x.size(0), -1)
        return self.classifier(x)


def run_comparison_benchmark(td_bids_root="scripts/bids_tower_defense", music_bids_root="scripts/bids_music", ses="01", win_len_s=3.0):
    print("=" * 80)
    print(" MUSIC-GUIDED TRANSFER LEARNING & DOMAIN ADAPTATION BENCHMARK ".center(80, "="))
    print("=" * 80)
    
    # 1. Load Music Listening Session Data
    X_music, y_music, raw_music = extract_music_listening_segments(music_bids_root, sub="01", ses="02", win_len_s=win_len_s, step_s=1.0)
    
    # 2. Load Tower Defense Recall Data
    raw_td, bp_td = load_and_preprocess_raw(td_bids_root, "01", ses, "recall")
    sfreq = raw_td.info['sfreq']
    
    events_tsv = os.path.join(bp_td.directory, f"sub-01_ses-{ses}_task-recall_events.tsv")
    events_arr, class_map = parse_td_recall_events(events_tsv, sfreq)
    
    # Epoching: 3.0s window during recall imagery (0.5s to 3.5s post-blinking stop)
    epochs_td = mne.Epochs(raw_td, events_arr, event_id=class_map, tmin=0.5, tmax=0.5 + win_len_s, baseline=None, preload=True, verbose=False)
    X_td = epochs_td.get_data() * 1e6  # (N_trials, n_ch, n_samples)
    y_td = epochs_td.events[:, 2]
    
    print(f"[+] Tower Defense Trials: {X_td.shape[0]} epochs across 4 classes: {np.bincount(y_td)}")
    
    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = {}
    
    # -------------------------------------------------------------
    # BASELINE 1: Standard Riemannian Tangent Space + LogReg (No Transfer)
    # -------------------------------------------------------------
    cov_est = Covariances(estimator='oas')
    C_td = cov_est.fit_transform(X_td)
    ts = TangentSpace(metric='riemann')
    feat_ts_base = ts.fit_transform(C_td)
    
    pipe_base = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=1.0))])
    scores_base = cross_val_score(pipe_base, feat_ts_base, y_td, cv=cv, scoring='accuracy')
    results['[Baseline] Riemannian TS + LogReg'] = (np.mean(scores_base), np.std(scores_base))
    print(f"[+] Baseline Riemannian TS:             Acc = {np.mean(scores_base)*100:5.2f}% ± {np.std(scores_base)*100:4.2f}%")
    
    # -------------------------------------------------------------
    # BASELINE 2: Standard CSP + Shrinkage LDA (No Transfer)
    # -------------------------------------------------------------
    def ovr_csp_lda(X_tr, y_tr, X_te, n_comp=4):
        classes = np.unique(y_tr)
        f_tr, f_te = [], []
        for c in classes:
            y_b = (y_tr == c).astype(int)
            csp = CSP(n_components=min(n_comp, X_tr.shape[1]//2), reg='oas', log=True, norm_trace=False)
            f_tr.append(csp.fit_transform(X_tr, y_b))
            f_te.append(csp.transform(X_te))
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf.fit(np.hstack(f_tr), y_tr)
        return clf.predict(np.hstack(f_te))
        
    preds_csp = np.zeros_like(y_td)
    for tr_idx, te_idx in cv.split(X_td, y_td):
        preds_csp[te_idx] = ovr_csp_lda(X_td[tr_idx], y_td[tr_idx], X_td[te_idx], n_comp=4)
    acc_csp = accuracy_score(y_td, preds_csp)
    results['[Baseline] One-vs-Rest CSP + LDA'] = (acc_csp, 0.0)
    print(f"[+] Baseline OvR CSP + LDA:             Acc = {acc_csp*100:5.2f}%")
    
    # -------------------------------------------------------------
    # BASELINE 3: EEGNet Trained from Scratch (No Transfer)
    # -------------------------------------------------------------
    def eval_eegnet(pretrained=False):
        accs = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for tr_idx, te_idx in cv.split(X_td, y_td):
            X_tr, y_tr = torch.tensor(X_td[tr_idx], dtype=torch.float32), torch.tensor(y_td[tr_idx], dtype=torch.long)
            X_te, y_te = torch.tensor(X_td[te_idx], dtype=torch.float32), torch.tensor(y_td[te_idx], dtype=torch.long)
            
            model = EEGNetTransfer(n_channels=X_td.shape[1], n_samples=X_td.shape[2], n_classes=4).to(device)
            criterion = nn.CrossEntropyLoss()
            
            if pretrained:
                # Pretrain on full music listening dataset
                opt_pre = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
                ds_m = TensorDataset(torch.tensor(X_music, dtype=torch.float32), torch.tensor(y_music, dtype=torch.long))
                loader_m = DataLoader(ds_m, batch_size=32, shuffle=True)
                model.train()
                for _ in range(5):  # 5 pretraining epochs over 919 music listening windows
                    for bx, by in loader_m:
                        bx, by = bx.to(device), by.to(device)
                        opt_pre.zero_grad()
                        loss = criterion(model(bx), by)
                        loss.backward()
                        opt_pre.step()
                
                # Freeze early conv layers slightly, train on recall
                opt_ft = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            else:
                opt_ft = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
                
            loader_tr = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True)
            model.train()
            for _ in range(25):
                for bx, by in loader_tr:
                    bx, by = bx.to(device), by.to(device)
                    opt_ft.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    opt_ft.step()
                    
            model.eval()
            with torch.no_grad():
                preds = model(X_te.to(device)).argmax(dim=1).cpu().numpy()
                accs.append(accuracy_score(y_te.numpy(), preds))
        return np.mean(accs), np.std(accs)

    acc_eeg_base, std_eeg_base = eval_eegnet(pretrained=False)
    results['[Baseline] EEGNet (Scratch)'] = (acc_eeg_base, std_eeg_base)
    print(f"[+] Baseline EEGNet (Scratch):          Acc = {acc_eeg_base*100:5.2f}% ± {std_eeg_base*100:4.2f}%")
    
    # -------------------------------------------------------------
    # METHOD 1: Music-Guided Pre-Trained EEGNet (Transfer Learning)
    # -------------------------------------------------------------
    acc_eeg_trans, std_eeg_trans = eval_eegnet(pretrained=True)
    results['[Transfer] EEGNet (Music Pretrained)'] = (acc_eeg_trans, std_eeg_trans)
    print(f"[+] Music-Pretrained EEGNet Transfer:   Acc = {acc_eeg_trans*100:5.2f}% ± {std_eeg_trans*100:4.2f}%")
    
    # -------------------------------------------------------------
    # METHOD 2: Riemannian Domain Adaptation & Reference Alignment
    # -------------------------------------------------------------
    # Compute Riemannian geometric mean of music listening session
    C_music = cov_est.fit_transform(X_music)
    C_ref_music = mean_riemann(C_music)
    C_ref_music_invsqrt = invsqrtm(C_ref_music)
    
    # Whiten both music and TD covariance matrices
    C_td_whitened = np.array([C_ref_music_invsqrt @ C @ C_ref_music_invsqrt for C in C_td])
    feat_ts_whitened = ts.fit_transform(C_td_whitened)
    
    pipe_trans_ts = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, penalty='l2'))
    ])
    scores_trans_ts = cross_val_score(pipe_trans_ts, feat_ts_whitened, y_td, cv=cv, scoring='accuracy')
    results['[Transfer] Riemannian Music-Whitened TS + LogReg'] = (np.mean(scores_trans_ts), np.std(scores_trans_ts))
    print(f"[+] Music-Whitened Riemannian TS:       Acc = {np.mean(scores_trans_ts)*100:5.2f}% ± {np.std(scores_trans_ts)*100:4.2f}%")
    
    # -------------------------------------------------------------
    # METHOD 3: Perception-Regularized CSP (RCSP)
    # -------------------------------------------------------------
    # Blends covariance of recall with high-SNR listening covariance per class
    C_music_by_class = {c: mean_riemann(C_music[y_music == c]) for c in range(4)}
    
    def regularized_perception_csp_lda(X_tr, y_tr, X_te, alpha=0.35, n_comp=4):
        classes = np.unique(y_tr)
        f_tr, f_te = [], []
        
        for c in classes:
            X_c = X_tr[y_tr == c]
            X_rest = X_tr[y_tr != c]
            
            cov_c_recall = np.mean(cov_est.fit_transform(X_c), axis=0)
            cov_rest_recall = np.mean(cov_est.fit_transform(X_rest), axis=0)
            
            # Regularize with clean perceptual music listening covariances
            cov_c_music = C_music_by_class[c]
            cov_rest_music = mean_riemann(np.array([C_music_by_class[o] for o in classes if o != c]))
            
            cov_c_reg = (1.0 - alpha) * cov_c_recall + alpha * cov_c_music
            cov_rest_reg = (1.0 - alpha) * cov_rest_recall + alpha * cov_rest_music
            
            from scipy.linalg import eigh
            vals, vecs = eigh(cov_c_reg, cov_c_reg + cov_rest_reg)
            m = min(n_comp // 2, X_tr.shape[1] // 2)
            filters = np.hstack([vecs[:, :m], vecs[:, -m:]])  # (n_ch, 2m)
            
            # Project trials: log variance of spatially filtered signals (shape: [N_trials, 2m])
            feat_tr_c = np.log(np.var(np.matmul(filters.T, X_tr), axis=-1) + 1e-10)
            feat_te_c = np.log(np.var(np.matmul(filters.T, X_te), axis=-1) + 1e-10)
            
            f_tr.append(feat_tr_c)
            f_te.append(feat_te_c)
            
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf.fit(np.hstack(f_tr), y_tr)
        return clf.predict(np.hstack(f_te))
        
    preds_rcsp = np.zeros_like(y_td)
    for tr_idx, te_idx in cv.split(X_td, y_td):
        preds_rcsp[te_idx] = regularized_perception_csp_lda(X_td[tr_idx], y_td[tr_idx], X_td[te_idx], alpha=0.35, n_comp=4)
    acc_rcsp = accuracy_score(y_td, preds_rcsp)
    results['[Transfer] Perception-Regularized CSP (RCSP) + LDA'] = (acc_rcsp, 0.0)
    print(f"[+] Perception-Regularized CSP (RCSP):   Acc = {acc_rcsp*100:5.2f}%")
    
    # -------------------------------------------------------------
    # METHOD 4: Multi-Model Ensemble with Music Perception Priors
    # -------------------------------------------------------------
    pipe_ensemble = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', ExtraTreesClassifier(n_estimators=200, max_depth=6, random_state=42))
    ])
    scores_ens = cross_val_score(pipe_ensemble, feat_ts_whitened, y_td, cv=cv, scoring='accuracy')
    results['[Transfer] Music-Whitened TS + ExtraTrees'] = (np.mean(scores_ens), np.std(scores_ens))
    print(f"[+] Music-Whitened TS + ExtraTrees:     Acc = {np.mean(scores_ens)*100:5.2f}% ± {np.std(scores_ens)*100:4.2f}%")
    
    # -------------------------------------------------------------
    # METHOD 5: Perception-Guided Filter Bank CSP (FBCSP + Shrinkage LDA)
    # -------------------------------------------------------------
    bands = [(4.0, 8.0), (8.0, 12.0), (13.0, 30.0), (30.0, 45.0)]
    def fbcsp_perception_transfer(raw_td, raw_music, events_arr, class_map, cv):
        sf = raw_td.info['sfreq']
        fb_features_td = []
        for (f_low, f_high) in bands:
            raw_b_td = raw_td.copy().filter(f_low, f_high, verbose=False)
            ep_b = mne.Epochs(raw_b_td, events_arr, event_id=class_map, tmin=0.5, tmax=0.5+win_len_s, baseline=None, preload=True, verbose=False)
            X_b = ep_b.get_data() * 1e6
            
            raw_b_m = raw_music.copy().filter(f_low, f_high, verbose=False)
            X_m, y_m, _ = extract_music_listening_segments("scripts/bids_music", "01", "02", win_len_s=win_len_s, step_s=2.0)
            C_m = cov_est.fit_transform(X_m)
            C_m_by_c = {c: mean_riemann(C_m[y_m == c]) for c in range(4)}
            
            # Extract RCSP for this band
            f_band = []
            for c in range(4):
                cov_c_m = C_m_by_c[c]
                cov_rest_m = mean_riemann(np.array([C_m_by_c[o] for o in range(4) if o != c]))
                cov_c_td = np.mean(cov_est.fit_transform(X_b[y_td == c]), axis=0)
                cov_rest_td = np.mean(cov_est.fit_transform(X_b[y_td != c]), axis=0)
                
                cov_c_reg = 0.65 * cov_c_td + 0.35 * cov_c_m
                cov_rest_reg = 0.65 * cov_rest_td + 0.35 * cov_rest_m
                
                from scipy.linalg import eigh
                vals, vecs = eigh(cov_c_reg, cov_c_reg + cov_rest_reg)
                filters = np.hstack([vecs[:, :2], vecs[:, -2:]])
                feat_b_c = np.log(np.var(np.matmul(filters.T, X_b), axis=-1) + 1e-10)
                f_band.append(feat_b_c)
            fb_features_td.append(np.hstack(f_band))
            
        X_fbcsp = np.hstack(fb_features_td)
        clf_fb = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        scores = cross_val_score(clf_fb, X_fbcsp, y_td, cv=cv, scoring='accuracy')
        return np.mean(scores), np.std(scores)
        
    acc_fb, std_fb = fbcsp_perception_transfer(raw_td, raw_music, events_arr, class_map, cv)
    results['[Transfer] Perception-Guided FBCSP + Shrinkage LDA'] = (acc_fb, std_fb)
    print(f"[+] Perception-Guided FBCSP + LDA:      Acc = {acc_fb*100:5.2f}% ± {std_fb*100:4.2f}%")
    
    print("\n" + "=" * 80)
    print(" SUMMARY BENCHMARK COMPARISON TABLE ".center(80, "="))
    print("=" * 80)
    print(f"{'Method / Model':<52} | {'Accuracy':<14} | {'Gain vs Chance (25%)':<20}")
    print("-" * 92)
    for name, (acc, std) in results.items():
        gain = (acc - 0.25) / 0.25 * 100
        std_str = f" ± {std*100:.1f}%" if std > 0 else ""
        print(f"{name:<52} | {acc*100:5.2f}%{std_str:<7} | {gain:+6.1f}%")
        
    return results


if __name__ == "__main__":
    run_comparison_benchmark()
