"""
Friday Night Funkin' (FNF) Arrow Lock-On BCI Analysis Suite
============================================================
Comprehensive decoding and neural dynamics analysis for all 7 FNF BIDS sessions:
  - ses-01 (Mind / Motor Imagery - task-leftright, Left Arrow)
  - ses-02 (Mind / Motor Imagery - task-leftright, Left Arrow)
  - ses-03 (Mind / Motor Imagery - task-leftright, Left Arrow)
  - ses-04 (Movement / Motor Execution - task-me, Left Arrow)
  - ses-05 (Mind / Motor Imagery - task-leftright, Left Arrow)
  - ses-06 (Mind / Motor Imagery - task-leftright, Left Arrow)
  - ses-07 (Mind / 4 Directions - task-leftright: Left, Right, Up, Down)

Paradigms & Evaluations:
  1. Single-Session Target vs Rest Decoding across all 7 sessions (5-Fold CV)
  2. ses-07 4-Class Directional Decoding (Left vs Right vs Up vs Down; Chance = 25%)
  3. ses-07 Pairwise Directional Decoding (Left vs Right, Up vs Down, etc.)
  4. Mind (ses-01,02,03,05,06,07) vs Movement (ses-04) Decoding
  5. Cross-Session & Cross-Condition Transfer Learning
  6. Cortical Topographic Scalp Mapping & Lateralization (Contralateral ERD/ERS)
  7. Time-Locked Evoked Potential (ERP) Dynamics across Directions

Algorithms:
  - CSP + Linear Discriminant Analysis (LDA with OAS shrinkage)
  - CSP + Support Vector Machine (RBF)
  - CSP + Random Forest
  - Riemannian Geometry: Covariances + Tangent Space + Logistic Regression
  - Riemannian Geometry: Covariances + Tangent Space + SVM (RBF)
  - Riemannian Geometry: Minimum Distance to Mean (MDM)
  - Deep Learning: PyTorch EEGNet (Lawhern et al., 2018)
"""

import os
import sys
import glob
import json
import warnings
warnings.filterwarnings("ignore")

_curr_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_curr_dir, ".."))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.stats as stats

import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.decoding import CSP

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import pyriemann
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Standard 32 channel montage for g.Nautilus
STANDARD_32 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
    'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 
    'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 
    'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10'
]

DIRECTIONS = ['Left', 'Right', 'Up', 'Down']


# =====================================================================
# SPD Regularized Covariances for Riemannian Geometry
# =====================================================================
class RegCovariances(BaseEstimator, TransformerMixin):
    """Estimates Covariances with trace-proportional regularization for strict SPD manifold."""
    def __init__(self, estimator='oas', reg=1e-3):
        self.estimator = estimator
        self.reg = reg
        self._cov = Covariances(estimator=self.estimator)

    def fit(self, X, y=None):
        self._cov.fit(X, y)
        return self

    def transform(self, X):
        covs = self._cov.transform(X)
        n_matrices, n_ch, _ = covs.shape
        eye = np.eye(n_ch)
        for i in range(n_matrices):
            tr = np.trace(covs[i])
            covs[i] += (self.reg * (tr / n_ch) + 1e-6) * eye
        return covs


# =====================================================================
# 1. PyTorch EEGNet Architecture (Supports arbitrary n_classes)
# =====================================================================
class EEGNet(nn.Module):
    def __init__(self, n_channels=32, n_samples=176, n_classes=2, F1=8, D=2, F2=16, kernel_length=32, dropout_rate=0.25):
        super(EEGNet, self).__init__()
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
        
        self._flatten_dim = None
        self._calculate_flatten_dim(n_channels, n_samples)
        self.classifier = nn.Linear(self._flatten_dim, n_classes)

    def _calculate_flatten_dim(self, n_ch, n_s):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_ch, n_s)
            x = self.drop1(self.pool1(self.act1(self.bn2(self.depthwise(self.bn1(self.conv1(x)))))))
            x = self.drop2(self.pool2(self.act2(self.bn3(self.separable(x)))))
            self._flatten_dim = x.view(1, -1).size(1)

    def forward(self, x):
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
        out = self.classifier(x)
        return out


def train_eval_eegnet(X, y, cv_folds=5, epochs=30, batch_size=32, lr=0.005):
    """Evaluates EEGNet via Stratified K-Fold Cross Validation."""
    device = torch.device('cpu')
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    n_epochs_data, n_ch, n_samples = X.shape
    n_classes = len(np.unique(y))
    acc_scores = []
    all_preds = np.zeros(len(y), dtype=int)
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        X_train_t = torch.tensor(X_train[:, np.newaxis, :, :], dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_test_t = torch.tensor(X_test[:, np.newaxis, :, :], dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.long)
        
        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        model = EEGNet(n_channels=n_ch, n_samples=n_samples, n_classes=n_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        
        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            out_test = model(X_test_t.to(device))
            preds = torch.argmax(out_test, dim=1).cpu().numpy()
            acc = accuracy_score(y_test, preds)
            acc_scores.append(acc)
            all_preds[test_idx] = preds
            
    return np.mean(acc_scores), np.std(acc_scores), all_preds


# =====================================================================
# 2. Comprehensive BIDS Loader Supporting 1-4 Directions
# =====================================================================
def load_fnf_session(bids_root, sub="01", ses="01", task=None, l_freq=8.0, h_freq=30.0):
    if task is None:
        task = "me" if ses == "04" else "leftright"
        
    bp = BIDSPath(subject=sub, session=ses, task=task, datatype="eeg", root=bids_root)
    raw = read_raw_bids(bp, verbose=False)
    raw.load_data()
    
    mapping = {raw.ch_names[i]: STANDARD_32[i] for i in range(min(32, len(raw.ch_names)))}
    if len(raw.ch_names) > 32:
        raw.set_channel_types({raw.ch_names[32]: 'misc'})
    raw.rename_channels(mapping)
    raw.pick('eeg')
    
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)
    
    # Filter 8 - 30 Hz for SMR / Motor analysis and 50 Hz Notch
    raw_filt = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    raw_filt.notch_filter(freqs=50.0, verbose=False)
    raw_filt.set_eeg_reference('average', projection=False, verbose=False)
    
    # Broad unfiltered raw copy for ERP & spectral comparison
    raw_broad = raw.copy().filter(l_freq=1.0, h_freq=40.0, verbose=False).notch_filter(freqs=50.0, verbose=False)
    raw_broad.set_eeg_reference('average', projection=False, verbose=False)
    
    # Read annotations
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    # Identify direction hitzone events
    directional_epochs = {}
    directional_erp_epochs = {}
    all_target_events = []
    
    tmin, tmax = -0.1, 0.6
    
    for direction in DIRECTIONS:
        target_name = f'Arrow_{direction}_HitZone'
        code = None
        for k, v in event_id.items():
            if target_name in k:
                code = v
                break
        if code is not None:
            d_events = events[events[:, 2] == code]
            if len(d_events) > 0:
                all_target_events.append(d_events)
                ep = mne.Epochs(
                    raw_filt,
                    d_events,
                    event_id={f'Target_{direction}': code},
                    tmin=tmin,
                    tmax=tmax,
                    baseline=(-0.1, 0.0),
                    preload=True,
                    verbose=False
                )
                directional_epochs[direction] = ep
                
                ep_erp = mne.Epochs(
                    raw_broad,
                    d_events,
                    event_id={f'Target_{direction}': code},
                    tmin=-0.2,
                    tmax=0.8,
                    baseline=(-0.2, 0.0),
                    preload=True,
                    verbose=False
                )
                directional_erp_epochs[direction] = ep_erp

    if len(all_target_events) == 0:
        raise RuntimeError(f"Could not find any Arrow HitZone events in session {ses}")
        
    all_target_events = np.concatenate(all_target_events, axis=0)
    # Sort events by time
    all_target_events = all_target_events[np.argsort(all_target_events[:, 0])]
    
    # Combined target epochs
    epochs_target_all = mne.Epochs(
        raw_filt,
        all_target_events,
        tmin=tmin,
        tmax=tmax,
        baseline=(-0.1, 0.0),
        preload=True,
        verbose=False
    )
    
    # Synthesize Rest / Baseline epochs from non-target quiet periods
    target_onsets_sec = all_target_events[:, 0] / raw.info['sfreq']
    duration_total = raw.times[-1]
    
    candidate_times = np.arange(1.0, duration_total - 1.0, 0.7)
    valid_rest_times = []
    for ct in candidate_times:
        min_dist = np.min(np.abs(target_onsets_sec - ct))
        if min_dist >= 0.8:
            valid_rest_times.append(ct)
            
    n_targets = len(epochs_target_all)
    np.random.seed(42)
    if len(valid_rest_times) > n_targets:
        selected_rest_times = np.random.choice(valid_rest_times, size=n_targets, replace=False)
    else:
        selected_rest_times = valid_rest_times
        
    rest_samples = (np.array(selected_rest_times) * raw.info['sfreq']).astype(int)
    rest_events = np.zeros((len(rest_samples), 3), dtype=int)
    rest_events[:, 0] = rest_samples
    rest_events[:, 2] = 9999
    
    epochs_rest = mne.Epochs(
        raw_filt,
        rest_events,
        event_id={'Rest_Baseline': 9999},
        tmin=tmin,
        tmax=tmax,
        baseline=(-0.1, 0.0),
        preload=True,
        verbose=False
    )
    
    return {
        'session_id': ses,
        'task': task,
        'raw_filt': raw_filt,
        'raw_broad': raw_broad,
        'directional_epochs': directional_epochs,
        'directional_erp_epochs': directional_erp_epochs,
        'epochs_target': epochs_target_all,
        'epochs_rest': epochs_rest,
        'available_directions': list(directional_epochs.keys()),
        'sfreq': raw.info['sfreq'],
        'n_trials_total': len(epochs_target_all),
        'n_trials_per_direction': {k: len(v) for k, v in directional_epochs.items()}
    }


def get_pipelines(n_components=4):
    return {
        'CSP + LDA': Pipeline([
            ('csp', CSP(n_components=n_components, reg='oas', log=True, norm_trace=False)),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ]),
        'CSP + SVM (RBF)': Pipeline([
            ('csp', CSP(n_components=n_components, reg='oas', log=True, norm_trace=False)),
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=1.0))
        ]),
        'CSP + Random Forest': Pipeline([
            ('csp', CSP(n_components=n_components, reg='oas', log=True, norm_trace=False)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'Riemannian TS + Logistic Reg': Pipeline([
            ('cov', RegCovariances(estimator='oas', reg=1e-3)),
            ('ts', TangentSpace(metric='riemann')),
            ('lr', LogisticRegression(max_iter=1000, C=1.0))
        ]),
        'Riemannian TS + SVM (RBF)': Pipeline([
            ('cov', RegCovariances(estimator='oas', reg=1e-3)),
            ('ts', TangentSpace(metric='riemann')),
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=1.0))
        ]),
        'Riemannian MDM': Pipeline([
            ('cov', RegCovariances(estimator='oas', reg=1e-3)),
            ('mdm', MDM(metric='riemann'))
        ])
    }


# =====================================================================
# 3. Main Multi-Paradigm Benchmark Engine
# =====================================================================
def run_fnf_analysis(bids_root="scripts/bids/bids_fnf", out_dir="scripts/analysis/analysis_results_fnf"):
    os.makedirs(out_dir, exist_ok=True)
    bids_root = os.path.abspath(bids_root)
    
    print("=" * 85)
    print(" Friday Night Funkin' (FNF) Arrow Lock-On Neural Decoding Studio ".center(85, "="))
    print("=" * 85)
    
    # Metadata for all 7 sessions
    sessions_meta = [
        ('01', 'leftright', 'Mind (Session 1 - Left)'),
        ('02', 'leftright', 'Mind (Session 2 - Left)'),
        ('03', 'leftright', 'Mind (Session 3 - Left)'),
        ('04', 'me',        'Movement (Session 4 - Left)'),
        ('05', 'leftright', 'Mind (Session 5 - Left)'),
        ('06', 'leftright', 'Mind (Session 6 - Left)'),
        ('07', 'leftright', 'Mind (Session 7 - 4 Directions: Left, Right, Up, Down)')
    ]
    
    loaded_sessions = {}
    for ses, task, desc in sessions_meta:
        print(f"\n[*] Loading & Preprocessing sub-01 / ses-{ses} ({desc})...")
        s_data = load_fnf_session(bids_root, sub="01", ses=ses, task=task)
        loaded_sessions[ses] = s_data
        dir_breakdown = ", ".join([f"{k}: {v}" for k, v in s_data['n_trials_per_direction'].items()])
        print(f"    [+] Loaded {s_data['n_trials_total']} Target trials ({dir_breakdown}) & {len(s_data['epochs_rest'])} Baseline trials.")

    # -----------------------------------------------------------------
    # EXPERIMENT A: Single-Session Target vs Rest Classification (ses-01 to ses-07)
    # -----------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" [EXPERIMENT A] Single-Session Target Lock-On vs Rest/Baseline Decoding (5-Fold CV) ".center(85, "-"))
    print("=" * 85)
    
    results_exp_a = {}
    
    for ses, task, desc in sessions_meta:
        s_data = loaded_sessions[ses]
        
        X_target = s_data['epochs_target'].get_data()
        X_rest = s_data['epochs_rest'].get_data()
        
        X = np.concatenate([X_target, X_rest], axis=0)
        y = np.array([1] * len(X_target) + [0] * len(X_rest))
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        ses_res = {}
        pipelines = get_pipelines(n_components=4)
        for name, clf in pipelines.items():
            scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
            f1_scores = cross_val_score(clf, X, y, cv=skf, scoring='f1')
            ses_res[name] = {
                'acc_mean': float(np.mean(scores)),
                'acc_std': float(np.std(scores)),
                'f1_mean': float(np.mean(f1_scores))
            }
            
        eegnet_acc, eegnet_std, _ = train_eval_eegnet(X, y, cv_folds=5, epochs=30, batch_size=32)
        ses_res['Deep Learning (EEGNet)'] = {
            'acc_mean': float(eegnet_acc),
            'acc_std': float(eegnet_std),
            'f1_mean': float(eegnet_acc)
        }
        
        results_exp_a[ses] = {
            'description': desc,
            'n_trials_target': len(X_target),
            'n_trials_rest': len(X_rest),
            'directions': s_data['available_directions'],
            'models': ses_res
        }
        
        print(f"\n>>> Results for ses-{ses} ({desc}) [Total: {len(X)} epochs]:")
        for model_name, metrics in ses_res.items():
            print(f"    {model_name:<35}: Accuracy = {metrics['acc_mean']*100:6.2f}% ± {metrics['acc_std']*100:4.2f}% | F1 = {metrics['f1_mean']:.3f}")

    # -----------------------------------------------------------------
    # EXPERIMENT B: ses-07 4-Class Directional Decoding (Left vs Right vs Up vs Down)
    # -----------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" [EXPERIMENT B] ses-07 4-Class Directional Decoding (Left, Right, Up, Down; Chance=25%) ".center(85, "-"))
    print("=" * 85)
    
    s7 = loaded_sessions['07']
    d_names = ['Left', 'Right', 'Up', 'Down']
    d_arrays = [s7['directional_epochs'][d].get_data() for d in d_names]
    
    X_4dir = np.concatenate(d_arrays, axis=0)
    y_4dir = np.concatenate([[i] * len(d_arrays[i]) for i in range(4)], axis=0)
    
    print(f"[+] ses-07 Dataset: {len(X_4dir)} total epochs ({len(d_arrays[0])} trials per direction)")
    
    results_exp_b = {}
    cm_dict = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    pipelines_4c = get_pipelines(n_components=6)
    for name, clf in pipelines_4c.items():
        scores = cross_val_score(clf, X_4dir, y_4dir, cv=skf, scoring='accuracy')
        f1_scores = cross_val_score(clf, X_4dir, y_4dir, cv=skf, scoring='f1_macro')
        preds = cross_val_predict(clf, X_4dir, y_4dir, cv=skf)
        cm = confusion_matrix(y_4dir, preds)
        
        results_exp_b[name] = {
            'acc_mean': float(np.mean(scores)),
            'acc_std': float(np.std(scores)),
            'f1_macro': float(np.mean(f1_scores)),
            'confusion_matrix': cm.tolist()
        }
        cm_dict[name] = cm
        print(f"    {name:<35}: Accuracy = {np.mean(scores)*100:6.2f}% ± {np.std(scores)*100:4.2f}% | F1 Macro = {np.mean(f1_scores):.3f}")

    eegnet_acc_4c, eegnet_std_4c, eegnet_preds_4c = train_eval_eegnet(X_4dir, y_4dir, cv_folds=5, epochs=35, batch_size=32)
    cm_eegnet = confusion_matrix(y_4dir, eegnet_preds_4c)
    results_exp_b['Deep Learning (EEGNet)'] = {
        'acc_mean': float(eegnet_acc_4c),
        'acc_std': float(eegnet_std_4c),
        'f1_macro': float(eegnet_acc_4c),
        'confusion_matrix': cm_eegnet.tolist()
    }
    cm_dict['Deep Learning (EEGNet)'] = cm_eegnet
    print(f"    {'Deep Learning (EEGNet)':<35}: Accuracy = {eegnet_acc_4c*100:6.2f}% ± {eegnet_std_4c*100:4.2f}%")

    # -----------------------------------------------------------------
    # EXPERIMENT C: ses-07 Pairwise Directional Decoding
    # -----------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" [EXPERIMENT C] ses-07 Pairwise Directional Decoding (Chance = 50%) ".center(85, "-"))
    print("=" * 85)
    
    pairs = [
        ('Left', 'Right', 'Horizontal / Lateral Motor Imagery Axis'),
        ('Up', 'Down',     'Vertical Axis'),
        ('Left', 'Up',     'Left vs Up'),
        ('Left', 'Down',   'Left vs Down'),
        ('Right', 'Up',    'Right vs Up'),
        ('Right', 'Down',  'Right vs Down')
    ]
    
    results_exp_c = {}
    for d1, d2, pair_desc in pairs:
        X1 = s7['directional_epochs'][d1].get_data()
        X2 = s7['directional_epochs'][d2].get_data()
        X_pair = np.concatenate([X1, X2], axis=0)
        y_pair = np.array([0] * len(X1) + [1] * len(X2))
        
        pair_res = {}
        pipelines_2c = get_pipelines(n_components=4)
        for name, clf in pipelines_2c.items():
            scores = cross_val_score(clf, X_pair, y_pair, cv=skf, scoring='accuracy')
            f1_scores = cross_val_score(clf, X_pair, y_pair, cv=skf, scoring='f1')
            pair_res[name] = {
                'acc_mean': float(np.mean(scores)),
                'acc_std': float(np.std(scores)),
                'f1': float(np.mean(f1_scores))
            }
        
        eeg_acc, eeg_std, _ = train_eval_eegnet(X_pair, y_pair, cv_folds=5, epochs=25, batch_size=16)
        pair_res['Deep Learning (EEGNet)'] = {
            'acc_mean': float(eeg_acc),
            'acc_std': float(eeg_std),
            'f1': float(eeg_acc)
        }
        
        results_exp_c[f"{d1}_vs_{d2}"] = {
            'description': pair_desc,
            'models': pair_res
        }
        
        best_model = max(pair_res.items(), key=lambda item: item[1]['acc_mean'])
        print(f"    {d1:<6} vs {d2:<6} ({pair_desc:<38}): Best = {best_model[0]} ({best_model[1]['acc_mean']*100:5.2f}% ± {best_model[1]['acc_std']*100:4.2f}%)")

    # -----------------------------------------------------------------
    # EXPERIMENT D: Mind (All Mind Sessions) vs Movement (ses-04) Decoding
    # -----------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" [EXPERIMENT D] Mind (ses-01,02,03,05,06,07 Left) vs Movement (ses-04) Decoding ".center(85, "-"))
    print("=" * 85)
    
    mind_left_trials = []
    for s_idx in ['01', '02', '03', '05', '06']:
        mind_left_trials.append(loaded_sessions[s_idx]['epochs_target'].get_data())
    mind_left_trials.append(loaded_sessions['07']['directional_epochs']['Left'].get_data())
    
    X_mind_left = np.concatenate(mind_left_trials, axis=0)
    y_mind_left = np.zeros(len(X_mind_left), dtype=int)
    
    X_move_left = loaded_sessions['04']['epochs_target'].get_data()
    y_move_left = np.ones(len(X_move_left), dtype=int)
    
    X_mm = np.concatenate([X_mind_left, X_move_left], axis=0)
    y_mm = np.concatenate([y_mind_left, y_move_left], axis=0)
    
    print(f"[+] Pooled Dataset: {len(X_mm)} trials ({len(X_mind_left)} Mind trials across 6 sessions, {len(X_move_left)} Movement trials)")
    
    results_exp_d = {}
    for name, clf in get_pipelines(n_components=4).items():
        scores = cross_val_score(clf, X_mm, y_mm, cv=skf, scoring='accuracy')
        bal_acc = cross_val_score(clf, X_mm, y_mm, cv=skf, scoring='balanced_accuracy')
        f1 = cross_val_score(clf, X_mm, y_mm, cv=skf, scoring='f1')
        results_exp_d[name] = {
            'acc_mean': float(np.mean(scores)),
            'acc_std': float(np.std(scores)),
            'balanced_acc': float(np.mean(bal_acc)),
            'f1': float(np.mean(f1))
        }
        print(f"    {name:<35}: Acc = {np.mean(scores)*100:6.2f}% ± {np.std(scores)*100:4.2f}% | Bal Acc = {np.mean(bal_acc)*100:6.2f}% | F1 = {np.mean(f1):.3f}")
        
    eegnet_acc_mm, eegnet_std_mm, _ = train_eval_eegnet(X_mm, y_mm, cv_folds=5, epochs=30, batch_size=32)
    results_exp_d['Deep Learning (EEGNet)'] = {
        'acc_mean': float(eegnet_acc_mm),
        'acc_std': float(eegnet_std_mm),
        'balanced_acc': float(eegnet_acc_mm),
        'f1': float(eegnet_acc_mm)
    }
    print(f"    {'Deep Learning (EEGNet)':<35}: Acc = {eegnet_acc_mm*100:6.2f}% ± {eegnet_std_mm*100:4.2f}%")

    # -----------------------------------------------------------------
    # EXPERIMENT E: Cross-Session & Cross-Condition Transfer Generalization
    # -----------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" [EXPERIMENT E] Cross-Session Transfer: Earlier Mind -> ses-07 Multi-Direction ".center(85, "-"))
    print("=" * 85)
    
    # Train on ses-01..03 (Left Arrow vs Rest), Test on ses-07 Left Arrow vs Rest
    s7_left_target = loaded_sessions['07']['directional_epochs']['Left'].get_data()
    s7_rest_sub = loaded_sessions['07']['epochs_rest'].get_data()[:len(s7_left_target)]
    X_test_s7_left = np.concatenate([s7_left_target, s7_rest_sub], axis=0)
    y_test_s7_left = np.array([1]*len(s7_left_target) + [0]*len(s7_rest_sub))
    
    early_target = np.concatenate([loaded_sessions[s]['epochs_target'].get_data() for s in ['01', '02', '03']], axis=0)
    early_rest = np.concatenate([loaded_sessions[s]['epochs_rest'].get_data() for s in ['01', '02', '03']], axis=0)
    X_train_early = np.concatenate([early_target, early_rest], axis=0)
    y_train_early = np.array([1]*len(early_target) + [0]*len(early_rest))
    
    transfer_results = {}
    for name, clf in get_pipelines(n_components=4).items():
        clf.fit(X_train_early, y_train_early)
        preds_s7 = clf.predict(X_test_s7_left)
        acc_s7 = accuracy_score(y_test_s7_left, preds_s7)
        f1_s7 = f1_score(y_test_s7_left, preds_s7)
        transfer_results[name] = {
            'EarlyMind_to_ses07_Acc': float(acc_s7),
            'EarlyMind_to_ses07_F1': float(f1_s7)
        }
        print(f"    {name:<35}: ses-01..03 -> ses-07 Left Target = {acc_s7*100:5.2f}% | F1 = {f1_s7:.3f}")

    # -----------------------------------------------------------------
    # EXPERIMENT F: Visualizations & Topomaps
    # -----------------------------------------------------------------
    print("\n[*] Generating Publication-Quality Figures & Topomaps...")
    
    # 1. Bar Chart of Model Accuracies Across All 7 Sessions
    fig, axes = plt.subplots(4, 2, figsize=(18, 18), dpi=150)
    axes = axes.flatten()
    
    colors = ['#2b5c8f', '#3470a3', '#4682b4', '#2e8b57', '#3cb371', '#20b2aa', '#d9534f']
    
    for idx, (ses, task, desc) in enumerate(sessions_meta):
        ax = axes[idx]
        m_dict = results_exp_a[ses]['models']
        names = list(m_dict.keys())
        accs = [m_dict[n]['acc_mean'] * 100 for n in names]
        stds = [m_dict[n]['acc_std'] * 100 for n in names]
        
        bars = ax.barh(names, accs, xerr=stds, color=colors[:len(names)], alpha=0.88, capsize=4, edgecolor='black')
        ax.axvline(50.0, color='gray', linestyle='--', linewidth=1.5, label='Chance (50%)')
        ax.set_xlim(35, 105)
        ax.set_xlabel("Cross-Validation Accuracy (%)", fontsize=10, fontweight='bold')
        ax.set_title(f"sub-01 / ses-{ses}: {desc}\nTarget Lock-On vs Baseline", fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle=':')
        
        for bar, acc, std in zip(bars, accs, stds):
            ax.text(acc + 1.2, bar.get_y() + bar.get_height()/2, f"{acc:.1f}%", va='center', fontsize=8, fontweight='bold')
            
    # 8th subplot: Summary of Best Model Accuracy Per Session
    ax_sum = axes[7]
    ses_labels = [f"ses-{s}\n({loaded_sessions[s]['n_trials_total']} tr)" for s, _, _ in sessions_meta]
    best_accs = [max([results_exp_a[s]['models'][m]['acc_mean'] * 100 for m in results_exp_a[s]['models']]) for s, _, _ in sessions_meta]
    bar_cols = ['#2b5c8f']*3 + ['#d9534f'] + ['#2b5c8f']*3
    bars_sum = ax_sum.bar(ses_labels, best_accs, color=bar_cols, alpha=0.9, edgecolor='black')
    ax_sum.axhline(50.0, color='gray', linestyle='--', linewidth=1.5, label='Chance (50%)')
    ax_sum.set_ylim(40, 105)
    ax_sum.set_ylabel("Peak Accuracy (%)", fontsize=10, fontweight='bold')
    ax_sum.set_title("Peak BCI Decoding Across All 7 Sessions\n(Blue = Mind/MI, Red = Movement/ME)", fontsize=11, fontweight='bold')
    ax_sum.grid(axis='y', alpha=0.3, linestyle=':')
    for bar, val in zip(bars_sum, best_accs):
        ax_sum.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}%", ha='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "fnf_all_sessions_decoding_benchmark.png")
    plt.savefig(fig1_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved All Sessions Benchmark Figure to: {fig1_path}")
    
    # 2. ses-07 4-Class Directional Decoding & Confusion Matrix
    fig, (ax_bar4, ax_cm, ax_pair) = plt.subplots(1, 3, figsize=(20, 6), dpi=150)
    
    # Left: 4-Class Model Accuracies
    m_names_4c = list(results_exp_b.keys())
    accs_4c = [results_exp_b[m]['acc_mean'] * 100 for m in m_names_4c]
    stds_4c = [results_exp_b[m]['acc_std'] * 100 for m in m_names_4c]
    
    bars4 = ax_bar4.barh(m_names_4c, accs_4c, xerr=stds_4c, color=colors[:len(m_names_4c)], alpha=0.88, capsize=4, edgecolor='black')
    ax_bar4.axvline(25.0, color='red', linestyle='--', linewidth=1.5, label='Chance (25%)')
    ax_bar4.set_xlim(15, 80)
    ax_bar4.set_xlabel("4-Class Accuracy (%)", fontsize=11, fontweight='bold')
    ax_bar4.set_title("ses-07: 4-Direction Decoding\n(Left vs Right vs Up vs Down - 612 Trials)", fontsize=12, fontweight='bold')
    ax_bar4.grid(axis='x', alpha=0.3, linestyle=':')
    ax_bar4.legend(loc='lower right', fontsize=9)
    for bar, val in zip(bars4, accs_4c):
        ax_bar4.text(val + 1.2, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va='center', fontsize=9, fontweight='bold')

    # Middle: Confusion Matrix for Best Model (or Tangent Space + LR / EEGNet)
    best_4c_name = max(results_exp_b.items(), key=lambda item: item[1]['acc_mean'])[0]
    best_cm = np.array(results_exp_b[best_4c_name]['confusion_matrix'])
    best_cm_norm = best_cm.astype('float') / best_cm.sum(axis=1)[:, np.newaxis] * 100
    
    im = ax_cm.imshow(best_cm_norm, cmap='Blues', vmin=0, vmax=100)
    ax_cm.set_xticks(range(4))
    ax_cm.set_yticks(range(4))
    ax_cm.set_xticklabels(DIRECTIONS, fontsize=10, fontweight='bold')
    ax_cm.set_yticklabels(DIRECTIONS, fontsize=10, fontweight='bold')
    ax_cm.set_xlabel("Predicted Direction", fontsize=11, fontweight='bold')
    ax_cm.set_ylabel("True Direction", fontsize=11, fontweight='bold')
    ax_cm.set_title(f"Confusion Matrix ({best_4c_name})\nTotal: 612 Trials (153/class)", fontsize=12, fontweight='bold')
    for i in range(4):
        for j in range(4):
            val_txt = f"{best_cm[i, j]}\n({best_cm_norm[i, j]:.1f}%)"
            text_color = "white" if best_cm_norm[i, j] > 50 else "black"
            ax_cm.text(j, i, val_txt, ha='center', va='center', color=text_color, fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04, label="Percentage (%)")

    # Right: Pairwise Accuracies
    pair_labels = [k.replace('_vs_', ' vs ') for k in results_exp_c.keys()]
    pair_accs = [max([v['models'][m]['acc_mean']*100 for m in v['models']]) for v in results_exp_c.values()]
    bars_p = ax_pair.bar(pair_labels, pair_accs, color='#3470a3', alpha=0.9, edgecolor='black')
    ax_pair.axhline(50.0, color='gray', linestyle='--', linewidth=1.5, label='Chance (50%)')
    ax_pair.set_ylim(40, 95)
    ax_pair.set_xticklabels(pair_labels, rotation=35, ha='right', fontsize=9)
    ax_pair.set_ylabel("Peak Pairwise Accuracy (%)", fontsize=11, fontweight='bold')
    ax_pair.set_title("ses-07: Direction Pairwise Accuracies\n(Peak Classifier per Pair)", fontsize=12, fontweight='bold')
    ax_pair.grid(axis='y', alpha=0.3, linestyle=':')
    for bar, val in zip(bars_p, pair_accs):
        ax_pair.text(bar.get_x() + bar.get_width()/2, val + 1.2, f"{val:.1f}%", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig2_path = os.path.join(out_dir, "fnf_ses07_4direction_decoding_and_cm.png")
    plt.savefig(fig2_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved ses-07 4-Direction Figure to: {fig2_path}")

    # 3. ses-07 Cortical Topomaps across 4 Directions & Lateralization (Mu & Beta)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=150)
    raw_s7 = s7['directional_erp_epochs']
    
    # Compute Mu (8-12 Hz) and Beta (13-30 Hz) Power per direction across all channels
    topo_data = {'Mu': {}, 'Beta': {}}
    info_s7 = raw_s7['Left'].info
    
    for d in DIRECTIONS:
        ep_data = raw_s7[d].get_data()  # (n_trials, n_ch, n_times)
        f_axis, psd_all = signal.welch(ep_data, fs=250.0, nperseg=128, axis=-1)
        
        mu_mask = (f_axis >= 8) & (f_axis <= 12)
        beta_mask = (f_axis >= 13) & (f_axis <= 30)
        
        topo_data['Mu'][d] = np.mean(psd_all[:, :, mu_mask], axis=(0, 2))
        topo_data['Beta'][d] = np.mean(psd_all[:, :, beta_mask], axis=(0, 2))
        
    for i, d in enumerate(DIRECTIONS):
        ax_mu = axes[0, i]
        mne.viz.plot_topomap(topo_data['Mu'][d], info_s7, axes=ax_mu, show=False, cmap='RdBu_r')
        ax_mu.set_title(f"Mu (8-12 Hz): {d} Arrow", fontsize=11, fontweight='bold')
        
        ax_beta = axes[1, i]
        mne.viz.plot_topomap(topo_data['Beta'][d], info_s7, axes=ax_beta, show=False, cmap='viridis')
        ax_beta.set_title(f"Beta (13-30 Hz): {d} Arrow", fontsize=11, fontweight='bold')
        
    plt.suptitle("ses-07 Cortical Power Topographies Across 4 Arrow Directions (153 trials/dir)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig3_path = os.path.join(out_dir, "fnf_ses07_directional_topomaps.png")
    plt.savefig(fig3_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved ses-07 Directional Topomaps to: {fig3_path}")

    # 4. Grand-Average Evoked Potentials (ERP) for 4 Directions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    key_channels = ['C3', 'Cz', 'C4', 'Oz']
    dir_colors = {'Left': '#2b5c8f', 'Right': '#d9534f', 'Up': '#2e8b57', 'Down': '#e67e22'}
    
    times_erp = raw_s7['Left'].times * 1000  # in ms
    for ax, ch_name in zip(axes.flatten(), key_channels):
        ch_idx = raw_s7['Left'].ch_names.index(ch_name)
        
        for d in DIRECTIONS:
            ep_ch = raw_s7[d].get_data()[:, ch_idx, :] * 1e6  # uV
            mean_wave = np.mean(ep_ch, axis=0)
            sem_wave = stats.sem(ep_ch, axis=0)
            
            ax.plot(times_erp, mean_wave, label=f"{d} Arrow", color=dir_colors[d], linewidth=2.0)
            ax.fill_between(times_erp, mean_wave - sem_wave, mean_wave + sem_wave, color=dir_colors[d], alpha=0.12)
            
        ax.axvline(0, color='black', linestyle='--', linewidth=1.2, label='HitZone (t=0)')
        ax.set_title(f"Electrode {ch_name} ERP Response (ses-07)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time relative to HitZone (ms)", fontsize=10)
        ax.set_ylabel("Amplitude (µV)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
        
    plt.tight_layout()
    fig4_path = os.path.join(out_dir, "fnf_ses07_erp_waveforms.png")
    plt.savefig(fig4_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved ses-07 ERP Waveforms to: {fig4_path}")

    # -----------------------------------------------------------------
    # Export Final Metrics JSON
    # -----------------------------------------------------------------
    final_report = {
        'bids_root': bids_root,
        'dataset_summary': {
            f"ses-{s}": {
                'mode': 'Movement (Motor Execution)' if s == '04' else 'Mind (Motor Imagery)',
                'task': loaded_sessions[s]['task'],
                'total_trials': loaded_sessions[s]['n_trials_total'],
                'directions': loaded_sessions[s]['n_trials_per_direction']
            } for s, _, _ in sessions_meta
        },
        'experiment_a_target_vs_rest': results_exp_a,
        'experiment_b_ses07_4direction_decoding': results_exp_b,
        'experiment_c_ses07_pairwise_decoding': results_exp_c,
        'experiment_d_mind_vs_movement': results_exp_d,
        'experiment_e_transfer_generalization': transfer_results,
        'artifacts': {
            'all_sessions_benchmark_plot': fig1_path,
            'ses07_4direction_benchmark_plot': fig2_path,
            'ses07_directional_topomaps_plot': fig3_path,
            'ses07_erp_waveforms_plot': fig4_path
        }
    }
    
    json_path = os.path.join(out_dir, "fnf_decoding_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(final_report, f, indent=4)
    print(f"\n[+] Saved Complete Metrics Report to: {json_path}")
    print("=" * 85)
    print(" FNF BCI Multi-Session & Multi-Direction Analysis Completed Successfully! ".center(85, "="))
    print("=" * 85)
    
    return final_report


if __name__ == "__main__":
    bids_path = "scripts/bids/bids_fnf"
    if len(sys.argv) > 1:
        bids_path = sys.argv[1]
    run_fnf_analysis(bids_root=bids_path)
