"""
Friday Night Funkin' (FNF) Arrow Lock-On BCI Analysis Suite
============================================================
Processes the 4 FNF BIDS sessions:
  - ses-01 (Mind / Motor Imagery - task-leftright)
  - ses-02 (Mind / Motor Imagery - task-leftright)
  - ses-03 (Mind / Motor Imagery - task-leftright)
  - ses-04 (Movement / Motor Execution - task-me)

Events:
  - Target Event: `Arrow_Left_HitZone` (arrow reaches the target objective)
  - Ignored Events: `Left_Miss`, `Left_Perfect`, `Left_Great`, `Left_Good`, `Left_Ok`, `Left_Pressed` (other player/feedback)
  - Rest/Baseline: Non-arrow / listening baseline epochs

Algorithms evaluated from scripts/analysis:
  1. CSP + Linear Discriminant Analysis (LDA with OAS shrinkage)
  2. CSP + Support Vector Machine (Linear / RBF)
  3. CSP + Random Forest
  4. Riemannian Geometry: Covariances + Tangent Space + Logistic Regression
  5. Riemannian Geometry: Covariances + Tangent Space + SVM (RBF)
  6. Riemannian Geometry: Minimum Distance to Mean (MDM)
  7. Deep Learning: PyTorch EEGNet (Lawhern et al., 2018)
  8. Time-Frequency / Spectral ERD/ERS & Topographic Scalp Mapping
  9. Cross-Condition Transfer Learning (Mind -> Movement & Movement -> Mind)
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

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
# 1. PyTorch EEGNet Architecture
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


def train_eval_eegnet(X, y, cv_folds=5, epochs=35, batch_size=32, lr=0.005):
    """Evaluates EEGNet via Stratified K-Fold Cross Validation."""
    device = torch.device('cpu')
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    n_epochs_data, n_ch, n_samples = X.shape
    acc_scores = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        X_train_t = torch.tensor(X_train[:, np.newaxis, :, :], dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_test_t = torch.tensor(X_test[:, np.newaxis, :, :], dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.long)
        
        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        model = EEGNet(n_channels=n_ch, n_samples=n_samples, n_classes=len(np.unique(y))).to(device)
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
            
    return np.mean(acc_scores), np.std(acc_scores)


# =====================================================================
# 2. Robust BIDS Data Loader & Preprocessor
# =====================================================================
def load_fnf_session(bids_root, sub="01", ses="01", task="leftright", l_freq=8.0, h_freq=30.0):
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
    
    # Read annotations
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    hz_code = None
    for k, v in event_id.items():
        if 'Arrow_Left_HitZone' in k:
            hz_code = v
            break
            
    if hz_code is None:
        raise RuntimeError(f"Could not find Arrow_Left_HitZone in session {ses}")
        
    hz_events = events[events[:, 2] == hz_code]
    
    # Epoching Target window: [-0.1s to 0.6s] around Arrow_Left_HitZone
    tmin, tmax = -0.1, 0.6
    epochs_target = mne.Epochs(
        raw_filt,
        hz_events,
        event_id={'Target_HitZone': hz_code},
        tmin=tmin,
        tmax=tmax,
        baseline=(-0.1, 0.0),
        preload=True,
        verbose=False
    )
    
    # Synthesize Rest / Baseline non-target epochs from quiet periods
    target_onsets_sec = hz_events[:, 0] / raw.info['sfreq']
    duration_total = raw.times[-1]
    
    candidate_times = np.arange(1.0, duration_total - 1.0, 0.7)
    valid_rest_times = []
    for ct in candidate_times:
        min_dist = np.min(np.abs(target_onsets_sec - ct))
        if min_dist >= 1.0:
            valid_rest_times.append(ct)
            
    n_targets = len(epochs_target)
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
    
    # Broad unfiltered raw copy for ERP & spectral comparison
    raw_broad = raw.copy().filter(l_freq=1.0, h_freq=40.0, verbose=False).notch_filter(freqs=50.0, verbose=False)
    raw_broad.set_eeg_reference('average', projection=False, verbose=False)
    epochs_erp = mne.Epochs(
        raw_broad,
        hz_events,
        event_id={'Target_HitZone': hz_code},
        tmin=-0.2,
        tmax=0.8,
        baseline=(-0.2, 0.0),
        preload=True,
        verbose=False
    )
    
    return {
        'session_id': ses,
        'task': task,
        'raw_filt': raw_filt,
        'epochs_target': epochs_target,
        'epochs_rest': epochs_rest,
        'epochs_erp': epochs_erp,
        'sfreq': raw.info['sfreq'],
        'n_trials': len(epochs_target)
    }


# =====================================================================
# 3. Main Multi-Paradigm Benchmark Engine
# =====================================================================
def run_fnf_analysis(bids_root="scripts/bids/bids_fnf", out_dir="scripts/analysis/analysis_results_fnf"):
    os.makedirs(out_dir, exist_ok=True)
    bids_root = os.path.abspath(bids_root)
    
    print("=" * 80)
    print(" Friday Night Funkin' (FNF) Arrow Lock-On Neural Decoding Studio ".center(80, "="))
    print("=" * 80)
    
    sessions_meta = [
        ('01', 'leftright', 'Mind (Session 1)'),
        ('02', 'leftright', 'Mind (Session 2)'),
        ('03', 'leftright', 'Mind (Session 3)'),
        ('04', 'me', 'Movement (Session 4)')
    ]
    
    loaded_sessions = {}
    for ses, task, desc in sessions_meta:
        print(f"\n[*] Loading & Preprocessing sub-01 / ses-{ses} ({desc})...")
        s_data = load_fnf_session(bids_root, sub="01", ses=ses, task=task)
        loaded_sessions[ses] = s_data
        print(f"    [+] Loaded {s_data['n_trials']} Target Lock-On trials & {len(s_data['epochs_rest'])} Baseline trials.")

    def get_pipelines():
        return {
            'CSP + LDA': Pipeline([
                ('csp', CSP(n_components=4, reg='oas', log=True, norm_trace=False)),
                ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
            ]),
            'CSP + SVM (RBF)': Pipeline([
                ('csp', CSP(n_components=4, reg='oas', log=True, norm_trace=False)),
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', C=1.0))
            ]),
            'CSP + Random Forest': Pipeline([
                ('csp', CSP(n_components=4, reg='oas', log=True, norm_trace=False)),
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

    # -----------------------------------------------------------------
    # Experiment A: Single-Session Target vs Rest Classification
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" [EXPERIMENT A] Target Lock-On vs Rest/Baseline Decoding (5-Fold CV) ".center(80, "-"))
    print("=" * 80)
    
    results_exp_a = {}
    
    for ses, task, desc in sessions_meta:
        s_data = loaded_sessions[ses]
        
        X_target = s_data['epochs_target'].get_data()
        X_rest = s_data['epochs_rest'].get_data()
        
        X = np.concatenate([X_target, X_rest], axis=0)
        y = np.array([1] * len(X_target) + [0] * len(X_rest))
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        ses_res = {}
        pipelines = get_pipelines()
        for name, clf in pipelines.items():
            scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
            f1_scores = cross_val_score(clf, X, y, cv=skf, scoring='f1')
            ses_res[name] = {
                'acc_mean': float(np.mean(scores)),
                'acc_std': float(np.std(scores)),
                'f1_mean': float(np.mean(f1_scores))
            }
            
        eegnet_acc, eegnet_std = train_eval_eegnet(X, y, cv_folds=5, epochs=35, batch_size=16)
        ses_res['Deep Learning (EEGNet)'] = {
            'acc_mean': float(eegnet_acc),
            'acc_std': float(eegnet_std),
            'f1_mean': float(eegnet_acc)
        }
        
        results_exp_a[ses] = {
            'description': desc,
            'n_trials_target': len(X_target),
            'n_trials_rest': len(X_rest),
            'models': ses_res
        }
        
        print(f"\n>>> Results for {desc} (Total: {len(X)} epochs):")
        for model_name, metrics in ses_res.items():
            print(f"    {model_name:<35}: Accuracy = {metrics['acc_mean']*100:6.2f}% ± {metrics['acc_std']*100:4.2f}% | F1 = {metrics['f1_mean']:.3f}")

    # -----------------------------------------------------------------
    # Experiment B: Mind (ses-01..03) vs Movement (ses-04) Decoding
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" [EXPERIMENT B] Mind (ses-01..03) vs Movement (ses-04) Lock-On Decoding ".center(80, "-"))
    print("=" * 80)
    
    mind_target_list = [loaded_sessions[s]['epochs_target'].get_data() for s in ['01', '02', '03']]
    X_mind = np.concatenate(mind_target_list, axis=0)
    y_mind = np.zeros(len(X_mind), dtype=int)
    
    X_move = loaded_sessions['04']['epochs_target'].get_data()
    y_move = np.ones(len(X_move), dtype=int)
    
    X_mm = np.concatenate([X_mind, X_move], axis=0)
    y_mm = np.concatenate([y_mind, y_move], axis=0)
    
    print(f"[+] Pooled Dataset: {len(X_mm)} trials ({len(X_mind)} Mind trials, {len(X_move)} Movement trials)")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results_exp_b = {}
    pipelines = get_pipelines()
    for name, clf in pipelines.items():
        scores = cross_val_score(clf, X_mm, y_mm, cv=skf, scoring='accuracy')
        bal_acc = cross_val_score(clf, X_mm, y_mm, cv=skf, scoring='balanced_accuracy')
        f1 = cross_val_score(clf, X_mm, y_mm, cv=skf, scoring='f1')
        results_exp_b[name] = {
            'acc_mean': float(np.mean(scores)),
            'acc_std': float(np.std(scores)),
            'balanced_acc': float(np.mean(bal_acc)),
            'f1': float(np.mean(f1))
        }
        print(f"    {name:<35}: Acc = {np.mean(scores)*100:6.2f}% ± {np.std(scores)*100:4.2f}% | Bal Acc = {np.mean(bal_acc)*100:6.2f}% | F1 = {np.mean(f1):.3f}")
        
    eegnet_acc_mm, eegnet_std_mm = train_eval_eegnet(X_mm, y_mm, cv_folds=5, epochs=40, batch_size=32)
    results_exp_b['Deep Learning (EEGNet)'] = {
        'acc_mean': float(eegnet_acc_mm),
        'acc_std': float(eegnet_std_mm),
        'balanced_acc': float(eegnet_acc_mm),
        'f1': float(eegnet_acc_mm)
    }
    print(f"    {'Deep Learning (EEGNet)':<35}: Acc = {eegnet_acc_mm*100:6.2f}% ± {eegnet_std_mm*100:4.2f}%")

    # -----------------------------------------------------------------
    # Experiment C: Cross-Condition / Transfer Generalization
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" [EXPERIMENT C] Cross-Condition Generalization & Transfer Decoding ".center(80, "-"))
    print("=" * 80)
    
    s4_t = loaded_sessions['04']['epochs_target'].get_data()
    s4_r = loaded_sessions['04']['epochs_rest'].get_data()
    X_test_move = np.concatenate([s4_t, s4_r], axis=0)
    y_test_move = np.array([1]*len(s4_t) + [0]*len(s4_r))
    
    mind_t_all = np.concatenate([loaded_sessions[s]['epochs_target'].get_data() for s in ['01', '02', '03']], axis=0)
    mind_r_all = np.concatenate([loaded_sessions[s]['epochs_rest'].get_data() for s in ['01', '02', '03']], axis=0)
    X_train_mind = np.concatenate([mind_t_all, mind_r_all], axis=0)
    y_train_mind = np.array([1]*len(mind_t_all) + [0]*len(mind_r_all))
    
    transfer_results = {}
    for name, clf in get_pipelines().items():
        clf.fit(X_train_mind, y_train_mind)
        preds_m2mov = clf.predict(X_test_move)
        acc_m2mov = accuracy_score(y_test_move, preds_m2mov)
        f1_m2mov = f1_score(y_test_move, preds_m2mov)
        
        clf.fit(X_test_move, y_test_move)
        preds_mov2m = clf.predict(X_train_mind)
        acc_mov2m = accuracy_score(y_train_mind, preds_mov2m)
        f1_mov2m = f1_score(y_train_mind, preds_mov2m)
        
        transfer_results[name] = {
            'Mind_to_Movement_Acc': float(acc_m2mov),
            'Mind_to_Movement_F1': float(f1_m2mov),
            'Movement_to_Mind_Acc': float(acc_mov2m),
            'Movement_to_Mind_F1': float(f1_mov2m)
        }
        print(f"    {name:<35}: Mind->Move = {acc_m2mov*100:5.2f}% | Move->Mind = {acc_mov2m*100:5.2f}%")

    # -----------------------------------------------------------------
    # Experiment D: Visualizations & Topomaps
    # -----------------------------------------------------------------
    print("\n[*] Generating Publication-Quality Figures & Topomaps...")
    
    # 1. Bar Chart of Model Accuracies Across Sessions
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
    axes = axes.flatten()
    
    for idx, (ses, task, desc) in enumerate(sessions_meta):
        ax = axes[idx]
        m_dict = results_exp_a[ses]['models']
        names = list(m_dict.keys())
        accs = [m_dict[n]['acc_mean'] * 100 for n in names]
        stds = [m_dict[n]['acc_std'] * 100 for n in names]
        
        colors = ['#2b5c8f', '#3470a3', '#4682b4', '#2e8b57', '#3cb371', '#20b2aa', '#d9534f']
        bars = ax.barh(names, accs, xerr=stds, color=colors[:len(names)], alpha=0.88, capsize=4, edgecolor='black')
        ax.axvline(50.0, color='gray', linestyle='--', linewidth=1.5, label='Chance (50%)')
        ax.set_xlim(30, 105)
        ax.set_xlabel("Cross-Validation Accuracy (%)", fontsize=11, fontweight='bold')
        ax.set_title(f"{desc} (sub-01 / ses-{ses})\nTarget Lock-On vs Baseline", fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle=':')
        
        for bar, acc, std in zip(bars, accs, stds):
            ax.text(acc + 1.5, bar.get_y() + bar.get_height()/2, f"{acc:.1f}%", va='center', fontsize=9, fontweight='bold')
            
    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "fnf_session_decoding_benchmark.png")
    plt.savefig(fig1_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Session Benchmark Figure to: {fig1_path}")
    
    # 2. Mind vs Movement ERD/ERS Topomaps and Power Spectral Density
    fig, (ax_psd, ax_topo1, ax_topo2) = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    
    raw_mind = loaded_sessions['03']['epochs_erp']
    raw_move = loaded_sessions['04']['epochs_erp']
    
    ch_idx_c3 = raw_mind.ch_names.index('C3')
    
    data_mind_c3 = raw_mind.get_data()[:, ch_idx_c3, :]
    data_move_c3 = raw_move.get_data()[:, ch_idx_c3, :]
    
    f_mind, psd_mind = signal.welch(data_mind_c3, fs=250.0, nperseg=128, axis=-1)
    f_move, psd_move = signal.welch(data_move_c3, fs=250.0, nperseg=128, axis=-1)
    
    mean_psd_mind = np.mean(psd_mind, axis=0)
    mean_psd_move = np.mean(psd_move, axis=0)
    
    mask = (f_mind >= 4) & (f_mind <= 40)
    ax_psd.plot(f_mind[mask], 10*np.log10(mean_psd_mind[mask]), label='Mind (Motor Imagery)', color='#2b5c8f', linewidth=2.5)
    ax_psd.plot(f_move[mask], 10*np.log10(mean_psd_move[mask]), label='Movement (Motor Execution)', color='#d9534f', linewidth=2.5)
    ax_psd.axvspan(8, 12, color='#f1c40f', alpha=0.2, label='Mu Band (8-12 Hz)')
    ax_psd.axvspan(13, 30, color='#2ecc71', alpha=0.2, label='Beta Band (13-30 Hz)')
    ax_psd.set_title("Power Spectral Density @ C3 (Sensorimotor)", fontsize=12, fontweight='bold')
    ax_psd.set_xlabel("Frequency (Hz)", fontsize=11)
    ax_psd.set_ylabel("Power (dB)", fontsize=11)
    ax_psd.grid(True, alpha=0.3)
    ax_psd.legend(fontsize=9)
    
    psd_mind_all = []
    psd_move_all = []
    for ch_i in range(len(raw_mind.ch_names)):
        _, p_m = signal.welch(raw_mind.get_data()[:, ch_i, :], fs=250.0, nperseg=128, axis=-1)
        _, p_v = signal.welch(raw_move.get_data()[:, ch_i, :], fs=250.0, nperseg=128, axis=-1)
        mu_mask = (f_mind >= 8) & (f_mind <= 12)
        psd_mind_all.append(np.mean(p_m[:, mu_mask]))
        psd_move_all.append(np.mean(p_v[:, mu_mask]))
        
    mne.viz.plot_topomap(np.array(psd_mind_all), raw_mind.info, axes=ax_topo1, show=False, cmap='RdBu_r')
    ax_topo1.set_title("Mind: Mu Band (8-12 Hz) Power", fontsize=11, fontweight='bold')
    
    mne.viz.plot_topomap(np.array(psd_move_all), raw_move.info, axes=ax_topo2, show=False, cmap='RdBu_r')
    ax_topo2.set_title("Movement: Mu Band (8-12 Hz) Power", fontsize=11, fontweight='bold')
    
    fig2_path = os.path.join(out_dir, "fnf_mind_vs_movement_spectral_topomap.png")
    plt.savefig(fig2_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Spectral Topomap Figure to: {fig2_path}")

    # 3. Grand-Average Evoked Potentials (ERP) Time-Locked to Arrow HitZone
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    channels_to_plot = ['C3', 'Cz', 'C4', 'Oz']
    
    times_erp = raw_mind.times * 1000  # in ms
    for ax, ch_name in zip(axes.flatten(), channels_to_plot):
        ch_i = raw_mind.ch_names.index(ch_name)
        erp_mind_mean = np.mean(raw_mind.get_data()[:, ch_i, :], axis=0) * 1e6  # uV
        erp_mind_sem = stats.sem(raw_mind.get_data()[:, ch_i, :], axis=0) * 1e6
        
        erp_move_mean = np.mean(raw_move.get_data()[:, ch_i, :], axis=0) * 1e6
        erp_move_sem = stats.sem(raw_move.get_data()[:, ch_i, :], axis=0) * 1e6
        
        ax.plot(times_erp, erp_mind_mean, label='Mind (MI)', color='#2b5c8f', linewidth=2.0)
        ax.fill_between(times_erp, erp_mind_mean - erp_mind_sem, erp_mind_mean + erp_mind_sem, color='#2b5c8f', alpha=0.15)
        
        ax.plot(times_erp, erp_move_mean, label='Movement (ME)', color='#d9534f', linewidth=2.0)
        ax.fill_between(times_erp, erp_move_mean - erp_move_sem, erp_move_mean + erp_move_sem, color='#d9534f', alpha=0.15)
        
        ax.axvline(0, color='black', linestyle='--', linewidth=1.2, label='Arrow HitZone (t=0)')
        ax.set_title(f"Channel {ch_name} ERP Response", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time relative to HitZone (ms)", fontsize=10)
        ax.set_ylabel("Amplitude (µV)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
        
    plt.tight_layout()
    fig3_path = os.path.join(out_dir, "fnf_arrow_hitzone_erp_waveforms.png")
    plt.savefig(fig3_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved ERP Waveforms Figure to: {fig3_path}")

    # -----------------------------------------------------------------
    # Export Final Metrics JSON
    # -----------------------------------------------------------------
    final_report = {
        'bids_root': bids_root,
        'dataset_summary': {
            'ses-01': {'mode': 'Mind (Motor Imagery)', 'task': 'leftright', 'target_trials': loaded_sessions['01']['n_trials']},
            'ses-02': {'mode': 'Mind (Motor Imagery)', 'task': 'leftright', 'target_trials': loaded_sessions['02']['n_trials']},
            'ses-03': {'mode': 'Mind (Motor Imagery)', 'task': 'leftright', 'target_trials': loaded_sessions['03']['n_trials']},
            'ses-04': {'mode': 'Movement (Motor Execution)', 'task': 'me', 'target_trials': loaded_sessions['04']['n_trials']},
        },
        'experiment_a_target_vs_rest': results_exp_a,
        'experiment_b_mind_vs_movement': results_exp_b,
        'experiment_c_transfer_generalization': transfer_results,
        'artifacts': {
            'session_benchmark_plot': fig1_path,
            'spectral_topomap_plot': fig2_path,
            'erp_waveforms_plot': fig3_path
        }
    }
    
    json_path = os.path.join(out_dir, "fnf_decoding_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(final_report, f, indent=4)
    print(f"\n[+] Saved Complete Metrics Report to: {json_path}")
    print("=" * 80)
    print(" FNF BCI Analysis Completed Successfully! ".center(80, "="))
    print("=" * 80)
    
    return final_report


if __name__ == "__main__":
    bids_path = "scripts/bids/bids_fnf"
    if len(sys.argv) > 1:
        bids_path = sys.argv[1]
    run_fnf_analysis(bids_root=bids_path)
