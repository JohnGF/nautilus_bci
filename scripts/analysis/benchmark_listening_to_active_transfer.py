#!/usr/bin/env python3
"""
Benchmark: Passive Music Listening Prior -> Active Game Intent Few-Shot Transfer
================================================================================
Evaluates zero-shot (k=0) and few-shot (k in {1, 2, 3, 5, 8, 10}) transfer learning
using the newly streamlined, clean BIDS datasets:
  1. bids_td_modern (Subject 2): 4 Modern Songs Prior -> ses-03, ses-02, and All Sessions
  2. bids_td_classical (Subject 1): Continuous Classical Prior -> ses-04 and All Sessions
  3. bids_fnf (Subject 1): Auditory Prompts Prior -> Silent Directional HitZone Recall

Includes robust scientific controls:
  - Model Capacity Overfit Check
  - Shuffled Label Baseline (Permutation Empirical Chance)
  - Target-Only Baseline vs. Listening-Primed Transfer Model
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann

# Add analysis path for loaders
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "analysis"))

from load_td_data import load_classical_tower_defense, load_modern_tower_defense
import mne, mne_bids

OUTPUT_DIR = BASE_DIR.parent / "results" / "listening_to_active_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def invsqrtm(C):
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-9)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

def run_few_shot_experiment(X_src, y_src, X_tgt, y_tgt, exp_name, k_shots=[1, 2, 3, 5, 8, 10], n_mc=30, include_zero_shot=False):
    """
    Computes few-shot adaptation curves with Riemannian Procrustes Alignment.
    """
    print(f"\n=======================================================")
    print(f" EXPERIMENT: {exp_name}")
    print(f" Source (Listening Prior): {X_src.shape if X_src is not None else 'None'} | Target (Active): {X_tgt.shape}")
    print(f"=======================================================")

    n_ch = X_tgt.shape[1]
    classes = np.unique(y_tgt)
    n_classes = len(classes)

    # 1. Precompute OAS covariances upfront
    print("[*] Estimating regularized covariance matrices...")
    cov_est = Covariances(estimator='oas')
    C_tgt = cov_est.fit_transform(X_tgt)

    # Tangent space projection reference: Identity
    ts_identity = TangentSpace(metric='riemann', tsupdate=False)
    ts_identity.fit(np.array([np.eye(n_ch)]))

    # Center target domain
    C_bar_tgt = mean_riemann(C_tgt)
    W_tgt = invsqrtm(C_bar_tgt)
    C_tgt_aligned = np.array([W_tgt @ c @ W_tgt for c in C_tgt])
    feat_tgt = ts_identity.transform(C_tgt_aligned)

    feat_src = None
    if X_src is not None and y_src is not None and len(X_src) > 0:
        C_src = cov_est.fit_transform(X_src)
        C_bar_src = mean_riemann(C_src)
        W_src = invsqrtm(C_bar_src)
        C_src_aligned = np.array([W_src @ c @ W_src for c in C_src])
        feat_src = ts_identity.transform(C_src_aligned)

    # 2. Check Model Capacity (Overfit on 100% of data)
    clf_capacity = RidgeClassifier(alpha=1.0)
    clf_capacity.fit(feat_tgt, y_tgt)
    capacity_score = float(accuracy_score(y_tgt, clf_capacity.predict(feat_tgt)))
    print(f"[*] Model Capacity Overfit Check: {capacity_score * 100:.2f}% (Expected >85%)")

    # 3. Check Zero-Shot Transfer if source has labeled classes matching target
    zero_shot_acc = None
    if include_zero_shot and feat_src is not None and np.array_equal(np.unique(y_src), classes):
        clf_zero = RidgeClassifier(alpha=10.0)
        clf_zero.fit(feat_src, y_src)
        zero_shot_acc = float(accuracy_score(y_tgt, clf_zero.predict(feat_tgt)))
        print(f"[*] Pure Zero-Shot Transfer (k=0, trained ONLY on Listening Prior): {zero_shot_acc * 100:.2f}%")

    # 4. Few-Shot Monte Carlo Loop
    results = {
        'name': exp_name,
        'capacity_overfit': capacity_score,
        'zero_shot_k0': zero_shot_acc,
        'k': [],
        'baseline_mean': [],
        'baseline_std': [],
        'transfer_mean': [],
        'transfer_std': [],
        'shuffled_chance': [],
        'gain': []
    }

    min_class_count = min([np.sum(y_tgt == c) for c in classes])

    for k in k_shots:
        if k >= min_class_count - 1:
            continue

        acc_base, acc_trans, acc_shuff = [], [], []

        for seed in range(n_mc):
            rng = np.random.RandomState(seed + k * 500)

            # Sample k examples per class
            train_idx = []
            for c in classes:
                c_idx = np.where(y_tgt == c)[0]
                train_idx.extend(rng.choice(c_idx, size=k, replace=False))
            test_idx = np.setdiff1d(np.arange(len(y_tgt)), train_idx)

            X_tr, y_tr = feat_tgt[train_idx], y_tgt[train_idx]
            X_te, y_te = feat_tgt[test_idx], y_tgt[test_idx]

            # A. Baseline Model (Target k-shot only)
            clf_b = RidgeClassifier(alpha=10.0)
            clf_b.fit(X_tr, y_tr)
            acc_base.append(accuracy_score(y_te, clf_b.predict(X_te)))

            # B. Transfer Model (Listening Prior + k-shot target)
            if feat_src is not None and y_src is not None and len(feat_src) > 0:
                X_comb = np.vstack([feat_src, X_tr])
                y_comb = np.concatenate([y_src, y_tr])
                weights = np.ones(len(y_comb))
                # Weight target calibration samples appropriately
                ratio = len(feat_src) / max(len(X_tr), 1)
                weights[len(feat_src):] = min(ratio, 8.0)

                clf_t = RidgeClassifier(alpha=10.0)
                clf_t.fit(X_comb, y_comb, sample_weight=weights)
                acc_trans.append(accuracy_score(y_te, clf_t.predict(X_te)))
            else:
                acc_trans.append(acc_base[-1])

            # C. Shuffled Label Baseline (Permutation Chance)
            y_shuff = rng.permutation(y_tr)
            clf_s = RidgeClassifier(alpha=10.0)
            clf_s.fit(X_tr, y_shuff)
            acc_shuff.append(accuracy_score(y_te, clf_s.predict(X_te)))

        b_m, b_s = float(np.mean(acc_base)), float(np.std(acc_base))
        t_m, t_s = float(np.mean(acc_trans)), float(np.std(acc_trans))
        s_m = float(np.mean(acc_shuff))
        gain = float(t_m - b_m)

        results['k'].append(k)
        results['baseline_mean'].append(b_m)
        results['baseline_std'].append(b_s)
        results['transfer_mean'].append(t_m)
        results['transfer_std'].append(t_s)
        results['shuffled_chance'].append(s_m)
        results['gain'].append(gain)

        print(f"  k={k:2d} | Target Baseline: {b_m*100:5.2f}% +/- {b_s*100:4.2f}% | Listening Transfer: {t_m*100:5.2f}% +/- {t_s*100:4.2f}% | Gain: {gain*100:+5.2f}% | Chance: {s_m*100:5.2f}%")

    return results

def extract_fnf_dataset():
    """Extracts FNF auditory cues vs silent recall."""
    fnf_root = BASE_DIR / "bids_clean" / "bids_fnf"
    bp = mne_bids.BIDSPath(subject="01", session="07", task="leftright", datatype="eeg", root=fnf_root)
    raw = mne_bids.read_raw_bids(bp, verbose=False)
    raw.load_data()
    raw.pick_types(eeg=True)
    raw.filter(4.0, 40.0, fir_design='firwin', skip_by_annotation='edge', verbose=False)
    raw.set_eeg_reference('average', projection=True, verbose=False).apply_proj(verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    sfreq = raw.info['sfreq']
    ep_len = int(1.0 * sfreq)

    dir_map = {'Left': 0, 'Right': 1, 'Up': 2, 'Down': 3}
    spawn_events = {k: v for k, v in event_id.items() if '_Spawn' in k}
    hitzone_events = {k: v for k, v in event_id.items() if '_HitZone' in k}
    lis_id = event_id.get('Listen_Start') or event_id.get('Start Listen')

    X_lis, y_lis = [], []
    X_sil, y_sil = [], []

    for i, ev in enumerate(events):
        if lis_id and ev[2] == lis_id:
            for j in range(i+1, min(len(events), i+12)):
                for sname, sid in spawn_events.items():
                    if events[j, 2] == sid:
                        d = sname.split('_')[1]
                        if d in dir_map:
                            stop = ev[0] + ep_len
                            if stop <= len(raw.times):
                                ep = raw.get_data(start=ev[0], stop=stop)
                                if ep.shape[1] == ep_len:
                                    X_lis.append(ep)
                                    y_lis.append(dir_map[d])
                        break

        for hname, hid in hitzone_events.items():
            if ev[2] == hid:
                d = hname.split('_')[1]
                if d in dir_map:
                    stop = ev[0] + ep_len
                    if stop <= len(raw.times):
                        ep = raw.get_data(start=ev[0], stop=stop)
                        if ep.shape[1] == ep_len:
                            X_sil.append(ep)
                            y_sil.append(dir_map[d])
                break

    return (np.array(X_lis), np.array(y_lis)), (np.array(X_sil), np.array(y_sil))

def main():
    import os as _os, sys as _sys  # provenance bootstrap (stdlib only)
    _scripts_dir = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
    if _scripts_dir not in _sys.path:
        _sys.path.insert(0, _scripts_dir)
    from utils.provenance import write_provenance
    write_provenance(OUTPUT_DIR, script_file=__file__)
    all_experiments = []

    # -------------------------------------------------------------
    # 1. Subject 2: Modern Pop/Rock (Peak ses-03 Alone)
    # -------------------------------------------------------------
    (X_s03, y_s03, _), (X_lis_m, y_lis_m) = load_modern_tower_defense(sessions=["03"])
    res_s03 = run_few_shot_experiment(
        X_src=X_lis_m, y_src=y_lis_m,
        X_tgt=X_s03, y_tgt=y_s03,
        exp_name="Sub-02 Modern: 4-Song Prior -> ses-03 (Cz/F4 Restored Peak)",
        k_shots=[1, 2, 3, 5, 8],
        n_mc=30,
        include_zero_shot=True
    )
    all_experiments.append(res_s03)

    # -------------------------------------------------------------
    # 2. Subject 2: Modern Pop/Rock (ses-02 Alone)
    # -------------------------------------------------------------
    (X_s02, y_s02, _), _ = load_modern_tower_defense(sessions=["02"])
    res_s02 = run_few_shot_experiment(
        X_src=X_lis_m, y_src=y_lis_m,
        X_tgt=X_s02, y_tgt=y_s02,
        exp_name="Sub-02 Modern: 4-Song Prior -> ses-02",
        k_shots=[1, 2, 3, 5],
        n_mc=30,
        include_zero_shot=True
    )
    all_experiments.append(res_s02)

    # -------------------------------------------------------------
    # 3. Subject 1: Classical Paradigm (All Active Sessions)
    # -------------------------------------------------------------
    (X_rec_c, y_rec_c), (X_lis_c, y_lis_c) = load_classical_tower_defense()
    res_sub01 = run_few_shot_experiment(
        X_src=X_lis_c, y_src=y_lis_c,
        X_tgt=X_rec_c, y_tgt=y_rec_c,
        exp_name="Sub-01 Classical: Orchestral Listening Prior -> Active Spell Recall",
        k_shots=[1, 2, 3, 5, 8, 10],
        n_mc=30,
        include_zero_shot=True
    )
    all_experiments.append(res_sub01)

    # -------------------------------------------------------------
    # 4. FNF Directional Rhythm Game (Subject 1)
    # -------------------------------------------------------------
    (X_fnf_lis, y_fnf_lis), (X_fnf_sil, y_fnf_sil) = extract_fnf_dataset()
    res_fnf = run_few_shot_experiment(
        X_src=X_fnf_lis, y_src=y_fnf_lis,
        X_tgt=X_fnf_sil, y_tgt=y_fnf_sil,
        exp_name="FNF Rhythm Game: Auditory Cues Prior -> Silent Directional HitZone Recall",
        k_shots=[1, 2, 3, 5, 8, 10],
        n_mc=30,
        include_zero_shot=True
    )
    all_experiments.append(res_fnf)

    # -------------------------------------------------------------
    # Save Results & Plots
    # -------------------------------------------------------------
    json_path = OUTPUT_DIR / "benchmark_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_experiments, f, indent=4)
    print(f"\n[+] Saved quantitative benchmark to {json_path}")

    # Plot Comparison Grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, exp in enumerate(all_experiments):
        ax = axes[idx]
        k_vals = exp['k']
        ax.plot(k_vals, [m*100 for m in exp['transfer_mean']], 'o-', color='#10b981', lw=2.5, ms=7, label='Listening-Primed Transfer')
        ax.fill_between(k_vals, [(m-s)*100 for m,s in zip(exp['transfer_mean'], exp['transfer_std'])],
                                [(m+s)*100 for m,s in zip(exp['transfer_mean'], exp['transfer_std'])], color='#10b981', alpha=0.15)

        ax.plot(k_vals, [m*100 for m in exp['baseline_mean']], 's--', color='#6b7280', lw=2.0, ms=6, label='Active Baseline (No Prior)')
        ax.axhline(exp['shuffled_chance'][0]*100, color='#ef4444', linestyle=':', lw=1.8, label=f"Empirical Chance ({exp['shuffled_chance'][0]*100:.1f}%)")

        if exp['zero_shot_k0'] is not None:
            ax.scatter([0], [exp['zero_shot_k0']*100], color='#8b5cf6', s=90, zorder=5, label=f"Zero-Shot k=0 ({exp['zero_shot_k0']*100:.1f}%)")

        ax.set_title(exp['name'], fontsize=11, fontweight='bold')
        ax.set_xlabel('Calibration Shots per Class (k)', fontsize=10)
        ax.set_ylabel('4-Class Accuracy (%)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=9, loc='lower right')
        ax.set_ylim(15, 85)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "listening_to_active_transfer_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved publication-quality plot to {plot_path}")

    # Generate Markdown Report
    report_path = OUTPUT_DIR / "benchmark_report.md"
    with open(report_path, 'w') as f:
        f.write("# Benchmarking Listening Priors to Active Game Intent Transfer\n\n")
        f.write("Evaluates the core hypothesis: **Does passive music listening activity provide an effective spatial prior that aids active intent decoding in few-shot calibration?**\n\n")
        f.write("Datasets used: [`bids_td_modern`](file:///run/media/john/ssd_external/git/bci_projects/nautilus_bci/scripts/bids_clean/bids_td_modern), [`bids_td_classical`](file:///run/media/john/ssd_external/git/bci_projects/nautilus_bci/scripts/bids_clean/bids_td_classical), and [`bids_fnf`](file:///run/media/john/ssd_external/git/bci_projects/nautilus_bci/scripts/bids_clean/bids_fnf).\n\n")
        for exp in all_experiments:
            f.write(f"## {exp['name']}\n\n")
            f.write(f"- **Model Capacity Overfit**: `{exp['capacity_overfit']*100:.2f}%`\n")
            if exp['zero_shot_k0'] is not None:
                f.write(f"- **Pure Zero-Shot Transfer ($k=0$)**: `{exp['zero_shot_k0']*100:.2f}%`\n")
            f.write("\n| Shots ($k$) | Active Baseline (%) | Listening Transfer (%) | Net Gain (%) | Empirical Chance (%) |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: |\n")
            for i, k in enumerate(exp['k']):
                f.write(f"| **k={k}** | {exp['baseline_mean'][i]*100:.2f}% ± {exp['baseline_std'][i]*100:.2f}% | **{exp['transfer_mean'][i]*100:.2f}% ± {exp['transfer_std'][i]*100:.2f}%** | **{exp['gain'][i]*100:+.2f}%** | {exp['shuffled_chance'][i]*100:.2f}% |\n")
            f.write("\n---\n\n")

    print(f"[+] Generated comprehensive markdown report at {report_path}")

if __name__ == '__main__':
    main()
