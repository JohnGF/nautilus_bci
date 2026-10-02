#!/usr/bin/env python3
"""
Hierarchical Assistive BCI Decoder with Sequential Evidence Accumulation
========================================================================
Designed specifically for individuals with severe motor disabilities:
  1. 100% Pure Mental Imagery (No motor execution, no physical twitch needed).
  2. Hierarchical Binary Tree Decision:
       - Level 1: Macro-Intent (ENERGY: Fire/Electricity vs NATURE: Water/Wind)
       - Level 2a: Fire vs Electricity
       - Level 2b: Water vs Wind
  3. Sequential Evidence Accumulation (Drift Diffusion / Bayesian Belief Updating):
       - Accumulates posterior probabilities across time slices
       - Triggers action ONLY when decision certainty crosses threshold (e.g. 85%)
       - Eliminates false positive commands and reduces user frustration.
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_td_data import load_modern_tower_defense, load_classical_tower_defense

class HierarchicalAssistiveDecoder:
    """
    Two-tier hierarchical Riemannian classifier with Bayesian sequential accumulation.
    """
    def __init__(self, confidence_threshold=0.85, max_steps=3):
        self.confidence_threshold = confidence_threshold
        self.max_steps = max_steps
        
        # Pipelines for the 3 binary decision nodes
        # Node 0: Macro Group (Energy: {Fire, Electricity} vs Nature: {Water, Wind})
        # Node 1: Energy Subtree (Fire vs Electricity)
        # Node 2: Nature Subtree (Water vs Wind)
        self.clf_root = self._build_node_pipeline()
        self.clf_energy = self._build_node_pipeline()
        self.clf_nature = self._build_node_pipeline()
        
    def _build_node_pipeline(self):
        return make_pipeline(
            Covariances(estimator='oas'),
            TangentSpace(metric='riemann'),
            LogisticRegression(C=1.0, max_iter=500, solver='lbfgs')
        )
        
    def fit(self, X, y):
        """
        Fit the 3 hierarchical nodes.
        Classes:
          0: FIRE
          1: WATER
          2: WIND
          3: ELECTRICITY
        """
        # Node 0: Root Macro-Category (Energy: {0, 3} vs Nature: {1, 2})
        # 0 -> Energy, 1 -> Nature
        y_root = np.where(np.isin(y, [0, 3]), 0, 1)
        self.clf_root.fit(X, y_root)
        
        # Node 1: Energy Subtree (0: FIRE vs 3: ELECTRICITY)
        mask_energy = np.isin(y, [0, 3])
        if np.sum(mask_energy) > 0:
            y_energy = np.where(y[mask_energy] == 0, 0, 1) # 0: Fire, 1: Electricity
            self.clf_energy.fit(X[mask_energy], y_energy)
            
        # Node 2: Nature Subtree (1: WATER vs 2: WIND)
        mask_nature = np.isin(y, [1, 2])
        if np.sum(mask_nature) > 0:
            y_nature = np.where(y[mask_nature] == 1, 0, 1) # 0: Water, 1: Wind
            self.clf_nature.fit(X[mask_nature], y_nature)
            
        return self

    def predict_single_epoch(self, x_epoch):
        """
        Predicts 4-class probabilities for a single trial window.
        x_epoch: (1, Channels, Samples) or (Channels, Samples)
        """
        if x_epoch.ndim == 2:
            x_epoch = x_epoch[np.newaxis, ...]
            
        p_root = self.clf_root.predict_proba(x_epoch)[0] # [P(Energy), P(Nature)]
        p_energy = self.clf_energy.predict_proba(x_epoch)[0] # [P(Fire|Energy), P(Elec|Energy)]
        p_nature = self.clf_nature.predict_proba(x_epoch)[0] # [P(Water|Nature), P(Wind|Nature)]
        
        p_fire = p_root[0] * p_energy[0]
        p_elec = p_root[0] * p_energy[1]
        p_water = p_root[1] * p_nature[0]
        p_wind = p_root[1] * p_nature[1]
        
        # Probabilities for classes [0: FIRE, 1: WATER, 2: WIND, 3: ELECTRICITY]
        probs = np.array([p_fire, p_water, p_wind, p_elec])
        probs = probs / np.sum(probs)
        return probs

    def accumulate_and_decide(self, trial_slices):
        """
        Sequential evidence accumulation over time slices (Bayesian update).
        trial_slices: list of window slices for a single mental imagery attempt, e.g. [t1, t2, t3].
        Returns:
          final_class: int (0 to 3)
          final_confidence: float
          epochs_needed: int
          confirmed: bool (whether confidence >= threshold)
        """
        # Prior is uniform
        log_odds = np.zeros(4)
        
        for step, x_slice in enumerate(trial_slices[:self.max_steps], start=1):
            probs = self.predict_single_epoch(x_slice)
            # Bayesian update in log domain
            log_odds += np.log(np.clip(probs, 1e-4, 1.0 - 1e-4))
            
            # Normalize to posterior distribution
            exp_odds = np.exp(log_odds - np.max(log_odds))
            posterior = exp_odds / np.sum(exp_odds)
            
            top_class = int(np.argmax(posterior))
            top_conf = float(posterior[top_class])
            
            if top_conf >= self.confidence_threshold:
                return top_class, top_conf, step, True
                
        # If threshold not reached after max_steps, return best current candidate
        exp_odds = np.exp(log_odds - np.max(log_odds))
        posterior = exp_odds / np.sum(exp_odds)
        top_class = int(np.argmax(posterior))
        top_conf = float(posterior[top_class])
        return top_class, top_conf, len(trial_slices[:self.max_steps]), False

def evaluate_hierarchical_accumulation(X, y, n_splits=5, threshold=0.80):
    """
    Evaluates Single-Epoch Hierarchical vs Multi-Epoch Evidence Accumulation.
    Simulates real-world time slices: 
      Full trial (5.0s, 750 samples) is sliced into three 2.5s overlapping windows:
        Window 1: 0.0s - 2.5s (samples 0 - 625)
        Window 2: 1.0s - 3.5s (samples 250 - 875 -> clamped)
        Window 3: 2.0s - 4.5s (samples 500 - 750)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    single_epoch_accs = []
    accumulated_accs = []
    confirmed_accs = []
    confirmed_rates = []
    steps_list = []
    
    for tr_idx, te_idx in skf.split(X, y):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]
        
        decoder = HierarchicalAssistiveDecoder(confidence_threshold=threshold, max_steps=3)
        decoder.fit(X_tr, y_tr)
        
        # Evaluate Single-Epoch
        preds_single = []
        for i in range(len(X_te)):
            p = decoder.predict_single_epoch(X_te[i])
            preds_single.append(np.argmax(p))
        single_epoch_accs.append(np.mean(preds_single == y_te))
        
        # Evaluate Sequential Accumulation using temporal slices
        preds_acc = []
        conf_preds = []
        conf_true = []
        
        # Slice parameters: 3 overlapping 2.5s sub-windows from the 3s/5s trial
        sfreq = 250.0
        w_len = int(2.0 * sfreq) # 500 samples
        step_sz = int(0.5 * sfreq) # 125 samples
        
        for i in range(len(X_te)):
            trial = X_te[i] # (Channels, Time)
            n_samples = trial.shape[1]
            slices = []
            for start in range(0, max(1, n_samples - w_len + 1), step_sz):
                sl = trial[:, start:start+w_len]
                if sl.shape[1] == w_len:
                    slices.append(sl)
            if len(slices) == 0:
                slices = [trial]
                
            pred_c, conf, steps_used, is_confirmed = decoder.accumulate_and_decide(slices)
            preds_acc.append(pred_c)
            steps_list.append(steps_used)
            
            if is_confirmed:
                conf_preds.append(pred_c)
                conf_true.append(y_te[i])
                
        accumulated_accs.append(np.mean(preds_acc == y_te))
        if len(conf_preds) > 0:
            confirmed_accs.append(np.mean(np.array(conf_preds) == np.array(conf_true)))
            confirmed_rates.append(len(conf_preds) / len(y_te))
            
    return {
        'single_epoch_mean': float(np.mean(single_epoch_accs)),
        'accumulated_mean': float(np.mean(accumulated_accs)),
        'confirmed_mean': float(np.mean(confirmed_accs)) if confirmed_accs else 0.0,
        'confirmed_rate': float(np.mean(confirmed_rates)) if confirmed_rates else 0.0,
        'avg_steps': float(np.mean(steps_list))
    }

def main():
    print("=" * 85)
    print("HIERARCHICAL ASSISTIVE DECODER BENCHMARK: EVIDENCE ACCUMULATION")
    print("Pure Mental Imagery for Motor-Impaired Users")
    print("=" * 85)
    
    # 1. Evaluate on Subject 2 ses-03 Peak (69 trials)
    print("\n[1/2] Testing on Subject 2 Modern Session 03 (Dry Cz/F4 Reseated)...")
    (X_s03, y_s03, _), _ = load_modern_tower_defense(sessions=["03"])
    res_s03 = evaluate_hierarchical_accumulation(X_s03, y_s03, threshold=0.75)
    
    # 2. Evaluate on Subject 2 Golden Pooled (168 trials)
    print("\n[2/2] Testing on Subject 2 Golden Pooled (ses-01 + ses-02 + ses-03)...")
    (X_pool, y_pool, _), _ = load_modern_tower_defense(recommended_only=True)
    res_pool = evaluate_hierarchical_accumulation(X_pool, y_pool, threshold=0.75)
    
    print("\n" + "=" * 85)
    print(f"{'Metric':<40} | {'Sub-02 ses-03':<18} | {'Sub-02 Golden Pooled':<18}")
    print("=" * 85)
    print(f"{'Single-Epoch 4-Class Accuracy':<40} | {res_s03['single_epoch_mean']*100:6.2f}%            | {res_pool['single_epoch_mean']*100:6.2f}%")
    print(f"{'Accumulated Decision Accuracy':<40} | {res_s03['accumulated_mean']*100:6.2f}%            | {res_pool['accumulated_mean']*100:6.2f}%")
    print(f"{'Confirmed-Only Accuracy (>75% Cert)':<40} | {res_s03['confirmed_mean']*100:6.2f}%            | {res_pool['confirmed_mean']*100:6.2f}%")
    print(f"{'Confirmation Trigger Rate':<40} | {res_s03['confirmed_rate']*100:6.2f}%            | {res_pool['confirmed_rate']*100:6.2f}%")
    print(f"{'Average Decision Steps (Windows)':<40} | {res_s03['avg_steps']:6.2f} windows       | {res_pool['avg_steps']:6.2f} windows")
    print("=" * 85)
    
    # Save results
    out_dir = Path(__file__).resolve().parent.parent.parent / "results" / "assistive_decoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "hierarchical_decoder_results.json", "w") as f:
        json.dump({'sub02_ses03': res_s03, 'sub02_golden_pooled': res_pool}, f, indent=2)
    print(f"Results saved to {out_dir / 'hierarchical_decoder_results.json'}")

if __name__ == "__main__":
    main()
