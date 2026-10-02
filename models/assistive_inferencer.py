#!/usr/bin/env python3
"""
Assistive BCI Real-Time Inferencer
==================================
Drop-in inference engine for assistive communication boards, games (Tower Defense / FNF),
and neuro-rehabilitation interfaces.

Supports:
  1. 'hierarchical': 2-Tier Riemannian Tangent Space with Bayesian evidence accumulation (Default).
  2. 'eegnet': Pretrained compact deep learning model on GPU/CPU.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import joblib
import torch

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR
sys.path.insert(0, str(BASE_DIR.parent / "scripts" / "analysis"))
from hierarchical_assistive_decoder import HierarchicalAssistiveDecoder

CLASS_NAMES = {
    0: "FIRE",
    1: "WATER",
    2: "WIND",
    3: "ELECTRICITY"
}

class AssistiveBCIInferencer:
    """
    Real-time inference interface for 32-channel dry EEG mental imagery.
    """
    def __init__(self, model_type="hierarchical", confidence_threshold=0.80, max_steps=3, device=None):
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.max_steps = max_steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Metadata
        meta_path = MODELS_DIR / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            
        self.sfreq = self.metadata.get("sampling_rate_hz", 250.0)
        self.n_channels = self.metadata.get("channels", 32)
        self.window_samples = int(self.metadata.get("epoch_duration_s", 3.0) * self.sfreq)
        
        # Load Model
        if self.model_type == "hierarchical":
            model_file = MODELS_DIR / "hierarchical_riemannian_sub02.joblib"
            if not model_file.exists():
                raise FileNotFoundError(f"Missing model file: {model_file}")
            self.model = joblib.load(model_file)
            self.model.confidence_threshold = self.confidence_threshold
            self.model.max_steps = self.max_steps
        elif self.model_type == "eegnet":
            from scripts.analysis.benchmark_deep_learning import EEGNet
            self.model = EEGNet(n_classes=4, channels=self.n_channels, samples=self.window_samples).to(self.device)
            weights_file = MODELS_DIR / "pretrained_eegnet_sub02.pt"
            if not weights_file.exists():
                raise FileNotFoundError(f"Missing EEGNet weights: {weights_file}")
            self.model.load_state_dict(torch.load(weights_file, map_location=self.device))
            self.model.eval()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        # Streaming Buffer & Log-Odds Accumulator
        self.buffer = np.zeros((self.n_channels, 0))
        self.log_odds = np.zeros(4)
        self.steps_accumulated = 0

    def reset(self):
        """Resets streaming ring buffer and accumulated evidence."""
        self.buffer = np.zeros((self.n_channels, 0))
        self.log_odds = np.zeros(4)
        self.steps_accumulated = 0

    def predict_window(self, eeg_window):
        """
        Infers intent on a single window (Channels, Samples) or (Samples, Channels).
        Returns:
          probs: array of 4 probabilities [P(Fire), P(Water), P(Wind), P(Elec)]
        """
        eeg_arr = np.asarray(eeg_window, dtype=np.float64)
        # Transpose if passed (Samples, Channels)
        if eeg_arr.shape[0] != self.n_channels and eeg_arr.shape[1] == self.n_channels:
            eeg_arr = eeg_arr.T
            
        # Ensure 33 or 32 channels compatible
        if eeg_arr.shape[0] == 32 and self.n_channels == 33:
            # Pad 1 reference channel (reference electrode)
            ref_ch = np.zeros((1, eeg_arr.shape[1]))
            eeg_arr = np.vstack([eeg_arr, ref_ch])
        elif eeg_arr.shape[0] < self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {eeg_arr.shape[0]}")
            
        if self.model_type == "hierarchical":
            probs = self.model.predict_single_epoch(eeg_arr)
        else:
            # EEGNet
            norm_mean = np.mean(eeg_arr, axis=-1, keepdims=True)
            norm_std = np.std(eeg_arr, axis=-1, keepdims=True) + 1e-6
            normed = (eeg_arr - norm_mean) / norm_std
            t_x = torch.tensor(normed[np.newaxis, ...], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                logits = self.model(t_x)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                
        return probs

    def push_chunk(self, eeg_chunk):
        """
        Streams continuous real-time EEG chunks from hardware (e.g. 50-100ms chunks).
        Automatically accumulates evidence across overlapping sub-windows.
        Returns:
          result dict with:
            - 'intent': str or None
            - 'class_id': int or None
            - 'confidence': float
            - 'is_confirmed': bool
            - 'probabilities': dict
        """
        chunk = np.asarray(eeg_chunk, dtype=np.float64)
        if chunk.shape[0] != self.n_channels and chunk.shape[1] == self.n_channels:
            chunk = chunk.T
        if chunk.shape[0] == 32 and self.n_channels == 33:
            ref_ch = np.zeros((1, chunk.shape[1]))
            chunk = np.vstack([chunk, ref_ch])
            
        self.buffer = np.hstack([self.buffer, chunk])
        
        # If buffer exceeds window length, run evidence step
        if self.buffer.shape[1] >= self.window_samples:
            # Extract current window
            window = self.buffer[:, -self.window_samples:]
            probs = self.predict_window(window)
            
            # Bayesian update in log domain
            self.log_odds += np.log(np.clip(probs, 1e-4, 1.0 - 1e-4))
            self.steps_accumulated += 1
            
            exp_odds = np.exp(self.log_odds - np.max(self.log_odds))
            posterior = exp_odds / np.sum(exp_odds)
            
            top_class = int(np.argmax(posterior))
            top_conf = float(posterior[top_class])
            is_confirmed = (top_conf >= self.confidence_threshold) or (self.steps_accumulated >= self.max_steps)
            
            res = {
                "intent": CLASS_NAMES[top_class],
                "class_id": top_class,
                "confidence": top_conf,
                "is_confirmed": is_confirmed,
                "steps": self.steps_accumulated,
                "trial_buffer": window,
                "probabilities": {CLASS_NAMES[i]: float(posterior[i]) for i in range(4)}
            }
            
            if is_confirmed:
                # Reset accumulator for next intent
                self.reset()
                
            return res
            
        # Buffer still filling
        return {
            "intent": None,
            "class_id": None,
            "confidence": 0.0,
            "is_confirmed": False,
            "steps": 0,
            "trial_buffer": None,
            "probabilities": {CLASS_NAMES[i]: 0.25 for i in range(4)}
        }

    def adapt_online(self, confirmed_trial, true_label):
        """
        Ongoing training: updates active model with newly confirmed trials.
        Uses a rolling experience replay buffer (up to 45 trials) to prevent
        catastrophic forgetting and impedance drift.
        """
        if confirmed_trial is None:
            return
            
        trial = np.asarray(confirmed_trial, dtype=np.float64)
        if trial.shape[0] == 32 and self.n_channels == 33:
            trial = np.vstack([trial, np.zeros((1, trial.shape[1]))])
            
        if not hasattr(self, "replay_X"):
            self.replay_X = []
            self.replay_y = []
            
        self.replay_X.append(trial)
        self.replay_y.append(true_label)
        if len(self.replay_X) > 45:
            self.replay_X.pop(0)
            self.replay_y.pop(0)
            
        # If at least 12 trials and balanced classes exist, update hierarchical nodes
        if len(self.replay_X) >= 12 and len(np.unique(self.replay_y)) >= 2:
            if self.model_type == "hierarchical":
                try:
                    self.model.fit(np.array(self.replay_X), np.array(self.replay_y))
                except Exception:
                    pass

if __name__ == "__main__":
    print("Testing AssistiveBCIInferencer...")
    inferencer = AssistiveBCIInferencer(model_type="hierarchical", confidence_threshold=0.80)
    
    # Generate dummy 32-channel trial (3.0 seconds at 250Hz = 750 samples)
    dummy_eeg = np.random.randn(32, 750)
    probs = inferencer.predict_window(dummy_eeg)
    print(f"Single-window Prediction Probabilities:")
    for k, v in enumerate(probs):
        print(f"  {CLASS_NAMES[k]}: {v*100:.2f}%")
        
    print("\nSimulating streaming chunk ingestion (pushing 10 chunks of 100 samples)...")
    for step in range(10):
        chunk = np.random.randn(32, 100)
        res = inferencer.push_chunk(chunk)
        if res["is_confirmed"]:
            print(f"  -> INTENT CONFIRMED at step {step}: {res['intent']} (Confidence: {res['confidence']*100:.2f}%)")
            break
        else:
            print(f"  Step {step}: Buffer accumulating... top candidate: {res['intent']} (Conf: {res['confidence']*100:.2f}%)")
    print("Inference engine test passed successfully!")
