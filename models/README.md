# Pretrained Assistive BCI Models & Real-Time Inferencer

This directory contains trained model weights, configurations, and a drop-in real-time inference engine for 32-channel dry EEG mental imagery.

---

## 📁 Directory Files

| File | Type | Description |
| :--- | :--- | :--- |
| [`assistive_inferencer.py`](assistive_inferencer.py) | Python Module | Real-time inferencer with sliding window buffer, Bayesian evidence accumulator, and online rolling adaptation. |
| [`hierarchical_riemannian_sub02.joblib`](hierarchical_riemannian_sub02.joblib) | Scikit-Learn Pipeline | 2-Tier Hierarchical Riemannian Tangent Space model trained on golden dry pure MI trials. |
| [`pretrained_eegnet_sub02.pt`](pretrained_eegnet_sub02.pt) | PyTorch State Dict | Compact spatial-temporal depthwise separable CNN pre-trained on listening priors and fine-tuned on pure MI. |
| [`model_metadata.json`](model_metadata.json) | JSON | Channel count (33), sampling rate (250Hz), class mappings, and default accumulation thresholds. |

---

## ⚡ Real-Time Inference Quick-Start

### 1. Ingesting Streaming Chunks (Game Loops & LSL)

```python
from models.assistive_inferencer import AssistiveBCIInferencer

# Initialize with 80% decision confidence threshold
inferencer = AssistiveBCIInferencer(
    model_type="hierarchical",        # or "eegnet"
    confidence_threshold=0.80,         # Minimum posterior certainty required before firing
    max_steps=3                        # Max windows to accumulate (~6-7 seconds)
)

# In your continuous streaming loop (e.g. 50-100ms chunks from g.tec Nautilus):
while is_streaming:
    chunk = stream.get_data()  # Shape: (32, chunk_samples)
    result = inferencer.push_chunk(chunk)
    
    if result["is_confirmed"]:
        spell = result["intent"]            # "FIRE", "WATER", "WIND", or "ELECTRICITY"
        confidence = result["confidence"]   # e.g. 0.86
        print(f"Triggering Action: {spell} ({confidence*100:.1f}%)")
        execute_command(spell)
        
        # Ongoing online adaptation:
        inferencer.adapt_online(result["trial_buffer"], result["class_id"])
```

### 2. Single-Window Prediction

```python
# Evaluates a single 3-second window (32 or 33 channels, 750 samples)
probs = inferencer.predict_window(eeg_window)
print("Probabilities [Fire, Water, Wind, Electricity]:", probs)
```
