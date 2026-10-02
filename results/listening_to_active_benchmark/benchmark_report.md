# Benchmarking Listening Priors to Active Game Intent Transfer

Evaluates the core hypothesis: **Does passive music listening activity provide an effective spatial prior that aids active intent decoding in few-shot calibration?**

Datasets used: [`bids_td_modern`](file:///run/media/john/ssd_external/git/bci_projects/nautilus_bci/scripts/bids_clean/bids_td_modern), [`bids_td_classical`](file:///run/media/john/ssd_external/git/bci_projects/nautilus_bci/scripts/bids_clean/bids_td_classical), and [`bids_fnf`](file:///run/media/john/ssd_external/git/bci_projects/nautilus_bci/scripts/bids_clean/bids_fnf).

## Sub-02 Modern: 4-Song Prior -> ses-03 (Cz/F4 Restored Peak)

- **Model Capacity Overfit**: `100.00%`
- **Pure Zero-Shot Transfer ($k=0$)**: `23.19%`

| Shots ($k$) | Active Baseline (%) | Listening Transfer (%) | Net Gain (%) | Empirical Chance (%) |
| :---: | :---: | :---: | :---: | :---: |
| **k=1** | 25.33% ± 4.24% | **22.87% ± 4.49%** | **-2.46%** | 25.38% |
| **k=2** | 26.28% ± 4.72% | **25.57% ± 4.05%** | **-0.71%** | 26.50% |
| **k=3** | 27.08% ± 5.26% | **26.61% ± 5.23%** | **-0.47%** | 24.80% |
| **k=5** | 29.66% ± 6.84% | **27.96% ± 5.63%** | **-1.70%** | 24.83% |
| **k=8** | 32.97% ± 5.51% | **31.80% ± 6.65%** | **-1.17%** | 23.42% |

---

## Sub-02 Modern: 4-Song Prior -> ses-02

- **Model Capacity Overfit**: `100.00%`
- **Pure Zero-Shot Transfer ($k=0$)**: `20.59%`

| Shots ($k$) | Active Baseline (%) | Listening Transfer (%) | Net Gain (%) | Empirical Chance (%) |
| :---: | :---: | :---: | :---: | :---: |
| **k=1** | 24.33% ± 5.78% | **26.56% ± 6.96%** | **+2.22%** | 26.11% |
| **k=2** | 27.18% ± 6.43% | **27.69% ± 5.56%** | **+0.51%** | 25.77% |
| **k=3** | 30.76% ± 9.37% | **32.42% ± 9.66%** | **+1.67%** | 24.39% |
| **k=5** | 34.76% ± 8.40% | **34.76% ± 9.54%** | **-0.00%** | 25.95% |

---

## Sub-01 Classical: Orchestral Listening Prior -> Active Spell Recall

- **Model Capacity Overfit**: `78.87%`
- **Pure Zero-Shot Transfer ($k=0$)**: `25.77%`

| Shots ($k$) | Active Baseline (%) | Listening Transfer (%) | Net Gain (%) | Empirical Chance (%) |
| :---: | :---: | :---: | :---: | :---: |
| **k=1** | 25.72% ± 3.23% | **24.35% ± 2.65%** | **-1.37%** | 25.11% |
| **k=2** | 25.27% ± 2.99% | **25.75% ± 2.60%** | **+0.48%** | 24.78% |
| **k=3** | 24.85% ± 3.16% | **25.40% ± 3.37%** | **+0.55%** | 24.34% |
| **k=5** | 26.05% ± 3.25% | **27.13% ± 3.32%** | **+1.07%** | 24.60% |
| **k=8** | 24.55% ± 3.11% | **25.51% ± 3.03%** | **+0.97%** | 25.66% |
| **k=10** | 26.47% ± 3.32% | **26.93% ± 3.58%** | **+0.45%** | 26.17% |

---

## FNF Rhythm Game: Auditory Cues Prior -> Silent Directional HitZone Recall

- **Model Capacity Overfit**: `83.01%`
- **Pure Zero-Shot Transfer ($k=0$)**: `70.75%`

| Shots ($k$) | Active Baseline (%) | Listening Transfer (%) | Net Gain (%) | Empirical Chance (%) |
| :---: | :---: | :---: | :---: | :---: |
| **k=1** | 53.80% ± 7.71% | **72.00% ± 2.74%** | **+18.19%** | 25.69% |
| **k=2** | 60.04% ± 5.46% | **74.02% ± 2.81%** | **+13.98%** | 30.07% |
| **k=3** | 63.34% ± 3.38% | **74.82% ± 1.76%** | **+11.48%** | 22.73% |
| **k=5** | 66.55% ± 4.13% | **74.50% ± 2.38%** | **+7.96%** | 25.44% |
| **k=8** | 70.61% ± 3.13% | **76.40% ± 1.99%** | **+5.79%** | 21.86% |
| **k=10** | 71.60% ± 3.45% | **75.97% ± 1.67%** | **+4.38%** | 24.03% |

---

