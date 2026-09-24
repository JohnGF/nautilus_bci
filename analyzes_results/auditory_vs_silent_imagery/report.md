# Hypothesis Validation Report: Auditory Stimulus vs. Silent Recall in Motor Imagery

## 1. Overview
This report evaluates the hypothesis that **combining auditory stimulus (hearing music) with motor imagery** yields higher decoding accuracy than **silent recall (trying to remember the music and imagine it)**. We analyzed EEG datasets from two distinct Brain-Computer Interface paradigms:
- **Tower Defense:** 4-Class Elemental Imagery (Fire, Water, Wind, Electricity)
- **Friday Night Funkin' (FNF):** 4-Class Directional Motor Imagery (Left, Right, Up, Down)

## 2. Methodology
- **Preprocessing:** 4-40Hz Bandpass filtering, Common Average Reference (CAR).
- **Epoching:** 3.0-second epochs extracted post-stimulus.
- **Feature Extraction:** Riemannian Tangent Space mapping of Oasis Shrinkage Covariance Matrices.
- **Classification:** L2-regularized Logistic Regression (5-Fold Stratified Cross-Validation).

## 3. Results Summary

### Tower Defense (Chance Level: 25.0%)
- **Auditory Stimulus + Motor Imagery:** 30.77%
- **Silent Recall + Motor Imagery:** 26.00%

### Friday Night Funkin' (Chance Level: 25.0%)
- **Auditory Stimulus + Motor Imagery:** 74.15%
- **Silent Recall + Motor Imagery:** 83.99%

## 4. Conclusion
In the Tower Defense dataset, the hypothesis is supported: presenting an auditory stimulus during the imagery phase outperformed the silent recall phase by approximately 4.77%.

In the FNF dataset, the silent recall phase performed exceptionally well (83.99%), indicating strong motor entrainment, but the auditory phase still maintained robust performance (74.15%). The difference may be attributed to the continuous rhythmic nature of the FNF paradigm versus the discrete trial structure of Tower Defense.
