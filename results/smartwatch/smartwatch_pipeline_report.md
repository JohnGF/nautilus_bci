# Smartwatch Multimodal Prediction Pipeline Report

**Execution Script**: `scripts/analysis/smartwatch_prediction_pipeline.py`  
**Dataset Sources**: `scripts/bids/bids_baseline/sub-01/ses-02 (Motion + Physio PPG)`  
**Modalities**: Smartwatch IMU 6-Axis Motion (Accelerometer & Gyroscope), Smartwatch Optical PPG (Pulse & HRV)  
**Cross-Validation**: Stratified 5-Fold Cross-Validation  
**Samples Extracted**: 157 samples | **Feature Dimensions**: 35 features  

## 1. Benchmark Comparison Table

| Algorithm / Model                       | Accuracy   | Balanced Accuracy   |   F1-Score (Macro) |   Precision |   Recall |   ROC-AUC |
|:----------------------------------------|:-----------|:--------------------|-------------------:|------------:|---------:|----------:|
| Random Forest (RF)                      | 25.48%     | 25.53%              |             0.2558 |      0.2567 |   0.2553 |    0.506  |
| Gradient Boosting (GBM)                 | 24.84%     | 24.87%              |             0.2489 |      0.2504 |   0.2487 |    0.4765 |
| Support Vector Machine (SVC-RBF)        | 17.83%     | 17.87%              |             0.1611 |      0.1664 |   0.1787 |    0.4856 |
| L2 Logistic Regression (LogReg)         | 21.02%     | 21.01%              |             0.2111 |      0.2176 |   0.2101 |    0.4812 |
| Multi-Layer Perceptron (MLP Neural Net) | 27.39%     | 27.37%              |             0.2747 |      0.2772 |   0.2737 |    0.5154 |

## 2. Top 10 Informative Features

| Feature Name | Modality | Random Forest Importance |
| :--- | :--- | :--- |
| `imu_gyro_z_std` | IMU / Kinematic | 0.0578 |
| `imu_acc_x_std` | IMU / Kinematic | 0.0547 |
| `imu_acc_spectral_entropy` | IMU / Kinematic | 0.0522 |
| `imu_acc_x_mean` | IMU / Kinematic | 0.0461 |
| `imu_acc_mag_std` | IMU / Kinematic | 0.0454 |
| `imu_gyro_energy` | IMU / Kinematic | 0.0443 |
| `imu_gyro_mag_mean` | IMU / Kinematic | 0.0440 |
| `imu_acc_mag_max` | IMU / Kinematic | 0.0431 |
| `imu_gyro_x_mean` | IMU / Kinematic | 0.0428 |
| `imu_acc_z_mean` | IMU / Kinematic | 0.0421 |

## 3. Visual Artifacts Generated

- `smartwatch_model_benchmark.png`
- `smartwatch_confusion_matrices.png`
- `smartwatch_feature_importance.png`
