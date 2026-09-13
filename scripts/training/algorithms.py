"""
algorithms.py
=============
Scikit-Learn compatible classifier implementations for all 10 rhythm decoding algorithms
benchmarked in analyze_tower_defense_rhythm_decoding.py:

  1. CSP_ShrinkageLDA
  2. CSP_SVM_RBF
  3. FilterBank_CSP_LogReg
  4. Riemannian_TangentSpace_LogReg
  5. Riemannian_TangentSpace_SVM_Linear
  6. Riemannian_TangentSpace_Ridge
  7. Riemannian_TangentSpace_SVM_RBF
  8. Welch_PSD_RandomForest
  9. Welch_PSD_ShrinkageLDA
 10. Ensemble_Voting (combines CSP + FBCSP + Riemannian Tangent Space)

All models accept:
  - Batch epochs: (n_epochs, n_channels, n_samples)
  - Single real-time windows: (n_channels, n_samples)
and output calibrated 4-class probabilities [FIRE, WATER, WIND, ELECTRICITY].
"""

import numpy as np
import scipy.signal as signal
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

# Ensure classes serialize cleanly under 'algorithms' module namespace
__module__ = "algorithms"

ELEMENT_NAMES = {
    0: "FIRE",
    1: "WATER",
    2: "WIND",
    3: "ELECTRICITY"
}

DEFAULT_BANDS = [
    ('Theta', 4.0, 8.0),
    ('Alpha', 8.0, 12.0),
    ('Low-Beta', 12.0, 20.0),
    ('High-Beta', 20.0, 32.0),
    ('Gamma', 32.0, 45.0)
]


# ----------------------------------------------------------------------
# Core Spatial & Mathematical Routines
# ----------------------------------------------------------------------
def compute_ovr_csp(X, y, n_components=4):
    """
    Computes One-vs-Rest CSP spatial filters.
    X: shape (n_epochs, n_channels, n_samples)
    y: shape (n_epochs,)
    n_components: number of CSP filters per class (half from each end)
    """
    n_epochs, n_ch, _ = X.shape
    classes = np.unique(y)
    covs = [np.cov(X[i]) / (np.trace(np.cov(X[i])) + 1e-12) for i in range(n_epochs)]

    if len(classes) == 2:
        c1, c2 = classes[0], classes[1]
        cov1 = np.mean([covs[k] for k in range(len(covs)) if y[k] == c1], axis=0) + 1e-5 * np.eye(n_ch)
        cov2 = np.mean([covs[k] for k in range(len(covs)) if y[k] == c2], axis=0) + 1e-5 * np.eye(n_ch)
        vals, vecs = eigh(cov1, cov1 + cov2)
        half = max(1, min(n_components // 2, n_ch // 2))
        return np.hstack([vecs[:, -half:], vecs[:, :half]])

    filters = []
    for c_id in classes:
        mask = (y == c_id)
        if not np.any(mask) or np.all(mask):
            continue
        cov_target = np.mean([covs[k] for k in range(len(covs)) if mask[k]], axis=0) + 1e-5 * np.eye(n_ch)
        cov_rest = np.mean([covs[k] for k in range(len(covs)) if not mask[k]], axis=0) + 1e-5 * np.eye(n_ch)
        vals, vecs = eigh(cov_target, cov_target + cov_rest)
        half = max(1, min(n_components // 2, n_ch // 2))
        filters.append(np.hstack([vecs[:, -half:], vecs[:, :half]]))

    if not filters:
        raise ValueError("Could not extract CSP filters: check class distribution.")
    return np.hstack(filters)


def project_csp_features(X, W):
    """Projects epochs through CSP spatial filters to log-variance features."""
    n_epochs = len(X)
    feats = np.zeros((n_epochs, W.shape[1]), dtype=np.float32)
    for i in range(n_epochs):
        proj = np.dot(W.T, X[i])
        var = np.var(proj, axis=1)
        feats[i] = np.log(var + 1e-12)
    return feats


def compute_covariance_matrices(X):
    """Computes regularized trace-normalized covariance matrices."""
    n_epochs, n_ch, _ = X.shape
    covs = np.zeros((n_epochs, n_ch, n_ch), dtype=np.float64)
    for i in range(n_epochs):
        c = np.cov(X[i])
        c = c / (np.trace(c) + 1e-12)
        c += 1e-5 * np.eye(n_ch)
        covs[i] = c
    return covs


def compute_riemannian_mean(covmats, max_iter=25, tol=1e-6):
    """Fréchet geometric mean on SPD manifold under the affine-invariant Riemannian metric."""
    C_mean = np.mean(covmats, axis=0)
    for _ in range(max_iter):
        vals, vecs = eigh(C_mean)
        vals = np.maximum(vals, 1e-8)
        sqrt_C = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        inv_sqrt_C = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

        tangents = []
        for i in range(len(covmats)):
            m = inv_sqrt_C @ covmats[i] @ inv_sqrt_C
            v, w = eigh(m)
            v = np.maximum(v, 1e-8)
            log_m = w @ np.diag(np.log(v)) @ w.T
            tangents.append(log_m)

        mean_t = np.mean(tangents, axis=0)
        if np.linalg.norm(mean_t, ord='fro') < tol:
            break
        v, w = eigh(mean_t)
        exp_t = w @ np.diag(np.exp(v)) @ w.T
        C_mean = sqrt_C @ exp_t @ sqrt_C
    return C_mean


def project_to_riemannian_tangent_space(covmats, C_ref=None):
    """Projects covariance matrices onto Euclidean Tangent Space at reference point C_ref."""
    if C_ref is None:
        C_ref = compute_riemannian_mean(covmats)

    vals, vecs = eigh(C_ref)
    vals = np.maximum(vals, 1e-8)
    inv_sqrt_C = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

    n_epochs, n_ch, _ = covmats.shape
    triu_idx = np.triu_indices(n_ch)
    diag_mask = (triu_idx[0] == triu_idx[1])

    ts_vectors = []
    for i in range(n_epochs):
        m = inv_sqrt_C @ covmats[i] @ inv_sqrt_C
        v, w = eigh(m)
        v = np.maximum(v, 1e-8)
        log_m = w @ np.diag(np.log(v)) @ w.T
        vec = log_m[triu_idx].copy()
        vec[~diag_mask] *= np.sqrt(2.0)
        ts_vectors.append(vec)

    return np.array(ts_vectors, dtype=np.float64), C_ref


def extract_welch_bandpower_features(X, sfreq=250.0):
    """Relative Welch Power Spectral Density across 5 classical EEG bands."""
    bands = {
        'delta': (1.0, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 12.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 45.0)
    }
    nperseg = min(int(sfreq * 1.5), X.shape[-1])
    freqs, psd = signal.welch(X, fs=sfreq, nperseg=nperseg, axis=-1)
    tot = np.sum(psd, axis=-1, keepdims=True) + 1e-12
    rel_psd = psd / tot

    band_feats = []
    for fmin, fmax in bands.values():
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_feats.append(np.mean(rel_psd[:, :, mask], axis=-1))
    return np.hstack(band_feats)


def _ensure_3d(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        # Standardize (channels, samples)
        if X.shape[0] > X.shape[1] and X.shape[1] == 32:
            X = X.T
        X = X[np.newaxis, ...]
    elif X.ndim != 3:
        raise ValueError(f"Expected 2D or 3D EEG data, got shape {X.shape}")
    return X


# ----------------------------------------------------------------------
# 1. CSP + Shrinkage LDA
# ----------------------------------------------------------------------
class CSP_ShrinkageLDA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.W_csp_ = None
        self.classifier_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = _ensure_3d(X)
        self.classes_ = np.unique(y)
        self.W_csp_ = compute_ovr_csp(X, y, n_components=self.n_components)
        feats = project_csp_features(X, self.W_csp_)
        self.classifier_ = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        self.classifier_.fit(feats, y)
        return self

    def transform(self, X):
        X = _ensure_3d(X)
        return project_csp_features(X, self.W_csp_)

    def predict_proba(self, X):
        feats = self.transform(X)
        return self.classifier_.predict_proba(feats)

    def predict(self, X):
        feats = self.transform(X)
        return self.classifier_.predict(feats)


# ----------------------------------------------------------------------
# 2. CSP + SVM (RBF)
# ----------------------------------------------------------------------
class CSP_SVM_RBF(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=4, C=1.0, random_state=42):
        self.n_components = n_components
        self.C = C
        self.random_state = random_state
        self.W_csp_ = None
        self.scaler_ = None
        self.classifier_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = _ensure_3d(X)
        self.classes_ = np.unique(y)
        self.W_csp_ = compute_ovr_csp(X, y, n_components=self.n_components)
        feats = project_csp_features(X, self.W_csp_)
        self.scaler_ = StandardScaler()
        feats_scaled = self.scaler_.fit_transform(feats)
        self.classifier_ = SVC(C=self.C, kernel='rbf', probability=True, random_state=self.random_state)
        self.classifier_.fit(feats_scaled, y)
        return self

    def transform(self, X):
        X = _ensure_3d(X)
        feats = project_csp_features(X, self.W_csp_)
        return self.scaler_.transform(feats)

    def predict_proba(self, X):
        feats_scaled = self.transform(X)
        return self.classifier_.predict_proba(feats_scaled)

    def predict(self, X):
        feats_scaled = self.transform(X)
        return self.classifier_.predict(feats_scaled)


# ----------------------------------------------------------------------
# 3. Filter Bank CSP + Logistic Regression
# ----------------------------------------------------------------------
class FilterBank_CSP_LogReg(BaseEstimator, ClassifierMixin):
    def __init__(self, bands=None, sfreq=250.0, n_components=4, C=0.5, random_state=42):
        self.bands = bands if bands is not None else DEFAULT_BANDS
        self.sfreq = float(sfreq)
        self.n_components = int(n_components)
        self.C = float(C)
        self.random_state = random_state
        self.filters_ = []
        self.scaler_ = None
        self.classifier_ = None
        self.classes_ = None

    def _filter_epoch(self, epoch, b, a):
        return signal.filtfilt(b, a, epoch, axis=-1)

    def fit(self, X, y):
        X = _ensure_3d(X)
        self.classes_ = np.unique(y)
        nyq = self.sfreq / 2.0
        self.filters_ = []
        band_feats = []

        for band_name, fmin, fmax in self.bands:
            b, a = signal.butter(4, [fmin / nyq, fmax / nyq], btype='band')
            X_filt = np.array([self._filter_epoch(X[i], b, a) for i in range(len(X))])
            W_csp = compute_ovr_csp(X_filt, y, n_components=self.n_components)
            self.filters_.append({'band_name': band_name, 'b': b, 'a': a, 'W': W_csp})
            feats = project_csp_features(X_filt, W_csp)
            band_feats.append(feats)

        X_all_feats = np.hstack(band_feats)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_all_feats)
        self.classifier_ = LogisticRegression(C=self.C, max_iter=500, random_state=self.random_state, solver='lbfgs')
        self.classifier_.fit(X_scaled, y)
        return self

    def transform(self, X):
        X = _ensure_3d(X)
        band_feats = []
        for filt_info in self.filters_:
            b, a, W = filt_info['b'], filt_info['a'], filt_info['W']
            X_filt = np.array([self._filter_epoch(X[i], b, a) for i in range(len(X))])
            feats = project_csp_features(X_filt, W)
            band_feats.append(feats)
        all_feats = np.hstack(band_feats)
        return self.scaler_.transform(all_feats)

    def predict_proba(self, X):
        feats = self.transform(X)
        return self.classifier_.predict_proba(feats)

    def predict(self, X):
        feats = self.transform(X)
        return self.classifier_.predict(feats)


# ----------------------------------------------------------------------
# 4. Riemannian Tangent Space + Logistic Regression
# ----------------------------------------------------------------------
class Riemannian_TangentSpace_LogReg(BaseEstimator, ClassifierMixin):
    def __init__(self, C=0.1, max_iter=500, random_state=42):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.C_ref_ = None
        self.scaler_ = None
        self.classifier_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = _ensure_3d(X)
        self.classes_ = np.unique(y)
        covs = compute_covariance_matrices(X)
        ts_vecs, self.C_ref_ = project_to_riemannian_tangent_space(covs)
        self.scaler_ = StandardScaler()
        ts_scaled = self.scaler_.fit_transform(ts_vecs)
        self.classifier_ = LogisticRegression(
            C=self.C, max_iter=self.max_iter, fit_intercept=False, random_state=self.random_state, solver='lbfgs'
        )
        self.classifier_.fit(ts_scaled, y)
        return self

    def transform(self, X):
        X = _ensure_3d(X)
        covs = compute_covariance_matrices(X)
        ts_vecs, _ = project_to_riemannian_tangent_space(covs, C_ref=self.C_ref_)
        return self.scaler_.transform(ts_vecs)

    def predict_proba(self, X):
        feats = self.transform(X)
        return self.classifier_.predict_proba(feats)

    def predict(self, X):
        feats = self.transform(X)
        return self.classifier_.predict(feats)

    def recalibrate_reference(self, X_calib):
        X_calib = _ensure_3d(X_calib)
        covs_calib = compute_covariance_matrices(X_calib)
        self.C_ref_ = compute_riemannian_mean(covs_calib)


# ----------------------------------------------------------------------
# 5. Riemannian Tangent Space + Linear SVM (Calibrated Probabilities)
# ----------------------------------------------------------------------
class Riemannian_TangentSpace_SVM_Linear(BaseEstimator, ClassifierMixin):
    def __init__(self, C=0.1, random_state=42):
        self.C = float(C)
        self.random_state = random_state
        self.C_ref_ = None
        self.scaler_ = None
        self.classifier_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = _ensure_3d(X)
        self.classes_ = np.unique(y)
        covs = compute_covariance_matrices(X)
        ts_vecs, self.C_ref_ = project_to_riemannian_tangent_space(covs)
        self.scaler_ = StandardScaler()
        ts_scaled = self.scaler_.fit_transform(ts_vecs)
        self.classifier_ = SVC(C=self.C, kernel='linear', probability=True, random_state=self.random_state)
        self.classifier_.fit(ts_scaled, y)
        return self

    def transform(self, X):
        X = _ensure_3d(X)
        covs = compute_covariance_matrices(X)
        ts_vecs, _ = project_to_riemannian_tangent_space(covs, C_ref=self.C_ref_)
        return self.scaler_.transform(ts_vecs)

    def predict_proba(self, X):
        feats = self.transform(X)
        return self.classifier_.predict_proba(feats)

    def predict(self, X):
        feats = self.transform(X)
        return self.classifier_.predict(feats)


# ----------------------------------------------------------------------
# 6. Riemannian Tangent Space + Ridge Classifier
# ----------------------------------------------------------------------
class Riemannian_TangentSpace_Ridge(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=10.0, random_state=42):
        self.alpha = float(alpha)
        self.random_state = random_state
        self.C_ref_ = None
        self.scaler_ = None
        self.classifier_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = _ensure_3d(X)
        self.classes_ = np.unique(y)
        covs = compute_covariance_matrices(X)
        ts_vecs, self.C_ref_ = project_to_riemannian_tangent_space(covs)
        self.scaler_ = StandardScaler()
        ts_scaled = self.scaler_.fit_transform(ts_vecs)
        base_ridge = RidgeClassifier(alpha=self.alpha, random_state=self.random_state)
        self.classifier_ = CalibratedClassifierCV(base_ridge, cv=3)
        self.classifier_.fit(ts_scaled, y)
        return self

    def transform(self, X):
        X = _ensure_3d(X)
        covs = compute_covariance_matrices(X)
        ts_vecs, _ = project_to_riemannian_tangent_space(covs, C_ref=self.C_ref_)
        return self.scaler_.transform(ts_vecs)

    def predict_proba(self, X):
        feats = self.transform(X)
        return self.classifier_.predict_proba(feats)

    def predict(self, X):
        feats = self.transform(X)
        return self.classifier_.predict(feats)


# ----------------------------------------------------------------------
# 7. Riemannian Tangent Space + RBF SVM
# ----------------------------------------------------------------------
class Riemannian_TangentSpace_SVM_RBF(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, random_state=42):
        self.C = float(C)
        self.random_state = random_state
        self.C_ref_ = None
        self.scaler_ = None
        self.classifier_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = _ensure_3d(X)
        self.classes_ = np.unique(y)
        covs = compute_covariance_matrices(X)
        ts_vecs, self.C_ref_ = project_to_riemannian_tangent_space(covs)
        self.scaler_ = StandardScaler()
        ts_scaled = self.scaler_.fit_transform(ts_vecs)
        self.classifier_ = SVC(C=self.C, kernel='rbf', probability=True, random_state=self.random_state)
        self.classifier_.fit(ts_scaled, y)
        return self

    def transform(self, X):
        X = _ensure_3d(X)
        covs = compute_covariance_matrices(X)
        ts_vecs, _ = project_to_riemannian_tangent_space(covs, C_ref=self.C_ref_)
        return self.scaler_.transform(ts_vecs)

    def predict_proba(self, X):
        feats = self.transform(X)
        return self.classifier_.predict_proba(feats)

    def predict(self, X):
        feats = self.transform(X)
        return self.classifier_.predict(feats)


# ----------------------------------------------------------------------
# 8. Welch PSD + Random Forest
# ----------------------------------------------------------------------
class Welch_PSD_RandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=150, max_depth=6, sfreq=250.0, random_state=42):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth) if max_depth else None
        self.sfreq = float(sfreq)
        self.random_state = random_state
        self.classifier_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = _ensure_3d(X)
        self.classes_ = np.unique(y)
        psd_feats = extract_welch_bandpower_features(X, sfreq=self.sfreq)
        self.classifier_ = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state
        )
        self.classifier_.fit(psd_feats, y)
        return self

    def transform(self, X):
        X = _ensure_3d(X)
        return extract_welch_bandpower_features(X, sfreq=self.sfreq)

    def predict_proba(self, X):
        feats = self.transform(X)
        return self.classifier_.predict_proba(feats)

    def predict(self, X):
        feats = self.transform(X)
        return self.classifier_.predict(feats)


# ----------------------------------------------------------------------
# 9. Welch PSD + Shrinkage LDA
# ----------------------------------------------------------------------
class Welch_PSD_ShrinkageLDA(BaseEstimator, ClassifierMixin):
    def __init__(self, sfreq=250.0):
        self.sfreq = float(sfreq)
        self.classifier_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = _ensure_3d(X)
        self.classes_ = np.unique(y)
        psd_feats = extract_welch_bandpower_features(X, sfreq=self.sfreq)
        self.classifier_ = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        self.classifier_.fit(psd_feats, y)
        return self

    def transform(self, X):
        X = _ensure_3d(X)
        return extract_welch_bandpower_features(X, sfreq=self.sfreq)

    def predict_proba(self, X):
        feats = self.transform(X)
        return self.classifier_.predict_proba(feats)

    def predict(self, X):
        feats = self.transform(X)
        return self.classifier_.predict(feats)


# ----------------------------------------------------------------------
# 10. Ensemble Voting (Soft Voting: CSP + FBCSP + Riemannian)
# ----------------------------------------------------------------------
class Ensemble_Voting(BaseEstimator, ClassifierMixin):
    """
    Soft-voting ensemble combining the 3 best diverse feature families:
      1. CSP + Shrinkage LDA
      2. FilterBank CSP + Logistic Regression
      3. Riemannian Tangent Space + Logistic Regression
    """
    def __init__(self, sfreq=250.0, random_state=42):
        self.sfreq = sfreq
        self.random_state = random_state
        self.clf_csp_lda = CSP_ShrinkageLDA(n_components=4)
        self.clf_fbcsp_lr = FilterBank_CSP_LogReg(sfreq=sfreq, n_components=4, C=0.5, random_state=random_state)
        self.clf_riemann_lr = Riemannian_TangentSpace_LogReg(C=0.1, random_state=random_state)
        self.classes_ = None

    def fit(self, X, y):
        X = _ensure_3d(X)
        self.classes_ = np.unique(y)
        self.clf_csp_lda.fit(X, y)
        self.clf_fbcsp_lr.fit(X, y)
        self.clf_riemann_lr.fit(X, y)
        return self

    def predict_proba(self, X):
        X = _ensure_3d(X)
        p1 = self.clf_csp_lda.predict_proba(X)
        p2 = self.clf_fbcsp_lr.predict_proba(X)
        p3 = self.clf_riemann_lr.predict_proba(X)
        return (p1 + p2 + p3) / 3.0

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


# ----------------------------------------------------------------------
# Algorithm Registry & Factory
# ----------------------------------------------------------------------
ALGORITHMS = {
    'riemann_logreg': {
        'name': 'Riemannian Tangent Space + Logistic Regression',
        'class': Riemannian_TangentSpace_LogReg,
        'description': 'Fast, robust Fréchet geometric mean SPD covariance manifold projection with L2 LogReg'
    },
    'fbcsp_logreg': {
        'name': 'Filter Bank CSP + Logistic Regression',
        'class': FilterBank_CSP_LogReg,
        'description': 'Multi-band (Theta, Alpha, Beta, Gamma) spatial filtering + LogReg'
    },
    'csp_lda': {
        'name': 'CSP + Shrinkage LDA',
        'class': CSP_ShrinkageLDA,
        'description': 'Classic Common Spatial Pattern filters with Ledoit-Wolf shrinkage LDA'
    },
    'csp_svm': {
        'name': 'CSP + SVM (RBF)',
        'class': CSP_SVM_RBF,
        'description': 'CSP spatial filtering with non-linear RBF kernel Support Vector Machine'
    },
    'riemann_svm_lin': {
        'name': 'Riemannian Tangent Space + Linear SVM',
        'class': Riemannian_TangentSpace_SVM_Linear,
        'description': 'Riemannian Tangent Space with Linear SVM (Platt calibrated probabilities)'
    },
    'riemann_ridge': {
        'name': 'Riemannian Tangent Space + Ridge Classifier',
        'class': Riemannian_TangentSpace_Ridge,
        'description': 'Riemannian Tangent Space with L2 regularized Ridge Classifier'
    },
    'riemann_svm_rbf': {
        'name': 'Riemannian Tangent Space + RBF SVM',
        'class': Riemannian_TangentSpace_SVM_RBF,
        'description': 'Riemannian Tangent Space with non-linear RBF kernel SVM'
    },
    'welch_rf': {
        'name': 'Welch PSD + Random Forest',
        'class': Welch_PSD_RandomForest,
        'description': 'Relative spectral band power features with ensemble Random Forest'
    },
    'welch_lda': {
        'name': 'Welch PSD + Shrinkage LDA',
        'class': Welch_PSD_ShrinkageLDA,
        'description': 'Relative spectral band power features with shrinkage Linear Discriminant Analysis'
    },
    'ensemble_voting': {
        'name': 'Ensemble Soft Voting (CSP + FBCSP + Riemannian)',
        'class': Ensemble_Voting,
        'description': 'Averaged ensemble voting across top diverse neural representations'
    }
}


def create_classifier(alg_key, **kwargs):
    """Instantiates a classifier from its key."""
    if alg_key not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm key: {alg_key}. Available: {list(ALGORITHMS.keys())}")
    cls = ALGORITHMS[alg_key]['class']
    return cls(**kwargs)
