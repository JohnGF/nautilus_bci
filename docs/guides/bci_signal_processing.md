# Brain-Computer Interface Signal Processing: CSP and LDA

This document provides a mathematical and technical overview of Common Spatial Patterns (CSP) and Linear Discriminant Analysis (LDA), two fundamental algorithms used in Brain-Computer Interface (BCI) signal processing and motor imagery decoding.

*Note: Math equations in this document are formatted using KaTeX / MathJax LaTeX syntax (`$ ... $` for inline math and `$$ ... $$` for display equations), supported natively in GitHub, VS Code Markdown Preview, and browser MathJax/KaTeX extensions.*

---

## 1. Common Spatial Patterns (CSP)

### 1.1 Overview
Common Spatial Patterns (CSP) is a spatial filtering technique designed for multi-channel electroencephalography (EEG) data. It extracts spatial filters that maximize the variance (signal power) for one experimental condition (e.g., Left Hand Motor Imagery) while simultaneously minimizing the variance for another condition (e.g., Right Hand Motor Imagery).

Since motor imagery produces localized Event-Related Desynchronization (ERD) and Event-Related Synchronization (ERS) in the sensorimotor rhythm (alpha/mu band: 8-12 Hz, beta band: 12-30 Hz) over the motor cortex (channels C3, C4, Cz), CSP acts as a supervised spatial filter to accentuate these band-power changes across multi-channel electrode layouts.

### 1.1.1 Beginner Intuition: What is CSP?
Imagine listening to a crowded room where two people are speaking at the same time from different corners. Raw EEG electrodes record a mix of all brain activity across the scalp. 

CSP acts like a smart directional microphone array. It creates weighted combinations of all 32 electrode signals to "turn up the volume" on Left Hand mental activity while turning down Right Hand activity. 

By applying CSP spatial filters:
- When the participant imagines moving their **Left Hand**, the output signal variance becomes **very large**.
- When the participant imagines moving their **Right Hand**, the output signal variance becomes **very small**.

This huge difference in signal variance makes it effortless for machine learning algorithms to tell the two mental states apart.

---

### 1.2 Mathematical Formulation

Let $E \in \mathbb{R}^{N \times T}$ represent a single-trial bandpass-filtered EEG epoch, where $N$ is the number of channels (e.g., 32 channels) and $T$ is the number of time samples.

#### Step 1: Normalized Covariance Estimation
For each trial $k$ belonging to Class 1 ($C_1$) or Class 2 ($C_2$), the normalized spatial covariance matrix $R_k$ is computed:

$$R_k = \frac{E_k E_k^T}{\operatorname{trace}(E_k E_k^T)}$$

The average spatial covariance matrices for each class, $\bar{R}_1$ and $\bar{R}_2$, are calculated by averaging $R_k$ over all training trials in each class:

$$\bar{R}_1 = \frac{1}{M_1} \sum_{k \in C_1} R_k, \quad \bar{R}_2 = \frac{1}{M_2} \sum_{k \in C_2} R_k$$

#### Step 2: Composite Covariance and Whitening Transformation
The composite covariance matrix $R_c$ is formed by summing the class covariance matrices:

$$R_c = \bar{R}_1 + \bar{R}_2$$

Eigen-decomposition of $R_c$ yields:

$$R_c = U_c \Lambda_c U_c^T$$

where $U_c$ is the matrix of eigenvectors and $\Lambda_c$ is the diagonal matrix of corresponding eigenvalues sorted in descending order.

The whitening transformation matrix $P$ is constructed to equalize variances in all directions:

$$P = \Lambda_c^{-1/2} U_c^T$$

Applying $P$ to the class covariance matrices yields transformed matrices $S_1$ and $S_2$:

$$S_1 = P \bar{R}_1 P^T, \quad S_2 = P \bar{R}_2 P^T$$

$S_1$ and $S_2$ share common eigenvectors:

$$S_1 = B \Lambda_1 B^T, \quad S_2 = B \Lambda_2 B^T, \quad \text{where } \Lambda_1 + \Lambda_2 = I$$

Because the sum of corresponding eigenvalues equals 1 ($\lambda_{1,i} + \lambda_{2,i} = 1$), an eigenvector that has high variance for Class 1 (large $\lambda_{1,i} \to 1$) automatically has low variance for Class 2 ($\lambda_{2,i} \to 0$), and vice versa.

#### Step 3: Spatial Projection Matrix
The overall CSP spatial projection matrix $W \in \mathbb{R}^{N \times N}$ is defined as:

$$W = B^T P$$

The rows of $W$ are the spatial filters. When raw EEG trial data $E$ is multiplied by $W$, the projected trial signals $Z$ are obtained:

$$Z = W E$$

Typically, only the first $m$ and last $m$ spatial filters (rows of $W$) are selected (often $m=2$ or $m=3$), resulting in $2m$ spatial filter channels that capture maximum contrast between the two motor imagery conditions.

---

### 1.3 Feature Extraction

For each spatially filtered trial $Z \in \mathbb{R}^{2m \times T}$, the normalized log-variance features $f_p$ (for $p = 1, \dots, 2m$) are computed:

$$v_p = \operatorname{var}(Z_p) = \frac{1}{T} \sum_{t=1}^T Z_{p,t}^2$$

$$f_p = \log \left( \frac{v_p}{\sum_{i=1}^{2m} v_i} \right)$$

Taking the logarithm transforms the non-linearly distributed variance ratio into a normally distributed feature space suitable for linear classification models.

---

## 2. Linear Discriminant Analysis (LDA)

### 2.1 Overview
Linear Discriminant Analysis (LDA) is a supervised classification algorithm that finds a linear combination of features that separates two or more classes of objects. In BCI pipelines, LDA receives the log-variance feature vectors $f \in \mathbb{R}^{2m}$ generated by CSP and outputs a predicted class label (e.g., Left Hand vs. Right Hand).

### 2.1.1 Beginner Intuition: What is LDA?
Imagine plotting CSP feature points on a 2D graph, where Left Hand trials cluster in one region and Right Hand trials cluster in another region.

LDA draws a **single straight line (or hyperplane)** between the two clusters so that:
1. The distance between the center of the Left cluster and Right cluster is as large as possible.
2. The spread (variance) within each cluster is as tight as possible.

When a new unknown trial comes in, LDA checks which side of the dividing line it falls on. If it falls on the Left side, it classifies the trial as Left Hand; if on the Right side, as Right Hand.

---

### 2.2 Mathematical Formulation

Given training feature vectors $X = \{x_1, x_2, \dots, x_M\}$ with class labels $y_i \in \{+1, -1\}$:

#### Step 1: Mean and Covariance Estimation
Compute the class mean vectors $\mu_1$ and $\mu_2$:

$$\mu_1 = \frac{1}{M_1} \sum_{i \in C_1} x_i, \quad \mu_2 = \frac{1}{M_2} \sum_{i \in C_2} x_i$$

Compute the pooled within-class covariance matrix $S_W$:

$$S_W = \frac{1}{M - 2} \left( \sum_{i \in C_1} (x_i - \mu_1)(x_i - \mu_1)^T + \sum_{i \in C_2} (x_i - \mu_2)(x_i - \mu_2)^T \right)$$

#### Step 2: Linear Projection Vector
The optimal weight vector $w$ projecting the multi-dimensional feature space onto a 1D line to maximize Rayleigh's quotient is given by:

$$w = S_W^{-1} (\mu_1 - \mu_2)$$

The threshold bias $w_0$ is computed as:

$$w_0 = -\frac{1}{2} w^T (\mu_1 + \mu_2)$$

#### Step 3: Decision Boundary and Classification
For a new unlabelled test sample $x$, the LDA linear decision function $d(x)$ is evaluated:

$$d(x) = w^T x + w_0$$

- If $d(x) > 0$, assign sample to Class 1.
- If $d(x) < 0$, assign sample to Class 2.

The magnitude $|d(x)|$ indicates distance from the decision boundary and reflects classification confidence.

---

## 3. Implementation in Codebase

In the `nautilus_bci` repository, the CSP and LDA pipeline is implemented across the following scripts:

- **`scripts/analysis/eeg_features.py`**:
  Contains functions for Butterworth bandpass filtering (8-30 Hz mu/beta band), spatial covariance estimation, CSP matrix derivation, log-variance feature computation, and cross-validated LDA model training.
- **`scripts/analysis/compare_bci_paradigms.py`**:
  Evaluates 10-fold cross-validated decoding accuracy across different session datasets (e.g., Motor Imagery vs. Auditory Imagery) using CSP-LDA feature pipelines.
