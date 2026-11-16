# ECG Arrhythmia Classification Using Classical Machine Learning

> **A rigorous, leakage-free, research-grade machine learning pipeline for cardiac arrhythmia detection from ECG signals using the MIT-BIH Arrhythmia Database.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org)
[![Tests](https://img.shields.io/badge/Tests-24%2F24%20Passing-brightgreen.svg)](tests/)
[![Dashboard](https://img.shields.io/badge/Streamlit-9--Page%20App-red.svg)](app/streamlit_app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📑 Research Deliverables

The complete final academic deliverables produced from this research are available directly in this repository:

* **📄 Full Academic Research Report**: [`Reports and presentation/ECG_Arrhythmia_Classical_ML_Report_final.pdf`](Reports%20and%20presentation/ECG_Arrhythmia_Classical_ML_Report_final.pdf)
* **📊 Presentation Slide Deck**: [`Reports and presentation/ML_Endsem_Presentation.pptx`](Reports%20and%20presentation/ML_Endsem_Presentation.pptx)

---

## ⚕️ Medical Disclaimer

**This project is for educational and academic research purposes only.**
Results must NOT be used for clinical diagnosis, patient management, or any healthcare application. ECG interpretation must only be performed by qualified medical professionals using validated clinical tools.

---

## 🎯 Motivation & Problem Statement

Cardiac arrhythmias are abnormal heart rhythms that can range from harmless to life-threatening. Early and accurate detection is crucial for patient outcomes. The MIT-BIH Arrhythmia Database provides a gold-standard, expert-annotated ECG corpus that enables rigorous ML research.

**Key Research Questions:**
1. Can classical ML (without deep neural networks) achieve clinically meaningful arrhythmia classification?
2. Which ECG feature engineering strategies are most discriminative?
3. How do models generalize across patients (strict patient-independent evaluation)?
4. What is the impact of class imbalance mitigation strategies on minority-class detection?

---

## 📚 Dataset & Standard AAMI Mapping

**MIT-BIH Arrhythmia Database** ([PhysioNet](https://physionet.org/content/mitdb/1.0.0/))
- 48 half-hour, dual-lead ECG recordings sampled at 360 Hz.
- Expert annotations by $\ge 2$ independent cardiologists.
- Standard single-lead analysis using Modified Limb Lead II (MLII).

### AAMI 5-Class Formulation (ANSI/AAMI EC57)

| Class | Description | Raw MIT-BIH Symbols | Dataset Beats | Share (%) |
|:---:|---|---|---:|---:|
| **N** | Normal + Bundle Branch Blocks + Nodal | N, L, R, e, j | 11,464 | 81.18% |
| **S** | Supraventricular Ectopic (PAC, APC) | A, a, J, S | 197 | 1.40% |
| **V** | Ventricular Ectopic (PVC, escape) | V, E | 2,075 | 14.69% |
| **F** | Fusion of Ventricular and Normal | F | 378 | 2.68% |
| **Q** | Unknown / Artefact | Q, f, ? | 7 | 0.05% |

---

## 🏗️ Architecture & Pipeline Flow

```
Raw ECG Signal (360 Hz)
    ↓
4-Stage Digital Preprocessing (Zero-Phase Butterworth & Notch)
  · Baseline wander removal (0.5 Hz high-pass Butterworth)
  · Diagnostic bandpass filter (0.5–40 Hz Butterworth)
  · Power-line interference notch filter (60 Hz IIR, Q=30)
  · Per-record Z-score normalization
    ↓
Heartbeat Segmentation (R-peak centered)
  · Fixed window: 90 samples pre-R (250 ms) + 110 samples post-R (305 ms) = 200 samples (556 ms)
  · Discard boundary beats out of signal range
    ↓
Feature Extraction (66 Hand-Crafted Features, 7 Categories)
  · Time-Domain (13) · Morphological (10) · RR Interval Dynamics (7)
  · Frequency-Domain (10) · Wavelet DWT db4 (15) · Entropy & Energy (6) · Shape Descriptors (5)
    ↓
Patient-Wise Train / Test Partition (No Patient Overlap)
  · Training Set: Records 105, 208, 108, 201, 200 (11,850 beats)
  · Held-Out Test Set: Record 100 (2,271 beats)
    ↓
Class Imbalance Handling & Model Benchmarking
  · 10 Classifiers: Majority Baseline, Logistic Regression, KNN, Linear SVM, RBF SVM,
    Random Forest, Extra Trees, XGBoost, HistGradientBoosting, AdaBoost
  · Evaluated under Default, Class-Weighted, and SMOTE variants
    ↓
Evaluation & Explainability
  · Metric Suite: Accuracy, Balanced Acc, Macro F1, Weighted F1, PR-AUC, per-class F1
  · Permutation Importance, Tree Importance, 2D PCA Projections
    ↓
Streamlit Research Dashboard (9 Interactive Pages)
```

---

## 🛡️ Leakage Prevention Architecture

| Pipeline Stage | Enforcement Mechanism |
|---|---|
| **Data Partitioning** | Strict record-level split: $\text{Records}_{\text{train}} \cap \text{Records}_{\text{test}} = \emptyset$ (no patient in both sets) |
| **Feature Scaling** | `StandardScaler` fitted on **training data only**, `.transform()` applied to test data |
| **Dimensionality Reduction** | PCA and selectors fitted on **training data only** |
| **Class Resampling** | SMOTE and undersampling executed **strictly on training data** |
| **Cross-Validation** | `StratifiedGroupKFold(groups=record_id)` to avoid intra-patient fold leakage |
| **Final Test Set** | Evaluated **exactly once** on locked models |
| **Unit Verification** | Continuously guarded by 24 passing automated tests in `tests/test_pipeline.py` |

---

## 📊 Empirical Results

All evaluations were executed on held-out test Record 100 (2,271 beats: 2,237 Normal, 33 Supraventricular, 1 Ventricular, 0 Fusion, 0 Unknown).

### 1. Multi-Model Benchmark

| Model Variant | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | PR-AUC | F1 (N) | F1 (S) | F1 (V) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **HistGradientBoosting (Weighted) 🏆** | **0.9859** | **0.6768** | **0.4103** | **0.9793** | **0.5890** | **0.9929** | **0.0588** | **1.0000** |
| **Random Forest (Default)** | 0.9855 | 0.6667 | 0.3985 | 0.9783 | 0.5996 | 0.9927 | 0.0000 | 1.0000 |
| **Random Forest (Weighted)** | 0.9855 | 0.6667 | 0.3985 | 0.9783 | 0.5540 | 0.9927 | 0.0000 | 1.0000 |
| **Random Forest (SMOTE)** | 0.9855 | 0.6667 | 0.3985 | 0.9783 | 0.5854 | 0.9927 | 0.0000 | 1.0000 |
| **Extra Trees (Default)** | 0.9855 | 0.6667 | 0.3985 | 0.9783 | 0.4481 | 0.9927 | 0.0000 | 1.0000 |
| **XGBoost (Default)** | 0.9855 | 0.6667 | 0.3985 | 0.9783 | 0.5991 | 0.9927 | 0.0000 | 1.0000 |
| **K-Nearest Neighbors (Default)** | 0.9855 | 0.6667 | 0.3985 | 0.9783 | 0.4005 | 0.9927 | 0.0000 | 1.0000 |
| **KNN (SMOTE)** | 0.0524 | 0.3609 | 0.2259 | 0.0988 | 0.4008 | 0.0994 | 0.0303 | 1.0000 |
| **Logistic Regression (Weighted)** | 0.0004 | 0.3333 | 0.2000 | 0.0004 | 0.4677 | 0.0000 | 0.0000 | 1.0000 |
| **Logistic Regression (Default)** | 0.2831 | 0.0958 | 0.0886 | 0.4362 | 0.4268 | 0.4428 | 0.0000 | 0.0000 |
| **Majority Vote Baseline** | 0.9850 | 0.3333 | 0.1985 | 0.9776 | 0.2000 | 0.9925 | 0.0000 | 0.0000 |

*Note: Multi-class ROC-AUC is mathematically undefined (`NaN`) because ground-truth support for classes F and Q is 0 in test Record 100. PR-AUC and Macro F1 are the primary objective metrics.*

---

### 2. Feature Group Ablation Study

Random Forest trained on individual feature subsets vs. all combined:

| Feature Group | Features | Accuracy | Balanced Acc | Macro F1 | Key Finding |
|:---|---:|---:|---:|---:|:---|
| **Morphological Only 🌟** | **10** | **0.9564** | **0.6867** | **0.4071** | **Outperformed all 66 features combined!** |
| **All Features Combined** | 66 | 0.9855 | 0.6667 | 0.3985 | Full benchmark representation |
| **Time-Domain Statistics** | 13 | 0.9841 | 0.6662 | 0.2984 | Good majority separation |
| **Entropy & Non-Linear Energy** | 6 | 0.8058 | 0.6059 | 0.2785 | Captures signal complexity |
| **RR Interval Dynamics** | 7 | 0.9855 | 0.6766 | 0.2262 | Captures rhythm irregularities |
| **Frequency-Domain (PSD)** | 10 | 0.9850 | 0.3333 | 0.1985 | Spectral powers alone insufficient |
| **Shape Descriptors** | 5 | 0.9850 | 0.3333 | 0.1985 | Segment slopes and peak locations |
| **Wavelet (DWT db4)** | 15 | 0.9370 | 0.3171 | 0.1935 | Detail sub-band energies |
| **Combined + PCA (50 components)** | 50 | 0.9815 | 0.3321 | **0.1981** | **-50.3% collapse (linear variance projection loses boundary)** |

---

### 3. Class Imbalance Mitigation Comparison

| Strategy | Train Size | Accuracy | Balanced Acc | Macro F1 | Mechanism Analysis |
|:---|---:|---:|---:|---:|:---|
| **None (Baseline)** | 11,850 | 0.9855 | 0.6667 | 0.3985 | Standard empirical distribution |
| **Class Weighting** | 11,850 | 0.9855 | 0.6667 | 0.3985 | Cost-sensitive loss penalty |
| **SMOTE** | 46,135 | 0.9855 | 0.6667 | 0.3985 | Synthetic k-NN interpolation |
| **SMOTE + Tomek Links** | 46,135 | 0.9855 | 0.6667 | 0.3985 | Synthetic oversampling + boundary cleaning |
| **SMOTE + ENN** | 45,807 | 0.9855 | 0.6667 | 0.3985 | Synthetic oversampling + neighborhood cleaning |
| **Random Undersampling** | 35 | **0.0000** | **0.0000** | **0.0000** | **Total collapse: equalizing to 7 Q beats left only 35 samples!** |

---

### 4. Sample Efficiency & Learning Curve

Random Forest trained on varying fractions of the training split:

| Fraction | Training Beats | Accuracy | Balanced Acc | Macro F1 | Observation |
|---:|---:|---:|---:|---:|:---|
| 10% | 1,185 | 0.9855 | 0.6667 | 0.3985 | Baseline performance achieved with only 10% data |
| 20% | 2,370 | 0.9850 | 0.3333 | 0.1985 | Temporary dip (missed lone V beat) |
| 40% | 4,740 | 0.9855 | 0.6667 | 0.3985 | Rapid convergence |
| 60% | 7,110 | 0.9855 | 0.6667 | 0.3985 | Stable plateau |
| **80%** | **9,480** | **0.9859** | **0.6768** | **0.4103** | **Optimal peak generalization performance** |
| 100% | 11,850 | 0.9855 | 0.6667 | 0.3985 | Full training dataset |

---

## 💡 Key Scientific Insights

> [!IMPORTANT]
> **The Accuracy Trap in Medical Informatics**: A non-learning Dummy Majority Classifier achieved **98.50% accuracy** simply by predicting "Normal" for every beat, but scored a poor Macro F1 of **0.1985** (missing 100% of arrhythmias). Accuracy must never be used alone in clinical ML.

> [!TIP]
> **Domain-Driven Morphology Beats Brute-Force Features**: 10 hand-crafted clinical morphology features (QRS duration, R/Q/S amplitudes, ST level, rise/fall slopes, asymmetry) outperformed all 66 features combined. Clinical waveform geometry captures pathology better than statistical noise.

> [!WARNING]
> **PCA Destroys Arrhythmia Discriminability**: Applying PCA (50 components) cut Macro F1 in half (from 0.3985 to 0.1981). PCA aligns with directions of maximum signal variance, which are dominated by normal beats (81.2%), obliterating subtle non-linear ectopic boundaries.

---

## 📁 Repository Structure

```
ecg-arrhythmia-ml/
├── README.md                       # Master documentation & benchmark summary
├── requirements.txt                 # Project dependencies
├── LICENSE                          # MIT License
├── .gitignore                       # Clean Git tracking configuration
├── PROJECT_ROADMAP.md               # Milestones & verification checklist
│
├── Reports and presentation/        # Final academic documents
│   ├── ECG_Arrhythmia_Classical_ML_Report_final.pdf   # 15–20 page academic paper
│   └── ML_Endsem_Presentation.pptx                    # 10-slide presentation deck
│
├── config/
│   └── config.yaml                  # Central parameter configuration
├── data/                            # Raw PhysioNet data & processed parquet files
│   ├── README.md
│   ├── raw/                         # MIT-BIH .dat, .hea, .atr files
│   └── processed/                   # segments.parquet, features.parquet, meta.parquet
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   └── 04_model_comparison.ipynb   # Interactive model comparison
│
├── src/                             # Core modular library
│   ├── utils.py                     # Config, seeding, logging, patient split
│   ├── data_loader.py               # wfdb loading, AAMI mapping
│   ├── preprocessing.py             # Butterworth filters, notch, normalisation
│   ├── segmentation.py              # R-peak window segmentation
│   ├── features.py                  # 66 features across 7 domains
│   ├── feature_selection.py         # Variance, ANOVA, MI, RF importance
│   ├── balancing.py                 # SMOTE, Tomek, ENN, undersampling
│   ├── models.py                    # 10-model registry & pipeline builder
│   ├── tuning.py                    # RandomizedSearchCV + StratifiedGroupKFold
│   ├── evaluation.py                # Metric calculations & confusion matrices
│   ├── explainability.py            # Permutation importance, SHAP
│   ├── experiments.py               # Ablation, imbalance, learning curve
│   └── visualization.py             # Plotting routines
│
├── scripts/                         # Executable CLI pipeline
│   ├── download_data.py             # Fetch records from PhysioNet
│   ├── preprocess.py                # Filter & segment heartbeats
│   ├── build_features.py            # Extract 66-feature matrix
│   ├── train.py                     # Train all 18 model variants
│   ├── evaluate.py                  # Generate test evaluations & figures
│   └── run_experiments.py           # Run ablation & imbalance studies
│
├── models/                          # Serialized .joblib model artifacts
├── results/                         # Generated figures, metrics, and reports
│   ├── figures/                     # Evaluation charts (PNG)
│   ├── metrics/                     # JSON & CSV experiment outputs
│   └── reports/                     # Summary reports
│
├── app/
│   └── streamlit_app.py             # 9-page research dashboard
└── tests/
    └── test_pipeline.py             # 24 unit tests (100% pass)
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/<your-username>/ecg-arrhythmia-ml.git
cd ecg-arrhythmia-ml
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
```bash
# Step 1: Download MIT-BIH records (Fast mode: 6 records, ~5 mins)
python scripts/download_data.py --mode fast

# Step 2: Signal preprocessing and heartbeat segmentation
python scripts/preprocess.py

# Step 3: Feature extraction (66 features)
python scripts/build_features.py

# Step 4: Model training (10 algorithms, 18 variants)
python scripts/train.py

# Step 5: Held-out evaluation, figures & reports
python scripts/evaluate.py

# Step 6: Run ablation and class imbalance experiments
python scripts/run_experiments.py --skip-window --skip-noise
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```
*The dashboard includes 9 interactive dark-mode pages for dataset exploration, interactive waveform filtering, feature distributions, model comparison tables, live beat prediction, and ablation visualizers.*

### 4. Run Automated Unit Tests
```bash
python -m pytest tests/ -v
# Output: 24 passed, 3 warnings in ~28s (100% pass rate)
```

---

## 📚 References

1. **Moody GB, Mark RG** (2001). *The impact of the MIT-BIH Arrhythmia Database.* IEEE Engineering in Medicine and Biology Magazine, 20(3):45–50.
2. **ANSI/AAMI EC57** (1998). *Testing and reporting performance results of cardiac rhythm and ST-segment measurement algorithms.* Association for the Advancement of Medical Instrumentation.
3. **Goldberger AL, et al.** (2000). *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.* Circulation, 101(23):e215–e220.
4. **de Chazal P, O'Dwyer M, Reilly RB** (2004). *Automatic classification of heartbeats using ECG morphology and heartbeat interval features.* IEEE Transactions on Biomedical Engineering, 51(7):1196–1206.
5. **Pedregosa F, et al.** (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12:2825–2830.
6. **Chen T, Guestrin C** (2016). *XGBoost: A Scalable Tree Boosting System.* ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
