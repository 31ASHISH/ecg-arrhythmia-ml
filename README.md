# ECG Arrhythmia Classification Using Machine Learning

> **A rigorous, research-grade ML pipeline for cardiac arrhythmia detection from ECG signals using the MIT-BIH Arrhythmia Database.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ⚕️ Medical Disclaimer

**This project is for educational and academic research purposes only.**
Results must NOT be used for clinical diagnosis, patient management, or any healthcare application.
ECG interpretation must only be performed by qualified medical professionals using validated clinical tools.

---

## 🎯 Motivation & Problem Statement

Cardiac arrhythmias are abnormal heart rhythms that can range from harmless to life-threatening. Early and accurate detection is crucial for patient outcomes. The MIT-BIH Arrhythmia Database provides a gold-standard, expert-annotated ECG corpus that enables rigorous ML research.

**This project addresses:**
- Can classical ML (without deep learning) achieve clinically meaningful arrhythmia classification?
- Which ECG feature engineering strategies are most discriminative?
- How do models generalise across patients (patient-independent evaluation)?
- What is the impact of class imbalance mitigation on minority-class recall?

---

## 📚 Dataset

**MIT-BIH Arrhythmia Database** ([PhysioNet](https://physionet.org/content/mitdb/1.0.0/))
- 48 half-hour, dual-lead ECG recordings
- 360 Hz sampling frequency
- Expert annotations by ≥ 2 cardiologists
- ~110,000 annotated beats

**AAMI 5-Class Formulation (ANSI/AAMI EC57):**

| Class | Description | Raw Symbols |
|-------|-------------|-------------|
| N | Normal + Bundle Branch Blocks + Nodal | N, L, R, e, j |
| S | Supraventricular Ectopic (PAC, APC) | A, a, J, S |
| V | Ventricular Ectopic (PVC, escape) | V, E |
| F | Fusion of Ventricular and Normal | F |
| Q | Unknown / Artefact | Q, f, ? |

---

## 🏗️ Architecture & Methodology

```
Raw ECG (360 Hz)
    ↓
Signal Preprocessing
  · Baseline wander removal (0.5 Hz high-pass)
  · Bandpass filter (0.5–40 Hz Butterworth)
  · Power-line notch (60 Hz)
  · Z-score normalisation (per-record)
    ↓
Heartbeat Segmentation
  · Expert R-peak annotations (.atr files)
  · Fixed window: 90 samples before + 110 samples after R-peak
  · Invalid beats discarded (boundary checks)
    ↓
Feature Engineering (66+ features, 7 categories)
  · Time-domain (13) · Morphological (10) · RR interval (7)
  · Frequency-domain (10) · Wavelet DWT (15)
  · Entropy (6) · Shape descriptors (5)
    ↓
Patient-wise Train/Test Split (80/20 by record)
    ↓
Feature Selection + Dimensionality Reduction (train-only)
    ↓
Class Imbalance Handling (SMOTE, class weights — train-only)
    ↓
Model Training + Hyperparameter Tuning (StratifiedGroupKFold CV)
  · 10 classifiers: Majority, LR, KNN, LinearSVM, RBFSVM,
    RF, ExtraTrees, XGBoost, HistGB, AdaBoost
    ↓
Final Evaluation on Held-out Test Set (touched once)
    ↓
Explainability
  · Feature importance · Permutation importance · SHAP
    ↓
Streamlit Dashboard
```

---

## 🛡️ Leakage Prevention

| Stage | Control |
|-------|---------|
| Data split | Patient-wise (record-level, no record in both train and test) |
| Cross-validation | `StratifiedGroupKFold(groups=record_id)` |
| StandardScaler | Fitted on **training data only**, `.transform()` on test |
| PCA | Fitted on **training data only** |
| Feature selection | Fitted on **training data only** |
| SMOTE / resampling | Applied to **training data only** |
| Hyperparameter tuning | Train+validation folds only |
| Final test set | **Evaluated exactly once** |

---

## 📊 Evaluation Metrics

- Accuracy, Balanced Accuracy
- Macro Precision/Recall/F1, Weighted F1
- Per-class Precision/Recall/F1/Support
- Confusion Matrix (normalised + raw)
- ROC-AUC (macro OvR), PR-AUC

**Primary metric: Macro F1** (robust to class imbalance)
**Secondary metric: Balanced Accuracy** (minority-class sensitive)

---

## 🚀 Installation & Usage

### Requirements
```bash
Python 3.10+
pip install -r requirements.txt
```

### Run the pipeline
```bash
# 1. Download data (FAST mode: 6 records, ~5 min end-to-end)
python scripts/download_data.py --mode fast

# 2. Preprocess + segment
python scripts/preprocess.py

# 3. Extract features
python scripts/build_features.py

# 4. Train all models
python scripts/train.py

# 5. Final evaluation + figures
python scripts/evaluate.py

# 6. Run ablation/robustness experiments
python scripts/run_experiments.py --skip-window --skip-noise

# 7. Launch dashboard
streamlit run app/streamlit_app.py
```

### Full mode (48 records, ~60 min)
```bash
# Edit config/config.yaml → mode: full
# Or use CLI flags:
python scripts/download_data.py --mode full
python scripts/preprocess.py --mode full
python scripts/train.py --tune
```

### Tests
```bash
python -m pytest tests/ -v
```

---

## 📁 Repository Structure

```
ecg-arrhythmia-ml/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── PROJECT_ROADMAP.md
├── METHODOLOGY.md
├── EXPERIMENTS.md
├── config/
│   └── config.yaml              # Central configuration
├── data/
│   ├── raw/                     # MIT-BIH .dat/.hea/.atr files
│   └── processed/               # segments.parquet, features.parquet
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_final_evaluation.ipynb
├── src/                         # Core library
│   ├── utils.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── features.py
│   ├── feature_selection.py
│   ├── balancing.py
│   ├── models.py
│   ├── tuning.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── experiments.py
│   └── visualization.py
├── scripts/                     # Executable pipeline
│   ├── download_data.py
│   ├── preprocess.py
│   ├── build_features.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_experiments.py
├── models/                      # Saved .joblib model files
├── results/
│   ├── figures/                 # Generated plots (PNG)
│   ├── metrics/                 # JSON metric files
│   └── reports/                 # Final summary reports
├── app/
│   └── streamlit_app.py         # 9-page Streamlit dashboard
└── tests/
    └── test_pipeline.py         # 24 unit tests
```

---

## 🔬 Experimental Design

### Feature Ablation
Train RF on each feature group alone vs. all combined:
- Time-domain only, Morphological only, RR only
- Frequency only, Wavelet only, Entropy only, Shape only
- All features combined, Combined + PCA(50)

### Class Imbalance Comparison
Same RF model under 6 strategies:
`none | class_weight | undersample | SMOTE | SMOTE+Tomek | SMOTE+ENN`

### Window Size Sensitivity
Beat windows: (60+80), (90+110), (120+140) samples

### Noise Robustness
Additive Gaussian noise at SNR: 40, 30, 20, 15, 10 dB

### Learning Curve
Training on 10%, 20%, 40%, 60%, 80%, 100% of training data

---

## 📈 Results

> Results are generated at runtime and stored in `results/metrics/`.
> Run the pipeline to populate actual metrics.

Key files after pipeline execution:
- `results/metrics/all_model_results.json` — all model metrics
- `results/metrics/model_comparison.csv`  — comparison table
- `results/metrics/best_model.json`       — best model by macro-F1
- `results/reports/final_summary.txt`     — human-readable report

---

## 💡 Explainability

- **Tree Feature Importance**: RF/ExtraTrees built-in importance
- **Permutation Importance**: Model-agnostic, computed on test set
- **SHAP**: TreeSHAP on 300-sample background (configurable)

---

## ⚠️ Limitations

- Single-lead (MLII) only
- Fixed-length beat windows
- No deep learning baseline (by design)
- Small dataset (48 records / ~47 patients)
- Class Q (unknown) performance unreliable due to rarity

---

## 🔮 Future Work

- Multi-lead feature fusion
- PTB-XL / CPSC2018 cross-dataset validation
- Real-time streaming classification
- Model calibration and uncertainty estimation
- Deep learning comparison study

---

## 📚 References

1. Moody GB, Mark RG (2001). *IEEE EMBS* 20(3):45-50. doi:10.1109/51.932724
2. Goldberger AL et al. (2000). *Circulation* 101(23):e215-e220.
3. de Chazal P et al. (2004). *IEEE Trans BME* 51(7):1196-1206.
4. ANSI/AAMI EC57 (1998). Testing ECG algorithm performance.
5. Pedregosa F et al. (2011). Scikit-learn. *JMLR* 12:2825-2830.
6. Chen T, Guestrin C (2016). XGBoost. *KDD 2016*.

---

*Course: Machine Learning (Undergraduate) · Semester-long project*
*Dataset: MIT-BIH Arrhythmia Database (PhysioNet)*
