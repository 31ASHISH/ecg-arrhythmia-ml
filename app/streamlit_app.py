"""
app/streamlit_app.py
====================
ECG Arrhythmia Classification — Professional ML Research Dashboard

9 pages:
  1. Home
  2. Dataset
  3. Signal Processing
  4. Feature Engineering
  5. Model Comparison
  6. Experiments
  7. Explainability
  8. Prediction Demo
  9. About
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import AAMI_CLASSES, AAMI_LABELS
from src.utils import load_config

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Arrhythmia ML",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0d1117; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stRadio label {
    color: #c9d1d9 !important;
    font-size: 0.95rem;
    padding: 6px 0;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 40%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border: 1px solid #30363d;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero h1 { color: #58a6ff; font-size: 2.6rem; font-weight: 700; margin: 0; }
.hero h3 { color: #8b949e; font-weight: 400; margin: 0.5rem 0 0; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #161b22, #21262d);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(88,166,255,0.15); }
.metric-value { font-size: 2rem; font-weight: 700; color: #58a6ff; }
.metric-label { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }

/* Section headers */
.section-header {
    font-size: 1.5rem; font-weight: 600; color: #e6edf3;
    border-left: 4px solid #58a6ff;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem;
}

/* Info box */
.info-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 4px solid #58a6ff;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    color: #c9d1d9;
    font-size: 0.92rem;
}

/* Pipeline steps */
.pipeline-step {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    margin: 3px;
    font-size: 0.85rem;
    color: #58a6ff;
    font-weight: 500;
}

/* Class badges */
.badge-N { background:#1a4731; color:#3fb950; border-radius:6px; padding:3px 10px; font-weight:600; }
.badge-S { background:#0c2d6b; color:#58a6ff; border-radius:6px; padding:3px 10px; font-weight:600; }
.badge-V { background:#4a1c1c; color:#f85149; border-radius:6px; padding:3px 10px; font-weight:600; }
.badge-F { background:#4a3200; color:#e3b341; border-radius:6px; padding:3px 10px; font-weight:600; }
.badge-Q { background:#2d1b69; color:#a5d6ff; border-radius:6px; padding:3px 10px; font-weight:600; }

/* DataFrames */
.stDataFrame { border-radius: 8px; overflow: hidden; }
thead tr th { background: #161b22 !important; color: #58a6ff !important; }

/* Plotly charts */
.js-plotly-plot { border-radius: 12px; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "N": "#3fb950", "S": "#58a6ff", "V": "#f85149",
    "F": "#e3b341", "Q": "#a5d6ff",
}
PLOTLY_TEMPLATE = "plotly_dark"

# ── Helper functions ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_segments() -> pd.DataFrame | None:
    p = ROOT / "data" / "processed" / "segments.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_features() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    fp = ROOT / "data" / "processed" / "features.parquet"
    mp = ROOT / "data" / "processed" / "meta.parquet"
    if not fp.exists():
        return None, None
    return pd.read_parquet(fp), pd.read_parquet(mp)


@st.cache_data(show_spinner=False)
def load_results() -> dict | None:
    p = ROOT / "results" / "metrics" / "all_model_results.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_experiment(name: str) -> dict | None:
    p = ROOT / "results" / "metrics" / f"experiment_{name}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>💓 {title}</h1><h3>{subtitle}</h3></div>',
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def info(text: str) -> None:
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str) -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>'
    )


def pipeline_html(steps: list[str]) -> str:
    arrows = " &nbsp;→&nbsp; ".join(
        f'<span class="pipeline-step">{s}</span>' for s in steps
    )
    return f"<div style='line-height:3;'>{arrows}</div>"


# ─────────────────────────────────────────────────────────────────────────────
#  Page 1 — Home
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    hero(
        "ECG Arrhythmia Classification",
        "Classical Machine Learning · MIT-BIH Database · AAMI 5-Class · Patient-wise Validation",
    )

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("MIT-BIH Records", "48"),
        ("AAMI Classes", "5"),
        ("Feature Groups", "7"),
        ("ML Models", "10"),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4], cards):
        col.markdown(metric_card(label, value), unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([1.6, 1])

    with col1:
        section("🎯 Project Objective")
        st.markdown("""
> Build a **rigorous, research-grade** Machine Learning pipeline for ECG arrhythmia
> classification using **classical ML methods**, comprehensive feature engineering,
> and **patient-wise validation** to prevent data leakage.

**Core Research Questions:**
- Which ECG feature categories are most discriminative for arrhythmia classification?
- How do classical ML models compare under patient-independent evaluation?
- What is the impact of class imbalance mitigation strategies?
- How robust are models to signal noise and different segmentation window sizes?
""")

        section("🔬 ML Pipeline Architecture")
        st.markdown(
            pipeline_html([
                "Raw ECG", "Preprocessing", "Segmentation", "Feature Engineering",
                "EDA", "Feature Selection", "Imbalance Handling",
                "Model Training", "Hyperparameter Tuning",
                "Patient-wise Eval", "Explainability",
            ]),
            unsafe_allow_html=True,
        )

        section("📊 AAMI 5-Class Formulation")
        aami_data = {
            "Class": ["N", "S", "V", "F", "Q"],
            "Description": [
                "Normal sinus rhythm + BBB + nodal",
                "Supraventricular ectopic (PAC, APC)",
                "Ventricular ectopic (PVC, escape)",
                "Fusion of V and N beats",
                "Unknown / Artefact",
            ],
            "Raw Symbols": ["N, L, R, e, j", "A, a, J, S", "V, E", "F", "Q, f, ?"],
        }
        st.dataframe(pd.DataFrame(aami_data), use_container_width=True, hide_index=True)

    with col2:
        section("🛡️ Leakage Prevention")
        st.markdown("""
| Stage | Leakage Control |
|-------|----------------|
| Data Split | Patient-wise (record-level) |
| Cross-Validation | StratifiedGroupKFold |
| Scaler | Fit on **train only** |
| PCA | Fit on **train only** |
| Feature Selection | Fit on **train only** |
| SMOTE | Applied to **train only** |
| Hyperparameter Tuning | Train+Val only |
| Final Test Set | **Touched once** |
""")

        section("⚙️ Configuration")
        cfg = load_config()
        st.json({
            "mode": cfg.get("mode"),
            "sampling_rate": cfg.get("sampling_rate"),
            "window_before": cfg.get("window_before"),
            "window_after":  cfg.get("window_after"),
            "random_seed":   cfg.get("random_seed"),
            "n_cv_folds":    cfg.get("n_cv_folds"),
        })


# ─────────────────────────────────────────────────────────────────────────────
#  Page 2 — Dataset
# ─────────────────────────────────────────────────────────────────────────────
def page_dataset():
    hero("Dataset", "MIT-BIH Arrhythmia Database · PhysioNet")

    section("📚 Database Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**MIT-BIH Arrhythmia Database** (PhysioNet):
- **48 half-hour** two-lead ambulatory ECG recordings
- Recorded at **Massachusetts Institute of Technology** and **Beth Israel Hospital**
- Sampling frequency: **360 Hz**
- **Dual-channel**: MLII + V1 (or V5)
- Expert annotations by **two or more cardiologists**
- Total beats annotated: **~110,000**
- Widely used **gold standard** for arrhythmia research
""")
    with col2:
        info(
            "⚠️ <b>Medical Disclaimer</b>: This project is for educational and research purposes only. "
            "No outputs should be used for clinical diagnosis or medical decisions."
        )
        st.markdown("""
**Reference:**
Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.*
IEEE Engineering in Medicine and Biology Magazine, 20(3):45-50, 2001.
doi: 10.1109/51.932724
""")

    st.markdown("---")
    segments = load_segments()

    if segments is None:
        st.warning("⚠️ No data found. Run the pipeline first: `python scripts/preprocess.py`")
        section("📋 Expected Statistics (after pipeline runs)")
        cols = st.columns(4)
        for col, (lbl, val) in zip(cols, [
            ("Records", "6 (fast) / 48 (full)"),
            ("Total Beats", "~13k (fast) / ~110k (full)"),
            ("Normal (N)", "~75%"),
            ("Minority Classes", "S+V+F+Q ~25%"),
        ]):
            col.metric(lbl, val)
        return

    # ── Live statistics ────────────────────────────────────────────────────
    section("📈 Dataset Statistics")
    n_records = segments["record_id"].nunique()
    n_beats   = len(segments)
    n_classes = segments["aami_label"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    for col, (lbl, val) in zip([c1, c2, c3, c4], [
        ("Records",      str(n_records)),
        ("Total Beats",  f"{n_beats:,}"),
        ("AAMI Classes", str(n_classes)),
        ("Avg Beats/Rec", f"{n_beats//n_records:,}"),
    ]):
        col.markdown(metric_card(lbl, val), unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        section("Class Distribution")
        counts = segments["aami_label"].value_counts().reindex(AAMI_CLASSES, fill_value=0)
        fig = px.bar(
            x=counts.index, y=counts.values,
            color=counts.index,
            color_discrete_map=CLASS_COLORS,
            labels={"x": "AAMI Class", "y": "Count"},
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("Class Proportion (Imbalance)")
        fig2 = px.pie(
            names=counts.index, values=counts.values,
            color=counts.index,
            color_discrete_map=CLASS_COLORS,
            template=PLOTLY_TEMPLATE,
            hole=0.4,
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    section("Beats per Record")
    bpr = segments.groupby("record_id").size().reset_index(name="n_beats")
    bpr_cls = segments.groupby(["record_id", "aami_label"]).size().reset_index(name="count")
    fig3 = px.bar(
        bpr_cls, x="record_id", y="count", color="aami_label",
        color_discrete_map=CLASS_COLORS,
        labels={"record_id": "Record", "count": "Beat Count"},
        template=PLOTLY_TEMPLATE,
        barmode="stack",
    )
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

    section("RR Interval Distribution")
    fig4 = px.histogram(
        segments, x="rr_prev", color="aami_label",
        color_discrete_map=CLASS_COLORS,
        nbins=80, barmode="overlay",
        opacity=0.6,
        labels={"rr_prev": "RR Interval (samples)", "aami_label": "Class"},
        template=PLOTLY_TEMPLATE,
    )
    fig4.update_layout(height=350)
    st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Page 3 — Signal Processing
# ─────────────────────────────────────────────────────────────────────────────
def page_signal_processing():
    hero("Signal Processing", "Baseline removal · Bandpass filtering · Beat segmentation")

    col1, col2 = st.columns(2)
    with col1:
        section("🔧 Preprocessing Pipeline")
        st.markdown("""
| Stage | Method | Parameters |
|-------|--------|------------|
| Baseline wander removal | High-pass Butterworth | 0.5 Hz, order 4 |
| Bandpass filter | Butterworth | 0.5–40 Hz, order 4 |
| Power-line notch | IIR Notch | 60 Hz |
| Normalisation | Z-score | Per-record |
""")
    with col2:
        section("📏 Beat Segmentation")
        cfg = load_config()
        wb = cfg.get("window_before", 90)
        wa = cfg.get("window_after",  110)
        st.markdown(f"""
- **Window**: `{wb}` samples before R-peak + `{wa}` samples after
- **Total window**: `{wb + wa}` samples = `{(wb+wa)/360*1000:.0f}` ms at 360 Hz
- **R-peak source**: Expert annotation (.atr files) — no automatic detection
- **Invalid beats**: Discarded if window exceeds signal boundary
""")

    segments = load_segments()

    if segments is None:
        st.warning("No segment data found. Run `python scripts/preprocess.py`")
        return

    st.markdown("---")
    section("🔭 Interactive ECG Explorer")

    records = sorted(segments["record_id"].unique())
    sel_rec = st.selectbox("Select Record", records)
    rec_seg = segments[segments["record_id"] == sel_rec].reset_index(drop=True)

    # Show raw vs filtered for first few seconds
    try:
        raw_data = ROOT / "data" / "raw"
        if raw_data.exists():
            import wfdb
            from src.preprocessing import full_pipeline
            record = wfdb.rdrecord(str(raw_data / sel_rec))
            sig_raw = record.p_signal[:, 0]
            sig_filt = full_pipeline(sig_raw, record.fs)
            fs = record.fs
            duration = st.slider("Display window (seconds)", 2, 15, 5)
            n_show = duration * fs
            t = np.arange(n_show) / fs

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=sig_raw[:n_show], name="Raw ECG",
                                     line=dict(color="#8b949e", width=0.8)))
            fig.add_trace(go.Scatter(x=t, y=sig_filt[:n_show], name="Filtered ECG",
                                     line=dict(color="#58a6ff", width=1.2)))
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                title=f"Record {sel_rec} — Raw vs Filtered",
                xaxis_title="Time (s)", yaxis_title="Amplitude",
                height=420, hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Raw signal display requires raw data files. ({e})")

    section("💓 Beat Window Gallery")
    n_cols = 5
    cols   = st.columns(n_cols)
    for ci, cls in enumerate(AAMI_CLASSES):
        cls_beats = rec_seg[rec_seg["aami_label"] == cls]
        if len(cls_beats) == 0:
            cols[ci].warning(f"Class {cls}: no beats in this record")
            continue
        beat = np.array(cls_beats.iloc[0]["filtered_window"])
        fig = go.Figure(go.Scatter(
            y=beat, mode="lines",
            line=dict(color=CLASS_COLORS[cls], width=1.5),
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title=dict(text=f"Class {cls}", x=0.5),
            height=200,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
        )
        cols[ci].plotly_chart(fig, use_container_width=True)

    section("📐 Segmentation Statistics")
    stats = rec_seg.groupby("aami_label").agg(
        count=("beat_id", "count"),
        rr_prev_mean=("rr_prev", "mean"),
        rr_prev_std=("rr_prev", "std"),
        rr_next_mean=("rr_next", "mean"),
    ).round(1)
    st.dataframe(stats, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Page 4 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def page_features():
    hero("Feature Engineering", "66+ hand-crafted ECG features across 7 categories")

    section("📚 Feature Categories")
    cats = {
        "⏱️ Time-domain (13)":    "Mean, std, min/max, range, skewness, kurtosis, zero-crossings, energy, percentiles, IQR",
        "🫀 Morphological (10)":  "R/Q/S amplitudes, QRS area, T-wave area, ST level, QRS duration, slopes, symmetry",
        "📈 RR Interval (7)":     "RR_prev, RR_next, ratio, mean, difference, normalized prev/next",
        "🎵 Frequency (10)":      "Dominant freq, centroid, spread, VLF/LF/HF band power, LF/HF ratio, spectral entropy",
        "🌊 Wavelet DWT (15)":    "Energy & std per level (db4, 5 levels), approximation energy, total energy, energy fractions",
        "🔢 Entropy (6)":         "Shannon entropy, log-energy entropy, approximate entropy, RMS, variance, mean-abs",
        "📐 Shape (5)":           "Start slope, end slope, peak position, trough position, curve area",
    }
    for cat, desc in cats.items():
        with st.expander(cat, expanded=False):
            st.markdown(f"**{desc}**")

    st.markdown("---")

    X, meta = load_features()
    if X is None:
        st.warning("No feature data found. Run `python scripts/build_features.py`")
        return

    y      = meta["label_int"].values
    labels = meta["aami_label"].values

    col1, col2, c3, c4 = st.columns(4)
    col1.markdown(metric_card("Total Features", str(X.shape[1])), unsafe_allow_html=True)
    col2.markdown(metric_card("Total Beats",    f"{X.shape[0]:,}"), unsafe_allow_html=True)
    c3.markdown(metric_card("NaN values",       str(X.isna().sum().sum())), unsafe_allow_html=True)
    c4.markdown(metric_card("Feature Groups",   "7"), unsafe_allow_html=True)

    st.markdown("---")

    section("📊 Feature Distributions")
    feat_choices = sorted(X.columns.tolist())
    sel_feat = st.selectbox("Select feature to visualise", feat_choices)

    X_plot = X[[sel_feat]].copy()
    X_plot["class"] = labels

    fig = px.histogram(
        X_plot, x=sel_feat, color="class",
        color_discrete_map=CLASS_COLORS,
        nbins=60, barmode="overlay", opacity=0.65,
        template=PLOTLY_TEMPLATE,
        labels={"class": "AAMI Class"},
        title=f"Distribution of {sel_feat} by class",
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    section("🔥 Correlation Heatmap (Top 25 features)")
    top_feats = X.std().nlargest(25).index.tolist()
    corr = X[top_feats].corr()

    fig_corr = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        template=PLOTLY_TEMPLATE,
        title="Feature Correlation Matrix",
    )
    fig_corr.update_layout(height=550)
    st.plotly_chart(fig_corr, use_container_width=True)

    section("🎯 PCA Scatter (2D)")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    max_sample = min(5000, len(X))
    idx = np.random.RandomState(42).choice(len(X), max_sample, replace=False)
    Xs  = StandardScaler().fit_transform(X.values[idx])
    pca = PCA(n_components=2, random_state=42)
    X2d = pca.fit_transform(Xs)
    var = pca.explained_variance_ratio_

    pca_df = pd.DataFrame({
        "PC1": X2d[:, 0], "PC2": X2d[:, 1],
        "Class": [AAMI_CLASSES[y_] for y_ in y[idx]],
    })
    fig_pca = px.scatter(
        pca_df, x="PC1", y="PC2", color="Class",
        color_discrete_map=CLASS_COLORS,
        opacity=0.5,
        template=PLOTLY_TEMPLATE,
        title=f"PCA — PC1 ({var[0]*100:.1f}%) vs PC2 ({var[1]*100:.1f}%)",
    )
    fig_pca.update_traces(marker_size=3)
    fig_pca.update_layout(height=480)
    st.plotly_chart(fig_pca, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Page 5 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
def page_models():
    hero("Model Comparison", "10 classifiers · Default, Weighted, SMOTE variants · Patient-wise test set")

    results = load_results()
    if results is None:
        st.warning("No results found. Run `python scripts/train.py` first.")
        section("📋 Models to be Trained")
        st.markdown("""
| Category | Model |
|---------|-------|
| Baseline | Majority Classifier |
| Linear | Logistic Regression, Linear SVM |
| Instance-based | K-Nearest Neighbours |
| Kernel | RBF SVM |
| Ensemble | Random Forest, Extra Trees |
| Boosting | XGBoost, HistGradientBoosting, AdaBoost |
""")
        return

    from src.evaluation import model_comparison_table
    comp_df = model_comparison_table(results)

    section("📊 Full Comparison Table")
    display_cols = ["Accuracy", "Balanced Acc", "Macro F1", "Weighted F1", "ROC-AUC",
                    "F1-N", "F1-S", "F1-V", "F1-F", "F1-Q"]
    show_cols    = [c for c in display_cols if c in comp_df.columns]
    styled       = comp_df[show_cols].style.background_gradient(
        cmap="YlOrRd", subset=["Macro F1", "Balanced Acc"]
    ).format("{:.4f}")
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        section("Macro F1 Comparison")
        if "Macro F1" in comp_df.columns:
            sorted_f1 = comp_df["Macro F1"].dropna().sort_values(ascending=True)
            fig = px.bar(
                x=sorted_f1.values, y=sorted_f1.index,
                orientation="h",
                color=sorted_f1.values,
                color_continuous_scale="RdYlGn",
                template=PLOTLY_TEMPLATE,
                labels={"x": "Macro F1", "y": "Model"},
            )
            fig.update_coloraxes(showscale=False)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("Balanced Accuracy Comparison")
        if "Balanced Acc" in comp_df.columns:
            sorted_ba = comp_df["Balanced Acc"].dropna().sort_values(ascending=True)
            fig2 = px.bar(
                x=sorted_ba.values, y=sorted_ba.index,
                orientation="h",
                color=sorted_ba.values,
                color_continuous_scale="Blues",
                template=PLOTLY_TEMPLATE,
                labels={"x": "Balanced Accuracy", "y": "Model"},
            )
            fig2.update_coloraxes(showscale=False)
            fig2.update_layout(height=500)
            st.plotly_chart(fig2, use_container_width=True)

    section("🎯 Confusion Matrix Viewer")
    model_sel = st.selectbox("Select model", list(results.keys()))
    if model_sel and "confusion_matrix" in results[model_sel]:
        cm = np.array(results[model_sel]["confusion_matrix"])
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums

        fig_cm = px.imshow(
            cm_norm,
            x=AAMI_CLASSES, y=AAMI_CLASSES,
            color_continuous_scale="Blues",
            zmin=0, zmax=1,
            template=PLOTLY_TEMPLATE,
            title=f"Confusion Matrix (normalised) — {model_sel}",
            labels={"x": "Predicted", "y": "True", "color": "Fraction"},
            text_auto=".2f",
        )
        fig_cm.update_layout(height=450)
        st.plotly_chart(fig_cm, use_container_width=True)

    section("📈 Per-class F1 Scores")
    per_class_rows = []
    for model_name, res in results.items():
        for cls in AAMI_CLASSES:
            f1 = res.get("per_class", {}).get(cls, {}).get("f1", 0)
            per_class_rows.append({"Model": model_name, "Class": cls, "F1": f1})
    if per_class_rows:
        pc_df = pd.DataFrame(per_class_rows)
        fig_pc = px.bar(
            pc_df, x="Class", y="F1", color="Model",
            barmode="group",
            template=PLOTLY_TEMPLATE,
            title="Per-class F1 by Model",
        )
        fig_pc.update_layout(height=420)
        st.plotly_chart(fig_pc, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Page 6 — Experiments
# ─────────────────────────────────────────────────────────────────────────────
def page_experiments():
    hero("Experiments", "Imbalance strategies · Feature ablation · Window sizes · Noise robustness · Learning curves")

    exp_tab1, exp_tab2, exp_tab3, exp_tab4, exp_tab5 = st.tabs([
        "⚖️ Imbalance", "🧩 Feature Ablation",
        "📐 Window Size", "📡 Noise", "📈 Learning Curve",
    ])

    # ── Imbalance experiment ───────────────────────────────────────────────
    with exp_tab1:
        data = load_experiment("imbalance")
        if data is None:
            st.warning("Run `python scripts/run_experiments.py` to generate results.")
        else:
            rows = []
            for strat, res in data.items():
                rows.append({
                    "Strategy":    strat,
                    "Train Size":  res.get("train_size", "—"),
                    "Macro F1":    res.get("macro_f1", 0),
                    "Balanced Acc": res.get("balanced_accuracy", 0),
                    "Accuracy":    res.get("accuracy", 0),
                    "F1-V":        res.get("per_class", {}).get("V", {}).get("f1", 0),
                    "F1-S":        res.get("per_class", {}).get("S", {}).get("f1", 0),
                })
            df_imb = pd.DataFrame(rows)
            st.dataframe(df_imb.set_index("Strategy").round(4), use_container_width=True)

            fig = px.bar(
                df_imb, x="Strategy", y=["Macro F1", "Balanced Acc"],
                barmode="group", template=PLOTLY_TEMPLATE,
                title="Imbalance Strategy Comparison",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # ── Feature ablation ───────────────────────────────────────────────────
    with exp_tab2:
        data = load_experiment("ablation")
        if data is None:
            st.warning("Run `python scripts/run_experiments.py` to generate results.")
        else:
            rows = []
            for grp, res in data.items():
                rows.append({
                    "Feature Group":  grp,
                    "N Features":     res.get("n_features", "—"),
                    "Macro F1":       res.get("macro_f1", 0),
                    "Balanced Acc":   res.get("balanced_accuracy", 0),
                    "F1-V":           res.get("per_class", {}).get("V", {}).get("f1", 0),
                })
            df_abl = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
            st.dataframe(df_abl.set_index("Feature Group").round(4), use_container_width=True)

            fig = px.bar(
                df_abl, x="Feature Group", y="Macro F1",
                color="Macro F1", color_continuous_scale="RdYlGn",
                template=PLOTLY_TEMPLATE, title="Feature Ablation — Macro F1",
            )
            fig.update_coloraxes(showscale=False)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # ── Window size ────────────────────────────────────────────────────────
    with exp_tab3:
        data = load_experiment("window_size")
        if data is None:
            st.info("Window size experiment data not yet generated.")
        else:
            rows = [{"Config": k, **{m: v for m, v in r.items()
                                     if m in ("macro_f1","balanced_accuracy","window_before","window_after")}}
                    for k, r in data.items()]
            df_w = pd.DataFrame(rows)
            st.dataframe(df_w.set_index("Config").round(4), use_container_width=True)

    # ── Noise robustness ───────────────────────────────────────────────────
    with exp_tab4:
        data = load_experiment("noise")
        if data is None:
            st.info("Noise experiment data not yet generated.")
        else:
            rows = []
            for k, r in data.items():
                snr = r.get("snr_db", "∞")
                rows.append({
                    "SNR (dB)": snr if snr != float("inf") else "Clean",
                    "Macro F1": r.get("macro_f1", 0),
                    "Balanced Acc": r.get("balanced_accuracy", 0),
                })
            df_n = pd.DataFrame(rows)
            st.dataframe(df_n.round(4), use_container_width=True)
            fig = px.line(
                df_n, x="SNR (dB)", y="Macro F1",
                template=PLOTLY_TEMPLATE,
                title="Model Robustness to Signal Noise",
                markers=True,
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    # ── Learning curve ─────────────────────────────────────────────────────
    with exp_tab5:
        data = load_experiment("training_size")
        if data is None:
            st.info("Training size experiment data not yet generated.")
        else:
            rows = []
            for k, r in data.items():
                rows.append({
                    "Train Fraction": r.get("train_fraction", 0),
                    "Train Size":     r.get("train_size", 0),
                    "Macro F1":       r.get("macro_f1", 0),
                    "Balanced Acc":   r.get("balanced_accuracy", 0),
                })
            df_lc = pd.DataFrame(rows).sort_values("Train Fraction")
            fig = px.line(
                df_lc, x="Train Fraction", y=["Macro F1", "Balanced Acc"],
                template=PLOTLY_TEMPLATE, title="Learning Curve",
                markers=True,
                labels={"value": "Score", "variable": "Metric"},
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Page 7 — Explainability
# ─────────────────────────────────────────────────────────────────────────────
def page_explainability():
    hero("Explainability", "Feature importance · Permutation importance · SHAP")

    fig_dir = ROOT / "results" / "figures"

    def show_fig(fname: str, caption: str):
        p = fig_dir / fname
        if p.exists():
            st.image(str(p), caption=caption, use_container_width=True)
        else:
            st.info(f"Figure not yet generated: {fname}. Run `python scripts/evaluate.py`")

    col1, col2 = st.columns(2)
    with col1:
        section("🌲 Tree Feature Importance")
        show_fig("feature_importance.png", "Random Forest Feature Importances (top 20)")
    with col2:
        section("🔀 Permutation Importance")
        show_fig("permutation_importance.png", "Permutation Importance on test set (top 20)")

    section("🎯 SHAP Summary")
    show_fig("shap_summary.png", "SHAP Feature Impact on Predictions")

    section("📈 Feature Group Insights")
    X, meta = load_features()
    if X is not None:
        from src.feature_selection import get_feature_groups
        groups = get_feature_groups(X.columns.tolist())
        group_sizes = {g: len(fs) for g, fs in groups.items() if fs}
        fig = px.bar(
            x=list(group_sizes.keys()), y=list(group_sizes.values()),
            template=PLOTLY_TEMPLATE,
            title="Number of Features per Category",
            labels={"x": "Category", "y": "# Features"},
            color=list(group_sizes.values()),
            color_continuous_scale="Viridis",
        )
        fig.update_coloraxes(showscale=False)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    section("🔬 ECG Features Explained")
    with st.expander("What do the most important features mean?"):
        from src.features import FEATURE_DICT
        feat_df = pd.DataFrame(
            [(k, v) for k, v in FEATURE_DICT.items()],
            columns=["Feature", "Description"],
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Page 8 — Prediction Demo
# ─────────────────────────────────────────────────────────────────────────────
def page_demo():
    hero("Prediction Demo", "Classify a heartbeat waveform using the best trained model")

    import joblib

    # Load best model
    met_dir = ROOT / "results" / "metrics"
    mdl_dir = ROOT / "models"
    best_model_file = met_dir / "best_model.json"

    model = None
    scaler = None
    if best_model_file.exists():
        with open(best_model_file) as f:
            best_name = json.load(f)["best_model"]
        model_path = mdl_dir / f"{best_name}.joblib"
        scaler_path = mdl_dir / "scaler.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)

    if model is None:
        st.warning("No trained model found. Run `python scripts/train.py` first.")
        return

    st.success(f"✅ Model loaded: **{best_name}**")

    # ── Beat selection ─────────────────────────────────────────────────────
    segments = load_segments()
    if segments is not None:
        section("🎲 Select a Beat from the Dataset")
        col1, col2 = st.columns(2)
        with col1:
            rec_sel = st.selectbox("Record", sorted(segments["record_id"].unique()))
        with col2:
            cls_sel = st.selectbox("AAMI Class (true label)", AAMI_CLASSES)

        sub = segments[
            (segments["record_id"] == rec_sel) &
            (segments["aami_label"] == cls_sel)
        ].reset_index(drop=True)

        if len(sub) == 0:
            st.warning("No beats found for this combination.")
            return

        beat_idx = st.slider("Beat index", 0, min(len(sub)-1, 50), 0)
        row = sub.iloc[beat_idx]
        window = np.array(row["filtered_window"])

        # Plot waveform
        fig = go.Figure(go.Scatter(
            y=window, mode="lines",
            line=dict(color=CLASS_COLORS.get(cls_sel, "#58a6ff"), width=2),
        ))
        fig.add_vline(x=int(row["window_before"]), line_dash="dash",
                      line_color="#f85149", annotation_text="R-peak")
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title=f"Beat waveform — Record {rec_sel}, True class: {cls_sel}",
            xaxis_title="Sample", yaxis_title="Amplitude (normalised)",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Feature extraction + prediction ───────────────────────────────
        if st.button("🔍 Classify this Beat", type="primary"):
            from src.features import extract_all_features

            feats = extract_all_features(
                window,
                rr_prev=row["rr_prev"],
                rr_next=row["rr_next"],
                fs=360,
                window_before=int(row.get("window_before", 90)),
            )
            feat_vec = np.array([feats[k] for k in sorted(feats.keys())]).reshape(1, -1)

            # Align feature order with trained model
            proc_dir = ROOT / "data" / "processed"
            X_ref = pd.read_parquet(proc_dir / "features.parquet").iloc[:1]
            feat_df = pd.DataFrame(feats, index=[0])[X_ref.columns]

            X_sc = scaler.transform(feat_df)
            pred_int = model.predict(X_sc)[0]
            pred_cls = AAMI_CLASSES[pred_int]

            st.markdown("---")
            c1, c2 = st.columns(2)
            c1.metric("True Class",      cls_sel)
            c2.metric("Predicted Class", pred_cls,
                      delta="✓ Correct" if pred_cls == cls_sel else "✗ Incorrect",
                      delta_color="normal" if pred_cls == cls_sel else "inverse")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_sc)[0]
                section("📊 Probability Distribution")
                fig_prob = px.bar(
                    x=AAMI_CLASSES, y=proba,
                    color=AAMI_CLASSES,
                    color_discrete_map=CLASS_COLORS,
                    template=PLOTLY_TEMPLATE,
                    labels={"x": "AAMI Class", "y": "Probability"},
                    title="Class Probabilities",
                )
                fig_prob.update_layout(showlegend=False, height=320)
                st.plotly_chart(fig_prob, use_container_width=True)

            section("🔧 Key Feature Values")
            top_feats = sorted(feats.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            feat_df_disp = pd.DataFrame(top_feats, columns=["Feature", "Value"])
            st.dataframe(feat_df_disp.round(4), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Page 9 — About
# ─────────────────────────────────────────────────────────────────────────────
def page_about():
    hero("About", "Limitations · References · Medical Disclaimer")

    col1, col2 = st.columns(2)

    with col1:
        section("⚠️ Limitations")
        st.markdown("""
- **Single-lead only**: Only the MLII channel (first channel) is used; multi-lead fusion could improve performance.
- **Fixed window size**: Beat windows are fixed-length; variable-length approaches might capture more context.
- **Classical ML focus**: Deep learning (CNNs, LSTMs) may achieve higher absolute performance on this task but is excluded by design.
- **Intra-patient correlation**: Even with patient-wise splits, intra-patient beat correlation means test performance may overestimate real-world generalisation.
- **Class Q is rare**: The "Unknown/Artefact" class is very rare in some records; per-class Q performance may be unreliable.
- **Dataset size**: MIT-BIH has 48 records from 47 patients — a larger database would be needed for clinical validation.
""")

        section("🔮 Future Work")
        st.markdown("""
- Multi-lead feature fusion (MLII + V1/V5)
- Deep learning comparison (CNN, LSTM, Transformer)
- Real-time classification interface
- Larger datasets (PTB-XL, CPSC2018)
- Continuous arrhythmia monitoring
- Model calibration and uncertainty quantification
- Clinical deployment and prospective validation
""")

    with col2:
        section("📚 References")
        st.markdown("""
1. **Moody GB, Mark RG** (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE EMBS*, 20(3):45-50.
2. **Goldberger AL et al.** (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23):e215-e220.
3. **de Chazal P et al.** (2004). Automatic classification of heartbeats using ECG morphology and heartbeat interval features. *IEEE Trans. Biomed. Eng.*, 51(7):1196-1206.
4. **ANSI/AAMI EC57** (1998). Testing and reporting performance results of cardiac rhythm and ST-segment measurement algorithms.
5. **Krasteva V et al.** (2022). Electrocardiogram-based heartbeat classification — a deep learning review. *Front. Physiology*.
6. **Sklearn** (2023). Scikit-learn: Machine Learning in Python. JMLR 12:2825-2830.
7. **Chen T, Guestrin C** (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
""")

        section("⚕️ Medical Disclaimer")
        st.error("""
**This project is strictly for educational and academic research purposes.**

The models, predictions, and results presented here **must not** be used for:
- Clinical diagnosis
- Medical decision-making
- Patient management
- Any healthcare application

ECG interpretation for clinical use must only be performed by qualified medical professionals using validated, regulatory-approved software.

**If you or someone else is experiencing cardiac symptoms, seek immediate medical attention.**
""")

    section("🏗️ Project Information")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
**Course**: Machine Learning (Undergraduate)
**Type**: Semester-long research project
**Stack**: Python · scikit-learn · XGBoost · WFDB · Streamlit · Plotly
**Dataset**: MIT-BIH Arrhythmia Database (PhysioNet)
**Reproducibility**: Fixed seed 42, requirements.txt, config.yaml
""")
    with col4:
        st.markdown("""
**Repository Structure**:
```
ecg-arrhythmia-ml/
├── src/          # Core ML modules
├── scripts/      # Pipeline scripts
├── app/          # Streamlit dashboard
├── notebooks/    # Research notebooks
├── config/       # Configuration
├── data/         # Raw + processed
├── models/       # Saved models
└── results/      # Metrics + figures
```
""")


# ─────────────────────────────────────────────────────────────────────────────
#  Navigation
# ─────────────────────────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        st.markdown(
            """
<div style="text-align:center; padding: 1rem 0;">
    <span style="font-size:2.5rem;">💓</span><br>
    <span style="color:#58a6ff; font-size:1rem; font-weight:600;">ECG Arrhythmia ML</span><br>
    <span style="color:#8b949e; font-size:0.8rem;">MIT-BIH · AAMI 5-class</span>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        pages = {
            "🏠 Home":                page_home,
            "📊 Dataset":             page_dataset,
            "🔬 Signal Processing":   page_signal_processing,
            "⚙️ Feature Engineering": page_features,
            "🤖 Model Comparison":    page_models,
            "🧪 Experiments":         page_experiments,
            "💡 Explainability":      page_explainability,
            "🎯 Prediction Demo":     page_demo,
            "ℹ️ About":               page_about,
        }
        selection = st.radio("Navigation", list(pages.keys()), label_visibility="collapsed")

        st.markdown("---")
        cfg = load_config()
        mode_color = "#3fb950" if cfg.get("mode") == "full" else "#e3b341"
        st.markdown(
            f'<div style="text-align:center;">'
            f'<span style="color:{mode_color}; font-size:0.8rem; font-weight:600;">'
            f'Mode: {cfg.get("mode","fast").upper()}</span></div>',
            unsafe_allow_html=True,
        )

        # Pipeline status
        st.markdown("---")
        st.markdown('<div style="color:#8b949e; font-size:0.8rem; font-weight:600;">Pipeline Status</div>',
                    unsafe_allow_html=True)
        checks = [
            ("Raw data",    (ROOT / "data" / "raw").exists() and any((ROOT / "data" / "raw").glob("*.hea"))),
            ("Segments",    (ROOT / "data" / "processed" / "segments.parquet").exists()),
            ("Features",    (ROOT / "data" / "processed" / "features.parquet").exists()),
            ("Models",      any((ROOT / "models").glob("*.joblib"))),
            ("Results",     (ROOT / "results" / "metrics" / "all_model_results.json").exists()),
        ]
        for label, ok in checks:
            icon = "✅" if ok else "⬜"
            st.markdown(
                f'<div style="font-size:0.8rem; color:#c9d1d9;">{icon} {label}</div>',
                unsafe_allow_html=True,
            )

    pages[selection]()


if __name__ == "__main__":
    main()
