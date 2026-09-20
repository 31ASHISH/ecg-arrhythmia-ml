"""
src/visualization.py
====================
Plotting utilities for ECG signals, EDA, and model results.
All functions return matplotlib Figure objects.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data_loader import AAMI_CLASSES
from src.utils import load_config

# ── colour palette for AAMI classes ──────────────────────────────────────────
CLASS_COLORS = {
    "N": "#4CAF50",   # green
    "S": "#2196F3",   # blue
    "V": "#F44336",   # red
    "F": "#FF9800",   # orange
    "Q": "#9C27B0",   # purple
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
})


# ─────────────────────────────────────────────────────────────────────────────
#  ECG signal plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_raw_vs_filtered(
    signal_raw: np.ndarray,
    signal_filtered: np.ndarray,
    fs: int = 360,
    duration_s: float = 5.0,
    record_id: str = "",
) -> plt.Figure:
    """Side-by-side raw and filtered ECG."""
    n_show = int(duration_s * fs)
    t = np.arange(n_show) / fs

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(t, signal_raw[:n_show], color="#666666", linewidth=0.8)
    axes[0].set_title(f"Raw ECG — Record {record_id}", fontweight="bold")
    axes[0].set_ylabel("Amplitude (mV)")

    axes[1].plot(t, signal_filtered[:n_show], color="#1565C0", linewidth=0.8)
    axes[1].set_title("Filtered ECG (0.5–40 Hz bandpass + notch)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude (normalised)")

    plt.tight_layout()
    return fig


def plot_beat_windows(
    beats_by_class: dict[str, list[np.ndarray]],
    n_examples: int = 5,
    fs: int = 360,
) -> plt.Figure:
    """Grid of beat windows, one column per AAMI class."""
    classes = AAMI_CLASSES
    n_rows  = n_examples
    n_cols  = len(classes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2), sharey=True)

    for ci, cls in enumerate(classes):
        beats = beats_by_class.get(cls, [])
        for ri in range(n_rows):
            ax = axes[ri, ci] if n_rows > 1 else axes[ci]
            if ri < len(beats):
                ax.plot(beats[ri], color=CLASS_COLORS[cls], linewidth=1.0)
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"Class {cls}", fontweight="bold",
                             color=CLASS_COLORS[cls], fontsize=11)

    fig.suptitle("Representative Beat Windows per AAMI Class", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  EDA plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(y: pd.Series | np.ndarray, title: str = "Class Distribution") -> plt.Figure:
    """Bar chart of AAMI class counts and percentages."""
    y_arr   = np.array(y)
    labels  = [AAMI_CLASSES[i] if i < len(AAMI_CLASSES) else str(i) for i in y_arr]
    counts  = pd.Series(labels).value_counts().reindex(AAMI_CLASSES, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = [CLASS_COLORS[c] for c in AAMI_CLASSES]
    bars    = ax.bar(AAMI_CLASSES, counts.values, color=colors, edgecolor="white", linewidth=0.5)

    total   = counts.sum()
    for bar, cnt in zip(bars, counts.values):
        pct = cnt / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.003,
                f"{cnt:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("AAMI Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig


def plot_rr_distributions(segments_df: pd.DataFrame) -> plt.Figure:
    """Kernel density plots of RR intervals per class."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ci, cls in enumerate(AAMI_CLASSES):
        sub = segments_df[segments_df["aami_label"] == cls]
        if len(sub) == 0:
            continue
        color = CLASS_COLORS[cls]
        axes[0].hist(sub["rr_prev"].dropna(), bins=50, alpha=0.5,
                     label=cls, color=color, density=True)
        axes[1].hist(sub["rr_next"].dropna(), bins=50, alpha=0.5,
                     label=cls, color=color, density=True)

    for ax, label in zip(axes, ["RR_prev", "RR_next"]):
        ax.set_title(f"{label} distribution by class")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(X: pd.DataFrame, top_n: int = 30) -> plt.Figure:
    """Correlation heatmap of top-n most-variable features."""
    top_cols = X.std().nlargest(top_n).index.tolist()
    corr = X[top_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr, ax=ax, cmap="coolwarm", center=0,
        square=True, linewidths=0.3, annot=False,
        xticklabels=True, yticklabels=True,
    )
    ax.set_title(f"Feature Correlation (top-{top_n} features)", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    return fig


def plot_pca_scatter(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    title: str = "PCA — First 2 Components",
) -> plt.Figure:
    """2-D PCA scatter coloured by AAMI class."""
    X_arr = np.array(X)
    y_arr = np.array(y)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_arr)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_sc)

    fig, ax = plt.subplots(figsize=(9, 7))
    for ci, cls in enumerate(AAMI_CLASSES):
        mask = y_arr == ci
        if mask.sum() == 0:
            continue
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   label=f"{cls} (n={mask.sum()})",
                   color=CLASS_COLORS[cls],
                   alpha=0.4, s=8, edgecolors="none")

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(markerscale=3, fontsize=9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Model result plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: list | np.ndarray,
    classes: list[str] = AAMI_CLASSES,
    title: str = "Confusion Matrix",
    normalise: bool = True,
) -> plt.Figure:
    cm_arr = np.array(cm)
    if normalise:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_plot = cm_arr / row_sums
        fmt_str = ".2f"
    else:
        cm_plot = cm_arr
        fmt_str = "d"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt_str,
        xticklabels=classes, yticklabels=classes,
        cmap="Blues", ax=ax, linewidths=0.5,
        vmin=0, vmax=1 if normalise else None,
    )
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "Macro F1",
    title: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart comparing models on a single metric."""
    if metric not in comparison_df.columns:
        metric = comparison_df.columns[0]

    sorted_df = comparison_df[metric].dropna().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(sorted_df)))
    sorted_df.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel(metric, fontsize=11)
    ax.set_title(title or f"Model Comparison — {metric}", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.05)
    for i, v in enumerate(sorted_df):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    return fig
