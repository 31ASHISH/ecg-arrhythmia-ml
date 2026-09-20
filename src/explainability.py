"""
src/explainability.py
=====================
Model interpretability: feature importance, permutation importance, SHAP.

Design notes
------------
- SHAP is computed on a small background sample (configurable) to keep
  runtime practical.
- All functions return matplotlib Figure objects (not plt.show()) so they
  can be embedded in notebooks or the Streamlit app.
- permutation importance uses the test set, which is valid (not training).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.data_loader import AAMI_CLASSES
from src.utils import load_config, logger

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  Feature importance (tree-based models)
# ─────────────────────────────────────────────────────────────────────────────

def feature_importance_plot(
    model,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "Feature Importances",
) -> plt.Figure:
    """
    Bar plot of tree-model feature importances.

    Works with RandomForest, ExtraTrees, XGBoost, HistGB (any model with
    `feature_importances_` attribute, or within a sklearn Pipeline).
    """
    # Handle Pipeline wrapper
    clf = model
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf", model)

    if not hasattr(clf, "feature_importances_"):
        raise AttributeError(f"{type(clf).__name__} has no feature_importances_")

    importances = pd.Series(clf.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top)))
    top.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Permutation importance
# ─────────────────────────────────────────────────────────────────────────────

def permutation_importance_plot(
    model,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    n_repeats: int = 10,
    seed: int = 42,
    scoring: str = "f1_macro",
    title: str = "Permutation Importance",
) -> plt.Figure:
    """Permutation importance — computed on test set (valid, not leakage)."""
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=seed,
        scoring=scoring,
        n_jobs=-1,
    )
    importances = pd.Series(result.importances_mean, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(top)))
    top.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"Δ {scoring} (mean over {n_repeats} permutations)", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  SHAP
# ─────────────────────────────────────────────────────────────────────────────

def shap_summary(
    model,
    X_sample: pd.DataFrame | np.ndarray,
    feature_names: list[str],
    class_names: list[str] = AAMI_CLASSES,
    title: str = "SHAP Feature Importance",
) -> plt.Figure:
    """
    Compute SHAP values (TreeExplainer) and return a summary bar plot.
    Works with RF, ExtraTrees, XGBoost, HistGB.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install shap: pip install shap")

    cfg = load_config()
    n_sample = min(cfg.get("shap_sample_size", 300), len(X_sample))
    rng = np.random.RandomState(cfg.get("random_seed", 42))
    idx = rng.choice(len(X_sample), n_sample, replace=False)

    X_bg = np.array(X_sample)[idx]

    # Get underlying clf from Pipeline
    clf = model
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf", model)
        if "scaler" in model.named_steps:
            X_bg = model.named_steps["scaler"].transform(X_bg)

    logger.info(f"Computing SHAP on {n_sample} samples …")
    try:
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_bg)
    except Exception as exc:
        logger.warning(f"TreeExplainer failed ({exc}); trying KernelExplainer.")
        explainer   = shap.KernelExplainer(clf.predict_proba, X_bg[:50])
        shap_values = explainer.shap_values(X_bg[:50])

    # For multiclass: mean |SHAP| across classes
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values)

    importance = pd.Series(mean_abs.mean(axis=0), index=feature_names)
    top = importance.nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(top)))
    top.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Prediction examples
# ─────────────────────────────────────────────────────────────────────────────

def prediction_examples(
    model,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    segments_test: pd.DataFrame,
    n_per_class: int = 2,
    correct: bool = True,
) -> plt.Figure:
    """
    Plot waveforms for correctly or incorrectly classified beats.
    """
    y_true = np.array(y_test)
    y_pred = model.predict(X_test)
    mask   = (y_pred == y_true) if correct else (y_pred != y_true)
    label  = "Correct" if correct else "Incorrect"

    n_classes = len(AAMI_CLASSES)
    fig, axes = plt.subplots(
        n_classes, n_per_class,
        figsize=(n_per_class * 4, n_classes * 2.5),
        sharex=True,
    )
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    for ci, cls in enumerate(AAMI_CLASSES):
        cls_mask = mask & (y_true == ci)
        idx_pool = np.where(cls_mask)[0]
        plotted  = 0
        for ax in axes[ci]:
            if plotted < len(idx_pool):
                i    = idx_pool[plotted]
                wave = np.array(segments_test.iloc[i]["filtered_window"])
                ax.plot(wave, color="royalblue", linewidth=1.2)
                pred_cls = AAMI_CLASSES[y_pred[i]]
                ax.set_title(f"True:{cls} Pred:{pred_cls}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "–", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            plotted += 1

    fig.suptitle(f"{label} Predictions per Class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig
