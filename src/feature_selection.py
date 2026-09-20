"""
src/feature_selection.py
========================
Leakage-safe feature selection utilities.

All selectors are fitted ONLY on training data and then applied
to transform both train and test sets.  No test-set information
is ever used during fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    mutual_info_classif,
)

from src.utils import load_config, logger


# ─────────────────────────────────────────────────────────────────────────────
def variance_filter(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Remove near-zero-variance features (fitted on train only)."""
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(X_train)
    kept_cols = X_train.columns[sel.get_support()].tolist()
    return (
        pd.DataFrame(sel.transform(X_train), columns=kept_cols),
        pd.DataFrame(sel.transform(X_test),  columns=kept_cols),
        kept_cols,
    )


def anova_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    k: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """ANOVA F-test top-k feature selection (fitted on train only)."""
    sel = SelectKBest(f_classif, k=min(k, X_train.shape[1]))
    sel.fit(X_train, y_train)
    kept_cols = X_train.columns[sel.get_support()].tolist()
    return (
        pd.DataFrame(sel.transform(X_train), columns=kept_cols),
        pd.DataFrame(sel.transform(X_test),  columns=kept_cols),
        kept_cols,
    )


def mutual_info_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    k: int = 40,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Mutual information top-k feature selection (fitted on train only)."""
    sel = SelectKBest(
        lambda X, y: mutual_info_classif(X, y, random_state=seed),
        k=min(k, X_train.shape[1]),
    )
    sel.fit(X_train, y_train)
    kept_cols = X_train.columns[sel.get_support()].tolist()
    return (
        pd.DataFrame(sel.transform(X_train), columns=kept_cols),
        pd.DataFrame(sel.transform(X_test),  columns=kept_cols),
        kept_cols,
    )


def rf_importance_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    threshold: float = 0.005,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Select features with RandomForest importance ≥ threshold.
    RF fitted on train only.
    """
    rf = RandomForestClassifier(
        n_estimators=100, random_state=seed, n_jobs=-1, max_depth=12
    )
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    kept_cols   = importances[importances >= threshold].index.tolist()
    if not kept_cols:
        kept_cols = importances.nlargest(20).index.tolist()
    logger.info(
        f"RF importance selection: {len(kept_cols)}/{len(X_train.columns)} features kept"
    )
    return (
        X_train[kept_cols].copy(),
        X_test[kept_cols].copy(),
        kept_cols,
    )


def apply_feature_subset(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict to a predefined subset of features (for ablation studies)."""
    available = [f for f in feature_names if f in X_train.columns]
    return X_train[available].copy(), X_test[available].copy()


# ─────────────────────────────────────────────────────────────────────────────
#  Feature group definitions for ablation experiments
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_GROUPS: dict[str, list[str]] = {
    "time_domain":   [f for f in [] if f.startswith("td_")],   # populated at runtime
    "morphological": [],
    "rr_interval":   [],
    "frequency":     [],
    "wavelet":       [],
}


def get_feature_groups(feature_names: list[str]) -> dict[str, list[str]]:
    """Partition feature names by their prefix."""
    groups: dict[str, list[str]] = {
        "time_domain":   [],
        "morphological": [],
        "rr_interval":   [],
        "frequency":     [],
        "wavelet":       [],
        "entropy":       [],
        "shape":         [],
    }
    prefix_map = {
        "td_":    "time_domain",
        "morph_": "morphological",
        "rr_":    "rr_interval",
        "fd_":    "frequency",
        "wt_":    "wavelet",
        "en_":    "entropy",
        "sh_":    "shape",
    }
    for feat in feature_names:
        for prefix, group in prefix_map.items():
            if feat.startswith(prefix):
                groups[group].append(feat)
                break
    return groups
