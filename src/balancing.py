"""
src/balancing.py
================
Class imbalance handling strategies.

CRITICAL: All resampling is applied ONLY on training data.
Test data is NEVER touched by any resampler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight

from src.utils import load_config, logger


def compute_class_weights(y: np.ndarray) -> dict:
    """Compute balanced class weights for use in sklearn estimators."""
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def apply_smote(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    seed: int = 42,
    k_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE oversampling (train only)."""
    # Ensure minimum samples per class for SMOTE
    y_arr = np.array(y)
    min_samples = np.min(np.bincount(y_arr))
    k = min(k_neighbors, min_samples - 1)
    if k < 1:
        logger.warning("Too few samples for SMOTE; falling back to no resampling.")
        return np.array(X), y_arr
    sm = SMOTE(random_state=seed, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X, y)
    logger.info(f"SMOTE: {len(y_arr)} → {len(y_res)} samples")
    return X_res, y_res


def apply_smote_tomek(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE + Tomek Links cleaning (train only)."""
    st = SMOTETomek(random_state=seed)
    X_res, y_res = st.fit_resample(X, y)
    logger.info(f"SMOTETomek: {len(y)} → {len(y_res)} samples")
    return X_res, y_res


def apply_smote_enn(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE + Edited Nearest Neighbours cleaning (train only)."""
    senn = SMOTEENN(random_state=seed)
    X_res, y_res = senn.fit_resample(X, y)
    logger.info(f"SMOTEENN: {len(y)} → {len(y_res)} samples")
    return X_res, y_res


def apply_random_undersample(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Random undersampling to majority class (train only)."""
    rus = RandomUnderSampler(random_state=seed)
    X_res, y_res = rus.fit_resample(X, y)
    logger.info(f"RandomUnderSample: {len(y)} → {len(y_res)} samples")
    return X_res, y_res


def get_strategy(
    strategy: str,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """
    Apply a named balancing strategy to training data.

    Parameters
    ----------
    strategy : one of {'none', 'class_weight', 'undersample',
                       'smote', 'smote_tomek', 'smote_enn'}

    Returns
    -------
    X_resampled, y_resampled, class_weight_dict (or None)
    """
    X = np.array(X_train)
    y = np.array(y_train)

    if strategy == "none":
        return X, y, None

    if strategy == "class_weight":
        cw = compute_class_weights(y)
        return X, y, cw

    if strategy == "undersample":
        X_r, y_r = apply_random_undersample(X, y, seed=seed)
        return X_r, y_r, None

    if strategy == "smote":
        X_r, y_r = apply_smote(X, y, seed=seed)
        return X_r, y_r, None

    if strategy == "smote_tomek":
        X_r, y_r = apply_smote_tomek(X, y, seed=seed)
        return X_r, y_r, None

    if strategy == "smote_enn":
        X_r, y_r = apply_smote_enn(X, y, seed=seed)
        return X_r, y_r, None

    raise ValueError(f"Unknown balancing strategy: {strategy}")


def compare_strategies(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategies: list[str] | None = None,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray, dict | None]]:
    """
    Apply all named strategies and return a dict of results.
    Used for the imbalance comparison experiment.
    """
    cfg = load_config()
    strats = strategies or cfg.get("balancing_strategies", ["none", "class_weight", "smote"])
    results: dict = {}
    for s in strats:
        try:
            X_r, y_r, cw = get_strategy(s, X_train, y_train, seed=seed)
            results[s] = (X_r, y_r, cw)
        except Exception as exc:
            logger.warning(f"Strategy '{s}' failed: {exc}")
    return results
