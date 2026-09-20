"""
src/tuning.py
=============
Hyperparameter tuning with StratifiedGroupKFold to prevent leakage.

All tuning happens on training data only.
Test data is never touched during this phase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline

from src.models import build_pipeline
from src.utils import load_config, logger

# ─────────────────────────────────────────────────────────────────────────────
#  Hyperparameter grids
# ─────────────────────────────────────────────────────────────────────────────
PARAM_GRIDS: dict[str, dict] = {
    "logreg": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs", "saga"],
        "clf__max_iter": [500, 1000],
    },
    "knn": {
        "clf__n_neighbors": [3, 5, 7, 11, 15],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
    },
    "linear_svm": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__max_iter": [1000, 3000],
    },
    "rbf_svm": {
        "clf__C": [0.1, 1.0, 10.0, 100.0],
        "clf__gamma": ["scale", "auto", 0.001, 0.01],
    },
    "random_forest": {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 10, 20, 30],
        "clf__min_samples_split": [2, 5, 10],
        "clf__max_features": ["sqrt", "log2"],
    },
    "extra_trees": {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
        "clf__max_features": ["sqrt", "log2"],
    },
    "xgboost": {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [4, 6, 8],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
    },
    "hist_gb": {
        "clf__max_iter": [100, 200, 300],
        "clf__max_depth": [None, 6, 10],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__l2_regularization": [0.0, 0.1, 1.0],
    },
    "adaboost": {
        "clf__n_estimators": [50, 100, 200],
        "clf__learning_rate": [0.5, 1.0, 1.5],
    },
}


def tune_model(
    model_name: str,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    groups_train: pd.Series | np.ndarray,
    n_iter: int = 20,
    n_cv_folds: int = 5,
    class_weight=None,
    seed: int = 42,
    scoring: str = "f1_macro",
) -> tuple[Pipeline, dict, pd.DataFrame]:
    """
    Hyperparameter tuning with RandomizedSearchCV + StratifiedGroupKFold.

    Parameters
    ----------
    model_name    : key in MODEL_REGISTRY
    X_train       : training features
    y_train       : training labels
    groups_train  : record_id for each sample (prevents record leakage in CV)
    n_iter        : number of random parameter combinations to try
    n_cv_folds    : number of CV folds
    class_weight  : class weight dict or None
    seed          : random seed
    scoring       : sklearn scoring metric

    Returns
    -------
    best_estimator : fitted Pipeline
    best_params    : dict of best hyperparameters
    cv_results     : DataFrame of all CV results
    """
    if model_name not in PARAM_GRIDS:
        logger.warning(f"No param grid for {model_name}; returning default model.")
        pipe = build_pipeline(model_name, class_weight=class_weight)
        pipe.fit(X_train, y_train)
        return pipe, {}, pd.DataFrame()

    cfg = load_config()
    seed = seed or cfg.get("random_seed", 42)
    n_folds = n_cv_folds or cfg.get("n_cv_folds", 5)

    pipe = build_pipeline(model_name, class_weight=class_weight)
    param_grid = PARAM_GRIDS[model_name]

    cv = StratifiedGroupKFold(n_splits=n_folds)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
        refit=True,
        return_train_score=False,
    )

    logger.info(f"Tuning {model_name} ({n_iter} iterations, {n_folds}-fold CV) …")
    search.fit(X_train, y_train, groups=groups_train)

    best_params = {
        k.replace("clf__", ""): v
        for k, v in search.best_params_.items()
    }
    cv_results = pd.DataFrame(search.cv_results_)

    logger.info(
        f"  Best {scoring}: {search.best_score_:.4f}  |  params: {best_params}"
    )
    return search.best_estimator_, best_params, cv_results
