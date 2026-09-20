"""
src/experiments.py
==================
Structured ablation and robustness experiments.

All experiments follow the same leakage-safe protocol:
  - Scaler / transforms fitted on train split only
  - Test split evaluated once at the end

Results are saved as JSON to results/metrics/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.balancing import compare_strategies
from src.data_loader import AAMI_CLASSES
from src.evaluation import evaluate_model
from src.feature_selection import get_feature_groups
from src.models import build_pipeline
from src.preprocessing import add_noise
from src.utils import get_project_root, load_config, logger, save_json, timer


def _results_dir() -> Path:
    cfg = load_config()
    return get_project_root() / cfg["paths"]["metrics"]


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: cross-val with GroupKFold
# ─────────────────────────────────────────────────────────────────────────────
def _cv_score(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
    class_weight=None,
) -> dict:
    pipe = build_pipeline(model_name, class_weight=class_weight)
    cv   = StratifiedGroupKFold(n_splits=n_folds)
    res  = cross_validate(
        pipe, X, y,
        cv=cv,
        groups=groups,
        scoring=["f1_macro", "balanced_accuracy", "accuracy"],
        n_jobs=-1,
    )
    return {
        "macro_f1_mean":       float(res["test_f1_macro"].mean()),
        "macro_f1_std":        float(res["test_f1_macro"].std()),
        "balanced_acc_mean":   float(res["test_balanced_accuracy"].mean()),
        "balanced_acc_std":    float(res["test_balanced_accuracy"].std()),
        "accuracy_mean":       float(res["test_accuracy"].mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  1. Class imbalance experiment
# ─────────────────────────────────────────────────────────────────────────────
def imbalance_strategy_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "random_forest",
    seed: int = 42,
) -> dict:
    """Train the same model under each balancing strategy and compare."""
    cfg      = load_config()
    strats   = cfg.get("balancing_strategies", ["none", "class_weight", "smote"])
    results  = {}

    logger.info("=== Imbalance Strategy Experiment ===")
    balanced_sets = compare_strategies(X_train, y_train, strategies=strats, seed=seed)

    for strategy, (X_r, y_r, cw) in balanced_sets.items():
        with timer(f"imbalance/{strategy}"):
            pipe = build_pipeline(model_name, class_weight=cw)
            pipe.fit(X_r, y_r)
            metrics = evaluate_model(pipe, X_test, y_test, model_name=f"{model_name}_{strategy}")
            metrics["strategy"] = strategy
            metrics["train_size"] = int(len(y_r))
            results[strategy] = metrics

    save_json(results, _results_dir() / "experiment_imbalance.json")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  2. Feature ablation experiment
# ─────────────────────────────────────────────────────────────────────────────
def feature_ablation_experiment(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_name: str = "random_forest",
) -> dict:
    """Train model on each feature group alone and all combined."""
    groups_feat = get_feature_groups(X_train.columns.tolist())
    results = {}

    logger.info("=== Feature Ablation Experiment ===")

    # Individual groups
    for group_name, feat_list in groups_feat.items():
        avail = [f for f in feat_list if f in X_train.columns]
        if len(avail) < 2:
            logger.info(f"  Skipping {group_name} (< 2 features)")
            continue
        with timer(f"ablation/{group_name}"):
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train[avail])
            X_te_sc = scaler.transform(X_test[avail])
            pipe = build_pipeline(model_name)
            pipe.fit(X_tr_sc, y_train)
            metrics = evaluate_model(pipe, X_te_sc, y_test,
                                     model_name=f"{group_name}_only")
            metrics["feature_group"] = group_name
            metrics["n_features"] = len(avail)
            results[group_name] = metrics

    # All combined
    with timer("ablation/all_combined"):
        scaler = StandardScaler()
        X_tr_all = scaler.fit_transform(X_train)
        X_te_all = scaler.transform(X_test)
        pipe = build_pipeline(model_name)
        pipe.fit(X_tr_all, y_train)
        metrics = evaluate_model(pipe, X_te_all, y_test, model_name="all_features")
        metrics["feature_group"] = "all"
        metrics["n_features"] = X_train.shape[1]
        results["all"] = metrics

    # Combined + PCA(50)
    with timer("ablation/combined_pca"):
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)
        n_comp  = min(50, X_train.shape[1])
        pca     = PCA(n_components=n_comp, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_sc)
        X_te_pca = pca.transform(X_te_sc)
        pipe = build_pipeline(model_name)
        pipe.fit(X_tr_pca, y_train)
        metrics = evaluate_model(pipe, X_te_pca, y_test, model_name="combined_pca")
        metrics["feature_group"] = "pca"
        metrics["n_features"] = n_comp
        results["pca"] = metrics

    save_json(results, _results_dir() / "experiment_ablation.json")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  3. Window size experiment
# ─────────────────────────────────────────────────────────────────────────────
def window_size_experiment(
    records_train: list[str],
    records_test: list[str],
    model_name: str = "random_forest",
    window_configs: list[tuple[int, int]] | None = None,
) -> dict:
    """Compare different heartbeat window sizes (window_before, window_after)."""
    from src.features import build_feature_matrix
    from src.segmentation import segment_all_records

    cfg = load_config()
    wc  = window_configs or [(60, 80), (90, 110), (120, 140)]
    results = {}

    logger.info("=== Window Size Experiment ===")
    for wb, wa in wc:
        key = f"wb{wb}_wa{wa}"
        logger.info(f"  Window: before={wb}, after={wa}")
        with timer(key):
            segs_tr = segment_all_records(records_train, config=cfg,
                                          window_before=wb, window_after=wa)
            segs_te = segment_all_records(records_test, config=cfg,
                                          window_before=wb, window_after=wa)
            X_tr, y_tr, _ = build_feature_matrix(segs_tr, cfg)
            X_te, y_te, _ = build_feature_matrix(segs_te, cfg)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            pipe = build_pipeline(model_name)
            pipe.fit(X_tr_s, y_tr)
            metrics = evaluate_model(pipe, X_te_s, y_te, model_name=key)
            metrics["window_before"] = wb
            metrics["window_after"]  = wa
            results[key] = metrics

    save_json(results, _results_dir() / "experiment_window_size.json")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  4. Noise robustness experiment
# ─────────────────────────────────────────────────────────────────────────────
def noise_robustness_experiment(
    segments_test: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "random_forest",
    snr_levels: list[float] | None = None,
) -> dict:
    """Evaluate model under increasing levels of additive noise on the test set."""
    from src.features import build_feature_matrix

    cfg    = load_config()
    snrs   = snr_levels or [40.0, 30.0, 20.0, 15.0, 10.0]
    results = {}

    # Train clean model
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    pipe = build_pipeline(model_name)
    pipe.fit(X_tr_s, y_train)

    logger.info("=== Noise Robustness Experiment ===")
    for snr in snrs:
        key = f"SNR_{int(snr)}dB"
        logger.info(f"  SNR = {snr} dB")
        noisy_segs = segments_test.copy()
        noisy_segs["filtered_window"] = noisy_segs["filtered_window"].apply(
            lambda w: add_noise(np.array(w), snr_db=snr).tolist()
        )
        X_noisy, y_noisy, _ = build_feature_matrix(noisy_segs, cfg)
        X_noisy_s = scaler.transform(X_noisy)
        metrics = evaluate_model(pipe, X_noisy_s, y_noisy, model_name=key)
        metrics["snr_db"] = snr
        results[key] = metrics

    # Clean baseline
    X_clean, y_clean, _ = build_feature_matrix(segments_test, cfg)
    X_clean_s = scaler.transform(X_clean)
    metrics = evaluate_model(pipe, X_clean_s, y_clean, model_name="clean")
    metrics["snr_db"] = float("inf")
    results["clean"] = metrics

    save_json(results, _results_dir() / "experiment_noise.json")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  5. Training size experiment (learning curve)
# ─────────────────────────────────────────────────────────────────────────────
def training_size_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "random_forest",
    fractions: list[float] | None = None,
    seed: int = 42,
) -> dict:
    """Train on increasing fractions of training data."""
    fracs   = fractions or [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = {}
    rng     = np.random.RandomState(seed)

    logger.info("=== Training Size Experiment ===")
    for frac in fracs:
        n = max(10, int(frac * len(y_train)))
        idx = rng.choice(len(y_train), n, replace=False)
        key = f"frac_{int(frac*100)}pct"

        scaler = StandardScaler()
        X_sub  = scaler.fit_transform(X_train[idx])
        X_te_s = scaler.transform(X_test)

        pipe = build_pipeline(model_name)
        pipe.fit(X_sub, y_train[idx])
        metrics = evaluate_model(pipe, X_te_s, y_test, model_name=key)
        metrics["train_fraction"] = frac
        metrics["train_size"]     = n
        results[key] = metrics

    save_json(results, _results_dir() / "experiment_training_size.json")
    return results
