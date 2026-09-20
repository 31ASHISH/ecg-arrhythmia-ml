#!/usr/bin/env python
"""
scripts/train.py
================
Patient-wise train/test split → train all models → save trained models
and CV metrics to disk.

Usage
-----
    python scripts/train.py [--tune] [--models logreg rf xgboost …]

Flags
-----
    --tune   : run RandomizedSearchCV hyperparameter tuning (slower)
    --models : space-separated list of model names to train (default: all)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.balancing import get_strategy
from src.evaluation import evaluate_model, model_comparison_table, save_evaluation_results
from src.models import MODEL_REGISTRY, build_pipeline
from src.tuning import tune_model
from src.utils import (
    ensure_dirs,
    get_patient_splits,
    get_project_root,
    get_records,
    load_config,
    logger,
    save_json,
    set_seed,
    timer,
)


def main():
    parser = argparse.ArgumentParser(description="Train ECG arrhythmia models")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--models", nargs="*", default=None, help="Models to train")
    parser.add_argument("--mode", choices=["fast", "full"], default=None)
    args = parser.parse_args()

    cfg = load_config()
    if args.mode:
        cfg["mode"] = args.mode
    seed = set_seed()
    ensure_dirs(cfg)

    root     = get_project_root()
    proc_dir = root / cfg["paths"]["processed_data"]
    mdl_dir  = root / cfg["paths"]["models"]
    met_dir  = root / cfg["paths"]["metrics"]
    mdl_dir.mkdir(parents=True, exist_ok=True)

    # ── Load features ──────────────────────────────────────────────────────
    feat_path = proc_dir / "features.parquet"
    meta_path = proc_dir / "meta.parquet"
    if not feat_path.exists():
        logger.error("features.parquet not found. Run build_features.py first.")
        sys.exit(1)

    logger.info("Loading features …")
    X    = pd.read_parquet(feat_path).astype(np.float32)
    meta = pd.read_parquet(meta_path)
    y    = meta["label_int"].values
    groups = meta["record_id"].values

    logger.info(f"  Shape: {X.shape}, Classes: {np.unique(y)}")

    # ── Patient-wise split ─────────────────────────────────────────────────
    all_records = get_records(cfg)
    train_recs, test_recs = get_patient_splits(
        all_records, test_size=cfg.get("test_size", 0.2), seed=seed
    )

    train_mask = np.isin(groups, train_recs)
    test_mask  = np.isin(groups, test_recs)

    X_train, y_train, g_train = X[train_mask], y[train_mask], groups[train_mask]
    X_test,  y_test            = X[test_mask],  y[test_mask]

    logger.info(f"Train: {len(y_train):,} beats | Test: {len(y_test):,} beats")

    # Save split info
    save_json(
        {"train_records": train_recs, "test_records": test_recs},
        met_dir / "data_split.json",
    )

    # ── Scaler — fitted ONLY on training data ──────────────────────────────
    logger.info("Fitting StandardScaler on training data …")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)   # scaler NOT re-fitted on test
    joblib.dump(scaler, mdl_dir / "scaler.joblib")

    # ── Imbalance: use class_weight for default training ───────────────────
    _, y_cw, class_weight = get_strategy("class_weight", X_train_s, y_train, seed=seed)

    # ── Select models to train ─────────────────────────────────────────────
    model_names = args.models or list(MODEL_REGISTRY.keys())
    # Skip majority classifier for tuning
    skip_tune = {"majority"}

    all_results: dict = {}

    for name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {name}")

        # --- Default model (no class weights) ---
        with timer(f"{name}_default"):
            pipe_default = build_pipeline(name)
            pipe_default.fit(X_train_s, y_train)
            res_default = evaluate_model(
                pipe_default, X_test_s, y_test,
                model_name=f"{name}_default",
            )
            all_results[f"{name}_default"] = res_default
            joblib.dump(pipe_default, mdl_dir / f"{name}_default.joblib")

        # --- Class-weighted model ---
        if name not in {"majority", "knn", "xgboost", "adaboost"}:
            with timer(f"{name}_weighted"):
                pipe_wt = build_pipeline(name, class_weight=class_weight)
                pipe_wt.fit(X_train_s, y_train)
                res_wt = evaluate_model(
                    pipe_wt, X_test_s, y_test,
                    model_name=f"{name}_weighted",
                )
                all_results[f"{name}_weighted"] = res_wt
                joblib.dump(pipe_wt, mdl_dir / f"{name}_weighted.joblib")

        # --- SMOTE + default model ---
        X_sm, y_sm, _ = get_strategy("smote", X_train_s, y_train, seed=seed)
        with timer(f"{name}_smote"):
            pipe_sm = build_pipeline(name)
            pipe_sm.fit(X_sm, y_sm)
            res_sm = evaluate_model(
                pipe_sm, X_test_s, y_test,
                model_name=f"{name}_smote",
            )
            all_results[f"{name}_smote"] = res_sm
            joblib.dump(pipe_sm, mdl_dir / f"{name}_smote.joblib")

        # --- Tuned model (optional) ---
        if args.tune and name not in skip_tune:
            with timer(f"{name}_tuned"):
                best_pipe, best_params, _ = tune_model(
                    name, X_train_s, y_train, g_train,
                    n_iter=15, seed=seed,
                )
                res_tuned = evaluate_model(
                    best_pipe, X_test_s, y_test,
                    model_name=f"{name}_tuned",
                )
                all_results[f"{name}_tuned"] = res_tuned
                joblib.dump(best_pipe, mdl_dir / f"{name}_tuned.joblib")
                save_json(best_params, met_dir / f"{name}_best_params.json")

    # ── Save all results ───────────────────────────────────────────────────
    save_json(all_results, met_dir / "all_model_results.json")

    # Comparison table
    comp_df = model_comparison_table(all_results)
    comp_df.to_csv(met_dir / "model_comparison.csv")
    logger.info("\n" + comp_df.to_string())

    # Best model by macro-F1
    from src.evaluation import select_best_model
    best_name = select_best_model(all_results, primary_metric="macro_f1")
    save_json({"best_model": best_name}, met_dir / "best_model.json")
    logger.info(f"\nBest model: {best_name}")

    logger.info("\nTraining complete. Results saved to results/metrics/")


if __name__ == "__main__":
    main()
