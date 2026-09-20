#!/usr/bin/env python
"""
scripts/run_experiments.py
==========================
Run all ablation and robustness experiments.

Usage
-----
    python scripts/run_experiments.py [--skip-window] [--skip-noise]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.experiments import (
    feature_ablation_experiment,
    imbalance_strategy_experiment,
    noise_robustness_experiment,
    training_size_experiment,
    window_size_experiment,
)
from src.utils import (
    ensure_dirs,
    get_patient_splits,
    get_project_root,
    get_records,
    load_config,
    load_json,
    logger,
    set_seed,
    timer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-window", action="store_true")
    parser.add_argument("--skip-noise", action="store_true")
    parser.add_argument("--model", default="random_forest")
    args = parser.parse_args()

    cfg  = load_config()
    seed = set_seed()
    ensure_dirs(cfg)

    root     = get_project_root()
    proc_dir = root / cfg["paths"]["processed_data"]
    met_dir  = root / cfg["paths"]["metrics"]

    # ── Load data ──────────────────────────────────────────────────────────
    X    = pd.read_parquet(proc_dir / "features.parquet").astype(np.float32)
    meta = pd.read_parquet(proc_dir / "meta.parquet")
    y    = meta["label_int"].values
    groups = meta["record_id"].values

    split_info  = load_json(met_dir / "data_split.json")
    train_recs  = split_info["train_records"]
    test_recs   = split_info["test_records"]

    train_mask = np.isin(groups, train_recs)
    test_mask  = np.isin(groups, test_recs)

    X_train_df = X[train_mask]
    y_train    = y[train_mask]
    X_test_df  = X[test_mask]
    y_test     = y[test_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_df)
    X_test_s  = scaler.transform(X_test_df)

    model = args.model
    logger.info(f"Running experiments with model: {model}")

    # ── 1. Imbalance experiment ────────────────────────────────────────────
    logger.info("\n[1/5] Imbalance Strategy Experiment")
    with timer("imbalance_exp"):
        imbalance_strategy_experiment(
            X_train_s, y_train, X_test_s, y_test, model_name=model, seed=seed
        )

    # ── 2. Feature ablation ────────────────────────────────────────────────
    logger.info("\n[2/5] Feature Ablation Experiment")
    with timer("ablation_exp"):
        feature_ablation_experiment(
            X_train_df, y_train, X_test_df, y_test, model_name=model
        )

    # ── 3. Training size (learning curve) ─────────────────────────────────
    logger.info("\n[3/5] Training Size Experiment")
    with timer("size_exp"):
        training_size_experiment(
            X_train_s, y_train, X_test_s, y_test, model_name=model, seed=seed
        )

    # ── 4. Window size (skippable for speed) ──────────────────────────────
    if not args.skip_window:
        logger.info("\n[4/5] Window Size Experiment")
        with timer("window_exp"):
            window_size_experiment(train_recs, test_recs, model_name=model)
    else:
        logger.info("\n[4/5] Window Size Experiment — SKIPPED")

    # ── 5. Noise robustness ────────────────────────────────────────────────
    if not args.skip_noise:
        logger.info("\n[5/5] Noise Robustness Experiment")
        segs = pd.read_parquet(proc_dir / "segments.parquet")
        segs_test = segs[segs["record_id"].isin(test_recs)].reset_index(drop=True)
        with timer("noise_exp"):
            noise_robustness_experiment(
                segs_test, X_train_s, y_train, model_name=model
            )
    else:
        logger.info("\n[5/5] Noise Robustness Experiment — SKIPPED")

    logger.info("\nAll experiments complete. Results saved to results/metrics/")


if __name__ == "__main__":
    main()
