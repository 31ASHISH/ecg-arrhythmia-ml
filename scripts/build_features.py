#!/usr/bin/env python
"""
scripts/build_features.py
=========================
Load segments.parquet → extract features → save features.parquet + meta.parquet.

Usage
-----
    python scripts/build_features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.features import build_feature_matrix, get_waveform_matrix
from src.utils import (
    ensure_dirs,
    get_project_root,
    load_config,
    logger,
    set_seed,
    timer,
)


def main():
    cfg = load_config()
    set_seed()
    ensure_dirs(cfg)

    proc_dir  = get_project_root() / cfg["paths"]["processed_data"]
    seg_path  = proc_dir / "segments.parquet"

    if not seg_path.exists():
        logger.error(f"segments.parquet not found at {seg_path}. Run preprocess.py first.")
        sys.exit(1)

    logger.info(f"Loading segments from {seg_path} …")
    segments_df = pd.read_parquet(seg_path)
    logger.info(f"  {len(segments_df):,} beats loaded")

    # ── Engineered features ────────────────────────────────────────────────
    logger.info("Extracting engineered features …")
    with timer("feature_extraction"):
        X, y, groups = build_feature_matrix(segments_df, cfg)

    meta_df = pd.DataFrame(
        {
            "beat_id":    segments_df["beat_id"].values,
            "record_id":  groups.values,
            "aami_label": segments_df["aami_label"].values,
            "label_int":  y.values,
        }
    )

    feat_path = proc_dir / "features.parquet"
    meta_path = proc_dir / "meta.parquet"

    X.to_parquet(feat_path, index=False)
    meta_df.to_parquet(meta_path, index=False)
    logger.info(f"Saved features  → {feat_path}  shape: {X.shape}")
    logger.info(f"Saved meta      → {meta_path}")

    # ── Waveform matrix ────────────────────────────────────────────────────
    logger.info("Extracting waveform matrix …")
    with timer("waveform_extraction"):
        X_wave, y_wave, groups_wave = get_waveform_matrix(segments_df)

    wave_path = proc_dir / "waveforms.parquet"
    X_wave.to_parquet(wave_path, index=False)
    logger.info(f"Saved waveforms → {wave_path}  shape: {X_wave.shape}")

    logger.info("\nFeature summary:")
    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  Beats:    {X.shape[0]:,}")
    nan_counts = X.isna().sum().sum()
    logger.info(f"  NaN cells: {nan_counts}")

    if nan_counts > 0:
        logger.warning("NaN values present — filling with 0 in features.parquet")
        X.fillna(0.0).to_parquet(feat_path, index=False)

    logger.info("Feature extraction complete.")


if __name__ == "__main__":
    main()
