#!/usr/bin/env python
"""
scripts/preprocess.py
=====================
Load raw records, apply signal preprocessing, segment heartbeats,
and save segments to data/processed/segments.parquet.

Usage
-----
    python scripts/preprocess.py [--mode fast|full]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.segmentation import segment_all_records
from src.utils import (
    ensure_dirs,
    get_project_root,
    get_records,
    load_config,
    logger,
    set_seed,
    timer,
)


def main():
    parser = argparse.ArgumentParser(description="Preprocess ECG records")
    parser.add_argument("--mode", choices=["fast", "full"], default=None)
    args = parser.parse_args()

    cfg = load_config()
    if args.mode:
        cfg["mode"] = args.mode

    set_seed(cfg.get("random_seed", 42))
    ensure_dirs(cfg)

    records = get_records(cfg)
    logger.info(f"Preprocessing {len(records)} records …")

    out_path = get_project_root() / cfg["paths"]["processed_data"] / "segments.parquet"

    with timer("segmentation"):
        segments_df = segment_all_records(records, config=cfg)

    # Save (lists stored as object dtype → parquet handles it via pyarrow)
    logger.info(f"Saving {len(segments_df)} segments → {out_path}")
    segments_df.to_parquet(out_path, index=False)

    # Quick summary
    logger.info("\n=== Segment Summary ===")
    logger.info(f"  Total beats   : {len(segments_df):,}")
    logger.info(f"  Records       : {segments_df['record_id'].nunique()}")
    logger.info(f"  Class distribution:")
    for cls, cnt in segments_df["aami_label"].value_counts().items():
        pct = cnt / len(segments_df) * 100
        logger.info(f"    {cls}: {cnt:,} ({pct:.1f}%)")

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
