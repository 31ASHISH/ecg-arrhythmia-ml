#!/usr/bin/env python
"""
scripts/download_data.py
========================
Download MIT-BIH Arrhythmia Database records from PhysioNet.

Usage
-----
    python scripts/download_data.py [--mode fast|full]

The downloaded files (.dat, .hea, .atr) are saved to data/raw/.
"""

import argparse
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import wfdb

from src.utils import get_project_root, get_records, load_config, logger, timer


def download_records(records: list[str], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    total = len(records)
    for i, rec_id in enumerate(records, 1):
        hea = dest / f"{rec_id}.hea"
        if hea.exists():
            logger.info(f"[{i}/{total}] Already downloaded: {rec_id}")
            continue
        logger.info(f"[{i}/{total}] Downloading record {rec_id} …")
        try:
            # wfdb 4.3+ API: dl_files no longer accepts 'annotators' kwarg
            wfdb.dl_files(
                "mitdb",
                str(dest),
                [f"{rec_id}.dat", f"{rec_id}.hea", f"{rec_id}.atr"],
            )
        except TypeError:
            # Fallback for any remaining API differences
            try:
                wfdb.dl_database("mitdb", str(dest), records=[rec_id])
            except Exception as exc2:
                logger.error(f"  Fallback also failed for {rec_id}: {exc2}")
        except Exception as exc:
            logger.error(f"  Failed to download {rec_id}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Download MIT-BIH records")
    parser.add_argument("--mode", choices=["fast", "full"], default=None,
                        help="Override config mode")
    args = parser.parse_args()

    cfg = load_config()
    if args.mode:
        cfg["mode"] = args.mode

    records = get_records(cfg)
    dest    = get_project_root() / cfg["paths"]["raw_data"]

    logger.info(f"Downloading {len(records)} records (mode={cfg['mode']}) → {dest}")

    with timer("download"):
        download_records(records, dest)

    # Verify
    downloaded = [p.stem for p in dest.glob("*.hea")]
    missing    = [r for r in records if r not in downloaded]
    if missing:
        logger.warning(f"Missing records: {missing}")
    else:
        logger.info(f"All {len(records)} records downloaded successfully.")


if __name__ == "__main__":
    main()
