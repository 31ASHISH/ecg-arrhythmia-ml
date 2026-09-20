"""
src/data_loader.py
==================
MIT-BIH Arrhythmia Database loading and AAMI label mapping.

Key design decisions
--------------------
- wfdb is used for all record I/O (PhysioNet native library).
- AAMI 5-class mapping is applied at load time so downstream code
  never sees raw annotation symbols.
- Record metadata (stats) is computed once and cached.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import wfdb

from src.utils import get_project_root, load_config, logger

# ── AAMI mapping ──────────────────────────────────────────────────────────────
# Standard AAMI EC57 / ANSI 5-class formulation
AAMI_MAPPING: dict[str, str] = {
    # Normal
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # Supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # Ventricular ectopic
    "V": "V", "E": "V",
    # Fusion
    "F": "F",
    # Unknown / artefact
    "Q": "Q", "f": "Q", "?": "Q",
}

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
AAMI_LABELS  = {c: i for i, c in enumerate(AAMI_CLASSES)}


def map_aami(symbol: str) -> Optional[str]:
    """Map raw MIT-BIH beat annotation symbol → AAMI class or None."""
    return AAMI_MAPPING.get(symbol, None)


# ── Raw data directory ────────────────────────────────────────────────────────
def _raw_dir() -> Path:
    cfg = load_config()
    root = get_project_root()
    return root / cfg["paths"]["raw_data"]


# ── Record loading ────────────────────────────────────────────────────────────
def load_record(record_id: str) -> tuple[np.ndarray, wfdb.Annotation, int]:
    """
    Load a single MIT-BIH record.

    Parameters
    ----------
    record_id : str  e.g. '100'

    Returns
    -------
    signal     : ndarray shape (n_samples, n_channels)
    annotation : wfdb.Annotation object
    fs         : sampling frequency (int)
    """
    raw_dir = _raw_dir()
    record_path = str(raw_dir / record_id)

    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    # Use first channel (MLII in most records)
    signal = record.p_signal[:, 0].astype(np.float32)
    fs = record.fs

    return signal, annotation, fs


def get_all_records(config: dict | None = None) -> list[str]:
    """Return record IDs according to the current run mode."""
    from src.utils import get_records
    return get_records(config)


def load_dataset_metadata(records: list[str]) -> pd.DataFrame:
    """
    Compute per-record metadata: total samples, duration, beat counts.

    Returns
    -------
    DataFrame with columns:
      record_id, total_samples, duration_s, fs,
      total_beats, N_beats, S_beats, V_beats, F_beats, Q_beats,
      other_beats
    """
    rows = []
    for rec_id in records:
        try:
            signal, ann, fs = load_record(rec_id)
            duration = len(signal) / fs
            symbols = ann.symbol

            counts: dict[str, int] = {c: 0 for c in AAMI_CLASSES}
            other = 0
            for sym in symbols:
                aami = map_aami(sym)
                if aami:
                    counts[aami] += 1
                else:
                    other += 1

            rows.append(
                {
                    "record_id": rec_id,
                    "total_samples": len(signal),
                    "duration_s": round(duration, 1),
                    "fs": fs,
                    "total_beats": sum(counts.values()),
                    **{f"{c}_beats": counts[c] for c in AAMI_CLASSES},
                    "other_beats": other,
                }
            )
        except Exception as exc:
            logger.warning(f"Could not load record {rec_id}: {exc}")

    return pd.DataFrame(rows)


# ── Convenience: check which records are downloaded ───────────────────────────
def available_records() -> list[str]:
    """Return record IDs that exist in data/raw/."""
    raw = _raw_dir()
    found = sorted({p.stem for p in raw.glob("*.hea")})
    return found
