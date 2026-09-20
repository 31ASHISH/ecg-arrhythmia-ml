"""
src/segmentation.py
===================
Heartbeat segmentation around annotated R-peaks.

Key design decisions
--------------------
- We rely entirely on expert annotations (wfdb .atr files) for R-peak
  positions — no automatic R-peak detection.  This removes one source
  of error and keeps the focus on ML methodology.
- Each beat window is [r_peak - window_before : r_peak + window_after].
- Beats whose window extends outside the signal boundary are discarded.
- The output is a pandas DataFrame containing both the raw and filtered
  waveform windows, plus metadata columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import wfdb

from src.data_loader import AAMI_CLASSES, map_aami
from src.utils import load_config, logger


def segment_beats(
    record_id: str,
    signal_raw: np.ndarray,
    signal_filtered: np.ndarray,
    annotation: "wfdb.Annotation",
    fs: int,
    window_before: int | None = None,
    window_after: int | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Extract fixed-length windows around annotated beats.

    Parameters
    ----------
    record_id        : e.g. '100'
    signal_raw       : unprocessed ECG (1-D float32)
    signal_filtered  : preprocessed ECG (1-D float32)
    annotation       : wfdb.Annotation object
    fs               : sampling frequency
    window_before    : samples before R-peak (default from config)
    window_after     : samples after  R-peak (default from config)
    config           : project config (loaded if None)

    Returns
    -------
    DataFrame with columns:
      beat_id, record_id, sample_idx, aami_label, label_int,
      raw_window (list), filtered_window (list), rr_prev, rr_next
    """
    cfg = config or load_config()
    wb = window_before if window_before is not None else cfg["window_before"]
    wa = window_after  if window_after  is not None else cfg["window_after"]
    total_window = wb + wa

    n_samples = len(signal_raw)
    rows = []

    r_peaks   = annotation.sample
    symbols   = annotation.symbol
    n_ann     = len(r_peaks)

    for idx in range(n_ann):
        sym  = symbols[idx]
        aami = map_aami(sym)
        if aami is None:
            continue  # skip non-beat annotations (rhythm, noise markers)

        r = r_peaks[idx]
        start = r - wb
        end   = r + wa

        # Discard beats that go out of bounds
        if start < 0 or end > n_samples:
            continue

        raw_win      = signal_raw[start:end].tolist()
        filtered_win = signal_filtered[start:end].tolist()

        # RR intervals (inter-beat intervals in samples)
        rr_prev = float(r - r_peaks[idx - 1]) if idx > 0       else np.nan
        rr_next = float(r_peaks[idx + 1] - r) if idx < n_ann-1 else np.nan

        rows.append(
            {
                "beat_id":         f"{record_id}_{idx:05d}",
                "record_id":       record_id,
                "sample_idx":      int(r),
                "raw_symbol":      sym,
                "aami_label":      aami,
                "label_int":       AAMI_CLASSES.index(aami),
                "raw_window":      raw_win,
                "filtered_window": filtered_win,
                "rr_prev":         rr_prev,
                "rr_next":         rr_next,
                "window_before":   wb,
                "window_after":    wa,
                "fs":              fs,
            }
        )

    df = pd.DataFrame(rows)
    logger.info(
        f"Record {record_id}: {len(df)} beats segmented "
        f"({dict(df['aami_label'].value_counts())})"
    )
    return df


def segment_all_records(
    records: list[str],
    config: dict | None = None,
    window_before: int | None = None,
    window_after: int | None = None,
) -> pd.DataFrame:
    """
    Run segmentation on all records and return a single concatenated DataFrame.

    This function imports preprocessing to apply the filter pipeline.
    """
    from src.data_loader import load_record
    from src.preprocessing import full_pipeline

    cfg = config or load_config()
    all_segments: list[pd.DataFrame] = []

    for rec_id in records:
        try:
            signal_raw, annotation, fs = load_record(rec_id)
            signal_filtered = full_pipeline(signal_raw, fs, cfg)
            df = segment_beats(
                record_id=rec_id,
                signal_raw=signal_raw,
                signal_filtered=signal_filtered,
                annotation=annotation,
                fs=fs,
                window_before=window_before,
                window_after=window_after,
                config=cfg,
            )
            all_segments.append(df)
        except Exception as exc:
            logger.error(f"Segmentation failed for record {rec_id}: {exc}")

    if not all_segments:
        raise RuntimeError("No segments produced — check that records are downloaded.")

    combined = pd.concat(all_segments, ignore_index=True)
    logger.info(
        f"Total beats: {len(combined)}  |  "
        f"Class distribution:\n{combined['aami_label'].value_counts()}"
    )
    return combined
