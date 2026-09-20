"""
src/utils.py
============
Shared utilities: config loading, seeding, logging, I/O helpers,
patient-wise split, timing.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ── Logger ────────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("ecg_ml")


# ── Config ────────────────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
_config_cache: dict | None = None


def load_config(path: str | Path | None = None) -> dict:
    """Load and cache the YAML config."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    cfg_path = Path(path) if path else _CONFIG_PATH
    with open(cfg_path, "r") as f:
        _config_cache = yaml.safe_load(f)
    return _config_cache


def get_records(config: dict | None = None) -> list[str]:
    """Return the record list according to the current mode."""
    cfg = config or load_config()
    mode = cfg.get("mode", "fast")
    if mode == "full":
        return [str(r) for r in cfg["all_records"]]
    return [str(r) for r in cfg["fast_records"]]


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def resolve_path(relative: str, config: dict | None = None) -> Path:
    """Resolve a config path relative to project root."""
    root = get_project_root()
    return root / relative


# ── Seeding ───────────────────────────────────────────────────────────────────
def set_seed(seed: int | None = None) -> int:
    cfg = load_config()
    seed = seed if seed is not None else cfg.get("random_seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


# ── I/O ───────────────────────────────────────────────────────────────────────
def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    logger.info(f"Saved JSON → {path}")


def load_json(path: str | Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} not JSON serializable")


# ── Patient-wise split ────────────────────────────────────────────────────────
def get_patient_splits(
    records: list[str],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Split records into train/test at the *patient* level so no record
    appears in both sets.

    Returns
    -------
    train_records, test_records
    """
    rng = np.random.RandomState(seed)
    records_arr = np.array(sorted(records))
    rng.shuffle(records_arr)
    n_test = max(1, int(len(records_arr) * test_size))
    test_records = list(records_arr[:n_test])
    train_records = list(records_arr[n_test:])
    logger.info(
        f"Patient split → train: {len(train_records)} records, "
        f"test: {len(test_records)} records"
    )
    logger.info(f"  Train records: {train_records}")
    logger.info(f"  Test  records: {test_records}")
    return train_records, test_records


# ── Timer ─────────────────────────────────────────────────────────────────────
@contextmanager
def timer(label: str = ""):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"[TIMER] {label}: {elapsed:.2f}s")


# ── Ensure directories exist ──────────────────────────────────────────────────
def ensure_dirs(config: dict | None = None) -> None:
    cfg = config or load_config()
    root = get_project_root()
    for key, rel in cfg.get("paths", {}).items():
        (root / rel).mkdir(parents=True, exist_ok=True)
