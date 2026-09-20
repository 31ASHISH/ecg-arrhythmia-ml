"""
src/preprocessing.py
====================
ECG signal conditioning pipeline.

Stages (applied in order)
--------------------------
1. Baseline-wander removal  – high-pass Butterworth (0.5 Hz)
2. Bandpass filter          – Butterworth 0.5–40 Hz
3. Power-line notch         – 60 Hz (IIR notch)
4. Per-record normalization – Z-score

All filter coefficients are designed with scipy.signal.

Design decision: filters are applied in cascade; no zero-phase distortion
because we use sosfiltfilt (forward-backward) throughout.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt

from src.utils import load_config, logger


def _butter_bandpass_sos(lowcut: float, highcut: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="band", output="sos")
    return sos


def _butter_highpass_sos(cutoff: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype="high", output="sos")
    return sos


def _notch_sos(freq: float, fs: int, quality: float = 30.0):
    b, a = iirnotch(freq / (0.5 * fs), quality)
    # Convert ba → sos for numerical stability
    from scipy.signal import tf2sos
    sos = tf2sos(b, a)
    return sos


# ── Public API ────────────────────────────────────────────────────────────────
def remove_baseline_wander(signal: np.ndarray, fs: int, cutoff: float = 0.5) -> np.ndarray:
    """High-pass filter to remove baseline wander."""
    sos = _butter_highpass_sos(cutoff, fs, order=4)
    return sosfiltfilt(sos, signal).astype(np.float32)


def bandpass_filter(
    signal: np.ndarray,
    fs: int,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Bandpass Butterworth filter."""
    sos = _butter_bandpass_sos(lowcut, highcut, fs, order)
    return sosfiltfilt(sos, signal).astype(np.float32)


def notch_filter(signal: np.ndarray, fs: int, freq: float = 60.0) -> np.ndarray:
    """IIR notch filter at power-line frequency."""
    sos = _notch_sos(freq, fs)
    return sosfiltfilt(sos, signal).astype(np.float32)


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    """Z-score normalise a 1-D signal."""
    mu = np.mean(signal)
    sigma = np.std(signal)
    if sigma < 1e-8:
        return signal - mu
    return ((signal - mu) / sigma).astype(np.float32)


def full_pipeline(
    signal: np.ndarray,
    fs: int,
    config: dict | None = None,
    apply_notch: bool = True,
) -> np.ndarray:
    """
    Apply the full preprocessing pipeline:
      baseline removal → bandpass → [notch] → z-score normalisation

    Parameters
    ----------
    signal      : raw 1-D ECG signal
    fs          : sampling frequency
    config      : project config dict (loaded automatically if None)
    apply_notch : whether to apply power-line notch filter

    Returns
    -------
    filtered, normalised 1-D ndarray (float32)
    """
    cfg = config or load_config()

    sig = remove_baseline_wander(signal, fs, cutoff=cfg.get("bandpass_low", 0.5))
    sig = bandpass_filter(
        sig,
        fs,
        lowcut=cfg.get("bandpass_low", 0.5),
        highcut=cfg.get("bandpass_high", 40.0),
        order=cfg.get("bandpass_order", 4),
    )
    if apply_notch:
        sig = notch_filter(sig, fs, freq=cfg.get("notch_freq", 60.0))
    sig = zscore_normalize(sig)
    return sig


def add_noise(
    signal: np.ndarray,
    snr_db: float = 20.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Add Gaussian white noise at a specified SNR (dB).
    Used for robustness experiments.
    """
    rng = np.random.RandomState(seed)
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.randn(len(signal)) * np.sqrt(noise_power)
    return (signal + noise).astype(np.float32)
