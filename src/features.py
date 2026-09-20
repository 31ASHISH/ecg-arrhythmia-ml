"""
src/features.py
===============
ECG heartbeat feature engineering.

Feature categories
------------------
1. Time-domain statistical features   (13 features)
2. Morphological / shape features     (10 features)
3. RR interval / heart-rate features  ( 7 features)
4. Frequency-domain (FFT/PSD)         (10 features)
5. Wavelet (DWT db4, levels 1-5)      (15 features)
6. Energy and entropy features         ( 6 features)
7. Beat-shape descriptors              ( 5 features)
   ─────────────────────────────────────────────────
   Total engineered features          ≈ 66

Additionally we expose the normalised flattened waveform as a separate
representation (200 values) for direct waveform models / ablation.

All computations are performed on the *filtered* window.
"""

from __future__ import annotations

import warnings
from tqdm import tqdm
from typing import Optional

import numpy as np
import pandas as pd
import pywt
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from src.utils import load_config, logger

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
#  Feature dictionary (human-readable descriptions)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_DICT: dict[str, str] = {
    # Time-domain
    "td_mean":         "Mean amplitude of the beat window",
    "td_std":          "Standard deviation of amplitude",
    "td_min":          "Minimum amplitude",
    "td_max":          "Maximum amplitude",
    "td_range":        "Peak-to-peak amplitude range",
    "td_skewness":     "Skewness of amplitude distribution",
    "td_kurtosis":     "Excess kurtosis of amplitude distribution",
    "td_zero_cross":   "Number of zero-crossings",
    "td_energy":       "Sum of squared amplitudes",
    "td_p25":          "25th percentile of amplitude",
    "td_p50":          "Median amplitude",
    "td_p75":          "75th percentile of amplitude",
    "td_iqr":          "Interquartile range (p75 - p25)",
    # Morphological
    "morph_r_amp":     "R-peak amplitude (centre sample)",
    "morph_q_amp":     "Q-wave amplitude (5 samples before R)",
    "morph_s_amp":     "S-wave amplitude (minimum after R in [0..40])",
    "morph_qrs_area":  "QRS complex area (20 samples either side of R)",
    "morph_t_area":    "T-wave area (40-120 samples after R)",
    "morph_st_level":  "ST segment level (mean 80-120 ms after R)",
    "morph_qrs_dur":   "Estimated QRS duration (threshold crossing width)",
    "morph_r_rise":    "Slope from Q to R (R amplitude / distance)",
    "morph_r_fall":    "Slope from R to S (S amplitude / distance)",
    "morph_symm":      "R-peak symmetry (|left_area - right_area| / total_area)",
    # RR interval
    "rr_prev":         "Previous RR interval (samples)",
    "rr_next":         "Next RR interval (samples)",
    "rr_ratio":        "rr_next / rr_prev",
    "rr_mean_local":   "Mean of rr_prev and rr_next",
    "rr_diff":         "rr_next - rr_prev",
    "rr_norm_prev":    "rr_prev normalised by rr_mean_local",
    "rr_norm_next":    "rr_next normalised by rr_mean_local",
    # Frequency domain
    "fd_dom_freq":     "Dominant frequency (Hz) from FFT",
    "fd_centroid":     "Spectral centroid (Hz)",
    "fd_spread":       "Spectral spread (Hz)",
    "fd_vlf_power":    "Power in VLF band (0-1 Hz)",
    "fd_lf_power":     "Power in LF band (1-5 Hz)",
    "fd_hf_power":     "Power in HF band (5-15 Hz)",
    "fd_total_power":  "Total spectral power",
    "fd_lf_hf_ratio":  "LF/HF power ratio",
    "fd_peak_mag":     "Magnitude at dominant frequency",
    "fd_spectral_entropy": "Spectral entropy",
    # Wavelet
    "wt_cD1_energy":   "DWT detail level 1 energy",
    "wt_cD2_energy":   "DWT detail level 2 energy",
    "wt_cD3_energy":   "DWT detail level 3 energy",
    "wt_cD4_energy":   "DWT detail level 4 energy",
    "wt_cD5_energy":   "DWT detail level 5 energy",
    "wt_cD1_std":      "DWT detail level 1 standard deviation",
    "wt_cD2_std":      "DWT detail level 2 standard deviation",
    "wt_cD3_std":      "DWT detail level 3 standard deviation",
    "wt_cD4_std":      "DWT detail level 4 standard deviation",
    "wt_cD5_std":      "DWT detail level 5 standard deviation",
    "wt_cA5_energy":   "DWT approximation level 5 energy",
    "wt_cA5_std":      "DWT approximation level 5 standard deviation",
    "wt_total_energy": "Total DWT energy across all levels",
    "wt_energy_frac2": "DWT level 2 energy fraction",
    "wt_energy_frac3": "DWT level 3 energy fraction",
    # Entropy & energy
    "en_shannon":      "Shannon entropy of normalised amplitude histogram",
    "en_log_energy":   "Log-energy entropy",
    "en_approx_ent":   "Approximate entropy (m=2, r=0.2*std)",
    "en_rms":          "Root mean square amplitude",
    "en_variance":     "Signal variance",
    "en_mean_abs":     "Mean absolute value",
    # Shape descriptors
    "sh_slope_start":  "Linear slope of first quarter of window",
    "sh_slope_end":    "Linear slope of last quarter of window",
    "sh_peak_pos":     "Relative position of maximum (0-1)",
    "sh_trough_pos":   "Relative position of minimum (0-1)",
    "sh_curve_area":   "Absolute area under the curve",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Individual feature groups
# ─────────────────────────────────────────────────────────────────────────────

def _time_domain(w: np.ndarray) -> dict:
    p25, p50, p75 = np.percentile(w, [25, 50, 75])
    n = len(w)
    zc = int(np.sum(np.diff(np.sign(w)) != 0))
    return {
        "td_mean":       float(np.mean(w)),
        "td_std":        float(np.std(w)),
        "td_min":        float(np.min(w)),
        "td_max":        float(np.max(w)),
        "td_range":      float(np.max(w) - np.min(w)),
        "td_skewness":   float(skew(w)),
        "td_kurtosis":   float(kurtosis(w)),
        "td_zero_cross": float(zc),
        "td_energy":     float(np.sum(w ** 2)),
        "td_p25":        float(p25),
        "td_p50":        float(p50),
        "td_p75":        float(p75),
        "td_iqr":        float(p75 - p25),
    }


def _morphological(w: np.ndarray, window_before: int) -> dict:
    n = len(w)
    r_idx = window_before  # R-peak position in local window

    r_amp = float(w[r_idx])

    # Q: look 5 samples before R
    q_idx = max(0, r_idx - 5)
    q_amp = float(w[q_idx])

    # S: minimum in first 40 samples after R
    s_end = min(n, r_idx + 40)
    s_amp = float(np.min(w[r_idx:s_end])) if s_end > r_idx else 0.0

    # QRS area: 20 samples either side
    qrs_start = max(0, r_idx - 20)
    qrs_end   = min(n, r_idx + 20)
    qrs_area  = float(np.trapezoid(np.abs(w[qrs_start:qrs_end])))

    # T-wave area: 40 to 120 samples after R
    t_start = min(n, r_idx + 40)
    t_end   = min(n, r_idx + 120)
    t_area  = float(np.trapezoid(np.abs(w[t_start:t_end]))) if t_end > t_start else 0.0

    # ST level: mean 80-120 samples after R (at 360 Hz ≈ 222-333 ms)
    st_start = min(n, r_idx + 29)   # ~80 ms at 360 Hz
    st_end   = min(n, r_idx + 43)   # ~120 ms
    st_level = float(np.mean(w[st_start:st_end])) if st_end > st_start else 0.0

    # QRS duration: width above 50% of R amplitude
    threshold = 0.5 * abs(r_amp)
    above = np.where(np.abs(w) > threshold)[0]
    qrs_dur = float(len(above))

    # Slopes
    r_rise = float((r_amp - q_amp) / max(1, r_idx - q_idx))
    r_fall = float((s_amp - r_amp) / max(1, 5))

    # Symmetry
    left_area  = float(np.sum(np.abs(w[qrs_start:r_idx])))
    right_area = float(np.sum(np.abs(w[r_idx:qrs_end])))
    total_area = left_area + right_area + 1e-8
    symm = float(abs(left_area - right_area) / total_area)

    return {
        "morph_r_amp":    r_amp,
        "morph_q_amp":    q_amp,
        "morph_s_amp":    s_amp,
        "morph_qrs_area": qrs_area,
        "morph_t_area":   t_area,
        "morph_st_level": st_level,
        "morph_qrs_dur":  qrs_dur,
        "morph_r_rise":   r_rise,
        "morph_r_fall":   r_fall,
        "morph_symm":     symm,
    }


def _rr_features(rr_prev: float, rr_next: float) -> dict:
    rr_p = rr_prev if not np.isnan(rr_prev) else 0.0
    rr_n = rr_next if not np.isnan(rr_next) else 0.0
    mean_rr = (rr_p + rr_n) / 2.0 + 1e-8
    return {
        "rr_prev":       float(rr_p),
        "rr_next":       float(rr_n),
        "rr_ratio":      float(rr_n / (rr_p + 1e-8)),
        "rr_mean_local": float(mean_rr),
        "rr_diff":       float(rr_n - rr_p),
        "rr_norm_prev":  float(rr_p / mean_rr),
        "rr_norm_next":  float(rr_n / mean_rr),
    }


def _frequency_domain(w: np.ndarray, fs: int) -> dict:
    n = len(w)
    # FFT
    fft_mag = np.abs(np.fft.rfft(w))
    freqs   = np.fft.rfftfreq(n, d=1.0 / fs)

    dom_idx  = int(np.argmax(fft_mag[1:]) + 1)  # skip DC
    dom_freq = float(freqs[dom_idx])
    peak_mag = float(fft_mag[dom_idx])

    # Spectral centroid & spread
    total_mag = np.sum(fft_mag) + 1e-8
    centroid  = float(np.sum(freqs * fft_mag) / total_mag)
    spread    = float(np.sqrt(np.sum((freqs - centroid) ** 2 * fft_mag) / total_mag))

    # PSD via Welch
    nperseg = min(64, n)
    f_psd, psd = welch(w, fs=fs, nperseg=nperseg)

    def band_power(f_low, f_high):
        mask = (f_psd >= f_low) & (f_psd < f_high)
        return float(np.trapezoid(psd[mask], f_psd[mask])) if mask.any() else 0.0

    vlf = band_power(0,  1)
    lf  = band_power(1,  5)
    hf  = band_power(5, 15)
    total_power = band_power(0, fs / 2)

    # Spectral entropy
    psd_norm = psd / (np.sum(psd) + 1e-8)
    sp_ent = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-8)))

    return {
        "fd_dom_freq":        dom_freq,
        "fd_centroid":        centroid,
        "fd_spread":          spread,
        "fd_vlf_power":       vlf,
        "fd_lf_power":        lf,
        "fd_hf_power":        hf,
        "fd_total_power":     total_power,
        "fd_lf_hf_ratio":     float(lf / (hf + 1e-8)),
        "fd_peak_mag":        peak_mag,
        "fd_spectral_entropy": sp_ent,
    }


def _wavelet_features(w: np.ndarray, wavelet: str = "db4", level: int = 5) -> dict:
    try:
        coeffs = pywt.wavedec(w, wavelet, level=level)
    except Exception:
        coeffs = pywt.wavedec(w, wavelet, level=pywt.dwt_max_level(len(w), wavelet))

    cA = coeffs[0]
    details = coeffs[1:]   # cD5, cD4, cD3, cD2, cD1 (highest → lowest freq)

    d_energies = [float(np.sum(c ** 2)) for c in details]
    d_stds     = [float(np.std(c)) for c in details]
    total_energy = sum(d_energies) + float(np.sum(cA ** 2)) + 1e-8

    feats: dict = {}
    for i, (en, sd) in enumerate(zip(d_energies, d_stds)):
        lv = len(details) - i   # level label: highest to lowest
        feats[f"wt_cD{lv}_energy"] = en
        feats[f"wt_cD{lv}_std"]    = sd

    # Pad to always have 5 levels even if DWT was truncated
    for lv in range(1, 6):
        feats.setdefault(f"wt_cD{lv}_energy", 0.0)
        feats.setdefault(f"wt_cD{lv}_std",    0.0)

    feats["wt_cA5_energy"]   = float(np.sum(cA ** 2))
    feats["wt_cA5_std"]      = float(np.std(cA))
    feats["wt_total_energy"] = total_energy
    feats["wt_energy_frac2"] = feats["wt_cD2_energy"] / total_energy
    feats["wt_energy_frac3"] = feats["wt_cD3_energy"] / total_energy
    return feats


def _entropy_energy(w: np.ndarray) -> dict:
    # Shannon entropy from histogram
    hist, _ = np.histogram(w, bins=32, density=True)
    hist = hist + 1e-10
    hist /= hist.sum()
    shannon = float(-np.sum(hist * np.log2(hist)))

    # Log-energy entropy
    log_en = float(np.sum(np.log2(w ** 2 + 1e-10)))

    # Fast approximate entropy proxy (avoids O(N^2) template matching)
    half = len(w) // 2
    var1 = float(np.var(w[:half]) + 1e-10)
    var2 = float(np.var(w[half:]) + 1e-10)
    approx_ent = float(abs(np.log(var1 / var2)))

    return {
        "en_shannon":    shannon,
        "en_log_energy": log_en,
        "en_approx_ent": approx_ent,
        "en_rms":        float(np.sqrt(np.mean(w ** 2))),
        "en_variance":   float(np.var(w)),
        "en_mean_abs":   float(np.mean(np.abs(w))),
    }


def _shape_descriptors(w: np.ndarray) -> dict:
    n = len(w)
    q = n // 4

    # Slopes via linear regression
    def slope(segment):
        x = np.arange(len(segment))
        if len(x) < 2:
            return 0.0
        return float(np.polyfit(x, segment, 1)[0])

    slope_start = slope(w[:q])
    slope_end   = slope(w[3 * q:])
    peak_pos    = float(np.argmax(w) / (n - 1))
    trough_pos  = float(np.argmin(w) / (n - 1))
    curve_area  = float(np.trapezoid(np.abs(w)))

    return {
        "sh_slope_start": slope_start,
        "sh_slope_end":   slope_end,
        "sh_peak_pos":    peak_pos,
        "sh_trough_pos":  trough_pos,
        "sh_curve_area":  curve_area,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Master feature extraction function
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(
    window: np.ndarray,
    rr_prev: float,
    rr_next: float,
    fs: int = 360,
    window_before: int = 90,
) -> dict:
    """
    Extract the full feature vector from a single filtered beat window.

    Parameters
    ----------
    window        : 1-D array (filtered heartbeat)
    rr_prev       : samples to previous beat
    rr_next       : samples to next beat
    fs            : sampling frequency
    window_before : R-peak position within window

    Returns
    -------
    dict of feature_name → float
    """
    feats: dict = {}
    feats.update(_time_domain(window))
    feats.update(_morphological(window, window_before))
    feats.update(_rr_features(rr_prev, rr_next))
    feats.update(_frequency_domain(window, fs))
    feats.update(_wavelet_features(window))
    feats.update(_entropy_energy(window))
    feats.update(_shape_descriptors(window))
    return feats


# ─────────────────────────────────────────────────────────────────────────────
#  Build feature matrix from segments DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    segments_df: pd.DataFrame,
    config: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Extract features for every beat in segments_df.

    Returns
    -------
    X          : feature DataFrame (float32)
    y          : label Series (int)
    groups     : record_id Series (for GroupKFold)
    """
    cfg = config or load_config()
    fs = cfg.get("sampling_rate", 360)
    wb = cfg.get("window_before", 90)

    feature_rows: list[dict] = []
    meta_rows:    list[dict] = []

    logger.info(f"Extracting features for {len(segments_df)} beats …")

    for _, row in tqdm(segments_df.iterrows(), total=len(segments_df), desc="Features", ncols=80):
        window = np.array(row["filtered_window"], dtype=np.float32)
        feats  = extract_all_features(
            window,
            rr_prev=row["rr_prev"],
            rr_next=row["rr_next"],
            fs=fs,
            window_before=wb,
        )
        feature_rows.append(feats)
        meta_rows.append(
            {
                "beat_id":    row["beat_id"],
                "record_id":  row["record_id"],
                "aami_label": row["aami_label"],
                "label_int":  row["label_int"],
            }
        )

    X      = pd.DataFrame(feature_rows).astype(np.float32)
    meta   = pd.DataFrame(meta_rows)
    y      = meta["label_int"]
    groups = meta["record_id"]

    # Fill any remaining NaN (e.g., RR at record boundaries)
    X.fillna(0.0, inplace=True)

    logger.info(f"Feature matrix: {X.shape[0]} beats × {X.shape[1]} features")
    return X, y, groups


def get_waveform_matrix(segments_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Return the flattened normalised waveform as an alternative representation.
    Column names are w_000, w_001, …, w_N.
    """
    n_cols = len(segments_df.iloc[0]["filtered_window"])
    cols   = [f"w_{i:03d}" for i in range(n_cols)]
    X_wave = pd.DataFrame(
        [row["filtered_window"] for _, row in segments_df.iterrows()],
        columns=cols,
        dtype=np.float32,
    )
    y      = segments_df["label_int"].values
    groups = segments_df["record_id"].values
    return X_wave, pd.Series(y), pd.Series(groups)
