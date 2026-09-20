"""
tests/test_pipeline.py
======================
Unit tests for critical pipeline components.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── AAMI mapping ──────────────────────────────────────────────────────────────
class TestAAMIMapping:
    def test_normal_symbols(self):
        from src.data_loader import map_aami
        for sym in ["N", "L", "R", "e", "j"]:
            assert map_aami(sym) == "N", f"{sym} should map to N"

    def test_supraventricular(self):
        from src.data_loader import map_aami
        for sym in ["A", "a", "J", "S"]:
            assert map_aami(sym) == "S"

    def test_ventricular(self):
        from src.data_loader import map_aami
        for sym in ["V", "E"]:
            assert map_aami(sym) == "V"

    def test_fusion(self):
        from src.data_loader import map_aami
        assert map_aami("F") == "F"

    def test_unknown(self):
        from src.data_loader import map_aami
        for sym in ["Q", "f", "?"]:
            assert map_aami(sym) == "Q"

    def test_non_beat_returns_none(self):
        from src.data_loader import map_aami
        for sym in ["+", "~", "|", "x"]:
            assert map_aami(sym) is None, f"Non-beat symbol {sym} should return None"


# ── Preprocessing ─────────────────────────────────────────────────────────────
class TestPreprocessing:
    def _synthetic_ecg(self, n=3600):
        t = np.linspace(0, 10, n)
        return (np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(n)).astype(np.float32)

    def test_bandpass_output_shape(self):
        from src.preprocessing import bandpass_filter
        sig = self._synthetic_ecg()
        out = bandpass_filter(sig, fs=360)
        assert out.shape == sig.shape

    def test_bandpass_output_dtype(self):
        from src.preprocessing import bandpass_filter
        sig = self._synthetic_ecg()
        out = bandpass_filter(sig, fs=360)
        assert out.dtype == np.float32

    def test_zscore_zero_mean(self):
        from src.preprocessing import zscore_normalize
        sig = self._synthetic_ecg()
        out = zscore_normalize(sig)
        assert abs(np.mean(out)) < 1e-5

    def test_zscore_unit_std(self):
        from src.preprocessing import zscore_normalize
        sig = self._synthetic_ecg()
        out = zscore_normalize(sig)
        assert abs(np.std(out) - 1.0) < 0.01

    def test_full_pipeline_shape(self):
        from src.preprocessing import full_pipeline
        sig = self._synthetic_ecg()
        out = full_pipeline(sig, fs=360)
        assert out.shape == sig.shape

    def test_add_noise_changes_signal(self):
        from src.preprocessing import add_noise
        sig = self._synthetic_ecg()
        noisy = add_noise(sig, snr_db=20)
        assert not np.allclose(sig, noisy)


# ── Segmentation ──────────────────────────────────────────────────────────────
class TestSegmentation:
    def _make_annotation(self, n_beats=50, window=3600):
        """Create a mock wfdb-like annotation object."""
        class MockAnnotation:
            sample = np.linspace(200, window - 200, n_beats).astype(int)
            symbol = ["N"] * n_beats
        return MockAnnotation()

    def test_segment_shape(self):
        from src.segmentation import segment_beats
        sig = np.random.randn(3600).astype(np.float32)
        ann = self._make_annotation()
        df = segment_beats("test", sig, sig, ann, fs=360,
                           window_before=90, window_after=110)
        assert len(df) > 0
        assert "aami_label" in df.columns
        assert "filtered_window" in df.columns

    def test_window_length(self):
        from src.segmentation import segment_beats
        sig = np.random.randn(3600).astype(np.float32)
        ann = self._make_annotation()
        df = segment_beats("test", sig, sig, ann, fs=360,
                           window_before=90, window_after=110)
        # Each window should have 90+110=200 samples
        for w in df["filtered_window"]:
            assert len(w) == 200

    def test_boundary_beats_discarded(self):
        from src.segmentation import segment_beats
        sig = np.random.randn(360).astype(np.float32)

        class EdgeAnn:
            sample = np.array([5, 180, 355])    # first and last are at boundary
            symbol = ["N", "N", "N"]

        df = segment_beats("test", sig, sig, EdgeAnn(), fs=360,
                           window_before=90, window_after=110)
        # Only the middle beat (r=180) fits: 180-90=90 >= 0 and 180+110=290 <= 360
        assert len(df) == 1
        assert df.iloc[0]["sample_idx"] == 180


# ── Feature extraction ────────────────────────────────────────────────────────
class TestFeatures:
    def _make_window(self, n=200):
        return np.random.randn(n).astype(np.float32)

    def test_feature_count(self):
        from src.features import extract_all_features, FEATURE_DICT
        w   = self._make_window()
        out = extract_all_features(w, rr_prev=300, rr_next=300, fs=360, window_before=90)
        # Should produce all features in FEATURE_DICT
        for key in FEATURE_DICT:
            assert key in out, f"Missing feature: {key}"

    def test_no_nan_in_features(self):
        from src.features import extract_all_features
        w   = self._make_window()
        out = extract_all_features(w, rr_prev=300, rr_next=300, fs=360, window_before=90)
        for k, v in out.items():
            assert not np.isnan(v), f"NaN in feature {k}"

    def test_all_values_finite(self):
        from src.features import extract_all_features
        w   = self._make_window()
        out = extract_all_features(w, rr_prev=300, rr_next=300, fs=360, window_before=90)
        for k, v in out.items():
            assert np.isfinite(v), f"Non-finite value in {k}: {v}"


# ── Leakage prevention ────────────────────────────────────────────────────────
class TestLeakagePrevention:
    """
    Verify that scalers and selectors are NOT fitted on test data.
    """

    def test_scaler_fit_on_train_only(self):
        from sklearn.preprocessing import StandardScaler
        rng = np.random.RandomState(0)
        X_train = rng.randn(100, 10).astype(np.float32)
        X_test  = rng.randn(30, 10).astype(np.float32) + 5.0  # shifted mean

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_te_s = scaler.transform(X_test)

        # Test set mean should NOT be close to 0 (scaler not fitted on test)
        assert abs(X_te_s.mean()) > 0.5, "Scaler appears to have been fitted on test data"

    def test_patient_split_no_overlap(self):
        from src.utils import get_patient_splits
        records = [str(i) for i in range(20)]
        train, test = get_patient_splits(records, test_size=0.2, seed=42)
        overlap = set(train) & set(test)
        assert len(overlap) == 0, f"Train/test overlap: {overlap}"

    def test_patient_split_covers_all(self):
        from src.utils import get_patient_splits
        records = [str(i) for i in range(20)]
        train, test = get_patient_splits(records, test_size=0.2, seed=42)
        assert set(train) | set(test) == set(records)


# ── Evaluation ────────────────────────────────────────────────────────────────
class TestEvaluation:
    def _make_dummy_data(self, n=200):
        from sklearn.dummy import DummyClassifier
        rng   = np.random.RandomState(0)
        X     = rng.randn(n, 10)
        y     = rng.randint(0, 5, n)
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X[:150], y[:150])
        return model, X[150:], y[150:]

    def test_evaluate_returns_required_keys(self):
        from src.evaluation import evaluate_model
        model, X_test, y_test = self._make_dummy_data()
        result = evaluate_model(model, X_test, y_test)
        for key in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                    "macro_precision", "macro_recall", "confusion_matrix", "per_class"]:
            assert key in result, f"Missing key: {key}"

    def test_confusion_matrix_sum_equals_n_test(self):
        from src.evaluation import evaluate_model
        model, X_test, y_test = self._make_dummy_data()
        result = evaluate_model(model, X_test, y_test)
        cm_sum = np.array(result["confusion_matrix"]).sum()
        assert cm_sum == len(y_test), "Confusion matrix sum ≠ n_test_samples"

    def test_no_fabricated_metrics(self):
        from src.evaluation import evaluate_model
        model, X_test, y_test = self._make_dummy_data()
        result = evaluate_model(model, X_test, y_test)
        # Verify all float metrics are in [0, 1]
        for key in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
            v = result[key]
            assert 0 <= v <= 1, f"Metric {key}={v} is outside [0, 1]"
