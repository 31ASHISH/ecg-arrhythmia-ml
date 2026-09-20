"""
src/evaluation.py
=================
Model evaluation: metrics, confusion matrices, ROC/PR curves.

Design: All metric computation happens from actual model predictions.
No metric values are hard-coded anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.data_loader import AAMI_CLASSES
from src.utils import logger, save_json


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    model,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    model_name: str = "model",
    classes: list[str] = AAMI_CLASSES,
) -> dict:
    """
    Compute the full evaluation metric suite for a fitted model.

    Returns
    -------
    dict with keys:
      accuracy, balanced_accuracy,
      macro_precision, macro_recall, macro_f1,
      weighted_f1,
      per_class_{class}: {precision, recall, f1, support},
      confusion_matrix,
      roc_auc (if model has predict_proba),
      pr_auc  (if model has predict_proba),
      model_name, n_test_samples
    """
    y_true = np.array(y_test)
    y_pred = model.predict(X_test)

    # Ensure we score over all AAMI classes (0..4) even if some absent in test
    all_labels = list(range(len(classes)))

    # Basic metrics (labels= keeps macro consistent across all 5 classes)
    acc      = float(accuracy_score(y_true, y_pred))
    bal_acc  = float(balanced_accuracy_score(y_true, y_pred))
    mac_prec = float(precision_score(y_true, y_pred, average="macro",
                                     labels=all_labels, zero_division=0))
    mac_rec  = float(recall_score(y_true, y_pred, average="macro",
                                  labels=all_labels, zero_division=0))
    mac_f1   = float(f1_score(y_true, y_pred, average="macro",
                               labels=all_labels, zero_division=0))
    wt_f1    = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class metrics — force all 5 labels so missing classes show 0
    report = classification_report(
        y_true, y_pred,
        labels=all_labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        cls: {
            "precision": float(report[cls]["precision"]),
            "recall":    float(report[cls]["recall"]),
            "f1":        float(report[cls]["f1-score"]),
            "support":   int(report[cls]["support"]),
        }
        for cls in classes
        if cls in report
    }

    # Confusion matrix — all 5 labels
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    result = {
        "model_name":        model_name,
        "n_test_samples":    int(len(y_true)),
        "accuracy":          acc,
        "balanced_accuracy": bal_acc,
        "macro_precision":   mac_prec,
        "macro_recall":      mac_rec,
        "macro_f1":          mac_f1,
        "weighted_f1":       wt_f1,
        "per_class":         per_class,
        "confusion_matrix":  cm.tolist(),
    }

    # ROC-AUC and PR-AUC (require predict_proba)
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
            y_bin  = label_binarize(y_true, classes=all_labels)
            if y_bin.shape[1] > 1 and y_prob.shape[1] == len(classes):
                roc_auc = float(
                    roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
                )
                pr_auc = float(
                    average_precision_score(y_bin, y_prob, average="macro")
                )
                result["roc_auc"] = roc_auc
                result["pr_auc"]  = pr_auc
        except Exception as exc:
            logger.warning(f"ROC/PR AUC computation failed: {exc}")

    logger.info(
        f"[{model_name}] macro_F1={mac_f1:.4f}  "
        f"balanced_acc={bal_acc:.4f}  "
        f"acc={acc:.4f}"
    )
    return result


def model_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Convert a dict of evaluation results into a comparison DataFrame."""
    rows = []
    for name, r in results.items():
        row = {
            "Model":            name,
            "Accuracy":         r.get("accuracy", np.nan),
            "Balanced Acc":     r.get("balanced_accuracy", np.nan),
            "Macro Precision":  r.get("macro_precision", np.nan),
            "Macro Recall":     r.get("macro_recall", np.nan),
            "Macro F1":         r.get("macro_f1", np.nan),
            "Weighted F1":      r.get("weighted_f1", np.nan),
            "ROC-AUC":          r.get("roc_auc", np.nan),
            "PR-AUC":           r.get("pr_auc", np.nan),
        }
        # Per-class F1
        for cls in AAMI_CLASSES:
            row[f"F1-{cls}"] = r.get("per_class", {}).get(cls, {}).get("f1", np.nan)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    df = df.round(4)
    return df


def select_best_model(
    results: dict[str, dict],
    primary_metric: str = "macro_f1",
) -> str:
    """
    Select the best model by primary_metric.

    Returns model name. Does NOT hard-code any choice.
    """
    best_name = max(results, key=lambda n: results[n].get(primary_metric, 0))
    logger.info(
        f"Best model by {primary_metric}: {best_name} "
        f"({results[best_name].get(primary_metric, 0):.4f})"
    )
    return best_name


def save_evaluation_results(
    results: dict,
    output_path: str | Path,
) -> None:
    """Save evaluation dict to JSON."""
    save_json(results, output_path)
