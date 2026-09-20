#!/usr/bin/env python
"""
scripts/evaluate.py
===================
Load the best trained model, evaluate on held-out test set,
generate all evaluation figures, and write the final report.

Usage
-----
    python scripts/evaluate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import AAMI_CLASSES
from src.evaluation import evaluate_model, model_comparison_table
from src.explainability import (
    feature_importance_plot,
    permutation_importance_plot,
    shap_summary,
)
from src.utils import (
    ensure_dirs,
    get_patient_splits,
    get_project_root,
    get_records,
    load_config,
    load_json,
    logger,
    save_json,
    set_seed,
    timer,
)
from src.visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_pca_scatter,
)


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure → {path.name}")


def main():
    cfg  = load_config()
    seed = set_seed()
    ensure_dirs(cfg)

    root     = get_project_root()
    proc_dir = root / cfg["paths"]["processed_data"]
    mdl_dir  = root / cfg["paths"]["models"]
    met_dir  = root / cfg["paths"]["metrics"]
    fig_dir  = root / cfg["paths"]["figures"]
    rep_dir  = root / cfg["paths"]["reports"]
    rep_dir.mkdir(parents=True, exist_ok=True)

    # ── Load features & meta ───────────────────────────────────────────────
    logger.info("Loading features …")
    X    = pd.read_parquet(proc_dir / "features.parquet").astype(np.float32)
    meta = pd.read_parquet(proc_dir / "meta.parquet")
    y    = meta["label_int"].values
    groups = meta["record_id"].values
    feature_names = X.columns.tolist()

    # ── Reconstruct test split ─────────────────────────────────────────────
    split_info  = load_json(met_dir / "data_split.json")
    test_recs   = split_info["test_records"]
    test_mask   = np.isin(groups, test_recs)
    X_test      = X[test_mask]
    y_test      = y[test_mask]

    scaler = joblib.load(mdl_dir / "scaler.joblib")
    X_test_s = scaler.transform(X_test)

    # ── Load all model results ─────────────────────────────────────────────
    all_results = load_json(met_dir / "all_model_results.json")
    best_name   = load_json(met_dir / "best_model.json")["best_model"]
    logger.info(f"Best model: {best_name}")

    # ── Figures: model comparison ──────────────────────────────────────────
    comp_df = model_comparison_table(all_results)
    for metric in ["Macro F1", "Balanced Acc", "Accuracy"]:
        if metric in comp_df.columns:
            fig = plot_model_comparison(comp_df, metric=metric)
            save_fig(fig, fig_dir / f"comparison_{metric.replace(' ', '_').lower()}.png")

    # ── Figures: confusion matrices for key models ─────────────────────────
    key_models = [k for k in all_results if "_default" in k or "_smote" in k]
    for mname in key_models[:6]:   # top 6 to avoid too many plots
        cm = all_results[mname].get("confusion_matrix", None)
        if cm:
            fig = plot_confusion_matrix(cm, title=f"Confusion Matrix — {mname}")
            save_fig(fig, fig_dir / f"cm_{mname}.png")

    # ── Load best model and generate explainability ────────────────────────
    best_path = mdl_dir / f"{best_name}.joblib"
    if best_path.exists():
        best_model = joblib.load(best_path)
        logger.info(f"Loaded best model from {best_path}")

        # Feature importance
        try:
            fig = feature_importance_plot(best_model, feature_names,
                                          title=f"Feature Importance — {best_name}")
            save_fig(fig, fig_dir / "feature_importance.png")
        except Exception as e:
            logger.warning(f"Feature importance failed: {e}")

        # Permutation importance
        try:
            with timer("permutation_importance"):
                fig = permutation_importance_plot(
                    best_model, X_test_s, y_test, feature_names,
                    n_repeats=5, seed=seed,
                    title=f"Permutation Importance — {best_name}",
                )
                save_fig(fig, fig_dir / "permutation_importance.png")
        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")

        # SHAP
        try:
            with timer("shap"):
                fig = shap_summary(best_model, X_test_s, feature_names,
                                   title=f"SHAP Summary — {best_name}")
                save_fig(fig, fig_dir / "shap_summary.png")
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")

    # ── PCA scatter of test features ───────────────────────────────────────
    try:
        fig = plot_pca_scatter(X_test_s, y_test, title="PCA — Test Set Features")
        save_fig(fig, fig_dir / "pca_test.png")
    except Exception as e:
        logger.warning(f"PCA plot failed: {e}")

    # ── Class distribution ─────────────────────────────────────────────────
    fig = plot_class_distribution(y, title="Overall Class Distribution")
    save_fig(fig, fig_dir / "class_distribution.png")

    # ── Write final report ─────────────────────────────────────────────────
    report = {
        "best_model":    best_name,
        "best_metrics":  all_results.get(best_name, {}),
        "all_models":    list(all_results.keys()),
        "test_records":  test_recs,
        "n_test_beats":  int(y_test.shape[0]),
        "n_features":    len(feature_names),
    }
    save_json(report, rep_dir / "final_report.json")

    # Human-readable summary
    bm = all_results.get(best_name, {})
    summary_lines = [
        "=" * 60,
        f"FINAL EVALUATION REPORT",
        "=" * 60,
        f"Best model       : {best_name}",
        f"Test beats       : {int(y_test.shape[0]):,}",
        f"Accuracy         : {bm.get('accuracy', 0):.4f}",
        f"Balanced Acc     : {bm.get('balanced_accuracy', 0):.4f}",
        f"Macro F1         : {bm.get('macro_f1', 0):.4f}",
        f"Weighted F1      : {bm.get('weighted_f1', 0):.4f}",
        f"ROC-AUC          : {bm.get('roc_auc', 'N/A')}",
        "",
        "Per-class F1:",
    ]
    for cls, m in bm.get("per_class", {}).items():
        summary_lines.append(f"  {cls}: {m.get('f1', 0):.4f}")

    summary = "\n".join(summary_lines)
    logger.info("\n" + summary)
    (rep_dir / "final_summary.txt").write_text(summary)

    logger.info("\nEvaluation complete. Figures saved to results/figures/")


if __name__ == "__main__":
    main()
