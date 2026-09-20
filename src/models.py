"""
src/models.py
=============
Model factory and sklearn Pipeline construction.

All models are constructed with fixed random seeds for reproducibility.
Class weights can be injected at pipeline-build time.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

from src.utils import load_config


def _seed():
    return load_config().get("random_seed", 42)


# ─────────────────────────────────────────────────────────────────────────────
#  Individual model constructors
# ─────────────────────────────────────────────────────────────────────────────

def majority_classifier(**kw) -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def logistic_regression(class_weight=None, **kw) -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        random_state=_seed(),
        class_weight=class_weight,
        solver="lbfgs",
        multi_class="auto",
        C=1.0,
    )


def knn(n_neighbors: int = 5, **kw) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)


def linear_svm(class_weight=None, **kw) -> LinearSVC:
    return LinearSVC(
        max_iter=2000,
        random_state=_seed(),
        class_weight=class_weight,
        C=1.0,
    )


def rbf_svm(class_weight=None, C: float = 1.0, gamma: str = "scale", **kw) -> SVC:
    return SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        random_state=_seed(),
        probability=True,
        cache_size=500,
    )


def random_forest(class_weight=None, n_estimators: int = 200, **kw) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=_seed(),
        class_weight=class_weight,
        n_jobs=-1,
        max_depth=None,
    )


def extra_trees(class_weight=None, n_estimators: int = 200, **kw) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        random_state=_seed(),
        class_weight=class_weight,
        n_jobs=-1,
    )


def xgboost_model(n_estimators: int = 200, **kw) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        random_state=_seed(),
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_jobs=-1,
        verbosity=0,
    )


def hist_gradient_boosting(class_weight=None, **kw) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        random_state=_seed(),
        class_weight=class_weight,
        max_iter=200,
    )


def adaboost(n_estimators: int = 100, **kw) -> AdaBoostClassifier:
    return AdaBoostClassifier(
        n_estimators=n_estimators,
        random_state=_seed(),
        algorithm="SAMME",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Model registry
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, callable] = {
    "majority":       majority_classifier,
    "logreg":         logistic_regression,
    "knn":            knn,
    "linear_svm":     linear_svm,
    "rbf_svm":        rbf_svm,
    "random_forest":  random_forest,
    "extra_trees":    extra_trees,
    "xgboost":        xgboost_model,
    "hist_gb":        hist_gradient_boosting,
    "adaboost":       adaboost,
}

# Which models require feature scaling
NEEDS_SCALING: set[str] = {"logreg", "knn", "linear_svm", "rbf_svm"}

# Which models support class_weight parameter
SUPPORTS_CLASS_WEIGHT: set[str] = {
    "logreg", "linear_svm", "rbf_svm",
    "random_forest", "extra_trees", "hist_gb",
}


def build_pipeline(
    model_name: str,
    class_weight=None,
    scale: bool | None = None,
    **model_kwargs,
) -> Pipeline:
    """
    Build a sklearn Pipeline: [StandardScaler] → model.

    Scaler is included only for distance/gradient-sensitive models unless
    overridden by `scale`.

    Parameters
    ----------
    model_name   : key in MODEL_REGISTRY
    class_weight : dict or 'balanced' or None
    scale        : override scaling decision
    **model_kwargs : extra kwargs forwarded to the model constructor

    Returns
    -------
    sklearn Pipeline
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY)}")

    if model_name in SUPPORTS_CLASS_WEIGHT:
        model_kwargs["class_weight"] = class_weight

    model = MODEL_REGISTRY[model_name](**model_kwargs)

    use_scale = scale if scale is not None else (model_name in NEEDS_SCALING)

    steps = []
    if use_scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", model))

    return Pipeline(steps)


def get_all_model_names() -> list[str]:
    return list(MODEL_REGISTRY.keys())
