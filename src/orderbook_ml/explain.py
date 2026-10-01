"""SHAP explanations for the trained champion model (computed on the test split)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from orderbook_ml.config import Workspace
from orderbook_ml.dataset import load_split
from orderbook_ml.logging_utils import get_logger
from orderbook_ml.modeling import ModelBundle
from orderbook_ml.storage import write_json_atomic

logger = get_logger(__name__)

DEPENDENCE_FEATURES = ("obi_level_5", "spread_relative", "bid_ask_volume_ratio")


def shap_values(bundle: ModelBundle, X: pd.DataFrame, background_size: int = 200) -> np.ndarray:
    """Returns an ``(n_samples, n_features)`` matrix of SHAP values for P(up) in log-odds."""
    model = bundle.model
    if isinstance(model, Pipeline):  # standardised logistic regression
        scaler, clf = model.named_steps["scaler"], model.named_steps["clf"]
        X_scaled = scaler.transform(X)
        background = X_scaled[: min(background_size, len(X_scaled))]
        explainer: Any = shap.LinearExplainer(clf, background)
        values = explainer.shap_values(X_scaled)
    else:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)

    if isinstance(values, list):  # older SHAP: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def explain(workspace: Workspace, max_samples: int = 5_000) -> dict[str, float]:
    """Computes SHAP values, writes importance JSON and summary/dependence plots."""
    bundle = ModelBundle.load(workspace.model_path)
    test = load_split(workspace, "test")
    if len(test) > max_samples:  # evenly spaced subsample keeps the whole time range
        test = test.iloc[np.linspace(0, len(test) - 1, max_samples).astype(int)]
    X = test[bundle.feature_names].reset_index(drop=True)

    logger.info("Computing SHAP values for %d test rows (%s)...", len(X), bundle.model_name)
    values = shap_values(bundle, X)
    mean_abs = np.abs(values).mean(axis=0)
    importance = dict(
        sorted(
            ((f, round(float(v), 6)) for f, v in zip(X.columns, mean_abs, strict=True)),
            key=lambda kv: -kv[1],
        )
    )

    out = workspace.reports
    write_json_atomic(
        {"model": bundle.model_name, "target": bundle.target, "mean_abs_shap": importance},
        out / "feature_importance.json",
    )
    np.save(workspace.processed / "shap_values.npy", values)
    _summary_plot(values, X, out / "shap_summary.png", plot_type="dot")
    _summary_plot(values, X, out / "shap_importance.png", plot_type="bar")
    for feature in DEPENDENCE_FEATURES:
        if feature in X.columns:
            _dependence_plot(feature, values, X, out / f"shap_dependence_{feature}.png")

    top = list(importance.items())[:5]
    logger.info("Top SHAP features: %s", ", ".join(f"{k} ({v:.4f})" for k, v in top))
    return importance


def _summary_plot(values: np.ndarray, X: pd.DataFrame, path: Path, plot_type: str) -> None:
    plt.figure()
    shap.summary_plot(values, X, plot_type=plot_type, max_display=15, show=False)
    plt.title("SHAP feature impact on P(up)" if plot_type == "dot" else "Mean |SHAP| value")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")


def _dependence_plot(feature: str, values: np.ndarray, X: pd.DataFrame, path: Path) -> None:
    shap.dependence_plot(feature, values, X, interaction_index=None, show=False)
    plt.title(f"SHAP dependence - {feature}")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
