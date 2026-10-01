"""Model training, evaluation and selection.

Protocol
--------
* Every candidate is fitted on the **train** split.
* Gradient-boosted models use the **validation** split for early stopping.
* The champion is chosen by **validation** ROC-AUC (ties broken by validation log loss).
* The **test** split is touched only once, to report the metrics of every candidate.

The champion is saved as a single :class:`ModelBundle` (estimator + feature list + target +
provenance) so inference can never pair a model with the wrong feature order or horizon.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from orderbook_ml.config import Workspace
from orderbook_ml.dataset import InsufficientDataError, load_meta, load_split
from orderbook_ml.labels import horizon_from_label
from orderbook_ml.logging_utils import get_logger
from orderbook_ml.storage import write_json_atomic

logger = get_logger(__name__)

CANDIDATES: dict[str, str] = {
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
    "logreg": "Logistic regression",
}
BUNDLE_VERSION = 1


@dataclass
class ModelBundle:
    """Everything needed to score new snapshots with a trained model."""

    model: Any
    model_name: str
    feature_names: list[str]
    target: str
    horizon_sec: int
    symbol: str
    synthetic: bool
    test_metrics: dict[str, Any]
    trained_at: str = field(default_factory=lambda: pd.Timestamp.now(tz="UTC").isoformat())
    version: int = BUNDLE_VERSION

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.model.predict_proba(X[self.feature_names])
        return np.asarray(proba)[:, 1]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path) -> ModelBundle:
        """Loads a bundle written by :meth:`save`.

        joblib/pickle can execute code on load: only load model files you created yourself.
        """
        if not path.exists():
            raise FileNotFoundError(f"No trained model at {path}. Run `obml train` first.")
        bundle = joblib.load(path)
        if not isinstance(bundle, cls):
            raise TypeError(f"{path} does not contain a ModelBundle")
        if bundle.version != BUNDLE_VERSION:
            raise ValueError(
                f"Model bundle version {bundle.version} is not supported (expected "
                f"{BUNDLE_VERSION}); retrain with `obml train`."
            )
        return bundle

    def describe(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("model")
        return data


def make_estimator(name: str, seed: int = 42) -> Any:
    if name == "catboost":
        return CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            early_stopping_rounds=100,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric=["logloss", "auc"],
            early_stopping_rounds=100,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=2000)),
            ]
        )
    raise ValueError(f"Unknown model {name!r}; choose from {sorted(CANDIDATES)}")


def fit_estimator(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int = 42,
) -> tuple[Any, dict[str, Any]]:
    """Fits one candidate; returns the estimator and its training history (if any)."""
    model = make_estimator(name, seed)
    history: dict[str, Any] = {}
    if name == "catboost":
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        evals = model.get_evals_result()
        history = {
            "best_iteration": int(model.get_best_iteration()),
            "train_logloss": evals.get("learn", {}).get("Logloss", []),
            "validation_logloss": evals.get("validation", {}).get("Logloss", []),
            "validation_auc": evals.get("validation", {}).get("AUC", []),
        }
    elif name == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
        evals = model.evals_result()
        history = {
            "best_iteration": int(model.best_iteration),
            "train_logloss": evals["validation_0"]["logloss"],
            "validation_logloss": evals["validation_1"]["logloss"],
            "validation_auc": evals["validation_1"]["auc"],
        }
    else:
        model.fit(X_train, y_train)
    return model, history


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def classification_metrics(
    y_true: np.ndarray | pd.Series, proba: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Probability and decision metrics, plus a naive baseline for context."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(proba, dtype=float), 1e-7, 1 - 1e-7)
    pred = (p >= threshold).astype(int)
    base_rate = float(y.mean()) if len(y) else float("nan")
    two_classes = len(np.unique(y)) == 2

    model_ll = float(log_loss(y, p, labels=[0, 1]))
    baseline_ll = float(
        log_loss(y, np.full_like(p, np.clip(base_rate, 1e-7, 1 - 1e-7)), labels=[0, 1])
    )

    bins = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    calibration = [
        {
            "bin": f"{bins[b]:.1f}-{bins[b + 1]:.1f}",
            "mean_predicted": _round(float(p[bin_idx == b].mean())),
            "observed_rate": _round(float(y[bin_idx == b].mean())),
            "count": int((bin_idx == b).sum()),
        }
        for b in range(10)
        if (bin_idx == b).any()
    ]

    return {
        "samples": len(y),
        "positive_rate": _round(base_rate),
        "auc": _round(roc_auc_score(y, p)) if two_classes else None,
        "log_loss": _round(model_ll),
        "baseline_log_loss": _round(baseline_ll),
        "log_loss_skill": _round(1 - model_ll / baseline_ll) if baseline_ll > 0 else None,
        "brier": _round(brier_score_loss(y, p)),
        "accuracy": _round(accuracy_score(y, pred)),
        "precision": _round(precision_score(y, pred, zero_division=0)),
        "recall": _round(recall_score(y, pred, zero_division=0)),
        "f1": _round(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "calibration": calibration,
    }


def native_importance(model: Any, feature_names: Sequence[str]) -> dict[str, float]:
    """Model-specific importance, normalised to sum to 1 (sorted descending)."""
    if isinstance(model, CatBoostClassifier):
        raw = np.asarray(model.get_feature_importance(), dtype=float)
    elif isinstance(model, XGBClassifier):
        raw = np.asarray(model.feature_importances_, dtype=float)
    elif isinstance(model, Pipeline):
        raw = np.abs(np.asarray(model.named_steps["clf"].coef_, dtype=float)).ravel()
    else:
        return {}
    total = raw.sum() or 1.0
    pairs = sorted(zip(feature_names, raw / total, strict=True), key=lambda kv: -kv[1])
    return {name: round(float(value), 6) for name, value in pairs}


def _require_both_classes(y: pd.Series, split: str, target: str) -> None:
    counts = y.value_counts()
    if len(counts) < 2:
        only = int(counts.index[0]) if len(counts) else "none"
        raise InsufficientDataError(
            f"The {split} split contains a single class ({only}) for {target}. The mid price "
            "barely moved during this recording; collect a longer / more active session or "
            "use a longer horizon (e.g. --target label_10s)."
        )


def train_models(
    workspace: Workspace,
    target: str = "label_1s",
    candidates: Sequence[str] = tuple(CANDIDATES),
    seed: int = 42,
) -> dict[str, Any]:
    """Trains all candidates, saves the champion bundle and a JSON training report."""
    from orderbook_ml import reporting  # local import: matplotlib is only needed here

    meta = load_meta(workspace)
    if target not in meta.label_columns:
        raise ValueError(f"Unknown target {target!r}; available: {meta.label_columns}")
    unknown = sorted(set(candidates) - set(CANDIDATES))
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}; choose from {sorted(CANDIDATES)}")

    splits = {name: load_split(workspace, name) for name in ("train", "validation", "test")}
    features = meta.feature_names
    X = {name: df[features] for name, df in splits.items()}
    y = {name: df[target].astype(int) for name, df in splits.items()}
    _require_both_classes(y["train"], "train", target)
    _require_both_classes(y["validation"], "validation", target)

    results: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Any] = {}
    for name in candidates:
        logger.info("Training %s on %d rows...", CANDIDATES[name], len(X["train"]))
        start = time.perf_counter()
        model, history = fit_estimator(
            name, X["train"], y["train"], X["validation"], y["validation"], seed
        )
        fit_seconds = time.perf_counter() - start
        fitted[name] = model
        results[name] = {
            "display_name": CANDIDATES[name],
            "fit_seconds": round(fit_seconds, 3),
            "validation": classification_metrics(
                y["validation"], model.predict_proba(X["validation"])[:, 1]
            ),
            "test": classification_metrics(y["test"], model.predict_proba(X["test"])[:, 1]),
            "history": history,
            "importance": native_importance(model, features),
        }
        logger.info(
            "%s: validation AUC=%s, test AUC=%s (%.1fs)",
            CANDIDATES[name],
            results[name]["validation"]["auc"],
            results[name]["test"]["auc"],
            fit_seconds,
        )

    def selection_key(name: str) -> tuple[float, float]:
        val = results[name]["validation"]
        return (-(val["auc"] or 0.0), val["log_loss"] or float("inf"))

    champion = min(candidates, key=selection_key)
    bundle = ModelBundle(
        model=fitted[champion],
        model_name=champion,
        feature_names=list(features),
        target=target,
        horizon_sec=horizon_from_label(target),
        symbol=meta.symbol,
        synthetic=meta.synthetic,
        test_metrics=results[champion]["test"],
    )
    bundle.save(workspace.model_path)

    report = {
        "target": target,
        "horizon_sec": bundle.horizon_sec,
        "symbol": meta.symbol,
        "synthetic": meta.synthetic,
        "selected_model": champion,
        "selection_rule": "highest validation ROC-AUC (ties: lowest validation log loss)",
        "trained_at": bundle.trained_at,
        "rows": meta.rows,
        "time_ranges": meta.time_ranges,
        "embargo_sec": meta.embargo_sec,
        "feature_names": list(features),
        "candidates": results,
    }
    workspace.reports.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report, workspace.reports / "training_report.json")
    reporting.write_training_plots(report, workspace.reports)
    reporting.write_model_card(report, workspace.reports / "model_card.md")
    logger.info("Champion: %s -> %s", CANDIDATES[champion], workspace.model_path)
    return report
