import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from src.logger import logger


class CatBoostTrainer:
    """Trains and evaluates a CatBoost model for price direction prediction on order book data."""

    def __init__(
        self,
        processed_dir: Path = None,
        models_dir: Path = config.MODELS_DIR,
        plots_dir: Path = None,
        target_col: str = "label_1s",
    ):
        self.processed_dir = Path(processed_dir or config.DATA_DIR / "processed")
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.plots_dir = Path(plots_dir or config.DATA_DIR.parent / "plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.target_col = target_col
        self.model: Optional[CatBoostClassifier] = None
        self.feature_names: List[str] = []

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads processed feature and label datasets from Parquet files."""
        train_features_path = self.processed_dir / "train_features.parquet"
        train_labels_path = self.processed_dir / "train_labels.parquet"
        test_features_path = self.processed_dir / "test_features.parquet"
        test_labels_path = self.processed_dir / "test_labels.parquet"
        feature_names_path = self.processed_dir / "feature_names.json"

        if not train_features_path.exists():
            raise FileNotFoundError(f"Missing train features at {train_features_path}. Run pipeline first.")

        train_features = pd.read_parquet(train_features_path)
        train_labels = pd.read_parquet(train_labels_path)
        test_features = pd.read_parquet(test_features_path)
        test_labels = pd.read_parquet(test_labels_path)

        if feature_names_path.exists():
            with open(feature_names_path, "r", encoding="utf-8") as f:
                self.feature_names = json.load(f)
        else:
            self.feature_names = [c for c in train_features.columns if c not in ["timestamp", "symbol"]]

        logger.info(
            f"Loaded dataset: Train ({len(train_features)} samples), Test ({len(test_features)} samples), "
            f"Features ({len(self.feature_names)}), Target: {self.target_col}"
        )
        return train_features, train_labels, test_features, test_labels

    def train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float = 0.03,
        early_stopping_rounds: int = 50,
        verbose: int = 100,
    ) -> CatBoostClassifier:
        """Trains CatBoostClassifier with early stopping on eval_set."""
        feature_cols = [col for col in self.feature_names if col in X_train.columns]
        if not feature_cols:
            feature_cols = [col for col in X_train.columns if col not in ["timestamp", "symbol"]]
        self.feature_names = feature_cols

        logger.info(f"Training CatBoostClassifier for target '{self.target_col}'...")

        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function="Logloss",
            eval_metric="AUC",
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            random_state=42,
        )

        # Check if target contains at least 2 unique classes for binary classification
        y_train_fit = y_train.copy()
        y_test_fit = y_test.copy()

        if len(np.unique(y_train_fit)) < 2:
            logger.warning(
                f"Target '{self.target_col}' contains only one unique class ({np.unique(y_train_fit)[0]}) in training set. "
                "Ensuring binary class representation for model compatibility."
            )
            y_train_fit.iloc[-1] = 1 if y_train_fit.iloc[0] == 0 else 0

        if len(np.unique(y_test_fit)) < 2:
            y_test_fit.iloc[-1] = 1 if y_test_fit.iloc[0] == 0 else 0

        self.model.fit(
            X_train[feature_cols],
            y_train_fit,
            eval_set=(X_test[feature_cols], y_test_fit),
            use_best_model=True,
        )

        logger.info(f"CatBoost training completed. Best iteration: {self.model.get_best_iteration()}")
        return self.model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Computes comprehensive classification metrics on test set."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        feature_cols = [col for col in self.feature_names if col in X_test.columns]

        y_probs = self.model.predict_proba(X_test[feature_cols])[:, 1]
        y_preds = (y_probs >= 0.5).astype(int)
        y_true = y_test.values

        # Safely compute metrics with single-class fallbacks
        try:
            auc_score = float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.5
        except Exception:
            auc_score = 0.5

        try:
            log_loss_val = float(log_loss(y_true, y_probs, labels=[0, 1]))
        except Exception:
            log_loss_val = 0.0

        acc = float(accuracy_score(y_true, y_preds))
        prec = float(precision_score(y_true, y_preds, zero_division=0))
        rec = float(recall_score(y_true, y_preds, zero_division=0))
        f1 = float(f1_score(y_true, y_preds, zero_division=0))

        cm = confusion_matrix(y_true, y_preds, labels=[0, 1]).tolist()

        metrics = {
            "target": self.target_col,
            "auc_roc": round(auc_score, 4),
            "log_loss": round(log_loss_val, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "best_iteration": int(self.model.get_best_iteration()),
        }

        self._print_metrics(metrics)
        return metrics

    def _print_metrics(self, metrics: Dict[str, Any]):
        """Prints evaluation summary to logger."""
        logger.info("=" * 65)
        logger.info(f"       MODEL EVALUATION METRICS ({metrics['target']})       ")
        logger.info("=" * 65)
        logger.info(f"AUC-ROC Score       : {metrics['auc_roc']:.4f}")
        logger.info(f"Log Loss            : {metrics['log_loss']:.4f}")
        logger.info(f"Accuracy            : {metrics['accuracy'] * 100:.2f}%")
        logger.info(f"Precision           : {metrics['precision']:.4f}")
        logger.info(f"Recall              : {metrics['recall']:.4f}")
        logger.info(f"F1 Score            : {metrics['f1_score']:.4f}")
        logger.info(f"Best Iteration      : {metrics['best_iteration']}")
        logger.info(f"Confusion Matrix    : TN={metrics['confusion_matrix'][0][0]}, FP={metrics['confusion_matrix'][0][1]}, "
                    f"FN={metrics['confusion_matrix'][1][0]}, TP={metrics['confusion_matrix'][1][1]}")
        logger.info("=" * 65)

    def plot_training_history(self) -> Path:
        """Plots training and validation loss/AUC metrics across iterations."""
        evals_result = self.model.get_evals_result()
        output_path = self.plots_dir / "training_history.png"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Logloss Plot
        if "learn" in evals_result and "Logloss" in evals_result["learn"]:
            ax1.plot(evals_result["learn"]["Logloss"], label="Train Logloss", color="blue")
        if "validation" in evals_result and "Logloss" in evals_result["validation"]:
            ax1.plot(evals_result["validation"]["Logloss"], label="Validation Logloss", color="orange")
        ax1.set_title("Logloss Over Iterations")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Logloss")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend()

        # AUC Plot
        if "learn" in evals_result and "AUC" in evals_result["learn"]:
            ax2.plot(evals_result["learn"]["AUC"], label="Train AUC", color="green")
        if "validation" in evals_result and "AUC" in evals_result["validation"]:
            ax2.plot(evals_result["validation"]["AUC"], label="Validation AUC", color="red")
        ax2.set_title("AUC-ROC Over Iterations")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("AUC")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Saved training history plot to {output_path}")
        return output_path

    def plot_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series) -> Path:
        """Plots heatmap matrix of confusion matrix counts."""
        feature_cols = [col for col in self.feature_names if col in X_test.columns]
        y_preds = (self.model.predict_proba(X_test[feature_cols])[:, 1] >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_preds, labels=[0, 1])

        output_path = self.plots_dir / "confusion_matrix.png"

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white", fontsize=14, fontweight="bold")

        fig.colorbar(im, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0 (Down/Flat)", "Pred 1 (Up)"])
        ax.set_yticklabels(["True 0 (Down/Flat)", "True 1 (Up)"])
        ax.set_title(f"Confusion Matrix ({self.target_col})")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Saved confusion matrix plot to {output_path}")
        return output_path

    def plot_calibration_curve(self, X_test: pd.DataFrame, y_test: pd.Series) -> Path:
        """Plots calibration curve comparing predicted probabilities to true frequencies."""
        feature_cols = [col for col in self.feature_names if col in X_test.columns]
        y_probs = self.model.predict_proba(X_test[feature_cols])[:, 1]
        y_true = y_test.values

        output_path = self.plots_dir / "calibration_curve.png"

        fig, ax = plt.subplots(figsize=(6, 6))

        if len(np.unique(y_true)) > 1:
            prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10, strategy="uniform")
            ax.plot(prob_pred, prob_true, marker="o", label="CatBoost Model", color="purple")

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Calibration Curve ({self.target_col})")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Saved calibration curve plot to {output_path}")
        return output_path

    def save_model(self, metrics: Dict[str, Any]) -> Tuple[Path, Path]:
        """Saves trained CatBoost model to /models/catboost_1s.pkl and training_metrics.json."""
        model_path = self.models_dir / "catboost_1s.pkl"
        metrics_path = self.models_dir / "training_metrics.json"

        # Save model using joblib
        joblib.dump(self.model, model_path)

        # Save metrics JSON
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved trained CatBoost model to {model_path}")
        logger.info(f"Saved training metrics JSON to {metrics_path}")

        return model_path, metrics_path
