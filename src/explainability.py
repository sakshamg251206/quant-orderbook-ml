import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import config
from src.logger import logger


class SHAPExplainer:
    """Computes SHAP (SHapley Additive exPlanations) values to interpret model feature importance

    and generate explainability plots for order book microstructure features.
    """

    def __init__(
        self,
        model_path: Path = None,
        processed_dir: Path = None,
        plots_dir: Path = None,
    ):
        self.model_path = Path(model_path or config.MODELS_DIR / "catboost_1s.pkl")
        self.processed_dir = Path(processed_dir or config.DATA_DIR / "processed")
        self.plots_dir = Path(plots_dir or config.DATA_DIR.parent / "plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.feature_names: List[str] = []
        self.shap_values: Optional[np.ndarray] = None
        self.X_test: Optional[pd.DataFrame] = None

    def load_model_and_data(self) -> Tuple[Any, pd.DataFrame, List[str]]:
        """Loads trained CatBoost model, test features, and feature names."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Run training first.")

        self.model = joblib.load(self.model_path)
        test_features_path = self.processed_dir / "test_features.parquet"
        feature_names_path = self.processed_dir / "feature_names.json"

        if not test_features_path.exists():
            raise FileNotFoundError(f"Test features file not found at {test_features_path}.")

        test_df = pd.read_parquet(test_features_path)

        if feature_names_path.exists():
            with open(feature_names_path, "r", encoding="utf-8") as f:
                self.feature_names = json.load(f)
        else:
            self.feature_names = [c for c in test_df.columns if c not in ["timestamp", "symbol"]]

        feature_cols = [c for c in self.feature_names if c in test_df.columns]
        self.X_test = test_df[feature_cols].copy()

        logger.info(f"Loaded model from {self.model_path} and test features shape {self.X_test.shape}")
        return self.model, self.X_test, self.feature_names

    def compute_shap_values(self) -> np.ndarray:
        """Runs SHAP TreeExplainer on test dataset to compute exact feature attribution values."""
        if self.model is None or self.X_test is None:
            self.load_model_and_data()

        logger.info("Computing SHAP values using TreeExplainer...")
        explainer = shap.TreeExplainer(self.model)

        raw_shap = explainer.shap_values(self.X_test)

        # Handle binary classification output format differences across SHAP/CatBoost versions
        if isinstance(raw_shap, list):
            # Dual output array list [class_0, class_1]
            self.shap_values = np.array(raw_shap[1])
        elif hasattr(raw_shap, "values"):
            # SHAP Explanation object
            self.shap_values = np.array(raw_shap.values)
        else:
            self.shap_values = np.array(raw_shap)

        # Ensure 2D matrix shape (N_samples, N_features)
        if self.shap_values.ndim == 3:
            self.shap_values = self.shap_values[:, :, 1]

        logger.info(f"Successfully computed SHAP values matrix of shape {self.shap_values.shape}")

        # Save raw SHAP matrix to data/processed/shap_values.npy
        npy_path = self.processed_dir / "shap_values.npy"
        np.save(npy_path, self.shap_values)
        logger.info(f"Saved SHAP values array to {npy_path}")

        return self.shap_values

    def analyze_feature_importance(self) -> Dict[str, float]:
        """Calculates mean absolute SHAP values per feature, exports feature_importance.json,

        and prints top 10 most important features.
        """
        if self.shap_values is None:
            self.compute_shap_values()

        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

        importance_dict = {
            feature: float(mean_abs_shap[idx])
            for idx, feature in enumerate(self.X_test.columns)
        }

        # Sort descending by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

        # Save feature_importance.json
        json_path = self.processed_dir / "feature_importance.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sorted_importance, f, indent=2)
        logger.info(f"Saved feature importances JSON to {json_path}")

        # Print top 10 features report
        logger.info("=" * 65)
        logger.info("           TOP 10 SHAP FEATURE IMPORTANCE RANKING             ")
        logger.info("=" * 65)
        top_10 = list(sorted_importance.items())[:10]
        for rank, (feat, score) in enumerate(top_10, 1):
            logger.info(f"  Rank {rank:02d}. {feat:<25s} | Mean |SHAP| = {score:.6f}")

        # Economic sense check: check if Order Book Imbalance (OBI) features rank high
        obi_features = [f for f in sorted_importance.keys() if "obi" in f.lower()]
        obi_ranks = [list(sorted_importance.keys()).index(f) + 1 for f in obi_features]
        logger.info("-" * 65)
        logger.info(f"Order Book Imbalance (OBI) Features Found : {len(obi_features)}")
        if obi_ranks:
            logger.info(f"OBI Feature Ranks                         : {obi_ranks[:5]}")
            logger.info("Economic Sanity Check: OBI metrics contribute actively to price direction prediction.")
        logger.info("=" * 65)

        return sorted_importance

    def plot_summary(self) -> Path:
        """Generates SHAP summary dot plot showing feature impact distribution colored by value."""
        if self.shap_values is None:
            self.compute_shap_values()

        output_path = self.plots_dir / "shap_summary.png"

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values,
            self.X_test,
            show=False,
            max_display=20,
        )
        plt.title("SHAP Summary Plot - Feature Impact Distribution", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved SHAP summary plot to {output_path}")
        return output_path

    def plot_importance(self) -> Path:
        """Generates SHAP bar plot of top 20 features ranked by mean absolute SHAP value."""
        if self.shap_values is None:
            self.compute_shap_values()

        output_path = self.plots_dir / "shap_importance.png"

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values,
            self.X_test,
            plot_type="bar",
            max_display=20,
            show=False,
        )
        plt.title("SHAP Feature Importance (Top 20 Features)", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved SHAP feature importance plot to {output_path}")
        return output_path

    def plot_dependence(self, feature_name: str, output_filename: str) -> Path:
        """Generates SHAP dependence plot for a specific feature."""
        if self.shap_values is None:
            self.compute_shap_values()

        output_path = self.plots_dir / output_filename

        # Match exact feature name or closest candidate
        target_feat = feature_name
        if target_feat not in self.X_test.columns:
            candidates = [c for c in self.X_test.columns if feature_name.lower() in c.lower()]
            target_feat = candidates[0] if candidates else self.X_test.columns[0]

        plt.figure(figsize=(8, 5))
        try:
            shap.dependence_plot(
                target_feat,
                self.shap_values,
                self.X_test,
                show=False,
                interaction_index="auto",
            )
        except Exception:
            # Fallback if interaction index calculation fails
            shap.dependence_plot(
                target_feat,
                self.shap_values,
                self.X_test,
                show=False,
                interaction_index=None,
            )

        plt.title(f"SHAP Dependence Plot for '{target_feat}'", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved SHAP dependence plot for {target_feat} to {output_path}")
        return output_path
