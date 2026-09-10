import json
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.model_training import CatBoostTrainer
from src.explainability import SHAPExplainer


class TestSHAPExplainer(unittest.TestCase):
    """Unit test suite for SHAPExplainer and visualization exports."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name)

        self.processed_dir = self.base_path / "processed"
        self.models_dir = self.base_path / "models"
        self.plots_dir = self.base_path / "plots"

        self.processed_dir.mkdir()
        self.models_dir.mkdir()
        self.plots_dir.mkdir()

        # Create synthetic dataset with OBI, spread, and volume ratio features
        np.random.seed(42)
        n_samples = 40
        feature_names = ["obi_level_5", "spread_relative", "bid_ask_volume_ratio", "mid_price", "total_bid_volume"]

        X_train = pd.DataFrame(np.random.randn(n_samples, 5), columns=feature_names)
        y_train = (X_train["obi_level_5"] > 0).astype(int)

        X_test = pd.DataFrame(np.random.randn(20, 5), columns=feature_names)
        y_test = (X_test["obi_level_5"] > 0).astype(int)

        X_train.to_parquet(self.processed_dir / "train_features.parquet")
        pd.DataFrame({"label_1s": y_train}).to_parquet(self.processed_dir / "train_labels.parquet")
        X_test.to_parquet(self.processed_dir / "test_features.parquet")
        pd.DataFrame({"label_1s": y_test}).to_parquet(self.processed_dir / "test_labels.parquet")

        with open(self.processed_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f)

        # Train sample CatBoost model
        trainer = CatBoostTrainer(
            processed_dir=self.processed_dir,
            models_dir=self.models_dir,
            plots_dir=self.plots_dir,
            target_col="label_1s",
        )
        trainer.train_catboost(X_train, y_train, X_test, y_test, iterations=20, depth=4, verbose=0)
        metrics = trainer.evaluate(X_test, y_test)
        trainer.save_model(metrics)

        self.explainer = SHAPExplainer(
            model_path=self.models_dir / "catboost_1s.pkl",
            processed_dir=self.processed_dir,
            plots_dir=self.plots_dir,
        )

    def tearDown(self):
        self.test_dir.cleanup()

    def test_shap_values_and_importance(self):
        """Verify SHAP values computation and feature importance dictionary generation."""
        shap_vals = self.explainer.compute_shap_values()
        self.assertEqual(shap_vals.shape, (20, 5))

        importance = self.explainer.analyze_feature_importance()
        self.assertIn("obi_level_5", importance)
        self.assertTrue((self.processed_dir / "shap_values.npy").exists())
        self.assertTrue((self.processed_dir / "feature_importance.json").exists())

    def test_shap_plots_export(self):
        """Verify export of SHAP summary, importance, and dependence plot files."""
        summary_plot = self.explainer.plot_summary()
        importance_plot = self.explainer.plot_importance()
        dep_obi_plot = self.explainer.plot_dependence("obi_level_5", "shap_dependence_obi.png")

        self.assertTrue(summary_plot.exists())
        self.assertTrue(importance_plot.exists())
        self.assertTrue(dep_obi_plot.exists())


if __name__ == "__main__":
    unittest.main()
