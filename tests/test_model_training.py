import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.model_training import CatBoostTrainer


class TestCatBoostTrainer(unittest.TestCase):
    """Unit test suite for CatBoost model training pipeline."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name)

        self.processed_dir = self.base_path / "processed"
        self.models_dir = self.base_path / "models"
        self.plots_dir = self.base_path / "plots"

        self.processed_dir.mkdir()
        self.models_dir.mkdir()
        self.plots_dir.mkdir()

        # Synthetic feature matrix (100 train, 25 test samples, 5 features)
        np.random.seed(42)
        n_train, n_test, n_feats = 100, 25, 5

        self.feature_names = [f"feat_{i}" for i in range(n_feats)]
        train_x_data = np.random.randn(n_train, n_feats)
        test_x_data = np.random.randn(n_test, n_feats)

        # Synthetic labels correlated with feat_0
        train_y_data = (train_x_data[:, 0] > 0).astype(int)
        test_y_data = (test_x_data[:, 0] > 0).astype(int)

        self.train_features = pd.DataFrame(train_x_data, columns=self.feature_names)
        self.train_labels = pd.DataFrame({"label_1s": train_y_data})

        self.test_features = pd.DataFrame(test_x_data, columns=self.feature_names)
        self.test_labels = pd.DataFrame({"label_1s": test_y_data})

        # Save synthetic Parquet datasets
        self.train_features.to_parquet(self.processed_dir / "train_features.parquet")
        self.train_labels.to_parquet(self.processed_dir / "train_labels.parquet")
        self.test_features.to_parquet(self.processed_dir / "test_features.parquet")
        self.test_labels.to_parquet(self.processed_dir / "test_labels.parquet")

        with open(self.processed_dir / "feature_names.json", "w") as f:
            import json
            json.dump(self.feature_names, f)

        self.trainer = CatBoostTrainer(
            processed_dir=self.processed_dir,
            models_dir=self.models_dir,
            plots_dir=self.plots_dir,
            target_col="label_1s",
        )

    def tearDown(self):
        self.test_dir.cleanup()

    def test_catboost_training_and_evaluation(self):
        """Verify CatBoost model training, metrics calculation, and plot generation."""
        X_train, y_train_df, X_test, y_test_df = self.trainer.load_data()

        # Train model with small iteration count for fast testing
        model = self.trainer.train_catboost(
            X_train,
            y_train_df["label_1s"],
            X_test,
            y_test_df["label_1s"],
            iterations=50,
            depth=4,
            learning_rate=0.1,
            early_stopping_rounds=10,
            verbose=0,
        )

        self.assertIsNotNone(model)

        # Compute metrics
        metrics = self.trainer.evaluate(X_test, y_test_df["label_1s"])

        self.assertIn("auc_roc", metrics)
        self.assertIn("accuracy", metrics)
        self.assertGreater(metrics["auc_roc"], 0.70)  # Strong signal in synthetic data

        # Plot generators
        hist_plot = self.trainer.plot_training_history()
        cm_plot = self.trainer.plot_confusion_matrix(X_test, y_test_df["label_1s"])
        cal_plot = self.trainer.plot_calibration_curve(X_test, y_test_df["label_1s"])

        self.assertTrue(hist_plot.exists())
        self.assertTrue(cm_plot.exists())
        self.assertTrue(cal_plot.exists())

        # Model persistence
        model_path, metrics_path = self.trainer.save_model(metrics)
        self.assertTrue(model_path.exists())
        self.assertTrue(metrics_path.exists())


if __name__ == "__main__":
    unittest.main()
