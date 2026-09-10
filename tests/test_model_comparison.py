import json
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.model_comparison import ModelComparator


class TestModelComparator(unittest.TestCase):
    """Unit test suite for multi-model training, evaluation comparison, and Model Card generation."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name)

        self.processed_dir = self.base_path / "processed"
        self.models_dir = self.base_path / "models"

        self.processed_dir.mkdir()
        self.models_dir.mkdir()

        # Create synthetic datasets (60 train samples, 20 test samples, 4 features)
        np.random.seed(42)
        n_train, n_test, n_feats = 60, 20, 4
        feature_names = ["obi_level_5", "spread_relative", "bid_ask_volume_ratio", "micro_price"]

        train_x = np.random.randn(n_train, n_feats)
        test_x = np.random.randn(n_test, n_feats)

        train_y = (train_x[:, 0] > 0).astype(int)
        test_y = (test_x[:, 0] > 0).astype(int)

        self.train_features = pd.DataFrame(train_x, columns=feature_names)
        self.train_labels = pd.DataFrame({"label_1s": train_y})
        self.test_features = pd.DataFrame(test_x, columns=feature_names)
        self.test_labels = pd.DataFrame({"label_1s": test_y})

        self.train_features.to_parquet(self.processed_dir / "train_features.parquet")
        self.train_labels.to_parquet(self.processed_dir / "train_labels.parquet")
        self.test_features.to_parquet(self.processed_dir / "test_features.parquet")
        self.test_labels.to_parquet(self.processed_dir / "test_labels.parquet")

        with open(self.processed_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f)

        self.comparator = ModelComparator(
            processed_dir=self.processed_dir,
            models_dir=self.models_dir,
            target_col="label_1s",
        )

    def tearDown(self):
        self.test_dir.cleanup()

    def test_train_and_compare_models(self):
        """Verify training of all 3 models, comparison table construction, best model selection, and model card export."""
        X_train, y_train_df, X_test, y_test_df = self.comparator.load_data()

        # Train all 3 candidate models
        models_dict = self.comparator.train_all_models(
            X_train, y_train_df["label_1s"], X_test, y_test_df["label_1s"]
        )

        self.assertIn("CatBoost", models_dict)
        self.assertIn("XGBoost", models_dict)
        self.assertIn("LogisticRegression", models_dict)

        # Performance comparison table
        df_comp = self.comparator.compare_performance(X_test, y_test_df["label_1s"])
        self.assertEqual(len(df_comp), 3)
        self.assertIn("AUC", df_comp.columns)
        self.assertIn("Training Time (s)", df_comp.columns)

        # Select best model and export best_model.pkl & comparison_report.json
        best_name, model_path, report_path = self.comparator.select_best(df_comp)
        self.assertIsNotNone(best_name)
        self.assertTrue(model_path.exists())
        self.assertTrue(report_path.exists())

        # Generate Model Card
        card_path = self.comparator.generate_model_card(X_train, X_test, df_comp)
        self.assertTrue(card_path.exists())


if __name__ == "__main__":
    unittest.main()
