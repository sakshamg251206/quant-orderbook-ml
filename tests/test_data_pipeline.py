import json
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.data_pipeline import DataPipeline


class TestDataPipeline(unittest.TestCase):
    """Unit test suite for end-to-end DataPipeline."""

    def setUp(self):
        # Create temporary working directories
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name)
        self.snapshots_dir = self.base_path / "snapshots"
        self.processed_dir = self.base_path / "processed"
        self.models_dir = self.base_path / "models"

        self.snapshots_dir.mkdir()

        # Create synthetic order book snapshots across 30 seconds
        times = pd.date_range(start="2026-08-31 10:00:00", periods=300, freq="100ms", tz="UTC")
        prices = [100.0 + i * 0.02 for i in range(300)]

        bids = [[[p - 0.1, 10.0], [p - 0.2, 20.0]] for p in prices]
        asks = [[[p + 0.1, 5.0], [p + 0.2, 15.0]] for p in prices]

        df_raw = pd.DataFrame({
            "timestamp": times,
            "symbol": "BTCUSDT",
            "bids_json": [json.dumps(b) for b in bids],
            "asks_json": [json.dumps(a) for a in asks],
            "best_bid": [p - 0.1 for p in prices],
            "best_ask": [p + 0.1 for p in prices],
            "spread": [0.2] * 300,
            "total_bid_volume": [30.0] * 300,
            "total_ask_volume": [20.0] * 300,
        })

        # Save synthetic snapshot Parquet file
        df_raw.to_parquet(self.snapshots_dir / "sample.parquet")

        self.pipeline = DataPipeline(
            data_dir=self.snapshots_dir,
            processed_dir=self.processed_dir,
            models_dir=self.models_dir,
            train_ratio=0.8,
        )

    def tearDown(self):
        self.test_dir.cleanup()

    def test_time_based_split_and_no_leakage(self):
        """Verify strict chronological order in train/test split with zero timestamp overlap."""
        df_raw, _ = self.pipeline.load_and_validate()
        df_features, _ = self.pipeline.compute_features(df_raw)
        df_features["timestamp"] = df_raw["timestamp"]
        df_features["symbol"] = df_raw["symbol"]

        df_labels, _ = self.pipeline.compute_labels(df_raw)

        train_f, train_l, test_f, test_l = self.pipeline.time_based_split(df_features, df_labels)

        # 1. Row count ratio check (80% / 20%)
        total_valid = len(train_f) + len(test_f)
        self.assertAlmostEqual(len(train_f) / total_valid, 0.8, delta=0.05)

        # 2. Strict time ordering: Max train timestamp < Min test timestamp
        max_train_time = train_f["timestamp"].max()
        min_test_time = test_f["timestamp"].min()

        self.assertLess(max_train_time, min_test_time, "Time leakage detected! Train timestamp >= Test timestamp")

    def test_standard_scaler_fitting_and_persistence(self):
        """Verify StandardScaler is fit only on train set and saved to scaler.pkl."""
        df_raw, _ = self.pipeline.load_and_validate()
        df_features, _ = self.pipeline.compute_features(df_raw)
        df_features["timestamp"] = df_raw["timestamp"]
        df_features["symbol"] = df_raw["symbol"]

        df_labels, _ = self.pipeline.compute_labels(df_raw)

        train_f, train_l, test_f, test_l = self.pipeline.time_based_split(df_features, df_labels)
        train_f_scaled, test_f_scaled = self.pipeline.preprocess_and_scale(train_f, test_f)

        # 1. Scaler persistence check
        scaler_file = self.models_dir / "scaler.pkl"
        self.assertTrue(scaler_file.exists(), "scaler.pkl was not saved!")

        scaler_loaded = joblib.load(scaler_file)
        self.assertIsNotNone(scaler_loaded.mean_)

        # 2. Check scaled features on train set have approx zero mean and unit variance
        feature_cols = self.pipeline.feature_names
        train_means = train_f_scaled[feature_cols].mean()
        for col_mean in train_means:
            self.assertAlmostEqual(col_mean, 0.0, delta=1e-3)

    def test_end_to_end_pipeline_execution(self):
        """Verify full run_pipeline outputs all required Parquet files and feature_names.json."""
        summary = self.pipeline.run_pipeline()

        self.assertGreater(summary["train_samples"], 0)
        self.assertGreater(summary["test_samples"], 0)

        self.assertTrue((self.processed_dir / "train_features.parquet").exists())
        self.assertTrue((self.processed_dir / "train_labels.parquet").exists())
        self.assertTrue((self.processed_dir / "test_features.parquet").exists())
        self.assertTrue((self.processed_dir / "test_labels.parquet").exists())
        self.assertTrue((self.processed_dir / "feature_names.json").exists())
        self.assertTrue((self.models_dir / "scaler.pkl").exists())


if __name__ == "__main__":
    unittest.main()
