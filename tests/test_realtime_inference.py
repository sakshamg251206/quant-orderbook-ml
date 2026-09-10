import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data_pipeline import DataPipeline
from src.model_training import CatBoostTrainer
from src.realtime_inference import RealTimePredictor


class TestRealTimePredictor(unittest.TestCase):
    """Unit test suite for RealTimePredictor real-time inference pipeline."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name)

        self.snapshots_dir = self.base_path / "snapshots"
        self.processed_dir = self.base_path / "processed"
        self.models_dir = self.base_path / "models"
        self.plots_dir = self.base_path / "plots"
        self.output_file = self.base_path / "predictions.parquet"

        self.snapshots_dir.mkdir()
        self.processed_dir.mkdir()
        self.models_dir.mkdir()
        self.plots_dir.mkdir()

        # Generate synthetic order book snapshot batch (200 snapshots = 20s > 10s label cutoff)
        timestamps = pd.date_range(start="2026-08-31 10:00:00", periods=200, freq="100ms", tz="UTC")
        self.sample_snapshots = []
        for idx, ts in enumerate(timestamps):
            price_offset = (idx % 4) * 0.05
            bids = [[100.0 + price_offset - i * 0.1, 1.0 + i * 0.1] for i in range(20)]
            asks = [[100.1 + price_offset + i * 0.1, 1.0 + i * 0.1] for i in range(20)]
            snap = {
                "timestamp": ts,
                "symbol": "BTCUSDT",
                "bids_json": json.dumps(bids),
                "asks_json": json.dumps(asks),
                "best_bid": 100.0 + price_offset,
                "best_ask": 100.1 + price_offset,
                "spread": 0.1,
                "total_bid_volume": sum(b[1] for b in bids),
                "total_ask_volume": sum(a[1] for a in asks),
            }
            self.sample_snapshots.append(snap)

        pd.DataFrame(self.sample_snapshots).to_parquet(self.snapshots_dir / "snap_1.parquet")

        # Run pipeline to create train features, labels, scaler.pkl, feature_names.json
        pipeline = DataPipeline(
            data_dir=self.snapshots_dir,
            processed_dir=self.processed_dir,
            models_dir=self.models_dir,
        )
        pipeline.run_pipeline()
        X_train = pd.read_parquet(self.processed_dir / "train_features.parquet")
        y_train_df = pd.read_parquet(self.processed_dir / "train_labels.parquet")
        X_test = pd.read_parquet(self.processed_dir / "test_features.parquet")
        y_test_df = pd.read_parquet(self.processed_dir / "test_labels.parquet")
        trainer = CatBoostTrainer(
            processed_dir=self.processed_dir,
            models_dir=self.models_dir,
            plots_dir=self.plots_dir,
            target_col="label_1s",
        )
        trainer.train_catboost(X_train, y_train_df["label_1s"], X_test, y_test_df["label_1s"], iterations=10, verbose=0)
        metrics = trainer.evaluate(X_test, y_test_df["label_1s"])
        trainer.save_model(metrics)

        # Save copy as best_model.pkl
        joblib.dump(trainer.model, self.models_dir / "best_model.pkl")

        self.predictor = RealTimePredictor(
            model_path=self.models_dir / "best_model.pkl",
            scaler_path=self.models_dir / "scaler.pkl",
            feature_names_path=self.processed_dir / "feature_names.json",
            output_file=self.output_file,
            buy_threshold=0.65,
            sell_threshold=0.35,
        )

    def tearDown(self):
        self.test_dir.cleanup()

    def test_signal_generation(self):
        """Verify signal thresholds: BUY if prob > 0.65, SELL if prob < 0.35, HOLD otherwise."""
        self.assertEqual(self.predictor.generate_signal(0.70), "BUY")
        self.assertEqual(self.predictor.generate_signal(0.20), "SELL")
        self.assertEqual(self.predictor.generate_signal(0.50), "HOLD")

    def test_predict_snapshot_and_parquet_export(self):
        """Verify real-time prediction on 100ms snapshots and predictions.parquet saving."""
        for snap in self.sample_snapshots:
            rec = self.predictor.predict_snapshot(snap)
            self.assertIn("prob_up_1s", rec)
            self.assertIn("signal", rec)
            self.assertIn(rec["signal"], ["BUY", "SELL", "HOLD"])

        self.predictor.save_predictions()
        self.assertTrue(self.output_file.exists())

        df_saved = pq.read_table(self.output_file).to_pandas()
        self.assertEqual(len(df_saved), len(self.sample_snapshots))
        self.assertIn("prob_up_1s", df_saved.columns)
        self.assertIn("signal", df_saved.columns)


if __name__ == "__main__":
    unittest.main()
