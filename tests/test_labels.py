import unittest
import pandas as pd
import numpy as np

from src.labels import LabelGenerator


class TestLabelGenerator(unittest.TestCase):
    """Unit test suite for LabelGenerator class and look-ahead bias prevention."""

    def setUp(self):
        # Create 20 seconds of 100ms synthetic order book data (200 snapshots)
        times = pd.date_range(start="2026-08-31 12:00:00", periods=200, freq="100ms", tz="UTC")
        
        # Synthetic upward price trend: mid_price increases steadily
        prices = [100.0 + i * 0.05 for i in range(200)]

        self.df_uptrend = pd.DataFrame({
            "timestamp": times,
            "best_bid": [p - 0.01 for p in prices],
            "best_ask": [p + 0.01 for p in prices],
            "mid_price": prices,
        })

        # Synthetic flat price trend
        self.df_flat = pd.DataFrame({
            "timestamp": times,
            "best_bid": [100.0] * 200,
            "best_ask": [100.1] * 200,
            "mid_price": [100.05] * 200,
        })

    def test_label_generation_and_lookahead_truncation(self):
        """Verify target creation and last 10s cutoff to prevent look-ahead bias."""
        generator = LabelGenerator(horizons_sec=[1, 5, 10], max_tolerance_ms=500.0)
        df_labels, label_cols = generator.generate_labels(self.df_uptrend)

        # 1. Target columns check
        self.assertEqual(sorted(label_cols), ["label_10s", "label_1s", "label_5s"])

        # 2. Look-ahead truncation check: Total duration is 20s. Last 10s (100 snapshots) must be dropped
        # Expected remaining snapshots: ~100 snapshots (for timestamps <= 12:00:10.000)
        max_valid_time = self.df_uptrend["timestamp"].max() - pd.Timedelta(seconds=10)
        self.assertTrue((df_labels["timestamp"] <= max_valid_time).all(), "Found snapshots past trailing 10s cutoff!")
        self.assertLessEqual(len(df_labels), 101)

        # 3. Uptrend should produce all 1s
        for col in label_cols:
            self.assertTrue((df_labels[col] == 1).all(), f"Expected all 1s for uptrend in {col}")

    def test_flat_trend_labels(self):
        """Verify flat price trend produces all 0s (since return <= 0)."""
        generator = LabelGenerator(horizons_sec=[1, 5, 10])
        df_labels, label_cols = generator.generate_labels(self.df_flat)

        for col in label_cols:
            self.assertTrue((df_labels[col] == 0).all(), f"Expected all 0s for flat trend in {col}")

    def test_class_balance_check(self):
        """Verify class balance calculation and imbalance warning threshold."""
        generator = LabelGenerator(horizons_sec=[1, 5, 10])
        
        # Create dataset with 70% 1s and 30% 0s
        mock_labels = pd.DataFrame({
            "label_1s": [1] * 70 + [0] * 30,
            "label_5s": [1] * 50 + [0] * 50,
            "label_10s": [0] * 70 + [1] * 30,
        })

        stats = generator.check_class_balance(mock_labels, ["label_1s", "label_5s", "label_10s"])

        self.assertEqual(stats["label_1s"]["positive_pct"], 70.0)
        self.assertEqual(stats["label_5s"]["positive_pct"], 50.0)
        self.assertEqual(stats["label_10s"]["positive_pct"], 30.0)


if __name__ == "__main__":
    unittest.main()
