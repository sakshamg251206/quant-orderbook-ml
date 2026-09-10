import json
import unittest
import numpy as np
import pandas as pd

from src.features import OrderBookFeatures


class TestOrderBookFeatures(unittest.TestCase):
    """Unit test suite for each feature calculation in OrderBookFeatures."""

    def setUp(self):
        # Synthetic L2 snapshot dataset for testing
        self.sample_bids = [[100.0 - i * 0.1, 10.0 + i] for i in range(20)]
        self.sample_asks = [[100.1 + i * 0.1, 5.0 + i] for i in range(20)]

        self.df_sample = pd.DataFrame([
            {
                "timestamp": pd.Timestamp("2026-08-31 10:00:00", tz="UTC"),
                "symbol": "BTCUSDT",
                "bids_json": json.dumps(self.sample_bids),
                "asks_json": json.dumps(self.sample_asks),
                "best_bid": 100.0,
                "best_ask": 100.1,
                "spread": 0.1,
                "total_bid_volume": sum(q for _, q in self.sample_bids),
                "total_ask_volume": sum(q for _, q in self.sample_asks),
            }
            for _ in range(10)
        ])

        self.calculator = OrderBookFeatures(self.df_sample)
        self.features_df, self.feature_names = self.calculator.compute_features()

    def test_feature_count_and_shape(self):
        """Verify that at least 30 features are generated with matching row count."""
        self.assertGreaterEqual(len(self.feature_names), 30)
        self.assertEqual(len(self.features_df), len(self.df_sample))
        self.assertFalse(self.features_df.isna().any().any(), "Output feature matrix contains NaN values")

    def test_obi_level_calculations(self):
        """Verify Order Book Imbalance across levels 1, 5, 10, 20."""
        # Level 1 OBI: bid=10, ask=5 -> (10-5)/(10+5) = 5/15 = 0.33333...
        expected_obi_1 = (10.0 - 5.0) / (10.0 + 5.0)
        self.assertAlmostEqual(self.features_df["obi_level_1"].iloc[0], expected_obi_1, places=4)

        # Level 5 OBI
        bid_vol_5 = sum(10.0 + i for i in range(5))
        ask_vol_5 = sum(5.0 + i for i in range(5))
        expected_obi_5 = (bid_vol_5 - ask_vol_5) / (bid_vol_5 + ask_vol_5)
        self.assertAlmostEqual(self.features_df["obi_level_5"].iloc[0], expected_obi_5, places=4)

        # Verify OBI bounds [-1.0, 1.0]
        for col in ["obi_level_1", "obi_level_5", "obi_level_10", "obi_level_20"]:
            self.assertTrue((self.features_df[col] >= -1.0).all() and (self.features_df[col] <= 1.0).all())

    def test_spread_metrics(self):
        """Verify spread_absolute, spread_relative, spread_log, and spread_ratio."""
        expected_spread_abs = 100.1 - 100.0
        expected_spread_rel = expected_spread_abs / 100.0
        expected_spread_log = np.log(expected_spread_abs)

        self.assertAlmostEqual(self.features_df["spread_absolute"].iloc[0], expected_spread_abs, places=4)
        self.assertAlmostEqual(self.features_df["spread_relative"].iloc[0], expected_spread_rel, places=4)
        self.assertAlmostEqual(self.features_df["spread_log"].iloc[0], expected_spread_log, places=4)
        self.assertAlmostEqual(self.features_df["bid_ask_spread_ratio"].iloc[0], 100.1 / 100.0, places=4)

    def test_micro_price_calculations(self):
        """Verify mid_price, micro_price, and micro_price_deviation."""
        mid_price = (100.0 + 100.1) / 2.0  # 100.05
        # Micro price = (100.0 * 5.0 + 100.1 * 10.0) / (10.0 + 5.0) = (500 + 1001) / 15 = 1501 / 15 = 100.0666...
        expected_micro = (100.0 * 5.0 + 100.1 * 10.0) / 15.0
        expected_dev = (expected_micro - mid_price) / mid_price

        self.assertAlmostEqual(self.features_df["mid_price"].iloc[0], mid_price, places=4)
        self.assertAlmostEqual(self.features_df["micro_price"].iloc[0], expected_micro, places=4)
        self.assertAlmostEqual(self.features_df["micro_price_deviation"].iloc[0], expected_dev, places=4)

    def test_depth_and_volume_ratios(self):
        """Verify total volume and volume ratios."""
        expected_bid_vol = sum(10.0 + i for i in range(20))
        expected_ask_vol = sum(5.0 + i for i in range(20))
        expected_vol_ratio = expected_bid_vol / expected_ask_vol

        self.assertAlmostEqual(self.features_df["total_bid_volume"].iloc[0], expected_bid_vol, places=4)
        self.assertAlmostEqual(self.features_df["total_ask_volume"].iloc[0], expected_ask_vol, places=4)
        self.assertAlmostEqual(self.features_df["bid_ask_volume_ratio"].iloc[0], expected_vol_ratio, places=4)

    def test_volume_distribution_statistics(self):
        """Verify volume standard deviation, skewness, and decay."""
        bid_vols = [10.0 + i for i in range(20)]
        expected_std = float(np.std(bid_vols))

        self.assertAlmostEqual(self.features_df["bid_volume_std"].iloc[0], expected_std, places=4)
        self.assertIn("bid_volume_skew", self.features_df.columns)
        self.assertIn("decayed_obi", self.features_df.columns)

    def test_momentum_features(self):
        """Verify momentum features are computed without NaN outputs."""
        self.assertIn("obi_momentum_100ms", self.features_df.columns)
        self.assertIn("obi_momentum_500ms", self.features_df.columns)
        self.assertIn("spread_momentum_100ms", self.features_df.columns)
        self.assertIn("mid_price_return_100ms", self.features_df.columns)
        self.assertIn("mid_price_return_500ms", self.features_df.columns)


if __name__ == "__main__":
    unittest.main()
