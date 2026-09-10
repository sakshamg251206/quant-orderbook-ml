import unittest
from pathlib import Path
import pandas as pd
import numpy as np

import config
from app import load_predictions_data, load_orderbook_features


class TestStreamlitApp(unittest.TestCase):
    def setUp(self):
        self.predictions_file = config.DATA_DIR / "predictions.parquet"

    def test_load_predictions_data(self):
        """Verifies loading of prediction dataset for Streamlit app."""
        df = load_predictions_data(self.predictions_file)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "Predictions DataFrame should not be empty.")
        self.assertIn("prob_up_1s", df.columns)
        self.assertIn("best_bid", df.columns)
        self.assertIn("best_ask", df.columns)

    def test_load_orderbook_features(self):
        """Verifies loading of microstructure feature dataset for OBI charts."""
        df_feat = load_orderbook_features()
        self.assertIsInstance(df_feat, pd.DataFrame)
        self.assertFalse(df_feat.empty, "Feature DataFrame should not be empty.")
        self.assertIn("obi_level_5", df_feat.columns)


if __name__ == "__main__":
    unittest.main()
