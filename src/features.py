import json
from typing import List, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
from scipy.stats import skew

from src.logger import logger


class OrderBookFeatures:
    """Computes 30+ microstructure features from L2 order book snapshot DataFrames."""

    def __init__(self, df: pd.DataFrame = None):
        self.df = df.copy() if df is not None else None
        self.feature_names: List[str] = []

    @staticmethod
    def parse_book_json(json_str: Union[str, List[List[float]], np.ndarray]) -> List[Tuple[float, float]]:
        """Safely parses JSON string or list representation of price-quantity pairs."""
        if isinstance(json_str, (list, np.ndarray)):
            return [(float(p), float(q)) for p, q in json_str]
        if isinstance(json_str, str):
            try:
                data = json.loads(json_str)
                return [(float(p), float(q)) for p, q in data]
            except Exception:
                return []
        return []

    def compute_features(self, df: pd.DataFrame = None) -> Tuple[pd.DataFrame, List[str]]:
        """Transforms order book snapshot DataFrame into a feature matrix of 30+ microstructure indicators."""
        if df is None:
            df = self.df

        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to OrderBookFeatures.")
            return pd.DataFrame(), []

        df_out = pd.DataFrame(index=df.index)

        # 1. Parse depth arrays across all rows
        num_rows = len(df)
        bids_matrix_price = np.zeros((num_rows, 20))
        bids_matrix_vol = np.zeros((num_rows, 20))
        asks_matrix_price = np.zeros((num_rows, 20))
        asks_matrix_vol = np.zeros((num_rows, 20))

        for idx in range(num_rows):
            row = df.iloc[idx]
            bids = self.parse_book_json(row.get("bids_json", []))
            asks = self.parse_book_json(row.get("asks_json", []))

            for level, (p, q) in enumerate(bids[:20]):
                bids_matrix_price[idx, level] = p
                bids_matrix_vol[idx, level] = q

            for level, (p, q) in enumerate(asks[:20]):
                asks_matrix_price[idx, level] = p
                asks_matrix_vol[idx, level] = q

        # Extract top level prices and volumes
        best_bid = bids_matrix_price[:, 0]
        best_ask = asks_matrix_price[:, 0]
        bid_vol_1 = bids_matrix_vol[:, 0]
        ask_vol_1 = asks_matrix_vol[:, 0]

        # Handle missing best bid/ask from raw column if matrix was empty
        if np.all(best_bid == 0) and "best_bid" in df.columns:
            best_bid = df["best_bid"].to_numpy()
        if np.all(best_ask == 0) and "best_ask" in df.columns:
            best_ask = df["best_ask"].to_numpy()

        # Cumulative volumes across depth levels
        bid_vol_5 = np.sum(bids_matrix_vol[:, :5], axis=1)
        ask_vol_5 = np.sum(asks_matrix_vol[:, :5], axis=1)

        bid_vol_10 = np.sum(bids_matrix_vol[:, :10], axis=1)
        ask_vol_10 = np.sum(asks_matrix_vol[:, :10], axis=1)

        bid_vol_20 = np.sum(bids_matrix_vol[:, :20], axis=1)
        ask_vol_20 = np.sum(asks_matrix_vol[:, :20], axis=1)

        total_bid_vol = df["total_bid_volume"].to_numpy() if "total_bid_volume" in df.columns else bid_vol_20
        total_ask_vol = df["total_ask_volume"].to_numpy() if "total_ask_volume" in df.columns else ask_vol_20

        # Mid-price and Micro-price
        mid_price = (best_bid + best_ask) / 2.0
        denom_1 = bid_vol_1 + ask_vol_1
        micro_price = np.where(denom_1 > 0, (best_bid * ask_vol_1 + best_ask * bid_vol_1) / denom_1, mid_price)

        # -------------------------------------------------------------
        # Feature Group 1: Order Book Imbalance (OBI - Multi-level)
        # -------------------------------------------------------------
        df_out["obi_level_1"] = np.where(denom_1 > 0, (bid_vol_1 - ask_vol_1) / denom_1, 0.0)
        
        denom_5 = bid_vol_5 + ask_vol_5
        df_out["obi_level_5"] = np.where(denom_5 > 0, (bid_vol_5 - ask_vol_5) / denom_5, 0.0)

        denom_10 = bid_vol_10 + ask_vol_10
        df_out["obi_level_10"] = np.where(denom_10 > 0, (bid_vol_10 - ask_vol_10) / denom_10, 0.0)

        denom_20 = bid_vol_20 + ask_vol_20
        df_out["obi_level_20"] = np.where(denom_20 > 0, (bid_vol_20 - ask_vol_20) / denom_20, 0.0)

        # -------------------------------------------------------------
        # Feature Group 2: Spread Metrics
        # -------------------------------------------------------------
        spread_abs = best_ask - best_bid
        df_out["spread_absolute"] = spread_abs
        df_out["spread_relative"] = np.where(best_bid > 0, spread_abs / best_bid, 0.0)
        df_out["spread_log"] = np.log(np.maximum(spread_abs, 1e-8))
        df_out["bid_ask_spread_ratio"] = np.where(best_bid > 0, best_ask / best_bid, 1.0)
        df_out["effective_spread"] = 2.0 * np.abs(micro_price - mid_price)

        # -------------------------------------------------------------
        # Feature Group 3: Depth & Volume Features
        # -------------------------------------------------------------
        df_out["total_bid_volume"] = total_bid_vol
        df_out["total_ask_volume"] = total_ask_vol
        df_out["bid_ask_volume_ratio"] = np.where(total_ask_vol > 0, total_bid_vol / total_ask_vol, 1.0)
        df_out["depth_ratio_level_5"] = np.where(ask_vol_5 > 0, bid_vol_5 / ask_vol_5, 1.0)
        df_out["depth_ratio_level_10"] = np.where(ask_vol_10 > 0, bid_vol_10 / ask_vol_10, 1.0)
        df_out["bid_vol_top1_share"] = np.where(total_bid_vol > 0, bid_vol_1 / total_bid_vol, 0.0)
        df_out["ask_vol_top1_share"] = np.where(total_ask_vol > 0, ask_vol_1 / total_ask_vol, 0.0)

        # -------------------------------------------------------------
        # Feature Group 4: Micro-Price Metrics
        # -------------------------------------------------------------
        df_out["mid_price"] = mid_price
        df_out["micro_price"] = micro_price
        df_out["micro_price_deviation"] = np.where(mid_price > 0, (micro_price - mid_price) / mid_price, 0.0)

        # Weighted mid price top 5 & top 10
        top5_price_vol_sum = np.sum(bids_matrix_price[:, :5] * asks_matrix_vol[:, :5] + asks_matrix_price[:, :5] * bids_matrix_vol[:, :5], axis=1)
        df_out["weighted_mid_price_5"] = np.where(denom_5 > 0, top5_price_vol_sum / denom_5, mid_price)

        top10_price_vol_sum = np.sum(bids_matrix_price[:, :10] * asks_matrix_vol[:, :10] + asks_matrix_price[:, :10] * bids_matrix_vol[:, :10], axis=1)
        df_out["weighted_mid_price_10"] = np.where(denom_10 > 0, top10_price_vol_sum / denom_10, mid_price)

        # -------------------------------------------------------------
        # Feature Group 5: Volume Distribution & Shape Statistics
        # -------------------------------------------------------------
        df_out["bid_volume_std"] = np.std(bids_matrix_vol, axis=1)
        df_out["ask_volume_std"] = np.std(asks_matrix_vol, axis=1)

        # Skewness across 20 depth levels
        bid_skews = [float(skew(bids_matrix_vol[i])) if np.std(bids_matrix_vol[i]) > 0 else 0.0 for i in range(num_rows)]
        ask_skews = [float(skew(asks_matrix_vol[i])) if np.std(asks_matrix_vol[i]) > 0 else 0.0 for i in range(num_rows)]
        df_out["bid_volume_skew"] = bid_skews
        df_out["ask_volume_skew"] = ask_skews

        # Exponential volume decay across levels
        decay_weights = np.exp(-0.1 * np.arange(20))
        bid_decay = np.dot(bids_matrix_vol, decay_weights)
        ask_decay = np.dot(asks_matrix_vol, decay_weights)
        df_out["bid_volume_decay"] = bid_decay
        df_out["ask_volume_decay"] = ask_decay
        df_out["decayed_obi"] = np.where((bid_decay + ask_decay) > 0, (bid_decay - ask_decay) / (bid_decay + ask_decay), 0.0)

        # -------------------------------------------------------------
        # Feature Group 6: Momentum & Time Dynamics (100ms & 500ms diffs)
        # -------------------------------------------------------------
        df_out["obi_momentum_100ms"] = df_out["obi_level_1"].diff(1)
        df_out["obi_momentum_500ms"] = df_out["obi_level_1"].diff(5)
        df_out["spread_momentum_100ms"] = df_out["spread_absolute"].diff(1)
        
        mid_price_s = df_out["mid_price"]
        df_out["mid_price_return_100ms"] = mid_price_s.pct_change(1)
        df_out["mid_price_return_500ms"] = mid_price_s.pct_change(5)
        df_out["mid_price_volatility_5"] = df_out["mid_price_return_100ms"].rolling(5).std()

        # -------------------------------------------------------------
        # Clean NaNs and Infinities (Requirement 3: Handle NaNs)
        # -------------------------------------------------------------
        df_out = df_out.ffill().bfill().fillna(0.0)
        df_out = df_out.replace([np.inf, -np.inf], 0.0)

        self.feature_names = list(df_out.columns)
        logger.info(f"Generated {len(self.feature_names)} microstructure features for {num_rows} snapshots.")

        return df_out, self.feature_names
