from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from src.logger import logger


class LabelGenerator:
    """Generates price direction classification targets (1s, 5s, 10s horizons)

    for order book snapshot datasets while preventing look-ahead bias.
    """

    def __init__(self, horizons_sec: List[int] = [1, 5, 10], max_tolerance_ms: float = 500.0):
        self.horizons_sec = sorted(horizons_sec)
        self.max_horizon_sec = max(self.horizons_sec)
        self.max_tolerance_ms = max_tolerance_ms

    def generate_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Computes target direction labels for each specified horizon.

        Drops the trailing period (max horizon = 10s) to prevent look-ahead bias and incomplete labels.
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to LabelGenerator.")
            return pd.DataFrame(), []

        df_work = df.copy()
        if "timestamp" not in df_work.columns:
            raise KeyError("DataFrame must contain a 'timestamp' column.")

        df_work["timestamp"] = pd.to_datetime(df_work["timestamp"])
        df_work = df_work.sort_values("timestamp").reset_index(drop=True)

        # Calculate mid_price if not present
        if "mid_price" not in df_work.columns:
            if "best_bid" in df_work.columns and "best_ask" in df_work.columns:
                df_work["mid_price"] = (df_work["best_bid"] + df_work["best_ask"]) / 2.0
            else:
                raise KeyError("DataFrame must contain 'mid_price' or ('best_bid', 'best_ask').")

        # 1. Truncate last 10 seconds to prevent look-ahead bias at dataset boundary
        max_time = df_work["timestamp"].max()
        cutoff_time = max_time - pd.Timedelta(seconds=self.max_horizon_sec)
        df_valid = df_work[df_work["timestamp"] <= cutoff_time].copy()

        if df_valid.empty:
            logger.warning(f"Dataset duration is less than {self.max_horizon_sec}s cutoff. No valid labels generated.")
            return pd.DataFrame(), []

        label_cols = []
        df_labels = pd.DataFrame(index=df_valid.index)
        df_labels["timestamp"] = df_valid["timestamp"]

        # 2. Match future prices using merge_asof forward lookup
        df_lookup = df_work[["timestamp", "mid_price"]].copy()

        for horizon in self.horizons_sec:
            col_name = f"label_{horizon}s"
            label_cols.append(col_name)

            # Target target_time = timestamp + horizon_seconds
            df_valid[f"target_time_{horizon}s"] = df_valid["timestamp"] + pd.Timedelta(seconds=horizon)

            # Perform forward merge_asof
            merged = pd.merge_asof(
                df_valid[[f"target_time_{horizon}s"]].rename(columns={f"target_time_{horizon}s": "lookup_time"}),
                df_lookup.rename(columns={"timestamp": "actual_time", "mid_price": "future_mid_price"}),
                left_on="lookup_time",
                right_on="actual_time",
                direction="nearest",
                tolerance=pd.Timedelta(milliseconds=self.max_tolerance_ms),
            )

            # Calculate price return: (P_future - P_now) / P_now
            price_now = df_valid["mid_price"].values
            price_future = merged["future_mid_price"].values

            price_return = (price_future - price_now) / price_now
            
            # Binary classification target: 1 if return > 0, 0 if return <= 0
            binary_labels = np.where(price_return > 0, 1, 0)

            # Invalidate labels where timestamp match exceeds tolerance
            valid_mask = ~merged["future_mid_price"].isna()
            df_labels[col_name] = np.where(valid_mask, binary_labels, np.nan)

        # Drop any remaining un-labeled rows if any gap occurred
        df_labels = df_labels.dropna().reset_index(drop=True)
        for col in label_cols:
            df_labels[col] = df_labels[col].astype(int)

        logger.info(
            f"Generated target labels ({', '.join(label_cols)}) for {len(df_labels)} snapshots "
            f"(Dropped trailing {self.max_horizon_sec}s to prevent look-ahead bias)."
        )

        self.check_class_balance(df_labels, label_cols)
        return df_labels, label_cols

    def check_class_balance(self, df_labels: pd.DataFrame, label_cols: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculates positive/negative class ratios and issues warnings if imbalance exceeds 60/40 threshold."""
        stats = {}
        logger.info("=" * 65)
        logger.info("           ORDER BOOK TARGET LABELS CLASS BALANCE             ")
        logger.info("=" * 65)

        for col in label_cols:
            if col not in df_labels.columns or df_labels.empty:
                continue

            total = len(df_labels)
            pos_count = (df_labels[col] == 1).sum()
            neg_count = (df_labels[col] == 0).sum()

            pos_pct = (pos_count / total) * 100.0 if total > 0 else 0.0
            neg_pct = (neg_count / total) * 100.0 if total > 0 else 0.0

            stats[col] = {
                "positive_pct": round(pos_pct, 2),
                "negative_pct": round(neg_pct, 2),
                "pos_count": int(pos_count),
                "neg_count": int(neg_count),
            }

            logger.info(
                f"Horizon [{col:>9s}] -> Positive (1): {pos_pct:6.2f}% ({pos_count:,}) | "
                f"Negative (0): {neg_pct:6.2f}% ({neg_count:,})"
            )

            # Check if class balance exceeds 60/40 or 40/60 threshold
            if pos_pct > 60.0 or pos_pct < 40.0:
                logger.warning(
                    f"CLASS IMBALANCE DETECTED on {col}: Positive class ratio is {pos_pct:.2f}% "
                    f"(Outside balanced 40%-60% threshold)."
                )

        logger.info("=" * 65)
        return stats
