from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import config
from src.logger import logger


class DataValidator:
    """Data quality validator for order book snapshot datasets stored in Parquet format."""

    def __init__(self, data_dir: Path = config.SNAPSHOTS_DIR):
        self.data_dir = Path(data_dir)

    def load_data(self) -> pd.DataFrame:
        """Loads all Parquet files from data_dir into a single DataFrame sorted by timestamp."""
        parquet_files = list(self.data_dir.glob("*.parquet"))
        if not parquet_files:
            # Fallback to checking parent DATA_DIR if snapshots dir is empty
            parquet_files = list(config.DATA_DIR.glob("**/*.parquet"))

        if not parquet_files:
            logger.warning(f"No Parquet files found in {self.data_dir}")
            return pd.DataFrame()

        logger.info(f"Loading {len(parquet_files)} Parquet file(s) from {self.data_dir}...")
        dfs = []
        for file_path in parquet_files:
            try:
                df_file = pd.read_parquet(file_path)
                dfs.append(df_file)
            except Exception as e:
                logger.error(f"Error reading Parquet file {file_path}: {e}")

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info(f"Successfully loaded {len(df)} total snapshot records.")
        return df

    def check_timestamp_gaps(self, df: pd.DataFrame, threshold_ms: float = 200.0) -> pd.DataFrame:
        """Finds snapshots with time gaps exceeding threshold_ms (default 200ms)."""
        if df.empty or len(df) < 2:
            return pd.DataFrame()

        time_diffs_ms = df["timestamp"].diff().dt.total_seconds() * 1000.0
        gaps_mask = time_diffs_ms > threshold_ms
        gaps_df = df[gaps_mask].copy()
        gaps_df["gap_ms"] = time_diffs_ms[gaps_mask]
        return gaps_df

    def check_spread_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Finds snapshots with crossed order books (best_bid >= best_ask) or negative spreads."""
        if df.empty:
            return pd.DataFrame()

        integrity_mask = (df["best_bid"] >= df["best_ask"]) | (df["spread"] <= 0)
        return df[integrity_mask].copy()

    def check_volume_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Finds snapshots with zero or invalid total volume on bid or ask side."""
        if df.empty:
            return pd.DataFrame()

        volume_mask = (df["total_bid_volume"] <= 0) | (df["total_ask_volume"] <= 0)
        return df[volume_mask].copy()

    def check_spread_outliers(self, df: pd.DataFrame, multiplier: float = 10.0) -> pd.DataFrame:
        """Finds spread anomalies exceeding multiplier times the median spread."""
        if df.empty:
            return pd.DataFrame()

        median_spread = df["spread"].median()
        if pd.isna(median_spread) or median_spread <= 0:
            median_spread = 0.01

        outlier_mask = df["spread"] > (multiplier * median_spread)
        outliers_df = df[outlier_mask].copy()
        outliers_df["median_spread"] = median_spread
        return outliers_df

    def generate_report(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """Generates comprehensive data quality statistics and computes overall data quality score."""
        if df is None:
            df = self.load_data()

        if df.empty:
            logger.warning("Empty dataset. Cannot generate validation report.")
            return {
                "total_snapshots": 0,
                "time_range": ("N/A", "N/A"),
                "gaps_count": 0,
                "integrity_violations_count": 0,
                "zero_volume_count": 0,
                "outliers_count": 0,
                "quality_score": 0.0,
            }

        total_snapshots = len(df)
        start_time = df["timestamp"].min()
        end_time = df["timestamp"].max()

        gaps_df = self.check_timestamp_gaps(df)
        integrity_df = self.check_spread_integrity(df)
        zero_vol_df = self.check_volume_integrity(df)
        outliers_df = self.check_spread_outliers(df)

        gaps_count = len(gaps_df)
        integrity_violations_count = len(integrity_df)
        zero_volume_count = len(zero_vol_df)
        outliers_count = len(outliers_df)

        # Quality score formula: 100% minus percentage of flawed snapshots
        flawed_snapshots = gaps_count + integrity_violations_count + zero_volume_count
        quality_score = max(0.0, min(100.0, (1.0 - (flawed_snapshots / total_snapshots)) * 100.0))

        report = {
            "total_snapshots": total_snapshots,
            "time_range": (str(start_time), str(end_time)),
            "gaps_count": gaps_count,
            "integrity_violations_count": integrity_violations_count,
            "zero_volume_count": zero_volume_count,
            "outliers_count": outliers_count,
            "quality_score": round(quality_score, 2),
            "median_spread": float(df["spread"].median()),
            "mean_spread": float(df["spread"].mean()),
        }

        self._print_report(report)
        return report

    def _print_report(self, report: Dict[str, Any]):
        """Prints formatted validation summary report to logger."""
        logger.info("=" * 65)
        logger.info("           ORDER BOOK DATA QUALITY VALIDATION REPORT          ")
        logger.info("=" * 65)
        logger.info(f"Total Snapshots Collected : {report['total_snapshots']:,}")
        logger.info(f"Time Range Start          : {report['time_range'][0]}")
        logger.info(f"Time Range End            : {report['time_range'][1]}")
        logger.info(f"Timestamp Gaps (>200ms)   : {report['gaps_count']}")
        logger.info(f"Integrity Violations      : {report['integrity_violations_count']} (Crossed/Negative Spreads)")
        logger.info(f"Zero Volume Snapshots     : {report['zero_volume_count']}")
        logger.info(f"Spread Outliers (>10x Med): {report['outliers_count']}")
        logger.info(f"Median Spread             : {report['median_spread']:.4f}")
        logger.info(f"DATA QUALITY SCORE        : {report['quality_score']:.2f}%")
        logger.info("=" * 65)

    def create_visualization(self, df: pd.DataFrame = None, output_path: Path = None) -> Path:
        """Plots spread and total volume over time, highlighting anomalies."""
        if df is None:
            df = self.load_data()

        if df.empty:
            logger.warning("Empty dataset. Skipping visualization generation.")
            return None

        output_path = Path(output_path or config.DATA_DIR / "data_quality_report.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        gaps_df = self.check_timestamp_gaps(df)
        integrity_df = self.check_spread_integrity(df)
        outliers_df = self.check_spread_outliers(df)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"Binance L2 Order Book Data Quality Report ({df['symbol'].iloc[0]})", fontsize=14, fontweight="bold")

        # Top Plot: Bid-Ask Spread Over Time
        ax1.plot(df["timestamp"], df["spread"], label="Bid-Ask Spread", color="#1f77b4", alpha=0.8, linewidth=1.2)
        ax1.axhline(df["spread"].median(), color="green", linestyle="--", alpha=0.7, label=f"Median Spread ({df['spread'].median():.2f})")

        # Highlight anomalies
        if not integrity_df.empty:
            ax1.scatter(integrity_df["timestamp"], integrity_df["spread"], color="red", s=40, zorder=5, label="Crossed/Negative Spread")
        if not outliers_df.empty:
            ax1.scatter(outliers_df["timestamp"], outliers_df["spread"], color="orange", s=30, zorder=4, label="Outlier Spread (>10x)")

        ax1.set_ylabel("Spread (USDT)")
        ax1.set_title("Bid-Ask Spread & Anomalies Over Time")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right")

        # Bottom Plot: Total Bid vs Ask Volume Over Time
        ax2.plot(df["timestamp"], df["total_bid_volume"], label="Total Bid Volume", color="#2ca02c", alpha=0.7, linewidth=1.2)
        ax2.plot(df["timestamp"], df["total_ask_volume"], label="Total Ask Volume", color="#d62728", alpha=0.7, linewidth=1.2)

        # Highlight timestamp gaps
        if not gaps_df.empty:
            for _, gap_row in gaps_df.iterrows():
                ax2.axvline(gap_row["timestamp"], color="magenta", linestyle=":", alpha=0.5, label="Timestamp Gap (>200ms)" if "Timestamp Gap (>200ms)" not in [l.get_label() for l in ax2.get_lines()] else "")

        ax2.set_xlabel("Timestamp (UTC)")
        ax2.set_ylabel("Volume (Base Asset)")
        ax2.set_title("Total Bid & Ask Depth Volume Over Time")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right")

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Data quality visualization saved to: {output_path}")
        return output_path
