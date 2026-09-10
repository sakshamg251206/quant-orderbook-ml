import sys
from pathlib import Path
import pandas as pd

import config
from src.logger import logger
from src.data_validator import DataValidator
from src.features import OrderBookFeatures


def main():
    logger.info("Initializing Order Book Feature Extraction Pipeline...")

    # 1. Load Parquet snapshots
    validator = DataValidator(data_dir=config.SNAPSHOTS_DIR)
    df = validator.load_data()

    if df.empty:
        logger.error("No snapshot data found. Run 'python3 run_collector.py' first.")
        sys.exit(1)

    logger.info(f"Loaded {len(df)} snapshots from {config.SNAPSHOTS_DIR}")

    # 2. Compute 30+ microstructure features
    feature_calculator = OrderBookFeatures(df)
    features_df, feature_names = feature_calculator.compute_features()

    logger.info("=" * 65)
    logger.info("           ORDER BOOK FEATURE ENGINEERING SUMMARY             ")
    logger.info("=" * 65)
    logger.info(f"Total Snapshots Processed : {len(features_df):,}")
    logger.info(f"Total Features Generated  : {len(feature_names)}")
    logger.info(f"Missing (NaN) Values      : {features_df.isna().sum().sum()}")
    logger.info("=" * 65)

    print("\nGenerated Microstructure Feature List (34 Features Total):")
    for idx, f_name in enumerate(feature_names, 1):
        print(f"  {idx:02d}. {f_name}")

    # 3. Save feature matrix to Parquet
    output_path = config.DATA_DIR / "features.parquet"
    features_df["timestamp"] = df["timestamp"]
    features_df["symbol"] = df["symbol"]
    features_df.to_parquet(output_path, index=False)

    logger.info(f"Feature matrix saved to: {output_path} (Shape: {features_df.shape})")


if __name__ == "__main__":
    main()
