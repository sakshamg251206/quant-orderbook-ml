import sys
from pathlib import Path
import pandas as pd

import config
from src.logger import logger
from src.data_validator import DataValidator
from src.features import OrderBookFeatures
from src.labels import LabelGenerator


def main():
    logger.info("Initializing Target Label Generation & Dataset Creation Pipeline...")

    # 1. Load Parquet snapshots
    validator = DataValidator(data_dir=config.SNAPSHOTS_DIR)
    df_raw = validator.load_data()

    if df_raw.empty:
        logger.error("No snapshot data found. Please run 'python3 run_collector.py' first.")
        sys.exit(1)

    # 2. Extract 34 microstructure features
    feature_calculator = OrderBookFeatures(df_raw)
    df_features, feature_names = feature_calculator.compute_features()

    df_features["timestamp"] = df_raw["timestamp"]
    df_features["symbol"] = df_raw["symbol"]

    # 3. Generate target labels (label_1s, label_5s, label_10s) with look-ahead bias prevention
    label_gen = LabelGenerator(horizons_sec=[1, 5, 10], max_tolerance_ms=500.0)
    df_labels, label_cols = label_gen.generate_labels(df_raw)

    if df_labels.empty:
        logger.error("Failed to generate target labels.")
        sys.exit(1)

    # 4. Merge features and labels on timestamp
    df_dataset = pd.merge(df_features, df_labels, on="timestamp", how="inner")

    # 5. Save complete ML dataset
    output_path = config.DATA_DIR / "dataset.parquet"
    df_dataset.to_parquet(output_path, index=False)

    logger.info("=" * 65)
    logger.info("              MACHINE LEARNING DATASET SUMMARY                ")
    logger.info("=" * 65)
    logger.info(f"Total ML Samples           : {len(df_dataset):,}")
    logger.info(f"Microstructure Features    : {len(feature_names)}")
    logger.info(f"Target Label Columns       : {', '.join(label_cols)}")
    logger.info(f"Missing Values (NaNs)      : {df_dataset.isna().sum().sum()}")
    logger.info(f"ML Dataset Output File     : {output_path} (Shape: {df_dataset.shape})")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
