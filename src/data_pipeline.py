import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from src.logger import logger
from src.data_validator import DataValidator
from src.features import OrderBookFeatures
from src.labels import LabelGenerator


class DataPipeline:
    """End-to-end data pipeline combining validation, feature engineering, label generation,

    strict time-based train/test splitting, and feature standardization.
    """

    def __init__(
        self,
        data_dir: Path = config.SNAPSHOTS_DIR,
        processed_dir: Path = None,
        models_dir: Path = config.MODELS_DIR,
        train_ratio: float = 0.8,
    ):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir or config.DATA_DIR / "processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.train_ratio = train_ratio
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []

    def load_and_validate(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Loads raw Parquet snapshots and runs data quality validation."""
        validator = DataValidator(data_dir=self.data_dir)
        df_raw = validator.load_data()

        if df_raw.empty:
            logger.error(f"No snapshot data found in {self.data_dir}.")
            return pd.DataFrame(), {}

        report = validator.generate_report(df_raw)
        if report.get("quality_score", 0) < 80.0:
            logger.warning(f"Data quality score is low ({report.get('quality_score')}%). Inspect validation report.")

        return df_raw, report

    def compute_features(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Computes 30+ microstructure features."""
        feature_calculator = OrderBookFeatures(df_raw)
        df_features, feature_names = feature_calculator.compute_features()
        self.feature_names = feature_names
        return df_features, feature_names

    def compute_labels(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Generates direction labels (1s, 5s, 10s) with look-ahead bias prevention."""
        label_generator = LabelGenerator(horizons_sec=[1, 5, 10], max_tolerance_ms=500.0)
        df_labels, label_cols = label_generator.generate_labels(df_raw)
        return df_labels, label_cols

    def time_based_split(
        self, df_features: pd.DataFrame, df_labels: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs a strict time-based train/test split (e.g. first 80% train, last 20% test).

        Preserves chronological ordering with NO shuffling to prevent temporal data leakage.
        """
        # Inner join on timestamp to guarantee index alignment
        merged = pd.merge(df_features, df_labels, on="timestamp", how="inner")
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        label_cols = [col for col in df_labels.columns if col != "timestamp"]
        feature_cols = [col for col in self.feature_names if col in merged.columns]

        n_samples = len(merged)
        split_idx = int(n_samples * self.train_ratio)

        train_df = merged.iloc[:split_idx].copy()
        test_df = merged.iloc[split_idx:].copy()

        logger.info(
            f"Time-based split completed (Train: {self.train_ratio*100:.0f}%, Test: {(1-self.train_ratio)*100:.0f}%): "
            f"Train samples = {len(train_df):,}, Test samples = {len(test_df):,}"
        )

        train_features = train_df[feature_cols + ["timestamp", "symbol"]]
        train_labels = train_df[label_cols + ["timestamp"]]

        test_features = test_df[feature_cols + ["timestamp", "symbol"]]
        test_labels = test_df[label_cols + ["timestamp"]]

        return train_features, train_labels, test_features, test_labels

    def preprocess_and_scale(
        self, train_features: pd.DataFrame, test_features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Standardizes feature columns (zero mean, unit variance).

        Fits StandardScaler on train set ONLY and applies the same transformation to test set.
        Saves the fitted scaler to /models/scaler.pkl.
        """
        feature_cols = [col for col in self.feature_names if col in train_features.columns]

        self.scaler = StandardScaler()

        # Fit on train features ONLY
        train_scaled_array = self.scaler.fit_transform(train_features[feature_cols])

        # Apply same fitted scaler to test features
        test_scaled_array = self.scaler.transform(test_features[feature_cols])

        # Construct output DataFrames with metadata preserved
        train_features_scaled = train_features.copy()
        train_features_scaled[feature_cols] = train_scaled_array

        test_features_scaled = test_features.copy()
        test_features_scaled[feature_cols] = test_scaled_array

        # Save scaler artifact
        scaler_path = self.models_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Fitted StandardScaler saved to {scaler_path}")

        return train_features_scaled, test_features_scaled

    def save_datasets(
        self,
        train_features: pd.DataFrame,
        train_labels: pd.DataFrame,
        test_features: pd.DataFrame,
        test_labels: pd.DataFrame,
    ) -> Dict[str, Path]:
        """Saves processed train and test datasets and feature_names.json to /data/processed/."""
        file_map = {
            "train_features": self.processed_dir / "train_features.parquet",
            "train_labels": self.processed_dir / "train_labels.parquet",
            "test_features": self.processed_dir / "test_features.parquet",
            "test_labels": self.processed_dir / "test_labels.parquet",
            "feature_names": self.processed_dir / "feature_names.json",
        }

        train_features.to_parquet(file_map["train_features"], index=False)
        train_labels.to_parquet(file_map["train_labels"], index=False)
        test_features.to_parquet(file_map["test_features"], index=False)
        test_labels.to_parquet(file_map["test_labels"], index=False)

        with open(file_map["feature_names"], "w", encoding="utf-8") as f:
            json.dump(self.feature_names, f, indent=2)

        logger.info(f"Successfully saved all datasets to {self.processed_dir}")
        return file_map

    def run_pipeline(self) -> Dict[str, Any]:
        """Orchestrates end-to-end execution of data loading, feature engineering, label generation,

        time-based train/test splitting, normalization, and persistence.
        """
        logger.info("=" * 65)
        logger.info("          EXECUTING END-TO-END ORDER BOOK DATA PIPELINE       ")
        logger.info("=" * 65)

        # 1. Load and validate
        df_raw, report = self.load_and_validate()
        if df_raw.empty:
            raise RuntimeError("Pipeline failed: empty raw dataset.")

        # 2. Compute features & labels
        df_features, feature_names = self.compute_features(df_raw)
        df_features["timestamp"] = df_raw["timestamp"]
        df_features["symbol"] = df_raw["symbol"]

        df_labels, label_cols = self.compute_labels(df_raw)

        # 3. Time-based split (80% train / 20% test)
        train_f, train_l, test_f, test_l = self.time_based_split(df_features, df_labels)

        # 4. Fit scaler on train and scale features
        train_f_scaled, test_f_scaled = self.preprocess_and_scale(train_f, test_f)

        # 5. Save processed datasets
        exported_files = self.save_datasets(train_f_scaled, train_l, test_f_scaled, test_l)

        summary = {
            "total_raw_snapshots": len(df_raw),
            "quality_score": report.get("quality_score", 100.0),
            "train_samples": len(train_f_scaled),
            "test_samples": len(test_f_scaled),
            "num_features": len(feature_names),
            "label_columns": label_cols,
            "exported_files": {k: str(v) for k, v in exported_files.items()},
        }

        logger.info("Pipeline execution finished successfully.")
        return summary
