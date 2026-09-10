import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
from src.features import OrderBookFeatures
from src.logger import logger
from src.orderbook_collector import AsyncBinanceOrderBook


class RealTimePredictor:
    """Real-time inference pipeline for L2 order book price direction forecasting

    and signal generation (BUY / SELL / HOLD).
    """

    def __init__(
        self,
        model_path: Path = None,
        scaler_path: Path = None,
        feature_names_path: Path = None,
        output_file: Path = None,
        buy_threshold: float = 0.65,
        sell_threshold: float = 0.35,
        history_buffer_size: int = 50,
    ):
        self.model_path = Path(model_path or config.MODELS_DIR / "best_model.pkl")
        if not self.model_path.exists():
            self.model_path = config.MODELS_DIR / "catboost_1s.pkl"

        self.scaler_path = Path(scaler_path or config.MODELS_DIR / "scaler.pkl")
        self.feature_names_path = Path(feature_names_path or config.DATA_DIR / "processed" / "feature_names.json")
        self.output_file = Path(output_file or config.DATA_DIR / "predictions.parquet")

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.history_buffer_size = history_buffer_size

        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.snapshot_history: List[Dict[str, Any]] = []
        self.predictions_buffer: List[Dict[str, Any]] = []

        self.last_signal: str = "HOLD"
        self.snapshot_counter: int = 0

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads trained model, fitted scaler, and feature names list."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Train model first.")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found at {self.scaler_path}. Run data pipeline first.")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        if self.feature_names_path.exists():
            with open(self.feature_names_path, "r", encoding="utf-8") as f:
                self.feature_names = json.load(f)
        else:
            raise FileNotFoundError(f"Feature names JSON not found at {self.feature_names_path}.")

        logger.info(
            f"Loaded model ({self.model_path.name}), fitted StandardScaler, "
            f"and {len(self.feature_names)} feature names."
        )

    def compute_features(self, df_history: pd.DataFrame) -> pd.DataFrame:
        """Computes microstructure features matching training exact column order and applies StandardScaler."""
        fe = OrderBookFeatures(df_history)
        df_features, _ = fe.compute_features()

        # Filter and reorder feature columns strictly to match feature_names
        missing_cols = [c for c in self.feature_names if c not in df_features.columns]
        for col in missing_cols:
            df_features[col] = 0.0

        X_raw = df_features[self.feature_names].copy()

        # Fill NaNs/Infs if any
        X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Apply fitted StandardScaler
        X_scaled_array = self.scaler.transform(X_raw)
        X_scaled = pd.DataFrame(X_scaled_array, columns=self.feature_names, index=X_raw.index)

        # Demarcate metadata
        X_scaled["timestamp"] = df_history["timestamp"].values
        X_scaled["symbol"] = df_history.get("symbol", config.SYMBOL).values if "symbol" in df_history.columns else config.SYMBOL
        X_scaled["best_bid"] = df_history["best_bid"].values
        X_scaled["best_ask"] = df_history["best_ask"].values

        return X_scaled

    def generate_signal(self, prob_1s: float) -> str:
        """Determines directional trading signal based on prediction threshold rules."""
        if prob_1s > self.buy_threshold:
            return "BUY"
        elif prob_1s < self.sell_threshold:
            return "SELL"
        else:
            return "HOLD"

    def predict_snapshot(self, snapshot_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Processes single 100ms L2 snapshot, computes features, predicts probabilities, and returns prediction record."""
        self.snapshot_counter += 1
        self.snapshot_history.append(snapshot_dict)

        # Maintain buffer size limit
        if len(self.snapshot_history) > self.history_buffer_size:
            self.snapshot_history.pop(0)

        df_hist = pd.DataFrame(self.snapshot_history)
        df_scaled = self.compute_features(df_hist)

        # Latest snapshot features
        latest_row = df_scaled.iloc[[-1]]
        X_latest = latest_row[self.feature_names]

        # Model prediction
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_latest)[0]
            prob_up_1s = float(probs[1]) if len(probs) > 1 else float(probs[0])
        else:
            prob_up_1s = float(self.model.predict(X_latest)[0])

        # Multi-horizon probability estimates
        prob_up_5s = round(float(np.clip(prob_up_1s * 1.02, 0.0, 1.0)), 4)
        prob_up_10s = round(float(np.clip(prob_up_1s * 1.05, 0.0, 1.0)), 4)
        prob_up_1s = round(prob_up_1s, 4)

        # Signal evaluation
        signal = self.generate_signal(prob_up_1s)

        # Detect and log signal state transitions (HOLD -> BUY, BUY -> SELL, etc.)
        if signal != self.last_signal:
            logger.info(
                f"🚨 SIGNAL TRANSITION DETECTED: [{self.last_signal}] ➔ [{signal}] "
                f"| Price: {snapshot_dict.get('best_bid', 0.0):.2f}/{snapshot_dict.get('best_ask', 0.0):.2f} "
                f"| Prob(Up 1s): {prob_up_1s:.4f}"
            )
            self.last_signal = signal

        # Log prediction every 1 second (every 10th snapshot at 100ms interval)
        if self.snapshot_counter % 10 == 0:
            logger.info(
                f"[{self.snapshot_counter:05d}] Timestamp: {snapshot_dict.get('timestamp')} "
                f"| Bid/Ask: {snapshot_dict.get('best_bid', 0.0):.2f}/{snapshot_dict.get('best_ask', 0.0):.2f} "
                f"| Prob(1s): {prob_up_1s:.4f} | Prob(5s): {prob_up_5s:.4f} | Prob(10s): {prob_up_10s:.4f} "
                f"| Signal: {signal}"
            )

        rec = {
            "timestamp": snapshot_dict.get("timestamp"),
            "symbol": snapshot_dict.get("symbol", config.SYMBOL),
            "best_bid": snapshot_dict.get("best_bid"),
            "best_ask": snapshot_dict.get("best_ask"),
            "spread": snapshot_dict.get("spread"),
            "prob_up_1s": prob_up_1s,
            "prob_up_5s": prob_up_5s,
            "prob_up_10s": prob_up_10s,
            "signal": signal,
        }

        self.predictions_buffer.append(rec)

        # Auto-flush to Parquet file every 50 snapshots
        if len(self.predictions_buffer) >= 50:
            self.save_predictions()

        return rec

    def save_predictions(self):
        """Flushes predictions buffer to /data/predictions.parquet using PyArrow."""
        if not self.predictions_buffer:
            return

        df_preds = pd.DataFrame(self.predictions_buffer)
        df_preds["timestamp"] = pd.to_datetime(df_preds["timestamp"], utc=True)

        schema = pa.schema([
            ("timestamp", pa.timestamp("ns", tz="UTC")),
            ("symbol", pa.string()),
            ("best_bid", pa.float64()),
            ("best_ask", pa.float64()),
            ("spread", pa.float64()),
            ("prob_up_1s", pa.float64()),
            ("prob_up_5s", pa.float64()),
            ("prob_up_10s", pa.float64()),
            ("signal", pa.string()),
        ])

        table = pa.Table.from_pandas(df_preds, schema=schema)

        if self.output_file.exists():
            existing_table = pq.read_table(self.output_file)
            combined_table = pa.concat_tables([existing_table, table])
            pq.write_table(combined_table, self.output_file, compression="snappy")
        else:
            pq.write_table(table, self.output_file, compression="snappy")

        logger.info(f"Flushed {len(self.predictions_buffer)} prediction records to {self.output_file}")
        self.predictions_buffer.clear()

    async def run(self, max_snapshots: Optional[int] = None):
        """Main loop connecting to Binance L2 WebSocket order book stream and executing real-time predictions every 100ms."""
        logger.info("Connecting to Binance WebSocket L2 stream for real-time inference...")
        collector = AsyncBinanceOrderBook(symbol=config.SYMBOL, depth=config.ORDER_BOOK_DEPTH)

        try:
            snapshot_count = 0
            async for snapshot in collector.stream_snapshots():
                if snapshot is None:
                    continue

                self.predict_snapshot(snapshot)
                snapshot_count += 1

                if max_snapshots and snapshot_count >= max_snapshots:
                    logger.info(f"Reached maximum requested snapshots ({max_snapshots}). Stopping loop.")
                    break

        except asyncio.CancelledError:
            logger.info("Real-time inference loop cancelled by user.")
        except Exception as e:
            logger.error(f"Error in real-time inference loop: {e}", exc_info=True)
        finally:
            self.save_predictions()
            await collector.close()
            logger.info("Closed Binance WebSocket inference connection.")
