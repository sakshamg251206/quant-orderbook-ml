"""Real-time scoring of order book snapshots.

Snapshots come either from the live Binance stream or from a replay of recorded Parquet files.
For every snapshot the predictor recomputes features over a short rolling history (exactly the
same code path as training), scores the model and converts the probability into a signal:

* ``BUY``  if P(up) >= buy_threshold
* ``SELL`` if P(up) <= sell_threshold
* ``HOLD`` otherwise

Outputs (under ``<workspace>/predictions``):

* ``live.parquet``        - rolling window of recent predictions (what the dashboard plots)
* ``latest_book.json``    - top-of-book levels of the most recent snapshot (depth chart)
* ``session.json``        - what is running: mode, model, thresholds, start time
* ``history/part-*.parquet`` - every prediction, in append-only batches
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa

from orderbook_ml.config import Settings
from orderbook_ml.features import MIN_HISTORY, compute_features
from orderbook_ml.logging_utils import get_logger
from orderbook_ml.modeling import ModelBundle
from orderbook_ml.storage import write_json_atomic, write_parquet_atomic

logger = get_logger(__name__)

SIGNALS = ("BUY", "HOLD", "SELL")
HISTORY_ROWS = 20  # >= MIN_HISTORY; enough context for every rolling feature

PREDICTION_SCHEMA = pa.schema(
    [
        ("timestamp", pa.timestamp("ns", tz="UTC")),
        ("symbol", pa.string()),
        ("best_bid", pa.float64()),
        ("best_ask", pa.float64()),
        ("mid_price", pa.float64()),
        ("spread", pa.float64()),
        ("obi_level_1", pa.float64()),
        ("obi_level_5", pa.float64()),
        ("decayed_obi", pa.float64()),
        ("prob_up", pa.float64()),
        ("signal", pa.string()),
        ("latency_ms", pa.float64()),
    ]
)


def signal_for(prob_up: float, buy_threshold: float, sell_threshold: float) -> str:
    if prob_up >= buy_threshold:
        return "BUY"
    if prob_up <= sell_threshold:
        return "SELL"
    return "HOLD"


class Predictor:
    """Stateful scorer: feed snapshots in time order, get prediction records out."""

    def __init__(
        self,
        bundle: ModelBundle,
        buy_threshold: float = 0.60,
        sell_threshold: float = 0.40,
    ) -> None:
        if not 0.0 < sell_threshold <= buy_threshold < 1.0:
            raise ValueError("Thresholds must satisfy 0 < sell <= buy < 1")
        self.bundle = bundle
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self._history: deque[dict] = deque(maxlen=HISTORY_ROWS)
        self.last_signal: str | None = None
        self.count = 0

    def update(self, snapshot: dict) -> dict | None:
        """Scores one snapshot. Returns ``None`` while the rolling history is warming up."""
        if self._history and snapshot["timestamp"] <= self._history[-1]["timestamp"]:
            logger.debug("Ignoring out-of-order snapshot at %s", snapshot["timestamp"])
            return None
        self._history.append(snapshot)
        if len(self._history) < MIN_HISTORY:
            return None

        start = time.perf_counter()
        features = compute_features(pd.DataFrame(list(self._history))).iloc[[-1]]
        prob_up = float(self.bundle.predict_proba_up(features)[0])
        latency_ms = (time.perf_counter() - start) * 1000

        signal = signal_for(prob_up, self.buy_threshold, self.sell_threshold)
        if signal != self.last_signal:
            if self.last_signal is not None:
                logger.debug(
                    "Signal %s -> %s at mid %.2f (P(up)=%.3f)",
                    self.last_signal,
                    signal,
                    (snapshot["best_bid"] + snapshot["best_ask"]) / 2,
                    prob_up,
                )
            self.last_signal = signal
        self.count += 1

        row = features.iloc[0]
        return {
            "timestamp": pd.Timestamp(snapshot["timestamp"]),
            "symbol": snapshot["symbol"],
            "best_bid": float(snapshot["best_bid"]),
            "best_ask": float(snapshot["best_ask"]),
            "mid_price": (float(snapshot["best_bid"]) + float(snapshot["best_ask"])) / 2,
            "spread": float(snapshot["spread"]),
            "obi_level_1": float(row["obi_level_1"]),
            "obi_level_5": float(row["obi_level_5"]),
            "decayed_obi": float(row["decayed_obi"]),
            "prob_up": prob_up,
            "signal": signal,
            "latency_ms": round(latency_ms, 3),
        }


class PredictionSink:
    """Persists prediction records for the dashboard and for later analysis."""

    def __init__(
        self,
        directory: Path,
        window_size: int = 600,
        part_size: int = 3_000,
        publish_every: int = 5,
    ) -> None:
        self.directory = Path(directory)
        self.history_dir = self.directory / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.window: deque[dict] = deque(maxlen=window_size)
        self.part: list[dict] = []
        self.part_size = part_size
        self.publish_every = publish_every
        self._latest_snapshot: dict | None = None
        self._since_publish = 0
        self.total = 0

    def start_session(self, info: dict[str, Any]) -> None:
        for stale in ("live.parquet", "latest_book.json"):
            (self.directory / stale).unlink(missing_ok=True)
        write_json_atomic(
            {**info, "started_at": pd.Timestamp.now(tz="UTC").isoformat(), "status": "running"},
            self.directory / "session.json",
        )

    def end_session(self, status: str = "finished") -> None:
        path = self.directory / "session.json"
        info = json.loads(path.read_text()) if path.exists() else {}
        info.update(
            status=status,
            ended_at=pd.Timestamp.now(tz="UTC").isoformat(),
            predictions=self.total,
        )
        write_json_atomic(info, path)

    def add(self, record: dict, snapshot: dict) -> None:
        self.window.append(record)
        self.part.append(record)
        self._latest_snapshot = snapshot
        self.total += 1
        self._since_publish += 1
        if self._since_publish >= self.publish_every:
            self.publish()
        if len(self.part) >= self.part_size:
            self._flush_part()

    def publish(self) -> None:
        """Atomically rewrites the live window and the latest book for the dashboard."""
        if self.window:
            table = pa.Table.from_pylist(list(self.window), schema=PREDICTION_SCHEMA)
            write_parquet_atomic(table, self.directory / "live.parquet")
        if self._latest_snapshot is not None:
            snap = self._latest_snapshot
            write_json_atomic(
                {
                    "timestamp": pd.Timestamp(snap["timestamp"]).isoformat(),
                    "symbol": snap["symbol"],
                    "bids": json.loads(snap["bids_json"]),
                    "asks": json.loads(snap["asks_json"]),
                },
                self.directory / "latest_book.json",
            )
        self._since_publish = 0

    def _flush_part(self) -> None:
        if not self.part:
            return
        table = pa.Table.from_pylist(self.part, schema=PREDICTION_SCHEMA)
        stamp = pd.Timestamp(self.part[0]["timestamp"]).strftime("%Y%m%dT%H%M%S%f")
        write_parquet_atomic(table, self.history_dir / f"part-{stamp}.parquet")
        self.part.clear()

    def close(self) -> None:
        self.publish()
        self._flush_part()


def _session_info(mode: str, predictor: Predictor) -> dict[str, Any]:
    bundle = predictor.bundle
    return {
        "mode": mode,
        "symbol": bundle.symbol,
        "synthetic": bundle.synthetic,
        "model_name": bundle.model_name,
        "target": bundle.target,
        "horizon_sec": bundle.horizon_sec,
        "buy_threshold": predictor.buy_threshold,
        "sell_threshold": predictor.sell_threshold,
    }


def _log_progress(predictor: Predictor, record: dict) -> None:
    if predictor.count % 100 == 0:
        logger.info(
            "#%d %s mid=%.2f P(up)=%.3f %s (%.1f ms)",
            predictor.count,
            record["timestamp"].strftime("%H:%M:%S.%f")[:-3],
            record["mid_price"],
            record["prob_up"],
            record["signal"],
            record["latency_ms"],
        )


def run_replay(
    snapshots: Iterable[dict],
    predictor: Predictor,
    sink: PredictionSink,
    realtime_interval_sec: float | None = None,
    max_snapshots: int | None = None,
) -> int:
    """Scores recorded snapshots in order. Optionally paces them at the original sampling rate."""
    sink.start_session(_session_info("replay", predictor))
    status = "finished"
    try:
        for i, snapshot in enumerate(snapshots):
            if max_snapshots is not None and i >= max_snapshots:
                break
            record = predictor.update(snapshot)
            if record is not None:
                sink.add(record, snapshot)
                _log_progress(predictor, record)
            if realtime_interval_sec:
                time.sleep(realtime_interval_sec)
    except KeyboardInterrupt:
        status = "stopped"
        raise
    finally:
        sink.close()
        sink.end_session(status)
    return predictor.count


async def run_live(
    stream: AsyncIterator[dict],
    predictor: Predictor,
    sink: PredictionSink,
    settings: Settings,
) -> int:
    """Scores snapshots from the live Binance stream until cancelled."""
    if predictor.bundle.symbol != settings.symbol:
        raise ValueError(
            f"The model was trained on {predictor.bundle.symbol} but OBML_SYMBOL is "
            f"{settings.symbol}. Train a model for {settings.symbol} first."
        )
    sink.start_session(_session_info("live", predictor))
    status = "stopped"
    try:
        async for snapshot in stream:
            record = predictor.update(snapshot)
            if record is not None:
                sink.add(record, snapshot)
                _log_progress(predictor, record)
        status = "finished"
    except asyncio.CancelledError:
        raise
    except Exception:
        status = "failed"
        raise
    finally:
        sink.close()
        sink.end_session(status)
    return predictor.count
