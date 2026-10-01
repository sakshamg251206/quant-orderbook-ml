"""Builds the supervised learning dataset from raw snapshots.

snapshots -> data-quality report -> features + labels -> chronological train / validation / test
split with an embargo gap between the parts -> Parquet files + metadata.

Why the embargo: a training row at time ``t`` has a label that looks ``h`` seconds into the
future. Without a gap, the last training labels would be computed from prices that fall inside
the validation/test period and leak information across the boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import pandas as pd

from orderbook_ml.config import Workspace
from orderbook_ml.features import FEATURE_NAMES, compute_features
from orderbook_ml.labels import class_balance, generate_labels, label_column
from orderbook_ml.logging_utils import get_logger
from orderbook_ml.storage import load_snapshots, read_json, write_json_atomic
from orderbook_ml.synthetic import is_synthetic_symbol
from orderbook_ml.validation import DataValidator

logger = get_logger(__name__)

SPLITS = ("train", "validation", "test")
META_COLUMNS = ["timestamp", "symbol", "best_bid", "best_ask"]
MIN_SNAPSHOTS = 1_000


class InsufficientDataError(RuntimeError):
    """Raised when there is not enough data to build a meaningful dataset or model."""


@dataclass
class DatasetMeta:
    symbol: str
    synthetic: bool
    feature_names: list[str]
    label_columns: list[str]
    horizons_sec: list[int]
    embargo_sec: float
    rows: dict[str, int]
    time_ranges: dict[str, list[str]]
    class_balance: dict[str, dict[str, dict[str, float]]]
    quality_score: float
    created_at: str = field(default_factory=lambda: pd.Timestamp.now(tz="UTC").isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    embargo: pd.Timedelta | None = None,
) -> dict[str, pd.DataFrame]:
    """Splits time-sorted rows into contiguous train/validation/test blocks (no shuffling).

    Rows within ``embargo`` before the start of the next block are dropped (purged).
    """
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("Fractions must be in (0, 1) and leave room for a test set")
    n = len(df)
    i_train, i_val = int(n * train_frac), int(n * (train_frac + val_frac))
    parts = {
        "train": df.iloc[:i_train],
        "validation": df.iloc[i_train:i_val],
        "test": df.iloc[i_val:],
    }
    if embargo is not None and embargo > pd.Timedelta(0):
        for current, following in (("train", "validation"), ("validation", "test")):
            if parts[following].empty:
                continue
            cutoff = parts[following]["timestamp"].iloc[0] - embargo
            parts[current] = parts[current].loc[parts[current]["timestamp"] < cutoff]
    return {name: part.reset_index(drop=True) for name, part in parts.items()}


def build_dataset(
    workspace: Workspace,
    symbol: str | None = None,
    horizons_sec: Sequence[int] = (1, 5, 10),
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> DatasetMeta:
    workspace.ensure()
    raw = load_snapshots(workspace.snapshots, symbol)
    if len(raw) < MIN_SNAPSHOTS:
        raise InsufficientDataError(
            f"Found {len(raw):,} snapshots in {workspace.snapshots} (need at least "
            f"{MIN_SNAPSHOTS:,}, ~2 minutes at 100 ms; 30+ minutes recommended). "
            "Run `obml collect` (live) or `obml simulate` (synthetic) first."
        )
    symbol = str(raw["symbol"].iloc[0])
    logger.info("Loaded %s snapshots for %s", f"{len(raw):,}", symbol)

    validator = DataValidator()
    quality = validator.report(raw)
    write_json_atomic(quality.to_dict(), workspace.reports / "data_quality.json")
    validator.plot(raw, workspace.reports / "data_quality.png")
    if quality.quality_score < 80:
        logger.warning(
            "Data quality score is low (%.1f%%); inspect reports/data_quality.png",
            quality.quality_score,
        )

    features = compute_features(raw)
    labels = generate_labels(raw, horizons_sec)
    label_cols = [label_column(h) for h in sorted(set(horizons_sec))]
    frame = pd.concat([raw[META_COLUMNS], features, labels], axis=1)
    frame = frame.dropna(subset=label_cols).reset_index(drop=True)
    frame[label_cols] = frame[label_cols].astype("int8")

    embargo = pd.Timedelta(seconds=max(horizons_sec))
    parts = chronological_split(frame, train_frac, val_frac, embargo)
    for name, part in parts.items():
        if len(part) < 50:
            raise InsufficientDataError(
                f"The {name} split has only {len(part)} rows; collect more data."
            )
        part.to_parquet(workspace.processed / f"{name}.parquet", index=False)

    meta = DatasetMeta(
        symbol=symbol,
        synthetic=is_synthetic_symbol(symbol),
        feature_names=list(FEATURE_NAMES),
        label_columns=label_cols,
        horizons_sec=sorted(set(horizons_sec)),
        embargo_sec=embargo.total_seconds(),
        rows={name: len(part) for name, part in parts.items()},
        time_ranges={
            name: [part["timestamp"].iloc[0].isoformat(), part["timestamp"].iloc[-1].isoformat()]
            for name, part in parts.items()
        },
        class_balance={name: class_balance(part[label_cols]) for name, part in parts.items()},
        quality_score=quality.quality_score,
    )
    write_json_atomic(meta.to_dict(), workspace.processed / "dataset_meta.json")
    logger.info(
        "Dataset written to %s (train=%d, validation=%d, test=%d rows, %d features)",
        workspace.processed,
        meta.rows["train"],
        meta.rows["validation"],
        meta.rows["test"],
        len(meta.feature_names),
    )
    return meta


def load_split(workspace: Workspace, split: str) -> pd.DataFrame:
    path = workspace.processed / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `obml build-dataset` first.")
    return pd.read_parquet(path)


def load_meta(workspace: Workspace) -> DatasetMeta:
    data = read_json(workspace.processed / "dataset_meta.json")
    if data is None:
        raise FileNotFoundError("Dataset metadata not found. Run `obml build-dataset` first.")
    return DatasetMeta(**data)
