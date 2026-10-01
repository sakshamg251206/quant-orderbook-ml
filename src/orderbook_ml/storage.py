"""Parquet persistence for order book snapshots and small JSON artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from orderbook_ml.logging_utils import get_logger
from orderbook_ml.orderbook import SNAPSHOT_SCHEMA

logger = get_logger(__name__)


class SnapshotWriter:
    """Buffers snapshot rows in memory and writes them as compressed Parquet batches.

    Files are named ``<SYMBOL>_<UTC timestamp>.parquet`` so data from different symbols can share
    a directory and still be loaded separately.
    """

    def __init__(self, directory: Path, symbol: str, batch_size: int = 600) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol.upper()
        self.batch_size = batch_size
        self._buffer: list[dict] = []
        self.rows_written = 0
        self.files_written = 0

    @property
    def pending(self) -> int:
        return len(self._buffer)

    def add(self, record: dict) -> Path | None:
        """Buffers a row; returns the file path when this row triggered a flush."""
        self._buffer.append(record)
        if len(self._buffer) >= self.batch_size:
            return self.flush()
        return None

    def flush(self) -> Path | None:
        if not self._buffer:
            return None
        table = pa.Table.from_pylist(self._buffer, schema=SNAPSHOT_SCHEMA)
        first_ts = pd.Timestamp(self._buffer[0]["timestamp"]).strftime("%Y%m%dT%H%M%S%f")
        path = self.directory / f"{self.symbol}_{first_ts}.parquet"
        write_parquet_atomic(table, path)
        self.rows_written += len(self._buffer)
        self.files_written += 1
        logger.debug("Wrote %d snapshots to %s", len(self._buffer), path.name)
        self._buffer.clear()
        return path

    def __enter__(self) -> SnapshotWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.flush()


def snapshot_files(directory: Path, symbol: str | None = None) -> list[Path]:
    directory = Path(directory)
    if directory.is_file():
        return [directory]
    pattern = f"{symbol.upper()}_*.parquet" if symbol else "*.parquet"
    return sorted(directory.glob(pattern))


def available_symbols(directory: Path) -> list[str]:
    """Symbols that have at least one snapshot file in ``directory``."""
    return sorted({p.name.rsplit("_", 1)[0] for p in snapshot_files(directory)})


def load_snapshots(directory: Path, symbol: str | None = None) -> pd.DataFrame:
    """Loads, de-duplicates and time-sorts all snapshot files for a symbol.

    If ``symbol`` is ``None`` the directory must contain exactly one symbol; mixing markets in
    one time series would silently corrupt every downstream feature and label.
    """
    files = snapshot_files(directory, symbol)
    if not files:
        return pd.DataFrame(columns=SNAPSHOT_SCHEMA.names)

    frames = []
    for path in files:
        try:
            frames.append(pd.read_parquet(path))
        except (OSError, pa.ArrowException) as exc:
            logger.warning("Skipping unreadable snapshot file %s: %s", path.name, exc)
    if not frames:
        return pd.DataFrame(columns=SNAPSHOT_SCHEMA.names)

    df = pd.concat(frames, ignore_index=True)
    symbols = df["symbol"].dropna().unique().tolist()
    if symbol is None and len(symbols) > 1:
        raise ValueError(
            f"Found snapshots for several symbols {symbols} in {directory}; pass a symbol."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = (
        df.sort_values("timestamp", kind="stable")
        .drop_duplicates(subset="timestamp", keep="last")
        .reset_index(drop=True)
    )
    return df


def write_parquet_atomic(table: pa.Table, path: Path) -> None:
    """Writes to a temp file and renames, so readers never observe a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression="snappy")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json_atomic(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any | None:
    """Reads a JSON file, returning ``None`` if it does not exist."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
