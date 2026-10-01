from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from orderbook_ml.config import Workspace
from orderbook_ml.dataset import build_dataset
from orderbook_ml.modeling import train_models
from orderbook_ml.storage import SnapshotWriter
from orderbook_ml.synthetic import generate_snapshots

START = pd.Timestamp("2026-01-05 09:00:00", tz="UTC")


@pytest.fixture(scope="session")
def snapshots() -> pd.DataFrame:
    """Six minutes of synthetic snapshots (3,600 rows)."""
    return generate_snapshots(360, seed=7, start=START)


def write_snapshots(df: pd.DataFrame, directory: Path) -> None:
    with SnapshotWriter(directory, str(df["symbol"].iloc[0]), batch_size=1_000) as writer:
        for record in df.to_dict("records"):
            writer.add(record)


@pytest.fixture(scope="session")
def trained_workspace(
    tmp_path_factory: pytest.TempPathFactory, snapshots: pd.DataFrame
) -> Workspace:
    """A workspace that has gone through build-dataset and train (fast model settings)."""
    ws = Workspace(tmp_path_factory.mktemp("workspace")).ensure()
    write_snapshots(snapshots, ws.snapshots)
    build_dataset(ws)
    train_models(ws, target="label_1s", candidates=("catboost", "logreg"))
    return ws
