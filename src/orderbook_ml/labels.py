"""Forward-looking price-direction targets.

``label_{h}s`` is 1 if the mid price ``h`` seconds later is strictly higher than now, and 0 if
it is equal or lower ("up" vs "not up"). The future price is the first snapshot at or after
``t + h``; if that snapshot is more than ``max_tolerance_ms`` late (e.g. a collection gap) the
label is left undefined and the row is dropped, rather than silently using a stale price.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from orderbook_ml.logging_utils import get_logger

logger = get_logger(__name__)


def label_column(horizon_sec: int) -> str:
    return f"label_{horizon_sec}s"


def horizon_from_label(label: str) -> int:
    try:
        return int(label.removeprefix("label_").removesuffix("s"))
    except ValueError as exc:
        raise ValueError(f"Not a label column name: {label!r}") from exc


def mid_price(df: pd.DataFrame) -> pd.Series:
    return (df["best_bid"].astype(float) + df["best_ask"].astype(float)) / 2.0


def generate_labels(
    df: pd.DataFrame,
    horizons_sec: Sequence[int] = (1, 5, 10),
    max_tolerance_ms: float = 500.0,
) -> pd.DataFrame:
    """Returns a frame indexed like ``df`` with one nullable integer column per horizon.

    ``df`` must be sorted by ``timestamp``. Rows whose label cannot be determined are ``<NA>``.
    """
    if df.empty:
        return pd.DataFrame(columns=[label_column(h) for h in horizons_sec], dtype="Int8")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Snapshots must be sorted by timestamp before labelling")

    now = pd.DataFrame({"timestamp": df["timestamp"], "mid": mid_price(df)}, index=df.index)
    lookup = now.rename(columns={"timestamp": "future_ts", "mid": "future_mid"})
    tolerance = pd.Timedelta(milliseconds=max_tolerance_ms)

    labels = pd.DataFrame(index=df.index)
    for horizon in sorted(set(horizons_sec)):
        target = pd.DataFrame({"target_ts": now["timestamp"] + pd.Timedelta(seconds=horizon)})
        matched = pd.merge_asof(
            target,
            lookup,
            left_on="target_ts",
            right_on="future_ts",
            direction="forward",
            tolerance=tolerance,
        )
        future_mid = matched["future_mid"].to_numpy()
        label = pd.Series(future_mid > now["mid"].to_numpy(), index=df.index, dtype="Int8")
        label[np.isnan(future_mid)] = pd.NA
        labels[label_column(horizon)] = label
    return labels


def class_balance(labels: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Share of positive labels per column (ignoring undefined rows)."""
    stats: dict[str, dict[str, float]] = {}
    for col in labels.columns:
        valid = labels[col].dropna()
        if valid.empty:
            continue
        pos = float(valid.mean())
        stats[col] = {"samples": len(valid), "positive_rate": round(pos, 4)}
        if not 0.4 <= pos <= 0.6:
            logger.warning(
                "%s is imbalanced: %.1f%% positive. Watch precision/recall, not accuracy.",
                col,
                pos * 100,
            )
    return stats
