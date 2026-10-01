"""Data quality checks for collected order book snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from orderbook_ml.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class QualityReport:
    symbol: str
    total_snapshots: int
    start: str | None
    end: str | None
    duration_sec: float
    median_interval_ms: float | None
    timestamp_gaps: int
    largest_gap_sec: float
    crossed_books: int
    empty_sides: int
    spread_outliers: int
    median_spread: float | None
    quality_score: float

    def to_dict(self) -> dict:
        return asdict(self)


class DataValidator:
    """Checks snapshot data for gaps, crossed books, missing liquidity and spread outliers."""

    def __init__(
        self,
        gap_threshold_ms: float = 200.0,
        outlier_multiplier: float = 10.0,
    ) -> None:
        self.gap_threshold_ms = gap_threshold_ms
        self.outlier_multiplier = outlier_multiplier

    def timestamp_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rows that arrived more than ``gap_threshold_ms`` after the previous row."""
        if len(df) < 2:
            return df.iloc[0:0].assign(gap_ms=pd.Series(dtype=float))
        diffs_ms = df["timestamp"].diff().dt.total_seconds() * 1000.0
        mask = diffs_ms > self.gap_threshold_ms
        return df.loc[mask].assign(gap_ms=diffs_ms[mask])

    @staticmethod
    def crossed_books(df: pd.DataFrame) -> pd.DataFrame:
        """Rows where best bid >= best ask (impossible in a consistent book)."""
        return df.loc[(df["best_bid"] >= df["best_ask"]) | (df["spread"] <= 0)]

    @staticmethod
    def empty_sides(df: pd.DataFrame) -> pd.DataFrame:
        """Rows with no resting volume on one side."""
        return df.loc[(df["total_bid_volume"] <= 0) | (df["total_ask_volume"] <= 0)]

    def spread_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rows whose spread exceeds ``outlier_multiplier`` x the median positive spread."""
        positive = df.loc[df["spread"] > 0, "spread"]
        if positive.empty:
            return df.iloc[0:0]
        return df.loc[df["spread"] > self.outlier_multiplier * positive.median()]

    def report(self, df: pd.DataFrame) -> QualityReport:
        if df.empty:
            return QualityReport(
                symbol="",
                total_snapshots=0,
                start=None,
                end=None,
                duration_sec=0.0,
                median_interval_ms=None,
                timestamp_gaps=0,
                largest_gap_sec=0.0,
                crossed_books=0,
                empty_sides=0,
                spread_outliers=0,
                median_spread=None,
                quality_score=0.0,
            )

        gaps = self.timestamp_gaps(df)
        crossed = self.crossed_books(df)
        empty = self.empty_sides(df)
        outliers = self.spread_outliers(df)

        # A snapshot is "flawed" if it is invalid (crossed / empty) or follows a gap.
        flawed_idx = set(gaps.index) | set(crossed.index) | set(empty.index)
        score = 100.0 * (1.0 - len(flawed_idx) / len(df))
        interval = df["timestamp"].diff().dt.total_seconds().median()

        return QualityReport(
            symbol=str(df["symbol"].iloc[0]),
            total_snapshots=len(df),
            start=df["timestamp"].iloc[0].isoformat(),
            end=df["timestamp"].iloc[-1].isoformat(),
            duration_sec=float(
                (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()
            ),
            median_interval_ms=None if pd.isna(interval) else round(float(interval) * 1000, 2),
            timestamp_gaps=len(gaps),
            largest_gap_sec=round(float(gaps["gap_ms"].max() / 1000), 3) if len(gaps) else 0.0,
            crossed_books=len(crossed),
            empty_sides=len(empty),
            spread_outliers=len(outliers),
            median_spread=float(df["spread"].median()),
            quality_score=round(score, 2),
        )

    def plot(self, df: pd.DataFrame, output_path: Path) -> Path | None:
        """Saves a two-panel chart of spread and resting volume, highlighting anomalies."""
        if df.empty:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_df = df
        if len(df) > 20_000:  # keep the figure light for long recordings
            plot_df = df.iloc[:: int(np.ceil(len(df) / 20_000))]

        crossed = self.crossed_books(df)
        outliers = self.spread_outliers(df)
        gaps = self.timestamp_gaps(df)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        fig.suptitle(f"Order book data quality - {df['symbol'].iloc[0]}", fontweight="bold")

        ax1.plot(plot_df["timestamp"], plot_df["spread"], lw=0.8, color="#2563eb", label="Spread")
        ax1.axhline(df["spread"].median(), ls="--", color="#16a34a", lw=1, label="Median spread")
        if not outliers.empty:
            ax1.scatter(
                outliers["timestamp"],
                outliers["spread"],
                s=12,
                color="#f59e0b",
                label=f"Outliers (>{self.outlier_multiplier:g}x median)",
                zorder=3,
            )
        if not crossed.empty:
            ax1.scatter(
                crossed["timestamp"],
                crossed["spread"],
                s=16,
                color="#dc2626",
                label="Crossed book",
                zorder=4,
            )
        ax1.set_ylabel("Spread (quote currency)")
        ax1.grid(alpha=0.3, ls=":")
        ax1.legend(loc="upper right", fontsize=8)

        ax2.plot(
            plot_df["timestamp"],
            plot_df["total_bid_volume"],
            lw=0.8,
            color="#16a34a",
            label="Bid volume",
        )
        ax2.plot(
            plot_df["timestamp"],
            plot_df["total_ask_volume"],
            lw=0.8,
            color="#dc2626",
            label="Ask volume",
        )
        for i, ts in enumerate(gaps["timestamp"].head(200)):
            ax2.axvline(
                ts,
                color="#a855f7",
                ls=":",
                alpha=0.5,
                label=f"Gap (>{self.gap_threshold_ms:g} ms)" if i == 0 else None,
            )
        ax2.set_ylabel("Resting volume (base asset)")
        ax2.set_xlabel("Time (UTC)")
        ax2.grid(alpha=0.3, ls=":")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return output_path
