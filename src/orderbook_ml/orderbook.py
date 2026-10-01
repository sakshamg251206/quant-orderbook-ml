"""Local L2 order book that is kept in sync with Binance diff-depth events.

This module is deliberately free of any networking so the synchronisation rules can be unit
tested exhaustively. It implements the procedure from the Binance Spot API documentation
("How to manage a local order book correctly"):

1. Load a REST depth snapshot with ``lastUpdateId = L``.
2. Ignore any event whose final update id ``u`` is ``<= L`` (already contained in the snapshot).
3. Every applied event must start at most one id after the previous one (``U <= L + 1``).
   Otherwise an update was missed, the book is no longer trustworthy and must be rebuilt.
"""

from __future__ import annotations

import heapq
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd
import pyarrow as pa

PriceLevel = tuple[float, float]

SNAPSHOT_SCHEMA = pa.schema(
    [
        ("timestamp", pa.timestamp("ns", tz="UTC")),
        ("symbol", pa.string()),
        ("last_update_id", pa.int64()),
        ("bids_json", pa.string()),
        ("asks_json", pa.string()),
        ("best_bid", pa.float64()),
        ("best_ask", pa.float64()),
        ("spread", pa.float64()),
        ("total_bid_volume", pa.float64()),
        ("total_ask_volume", pa.float64()),
    ]
)
"""On-disk schema of one order book snapshot (one row per sampling tick)."""


class ApplyResult(Enum):
    APPLIED = "applied"
    STALE = "stale"  # event already reflected in the snapshot; safely ignored
    GAP = "gap"  # an update was missed; the book must be rebuilt


@dataclass(frozen=True)
class DepthEvent:
    """A parsed ``depthUpdate`` event."""

    first_update_id: int
    final_update_id: int
    bids: Sequence[Sequence[str]]
    asks: Sequence[Sequence[str]]

    @classmethod
    def from_message(cls, message: dict[str, Any]) -> DepthEvent | None:
        """Parses a raw websocket payload; returns ``None`` for non-depth messages."""
        data = message.get("data", message)
        if not isinstance(data, dict) or data.get("e") != "depthUpdate":
            return None
        return cls(
            first_update_id=int(data["U"]),
            final_update_id=int(data["u"]),
            bids=data.get("b", []),
            asks=data.get("a", []),
        )


class OrderBookSyncError(RuntimeError):
    """Raised when the local book can no longer be trusted and must be re-synchronised."""


class LocalOrderBook:
    """In-memory price -> quantity maps for both sides of the book."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.upper()
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self.last_update_id: int | None = None

    @property
    def is_ready(self) -> bool:
        return self.last_update_id is not None and bool(self._bids) and bool(self._asks)

    def reset(self) -> None:
        self._bids.clear()
        self._asks.clear()
        self.last_update_id = None

    def load_snapshot(
        self,
        last_update_id: int,
        bids: Iterable[Sequence[str | float]],
        asks: Iterable[Sequence[str | float]],
    ) -> None:
        self._bids = {float(p): float(q) for p, q in bids if float(q) > 0}
        self._asks = {float(p): float(q) for p, q in asks if float(q) > 0}
        self.last_update_id = int(last_update_id)

    def apply(self, event: DepthEvent) -> ApplyResult:
        if self.last_update_id is None:
            raise OrderBookSyncError("Cannot apply events before a snapshot is loaded")
        if event.final_update_id <= self.last_update_id:
            return ApplyResult.STALE
        if event.first_update_id > self.last_update_id + 1:
            return ApplyResult.GAP

        self._apply_side(self._bids, event.bids)
        self._apply_side(self._asks, event.asks)
        self.last_update_id = event.final_update_id
        return ApplyResult.APPLIED

    @staticmethod
    def _apply_side(book: dict[float, float], levels: Iterable[Sequence[str | float]]) -> None:
        for price_raw, qty_raw in levels:
            price, qty = float(price_raw), float(qty_raw)
            if qty == 0.0:
                book.pop(price, None)
            else:
                book[price] = qty

    def top_levels(self, depth: int) -> tuple[list[PriceLevel], list[PriceLevel]]:
        """Best ``depth`` bids (descending price) and asks (ascending price)."""
        bids = heapq.nlargest(depth, self._bids.items())
        asks = heapq.nsmallest(depth, self._asks.items())
        return bids, asks

    def snapshot_record(self, depth: int, timestamp: pd.Timestamp | None = None) -> dict | None:
        """Builds a row matching :data:`SNAPSHOT_SCHEMA`, or ``None`` if the book is unusable."""
        if not self.is_ready:
            return None
        bids, asks = self.top_levels(depth)
        return build_snapshot_record(
            symbol=self.symbol,
            bids=bids,
            asks=asks,
            last_update_id=self.last_update_id or 0,
            timestamp=timestamp,
        )


def build_snapshot_record(
    symbol: str,
    bids: Sequence[PriceLevel],
    asks: Sequence[PriceLevel],
    last_update_id: int = 0,
    timestamp: pd.Timestamp | None = None,
) -> dict | None:
    """Creates a snapshot row from sorted price levels (best price first)."""
    if not bids or not asks:
        return None
    best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
    return {
        "timestamp": timestamp if timestamp is not None else pd.Timestamp.now(tz="UTC"),
        "symbol": symbol,
        "last_update_id": int(last_update_id),
        "bids_json": json.dumps([[p, q] for p, q in bids]),
        "asks_json": json.dumps([[p, q] for p, q in asks]),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": best_ask - best_bid,
        "total_bid_volume": float(sum(q for _, q in bids)),
        "total_ask_volume": float(sum(q for _, q in asks)),
    }
