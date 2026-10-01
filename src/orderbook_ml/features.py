"""Order book microstructure features.

Every feature at row ``t`` is computed only from snapshots ``<= t`` (no look-ahead), and every
feature is scale-free or spread/volume based rather than an absolute price level, so a model
cannot simply memorise "BTC was at 60k during training".

Time-based names (``_100ms``, ``_500ms``) assume the default 100 ms sampling interval; in
general they mean "1 step" and "5 steps".
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import skew

N_LEVELS = 20
MIN_HISTORY = 6  # rows needed before the 5-step momentum/volatility features are defined
_DECAY_WEIGHTS = np.exp(-0.1 * np.arange(N_LEVELS))

FEATURE_GROUPS: dict[str, list[str]] = {
    "Order book imbalance": [
        "obi_level_1",
        "obi_level_5",
        "obi_level_10",
        "obi_level_20",
        "decayed_obi",
    ],
    "Spread": [
        "spread_absolute",
        "spread_relative",
        "spread_log",
        "effective_spread",
    ],
    "Depth & volume": [
        "total_bid_volume",
        "total_ask_volume",
        "bid_ask_volume_ratio",
        "depth_ratio_level_5",
        "depth_ratio_level_10",
        "bid_vol_top1_share",
        "ask_vol_top1_share",
    ],
    "Micro-price": [
        "micro_price_deviation",
        "weighted_mid_deviation_5",
        "weighted_mid_deviation_10",
    ],
    "Book shape": [
        "bid_volume_std",
        "ask_volume_std",
        "bid_volume_skew",
        "ask_volume_skew",
        "bid_volume_decay",
        "ask_volume_decay",
    ],
    "Dynamics": [
        "obi_momentum_100ms",
        "obi_momentum_500ms",
        "spread_momentum_100ms",
        "mid_price_return_100ms",
        "mid_price_return_500ms",
        "mid_price_volatility_5",
    ],
}
FEATURE_NAMES: list[str] = [name for group in FEATURE_GROUPS.values() for name in group]


def _parse_levels(raw: object) -> list:
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []
    if isinstance(raw, (list, tuple, np.ndarray)):
        return list(raw)
    return []


def book_matrices(
    bids: Sequence[object], asks: Sequence[object], levels: int = N_LEVELS
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parses JSON price levels into dense ``(rows, levels)`` price and volume matrices.

    Missing levels are zero-filled.
    """
    n = len(bids)
    bid_p, bid_v = np.zeros((n, levels)), np.zeros((n, levels))
    ask_p, ask_v = np.zeros((n, levels)), np.zeros((n, levels))
    for i, (raw_b, raw_a) in enumerate(zip(bids, asks, strict=True)):
        for side_p, side_v, raw in ((bid_p, bid_v, raw_b), (ask_p, ask_v, raw_a)):
            parsed = _parse_levels(raw)[:levels]
            if parsed:
                arr = np.asarray(parsed, dtype=float).reshape(-1, 2)
                side_p[i, : len(arr)] = arr[:, 0]
                side_v[i, : len(arr)] = arr[:, 1]
    return bid_p, bid_v, ask_p, ask_v


def _safe_div(num: np.ndarray, den: np.ndarray, default: float = 0.0) -> np.ndarray:
    out = np.full(np.broadcast(num, den).shape, default, dtype=float)
    np.divide(num, den, out=out, where=den != 0)
    return out


def _imbalance(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    return _safe_div(bid - ask, bid + ask)


def _diff(values: np.ndarray, periods: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    out[periods:] = values[periods:] - values[:-periods]
    return out


def _pct_change(values: np.ndarray, periods: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[periods:] = values[periods:] / values[:-periods] - 1.0
    return out


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms snapshot rows into the :data:`FEATURE_NAMES` matrix (same index as ``df``)."""
    if df.empty:
        return pd.DataFrame(columns=FEATURE_NAMES, dtype=float)

    bid_p, bid_v, ask_p, ask_v = book_matrices(df["bids_json"].tolist(), df["asks_json"].tolist())
    best_bid = np.where(bid_p[:, 0] > 0, bid_p[:, 0], df["best_bid"].to_numpy(dtype=float))
    best_ask = np.where(ask_p[:, 0] > 0, ask_p[:, 0], df["best_ask"].to_numpy(dtype=float))

    bid_1, ask_1 = bid_v[:, 0], ask_v[:, 0]
    bid_5, ask_5 = bid_v[:, :5].sum(1), ask_v[:, :5].sum(1)
    bid_10, ask_10 = bid_v[:, :10].sum(1), ask_v[:, :10].sum(1)
    bid_20, ask_20 = bid_v.sum(1), ask_v.sum(1)

    mid = (best_bid + best_ask) / 2.0
    micro = np.where(
        bid_1 + ask_1 > 0, _safe_div(best_bid * ask_1 + best_ask * bid_1, bid_1 + ask_1), mid
    )

    def weighted_mid(k: int) -> np.ndarray:
        both = (bid_p[:, :k] > 0) & (ask_p[:, :k] > 0)
        num = ((bid_p[:, :k] * ask_v[:, :k] + ask_p[:, :k] * bid_v[:, :k]) * both).sum(1)
        den = ((bid_v[:, :k] + ask_v[:, :k]) * both).sum(1)
        return np.where(den > 0, _safe_div(num, den), mid)

    spread = best_ask - best_bid
    bid_decay, ask_decay = bid_v @ _DECAY_WEIGHTS, ask_v @ _DECAY_WEIGHTS
    bid_std, ask_std = bid_v.std(1), ask_v.std(1)

    out: dict[str, np.ndarray] = {}
    out["obi_level_1"] = _imbalance(bid_1, ask_1)
    out["obi_level_5"] = _imbalance(bid_5, ask_5)
    out["obi_level_10"] = _imbalance(bid_10, ask_10)
    out["obi_level_20"] = _imbalance(bid_20, ask_20)
    out["decayed_obi"] = _imbalance(bid_decay, ask_decay)

    out["spread_absolute"] = spread
    out["spread_relative"] = _safe_div(spread, mid)
    out["spread_log"] = np.log(np.maximum(spread, 1e-8))
    out["effective_spread"] = 2.0 * np.abs(micro - mid)

    out["total_bid_volume"] = bid_20
    out["total_ask_volume"] = ask_20
    out["bid_ask_volume_ratio"] = _safe_div(bid_20, ask_20, default=1.0)
    out["depth_ratio_level_5"] = _safe_div(bid_5, ask_5, default=1.0)
    out["depth_ratio_level_10"] = _safe_div(bid_10, ask_10, default=1.0)
    out["bid_vol_top1_share"] = _safe_div(bid_1, bid_20)
    out["ask_vol_top1_share"] = _safe_div(ask_1, ask_20)

    out["micro_price_deviation"] = _safe_div(micro - mid, mid)
    out["weighted_mid_deviation_5"] = _safe_div(weighted_mid(5) - mid, mid)
    out["weighted_mid_deviation_10"] = _safe_div(weighted_mid(10) - mid, mid)

    out["bid_volume_std"] = bid_std
    out["ask_volume_std"] = ask_std
    with np.errstate(all="ignore"):
        out["bid_volume_skew"] = np.where(bid_std > 0, skew(bid_v, axis=1), 0.0)
        out["ask_volume_skew"] = np.where(ask_std > 0, skew(ask_v, axis=1), 0.0)
    out["bid_volume_decay"] = bid_decay
    out["ask_volume_decay"] = ask_decay

    returns_1 = _pct_change(mid, 1)
    out["obi_momentum_100ms"] = _diff(out["obi_level_1"], 1)
    out["obi_momentum_500ms"] = _diff(out["obi_level_1"], 5)
    out["spread_momentum_100ms"] = _diff(spread, 1)
    out["mid_price_return_100ms"] = returns_1
    out["mid_price_return_500ms"] = _pct_change(mid, 5)
    out["mid_price_volatility_5"] = (
        pd.Series(returns_1).rolling(5).std().to_numpy()  # sample std (ddof=1)
    )

    # Leading rows have no history: fill with 0 ("no change"), never with future values.
    matrix = np.column_stack([out[name] for name in FEATURE_NAMES]).astype(float)
    matrix[~np.isfinite(matrix)] = 0.0
    return pd.DataFrame(matrix, index=df.index, columns=FEATURE_NAMES)
