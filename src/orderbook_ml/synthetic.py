"""Synthetic L2 order book generator.

Produces snapshots with exactly the same schema as the live collector so the whole pipeline
(validation -> features -> labels -> training -> inference -> dashboard) can be exercised offline,
in tests, in CI, or in regions where Binance market data is not reachable.

The generative model is intentionally simple and **its predictability is built in by
construction**: a hidden, slowly mean-reverting "order-flow pressure" both skews resting volume
towards one side of the book and tilts the probability of the next mid-price tick. A model that
learns "more bid volume -> price more likely to rise" will therefore score well on this data.
That demonstrates the pipeline works end to end; it says nothing about real-market performance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from orderbook_ml.orderbook import SNAPSHOT_SCHEMA

SYNTHETIC_SYMBOL = "SYNTH"


@dataclass(frozen=True)
class SyntheticMarket:
    symbol: str = SYNTHETIC_SYMBOL
    start_price: float = 60_000.0
    tick_size: float = 0.1
    levels: int = 20
    interval_ms: int = 100
    pressure_persistence: float = 0.985
    pressure_volatility: float = 0.07
    move_probability: float = 0.22
    signal_strength: float = 0.8
    base_volume: float = 1.5
    volume_decay: float = 0.08
    volume_noise: float = 0.35
    imbalance_sensitivity: float = 0.7


def is_synthetic_symbol(symbol: str | None) -> bool:
    return bool(symbol) and str(symbol).upper().startswith(SYNTHETIC_SYMBOL)


def generate_snapshots(
    duration_sec: float,
    seed: int = 42,
    start: pd.Timestamp | str | None = None,
    market: SyntheticMarket | None = None,
) -> pd.DataFrame:
    """Simulates ``duration_sec`` seconds of order book snapshots."""
    market = market or SyntheticMarket()
    n = int(duration_sec * 1000 / market.interval_ms)
    if n < 2:
        raise ValueError("duration_sec is too short to generate at least two snapshots")
    rng = np.random.default_rng(seed)

    # Hidden order-flow pressure: AR(1) squashed into (-1, 1).
    shocks = rng.normal(0.0, market.pressure_volatility, n)
    latent = np.empty(n)
    latent[0] = shocks[0]
    for t in range(1, n):
        latent[t] = market.pressure_persistence * latent[t - 1] + shocks[t]
    pressure = np.tanh(latent)

    # Mid-price ticks: moves are rare, and their direction leans with the pressure.
    moves = rng.random(n) < market.move_probability
    p_up = 0.5 + 0.5 * market.signal_strength * pressure
    direction = np.where(rng.random(n) < p_up, 1, -1)
    tick_steps = np.where(moves, direction, 0)
    tick_steps[0] = 0
    bid_ticks = round(market.start_price / market.tick_size) + np.cumsum(tick_steps)
    spread_ticks = rng.choice([1, 1, 1, 1, 2, 2, 3], size=n)

    # Resting volume: exponential decay away from the touch, skewed by pressure near the top.
    level_idx = np.arange(market.levels)
    shape = market.base_volume * np.exp(-market.volume_decay * level_idx)
    skew = market.imbalance_sensitivity * np.exp(-0.15 * level_idx)
    noise_b = rng.lognormal(0.0, market.volume_noise, (n, market.levels))
    noise_a = rng.lognormal(0.0, market.volume_noise, (n, market.levels))
    bid_vol = np.round(shape * noise_b * (1 + skew * pressure[:, None]), 5)
    ask_vol = np.round(shape * noise_a * (1 - skew * pressure[:, None]), 5)
    bid_vol = np.maximum(bid_vol, 0.00001)
    ask_vol = np.maximum(ask_vol, 0.00001)

    tick = market.tick_size
    decimals = max(0, round(-np.log10(tick)) + 1)
    bid_prices = np.round((bid_ticks[:, None] - level_idx) * tick, decimals)
    ask_prices = np.round((bid_ticks[:, None] + spread_ticks[:, None] + level_idx) * tick, decimals)

    if start is None:
        start_ts = pd.Timestamp.now(tz="UTC").floor("s") - pd.Timedelta(seconds=duration_sec)
    else:
        start_ts = pd.Timestamp(start)
        start_ts = (
            start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
        )
    timestamps = pd.date_range(start_ts, periods=n, freq=f"{market.interval_ms}ms")

    bids_json = [
        json.dumps(np.stack([bid_prices[i], bid_vol[i]], axis=1).tolist()) for i in range(n)
    ]
    asks_json = [
        json.dumps(np.stack([ask_prices[i], ask_vol[i]], axis=1).tolist()) for i in range(n)
    ]

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": market.symbol,
            "last_update_id": np.arange(1, n + 1, dtype=np.int64),
            "bids_json": bids_json,
            "asks_json": asks_json,
            "best_bid": bid_prices[:, 0],
            "best_ask": ask_prices[:, 0],
            "spread": np.round(ask_prices[:, 0] - bid_prices[:, 0], decimals),
            "total_bid_volume": bid_vol.sum(axis=1),
            "total_ask_volume": ask_vol.sum(axis=1),
        }
    )
    return df[SNAPSHOT_SCHEMA.names]
