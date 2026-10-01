"""Storage, synthetic data, validation, features, labels and dataset splitting."""

import json

import numpy as np
import pandas as pd
import pytest

from orderbook_ml.config import ConfigError, Settings, Workspace
from orderbook_ml.dataset import InsufficientDataError, build_dataset, chronological_split
from orderbook_ml.features import FEATURE_NAMES, compute_features
from orderbook_ml.labels import generate_labels, horizon_from_label
from orderbook_ml.orderbook import SNAPSHOT_SCHEMA
from orderbook_ml.storage import available_symbols, load_snapshots
from orderbook_ml.synthetic import generate_snapshots
from orderbook_ml.validation import DataValidator

from .conftest import START, write_snapshots

# ----------------------------------------------------------------------------- config


def test_settings_validation():
    with pytest.raises(ConfigError):
        Settings(symbol="btc/usdt")
    with pytest.raises(ConfigError):
        Settings(buy_threshold=0.3, sell_threshold=0.6)
    with pytest.raises(ConfigError):
        Settings(binance_ws_url="ws://insecure")
    assert Settings().with_overrides(symbol=None, depth=10).depth == 10


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OBML_SYMBOL", "ethusdt")
    monkeypatch.setenv("OBML_SNAPSHOT_INTERVAL_MS", "250")
    settings = Settings.from_env(env_file=None)
    assert settings.symbol == "ETHUSDT" and settings.snapshot_interval_sec == 0.25
    monkeypatch.setenv("OBML_DEPTH", "lots")
    with pytest.raises(ConfigError, match="OBML_DEPTH"):
        Settings.from_env(env_file=None)


# ----------------------------------------------------------------------------- synthetic


def test_synthetic_snapshots_are_valid_and_deterministic(snapshots):
    assert list(snapshots.columns) == SNAPSHOT_SCHEMA.names
    assert len(snapshots) == 3_600
    assert (snapshots["best_ask"] > snapshots["best_bid"]).all()
    assert snapshots["timestamp"].is_monotonic_increasing
    again = generate_snapshots(360, seed=7, start=START)
    pd.testing.assert_frame_equal(snapshots, again)


# ----------------------------------------------------------------------------- storage


def test_storage_round_trip_filters_symbols_and_dedupes(tmp_path, snapshots):
    write_snapshots(snapshots.iloc[:1500], tmp_path)
    write_snapshots(snapshots.iloc[1000:2000], tmp_path)  # overlapping batch
    other = snapshots.iloc[:10].assign(symbol="ETHUSDT")
    write_snapshots(other, tmp_path)

    assert available_symbols(tmp_path) == ["ETHUSDT", "SYNTH"]
    loaded = load_snapshots(tmp_path, "SYNTH")
    assert len(loaded) == 2_000
    assert loaded["timestamp"].is_unique and loaded["timestamp"].is_monotonic_increasing
    with pytest.raises(ValueError, match="several symbols"):
        load_snapshots(tmp_path)


def test_loading_empty_directory_returns_empty_frame(tmp_path):
    assert load_snapshots(tmp_path).empty


# ----------------------------------------------------------------------------- validation


def test_validator_flags_gaps_crossed_books_and_empty_sides(snapshots):
    df = snapshots.iloc[:100].copy()
    df.loc[10:, "timestamp"] += pd.Timedelta(seconds=2)  # one 2.1 s gap
    df.loc[20, ["best_bid", "best_ask", "spread"]] = [101.0, 100.0, -1.0]
    df.loc[30, "total_ask_volume"] = 0.0

    report = DataValidator().report(df)
    assert report.timestamp_gaps == 1 and report.largest_gap_sec == pytest.approx(2.1)
    assert report.crossed_books == 1 and report.empty_sides == 1
    assert report.quality_score == pytest.approx(97.0)


def test_validator_handles_empty_frame():
    assert DataValidator().report(pd.DataFrame()).total_snapshots == 0


# ----------------------------------------------------------------------------- features


def _book_row(ts, bids, asks):
    return {
        "timestamp": ts,
        "symbol": "X",
        "bids_json": json.dumps(bids),
        "asks_json": json.dumps(asks),
        "best_bid": bids[0][0],
        "best_ask": asks[0][0],
        "spread": asks[0][0] - bids[0][0],
        "total_bid_volume": sum(q for _, q in bids),
        "total_ask_volume": sum(q for _, q in asks),
    }


def test_feature_values_on_a_hand_built_book():
    row = _book_row(START, [[100.0, 3.0], [99.0, 1.0]], [[101.0, 1.0], [102.0, 1.0]])
    feats = compute_features(pd.DataFrame([row])).iloc[0]
    assert feats["obi_level_1"] == pytest.approx(0.5)  # (3 - 1) / 4
    assert feats["obi_level_5"] == pytest.approx(1 / 3)  # (4 - 2) / 6
    assert feats["spread_absolute"] == pytest.approx(1.0)
    assert feats["spread_relative"] == pytest.approx(1 / 100.5)
    # micro price = (100 * 1 + 101 * 3) / 4 = 100.75 -> deviation from mid 100.5
    assert feats["micro_price_deviation"] == pytest.approx(0.25 / 100.5)
    assert feats["bid_vol_top1_share"] == pytest.approx(0.75)


def test_features_are_finite_and_complete(snapshots):
    feats = compute_features(snapshots.iloc[:500])
    assert list(feats.columns) == FEATURE_NAMES
    assert np.isfinite(feats.to_numpy()).all()


def test_features_have_no_look_ahead(snapshots):
    """Changing the future must not change features of the past."""
    base = snapshots.iloc[:300].reset_index(drop=True)
    altered = base.copy()
    altered.loc[200:, "bids_json"] = altered.loc[200:, "asks_json"]
    a, b = compute_features(base), compute_features(altered)
    pd.testing.assert_frame_equal(a.iloc[:200], b.iloc[:200])


def test_empty_input_yields_empty_features():
    assert compute_features(pd.DataFrame(columns=SNAPSHOT_SCHEMA.names)).empty


# ----------------------------------------------------------------------------- labels


def _price_path(mids, freq="100ms"):
    ts = pd.date_range(START, periods=len(mids), freq=freq)
    return pd.DataFrame(
        {"timestamp": ts, "best_bid": np.array(mids) - 0.5, "best_ask": np.array(mids) + 0.5}
    )


def test_labels_look_exactly_one_horizon_ahead():
    mids = [100.0] * 10 + [101.0] * 10 + [100.0] * 15
    labels = generate_labels(_price_path(mids), horizons_sec=[1])["label_1s"]
    assert labels.iloc[0] == 1  # t=0.0 -> t=1.0 price 101
    assert labels.iloc[10] == 0  # t=1.0 -> t=2.0 price 100 (down)
    assert labels.iloc[20] == 0  # flat counts as "not up"
    assert labels.iloc[-10:].isna().all()  # no future price within reach


def test_labels_are_undefined_across_data_gaps():
    df = _price_path([100.0] * 20)
    df.loc[10:, "timestamp"] += pd.Timedelta(seconds=5)
    labels = generate_labels(df, horizons_sec=[1], max_tolerance_ms=500)["label_1s"]
    assert labels.iloc[:10].isna().all()


def test_labels_reject_unsorted_input():
    df = _price_path([100.0, 101.0]).iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        generate_labels(df)


def test_horizon_from_label():
    assert horizon_from_label("label_10s") == 10
    with pytest.raises(ValueError):
        horizon_from_label("price")


# ----------------------------------------------------------------------------- dataset


def test_chronological_split_with_embargo():
    df = pd.DataFrame({"timestamp": pd.date_range(START, periods=1000, freq="100ms")})
    parts = chronological_split(df, 0.7, 0.15, embargo=pd.Timedelta(seconds=5))
    train, val, test = parts["train"], parts["validation"], parts["test"]
    assert train["timestamp"].max() < val["timestamp"].min() - pd.Timedelta(
        seconds=5
    ) + pd.Timedelta("1ns")
    assert val["timestamp"].max() < test["timestamp"].min() - pd.Timedelta(
        seconds=5
    ) + pd.Timedelta("1ns")
    assert len(train) == 700 - 50 and len(test) == 150


def test_build_dataset_requires_enough_data(tmp_path, snapshots):
    ws = Workspace(tmp_path).ensure()
    write_snapshots(snapshots.iloc[:500], ws.snapshots)
    with pytest.raises(InsufficientDataError, match="500 snapshots"):
        build_dataset(ws)


def test_build_dataset_outputs(trained_workspace):
    ws = trained_workspace
    meta = json.loads((ws.processed / "dataset_meta.json").read_text())
    assert meta["symbol"] == "SYNTH" and meta["synthetic"] is True
    assert meta["feature_names"] == FEATURE_NAMES
    train = pd.read_parquet(ws.processed / "train.parquet")
    val = pd.read_parquet(ws.processed / "validation.parquet")
    assert train["timestamp"].max() + pd.Timedelta(seconds=10) < val["timestamp"].min()
    assert train[meta["label_columns"]].notna().all().all()
    assert (ws.reports / "data_quality.json").exists()
