from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from orderbook_ml.dashboard.data import (
    apply_thresholds,
    candidates_table,
    depth_frame,
    discover_workspaces,
    importance_frame,
    load_workspace,
    stream_state,
)
from orderbook_ml.inference import PredictionSink, Predictor, run_replay
from orderbook_ml.modeling import ModelBundle
from orderbook_ml.synthetic import generate_snapshots

from .conftest import START

APP = Path(__file__).parents[1] / "src" / "orderbook_ml" / "dashboard" / "app.py"


@pytest.fixture(scope="module")
def scored_workspace(trained_workspace):
    bundle = ModelBundle.load(trained_workspace.model_path)
    snaps = generate_snapshots(20, seed=11, start=START + pd.Timedelta(hours=3))
    run_replay(
        snaps.to_dict("records"), Predictor(bundle), PredictionSink(trained_workspace.predictions)
    )
    return trained_workspace


def test_empty_workspace_has_no_data(tmp_path):
    data = load_workspace(tmp_path)
    assert not data.has_predictions and data.training is None and not data.errors
    assert stream_state(data) == ("No predictions yet", "gray")


def test_corrupt_file_is_reported_not_raised(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "training_report.json").write_text("{broken")
    data = load_workspace(tmp_path)
    assert data.training is None and len(data.errors) == 1


def test_discover_workspaces(tmp_path):
    (tmp_path / "demo" / "models").mkdir(parents=True)
    (tmp_path / "unrelated").mkdir()
    assert discover_workspaces(tmp_path) == [tmp_path, tmp_path / "demo"]


def test_loaded_workspace(scored_workspace):
    data = load_workspace(scored_workspace.root)
    assert data.has_predictions and data.synthetic and data.symbol == "SYNTH"
    assert data.horizon_sec == 1
    assert stream_state(data) == ("Replay finished", "gray")
    assert set(candidates_table(data.training)["Model"]) == {"CatBoost", "Logistic regression"}
    imp, source = importance_frame(data)
    assert not imp.empty and source

    depth = depth_frame(data.book)
    bids = depth[depth["side"] == "Bids"]
    assert bids["cumulative"].is_monotonic_increasing


def test_live_stream_goes_stale(scored_workspace):
    data = load_workspace(scored_workspace.root)
    data.session = {**data.session, "mode": "live", "status": "running"}
    last = data.live["timestamp"].iloc[-1]
    assert stream_state(data, now=last + pd.Timedelta(seconds=1))[0] == "Live"
    assert stream_state(data, now=last + pd.Timedelta(minutes=5))[0].startswith("Stale")


def test_apply_thresholds_relabels_without_touching_probabilities(scored_workspace):
    live = load_workspace(scored_workspace.root).live
    everything_buys = apply_thresholds(live, buy=0.01, sell=0.005)
    assert (everything_buys["signal"] == "BUY").all()
    pd.testing.assert_series_equal(everything_buys["prob_up"], live["prob_up"])


def test_app_renders_onboarding_for_empty_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("OBML_DATA_DIR", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("Get started" in s.value for s in at.subheader)


def test_app_renders_populated_workspace(scored_workspace, monkeypatch):
    monkeypatch.setenv("OBML_DATA_DIR", str(scored_workspace.root))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    labels = {m.label for m in at.metric}
    assert {"Mid price", "P(up in 1s)", "ROC-AUC", "Quality score"} <= labels
