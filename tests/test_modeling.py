"""Training, selection, explanation and real-time inference."""

import json

import numpy as np
import pandas as pd
import pytest

from orderbook_ml.dataset import InsufficientDataError
from orderbook_ml.explain import explain
from orderbook_ml.features import MIN_HISTORY
from orderbook_ml.inference import PredictionSink, Predictor, run_replay, signal_for
from orderbook_ml.modeling import ModelBundle, classification_metrics, train_models
from orderbook_ml.synthetic import generate_snapshots

from .conftest import START


def test_metrics_include_baseline_and_handle_single_class():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, np.array([0.1, 0.4, 0.6, 0.9]))
    assert m["auc"] == 1.0 and m["accuracy"] == 1.0
    assert m["log_loss"] < m["baseline_log_loss"] == pytest.approx(np.log(2), abs=1e-4)
    single = classification_metrics(np.zeros(4), np.full(4, 0.3))
    assert single["auc"] is None  # undefined, not a misleading 0.5


def test_training_report_and_champion(trained_workspace):
    ws = trained_workspace
    report = json.loads((ws.reports / "training_report.json").read_text())
    assert report["selected_model"] in {"catboost", "logreg"}
    best = max(report["candidates"], key=lambda n: report["candidates"][n]["validation"]["auc"])
    assert report["selected_model"] == best
    # The simulator has built-in signal, so a working pipeline must beat a coin flip.
    assert report["candidates"][best]["test"]["auc"] > 0.55
    for name in ("model_card.md", "confusion_matrix.png", "calibration_curve.png"):
        assert (ws.reports / name).exists()
    assert "Synthetic data" in (ws.reports / "model_card.md").read_text()


def test_bundle_round_trip(trained_workspace):
    bundle = ModelBundle.load(trained_workspace.model_path)
    assert bundle.target == "label_1s" and bundle.horizon_sec == 1
    test = pd.read_parquet(trained_workspace.processed / "test.parquet")
    proba = bundle.predict_proba_up(test)
    assert proba.shape == (len(test),) and ((proba >= 0) & (proba <= 1)).all()


def test_bundle_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="obml train"):
        ModelBundle.load(tmp_path / "missing.joblib")
    import joblib

    joblib.dump({"not": "a bundle"}, tmp_path / "bad.joblib")
    with pytest.raises(TypeError):
        ModelBundle.load(tmp_path / "bad.joblib")


def test_training_refuses_single_class_data(trained_workspace, tmp_path):
    import shutil

    from orderbook_ml.config import Workspace

    ws = Workspace(tmp_path).ensure()
    shutil.copytree(trained_workspace.processed, ws.processed, dirs_exist_ok=True)
    train = pd.read_parquet(ws.processed / "train.parquet")
    train["label_1s"] = 0
    train.to_parquet(ws.processed / "train.parquet")
    with pytest.raises(InsufficientDataError, match="single class"):
        train_models(ws, candidates=("logreg",))


def test_unknown_target_is_rejected(trained_workspace):
    with pytest.raises(ValueError, match="Unknown target"):
        train_models(trained_workspace, target="label_7s")


def test_explain_writes_importance(trained_workspace):
    importance = explain(trained_workspace, max_samples=300)
    assert len(importance) == len(ModelBundle.load(trained_workspace.model_path).feature_names)
    assert list(importance.values()) == sorted(importance.values(), reverse=True)
    assert (trained_workspace.reports / "shap_summary.png").exists()


def test_signal_thresholds(trained_workspace):
    assert signal_for(0.70, 0.6, 0.4) == "BUY"
    assert signal_for(0.60, 0.6, 0.4) == "BUY"
    assert signal_for(0.50, 0.6, 0.4) == "HOLD"
    assert signal_for(0.40, 0.6, 0.4) == "SELL"
    with pytest.raises(ValueError):
        Predictor(
            ModelBundle.load(trained_workspace.model_path), buy_threshold=0.3, sell_threshold=0.7
        )


def test_predictor_matches_batch_features(trained_workspace):
    """Streaming inference must produce the same probabilities as offline scoring."""
    from orderbook_ml.features import compute_features

    bundle = ModelBundle.load(trained_workspace.model_path)
    snaps = generate_snapshots(10, seed=99, start=START + pd.Timedelta(hours=1))
    predictor = Predictor(bundle)
    streamed = [predictor.update(s) for s in snaps.to_dict("records")]
    assert all(r is None for r in streamed[: MIN_HISTORY - 1])
    probs = [r["prob_up"] for r in streamed if r is not None]
    offline = bundle.predict_proba_up(compute_features(snaps))[MIN_HISTORY - 1 :]
    np.testing.assert_allclose(probs, offline, rtol=1e-6)


def test_predictor_ignores_out_of_order_snapshots(trained_workspace):
    predictor = Predictor(ModelBundle.load(trained_workspace.model_path))
    snaps = generate_snapshots(2, seed=1, start=START).to_dict("records")
    for s in snaps[:10]:
        predictor.update(s)
    assert predictor.update(snaps[3]) is None


def test_replay_writes_dashboard_outputs(trained_workspace, tmp_path):
    bundle = ModelBundle.load(trained_workspace.model_path)
    sink = PredictionSink(tmp_path, window_size=100, part_size=150)
    snaps = generate_snapshots(30, seed=5, start=START + pd.Timedelta(hours=2))
    count = run_replay(snaps.to_dict("records"), Predictor(bundle), sink)

    assert count == len(snaps) - (MIN_HISTORY - 1)
    live = pd.read_parquet(tmp_path / "live.parquet")
    assert len(live) == 100 and set(live["signal"]) <= {"BUY", "HOLD", "SELL"}
    history = pd.concat(pd.read_parquet(p) for p in (tmp_path / "history").glob("*.parquet"))
    assert len(history) == count
    session = json.loads((tmp_path / "session.json").read_text())
    assert session["status"] == "finished" and session["predictions"] == count
    book = json.loads((tmp_path / "latest_book.json").read_text())
    assert len(book["bids"]) == 20
