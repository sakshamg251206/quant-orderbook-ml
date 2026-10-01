import pytest

from orderbook_ml.cli import main


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in ("collect", "simulate", "build-dataset", "train", "predict", "demo"):
        assert command in out


def test_invalid_config_is_a_usage_error(monkeypatch):
    monkeypatch.setenv("OBML_BUY_THRESHOLD", "abc")
    with pytest.raises(SystemExit) as exc:
        main(["status"])
    assert exc.value.code == 2


def test_missing_data_fails_with_clear_message(tmp_path, capsys):
    assert main(["build-dataset", "--data-dir", str(tmp_path)]) == 1
    assert "obml simulate" in capsys.readouterr().err


def test_demo_runs_end_to_end(tmp_path, capsys):
    code = main(
        [
            "demo",
            "--data-dir",
            str(tmp_path),
            "--minutes",
            "5",
            "--replay-minutes",
            "0.5",
            "--models",
            "logreg",
        ]
    )
    assert code == 0
    assert (tmp_path / "models" / "model.joblib").exists()
    assert (tmp_path / "predictions" / "live.parquet").exists()

    assert main(["status", "--data-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.count("[x]") == 5

    # A second run must not silently overwrite the workspace.
    assert main(["demo", "--data-dir", str(tmp_path)]) == 1


def test_live_prediction_refuses_synthetic_model(tmp_path):
    assert (
        main(
            [
                "demo",
                "--data-dir",
                str(tmp_path),
                "--minutes",
                "4",
                "--replay-minutes",
                "0.5",
                "--models",
                "logreg",
            ]
        )
        == 0
    )
    assert main(["predict", "--data-dir", str(tmp_path)]) == 1
