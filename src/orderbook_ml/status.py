"""Pipeline progress for a workspace (used by ``obml status`` and the dashboard)."""

from __future__ import annotations

from dataclasses import dataclass

from orderbook_ml.config import Workspace
from orderbook_ml.storage import available_symbols, read_json, snapshot_files


@dataclass(frozen=True)
class Step:
    title: str
    description: str
    command: str
    done: bool
    detail: str = ""
    optional: bool = False


def pipeline_status(workspace: Workspace) -> list[Step]:
    files = snapshot_files(workspace.snapshots) if workspace.snapshots.exists() else []
    symbols = available_symbols(workspace.snapshots) if files else []
    meta = read_json(workspace.processed / "dataset_meta.json")
    report = read_json(workspace.reports / "training_report.json")
    session = read_json(workspace.predictions / "session.json")

    return [
        Step(
            "Record order book snapshots",
            "Stream the Binance L2 book (or simulate one) and save a snapshot every 100 ms.",
            "obml collect --minutes 30   # or: obml simulate",
            bool(files),
            f"{len(files)} file(s) for {', '.join(symbols)}" if files else "",
        ),
        Step(
            "Build the dataset",
            "Check data quality, compute features and labels, split chronologically.",
            "obml build-dataset",
            meta is not None,
            f"{sum(meta['rows'].values()):,} labelled rows" if meta else "",
        ),
        Step(
            "Train and select a model",
            "Fit CatBoost, XGBoost and logistic regression; keep the best on validation data.",
            "obml train",
            workspace.model_path.exists() and report is not None,
            f"{report['selected_model']} for {report['target']}" if report else "",
        ),
        Step(
            "Explain the model",
            "Compute SHAP values to see which order book signals drive predictions.",
            "obml explain",
            (workspace.reports / "feature_importance.json").exists(),
            optional=True,
        ),
        Step(
            "Score snapshots in real time",
            "Run the model on the live stream (or replay recorded data) and publish signals.",
            "obml predict   # or: obml predict --replay <dir>",
            (workspace.predictions / "live.parquet").exists(),
            f"{session.get('mode', '')} session, {session.get('status', '')}" if session else "",
        ),
    ]
