"""Data access for the dashboard (kept free of Streamlit so it can be unit tested)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from orderbook_ml.config import Workspace
from orderbook_ml.inference import signal_for
from orderbook_ml.storage import read_json

STALE_AFTER_SEC = 10.0


def looks_like_workspace(path: Path) -> bool:
    return any((path / name).is_dir() for name in ("snapshots", "models", "predictions"))


def discover_workspaces(default_root: Path) -> list[Path]:
    """The configured workspace plus any workspace-shaped sub-directories (e.g. ``data/demo``)."""
    found = [default_root]
    if default_root.is_dir():
        found += sorted(
            child
            for child in default_root.iterdir()
            if child.is_dir() and looks_like_workspace(child)
        )
    return found


@dataclass
class WorkspaceData:
    workspace: Workspace
    session: dict[str, Any] | None
    live: pd.DataFrame | None
    book: dict[str, Any] | None
    training: dict[str, Any] | None
    quality: dict[str, Any] | None
    shap: dict[str, Any] | None
    errors: list[str]

    @property
    def has_predictions(self) -> bool:
        return self.live is not None and not self.live.empty

    @property
    def synthetic(self) -> bool:
        for source in (self.session, self.training):
            if source and "synthetic" in source:
                return bool(source["synthetic"])
        return False

    @property
    def symbol(self) -> str | None:
        for source in (self.session, self.training, self.quality):
            if source and source.get("symbol"):
                return str(source["symbol"])
        return None

    @property
    def horizon_sec(self) -> int | None:
        for source in (self.session, self.training):
            if source and source.get("horizon_sec"):
                return int(source["horizon_sec"])
        return None


def _safe(loader: Any, path: Path, errors: list[str]) -> Any:
    try:
        return loader(path)
    except Exception as exc:  # a corrupt file must not take the whole dashboard down
        errors.append(f"Could not read {path.name}: {exc}")
        return None


def _read_live(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_workspace(root: Path) -> WorkspaceData:
    ws = Workspace(root)
    errors: list[str] = []
    return WorkspaceData(
        workspace=ws,
        session=_safe(read_json, ws.predictions / "session.json", errors),
        live=_safe(_read_live, ws.predictions / "live.parquet", errors),
        book=_safe(read_json, ws.predictions / "latest_book.json", errors),
        training=_safe(read_json, ws.reports / "training_report.json", errors),
        quality=_safe(read_json, ws.reports / "data_quality.json", errors),
        shap=_safe(read_json, ws.reports / "feature_importance.json", errors),
        errors=errors,
    )


def apply_thresholds(live: pd.DataFrame, buy: float, sell: float) -> pd.DataFrame:
    """Recomputes signals for user-chosen thresholds (the model output is unchanged)."""
    out = live.copy()
    out["signal"] = [signal_for(p, buy, sell) for p in out["prob_up"]]
    return out


def seconds_since(ts: pd.Timestamp, now: pd.Timestamp | None = None) -> float:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    return max(0.0, (now - ts).total_seconds())


def stream_state(data: WorkspaceData, now: pd.Timestamp | None = None) -> tuple[str, str]:
    """A short status label and a Streamlit colour name for the current stream."""
    if not data.has_predictions or data.live is None:
        return "No predictions yet", "gray"
    session = data.session or {}
    mode, status = session.get("mode", "live"), session.get("status", "running")
    if mode == "replay":
        return (
            ("Replaying recorded data", "blue")
            if status == "running"
            else ("Replay finished", "gray")
        )
    if status != "running":
        return f"Stream {status}", "red" if status == "failed" else "gray"
    age = seconds_since(data.live["timestamp"].iloc[-1], now)
    if age > STALE_AFTER_SEC:
        return f"Stale - no update for {age:.0f}s", "orange"
    return "Live", "green"


def depth_frame(book: dict[str, Any]) -> pd.DataFrame:
    """Cumulative depth per price level for both sides of the book."""
    rows = []
    for side, levels in (("Bids", book.get("bids", [])), ("Asks", book.get("asks", []))):
        cumulative = 0.0
        for price, qty in levels:
            cumulative += float(qty)
            rows.append(
                {
                    "side": side,
                    "price": float(price),
                    "cumulative": cumulative,
                    "quantity": float(qty),
                }
            )
    return pd.DataFrame(rows, columns=["side", "price", "cumulative", "quantity"])


def candidates_table(training: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, result in training["candidates"].items():
        test, val = result["test"], result["validation"]
        rows.append(
            {
                "Model": result["display_name"],
                "Selected": name == training["selected_model"],
                "Validation AUC": val["auc"],
                "Test AUC": test["auc"],
                "Test log loss": test["log_loss"],
                "Baseline log loss": test["baseline_log_loss"],
                "Test Brier": test["brier"],
                "Test accuracy": test["accuracy"],
                "Fit time (s)": result["fit_seconds"],
            }
        )
    return pd.DataFrame(rows)


def importance_frame(data: WorkspaceData, top: int = 15) -> tuple[pd.DataFrame, str]:
    """Feature importance, preferring SHAP when it has been computed."""
    if data.shap and data.shap.get("mean_abs_shap"):
        values, source = data.shap["mean_abs_shap"], "Mean |SHAP value| on the test set"
    elif data.training:
        champion = data.training["candidates"][data.training["selected_model"]]
        values, source = champion["importance"], "Model-native importance (share of total)"
    else:
        return pd.DataFrame(columns=["feature", "importance"]), ""
    df = pd.DataFrame(list(values.items())[:top], columns=["feature", "importance"])
    return df, source
