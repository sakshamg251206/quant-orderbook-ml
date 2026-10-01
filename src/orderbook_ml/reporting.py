"""Static training reports: PNG charts and a Markdown model card."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def write_training_plots(report: dict[str, Any], out_dir: Path) -> list[Path]:
    champion = report["selected_model"]
    result = report["candidates"][champion]
    title = f"{result['display_name']} - {report['target']}"
    written = [
        _plot_confusion(
            result["test"]["confusion_matrix"], title, out_dir / "confusion_matrix.png"
        ),
        _plot_calibration(report, out_dir / "calibration_curve.png"),
    ]
    if result["history"].get("validation_logloss"):
        written.append(_plot_history(result["history"], title, out_dir / "training_history.png"))
    return written


def _plot_confusion(cm: list[list[int]], title: str, path: Path) -> Path:
    matrix = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        color = "white" if value > matrix.max() / 2 else "black"
        ax.text(j, i, f"{value:,}", ha="center", va="center", color=color, fontweight="bold")
    ax.set_xticks([0, 1], ["Pred: not up", "Pred: up"])
    ax.set_yticks([0, 1], ["True: not up", "True: up"])
    ax.set_title(f"Test confusion matrix\n{title}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_calibration(report: dict[str, Any], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="Perfect calibration")
    for name, result in report["candidates"].items():
        bins = result["test"]["calibration"]
        ax.plot(
            [b["mean_predicted"] for b in bins],
            [b["observed_rate"] for b in bins],
            marker="o",
            lw=2.2 if name == report["selected_model"] else 1,
            label=result["display_name"],
        )
    ax.set_xlabel("Mean predicted P(up)")
    ax.set_ylabel("Observed frequency of up moves")
    ax.set_title(f"Test calibration - {report['target']}", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_history(history: dict[str, Any], title: str, path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history["train_logloss"], label="Train")
    ax1.plot(history["validation_logloss"], label="Validation")
    ax1.axvline(history["best_iteration"], ls=":", color="grey", label="Best iteration")
    ax1.set_title("Log loss")
    ax1.set_xlabel("Boosting iteration")
    ax1.legend()
    ax1.grid(alpha=0.3, ls=":")
    ax2.plot(history["validation_auc"], color="#16a34a")
    ax2.axvline(history["best_iteration"], ls=":", color="grey")
    ax2.set_title("Validation ROC-AUC")
    ax2.set_xlabel("Boosting iteration")
    ax2.grid(alpha=0.3, ls=":")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_model_card(report: dict[str, Any], path: Path) -> Path:
    champion = report["selected_model"]
    best = report["candidates"][champion]
    test = best["test"]
    rows = [
        [
            r["display_name"] + (" (selected)" if name == champion else ""),
            r["validation"]["auc"],
            r["test"]["auc"],
            r["test"]["log_loss"],
            r["test"]["baseline_log_loss"],
            r["test"]["brier"],
            r["test"]["accuracy"],
            r["fit_seconds"],
        ]
        for name, r in report["candidates"].items()
    ]
    table = markdown_table(
        [
            "Model",
            "Val AUC",
            "Test AUC",
            "Test log loss",
            "Baseline log loss",
            "Test Brier",
            "Test accuracy",
            "Fit (s)",
        ],
        rows,
    )
    top_features = list(best["importance"].items())[:10]
    importance_md = "\n".join(
        f"{i}. `{name}` ({share:.1%})" for i, (name, share) in enumerate(top_features, 1)
    )
    synthetic_note = (
        "\n> **Synthetic data.** This model was trained on simulated order books whose "
        "predictability is built into the simulator. These numbers validate the pipeline, "
        "not a trading edge.\n"
        if report["synthetic"]
        else ""
    )
    ranges = report["time_ranges"]
    split_rows = "\n".join(
        f"| {name.title()} | {report['rows'][name]:,} | {ranges[name][0]} | {ranges[name][1]} |"
        for name in ("train", "validation", "test")
    )

    content = f"""# Model card - {best["display_name"]} for `{report["target"]}`
{synthetic_note}
_Generated automatically by `obml train` on {report["trained_at"]}._

## What the model predicts

The probability that the **mid price of {report["symbol"]} will be strictly higher
{report["horizon_sec"]} second(s) from now** than it is at the current snapshot
(label 1 = up, 0 = unchanged or down).

## Data

| Split | Rows | From (UTC) | To (UTC) |
|---|---|---|---|
{split_rows}

Splits are contiguous in time (no shuffling) and separated by a {report["embargo_sec"]:g}s
embargo so that no training label peeks into the evaluation period.
{len(report["feature_names"])} order book features are used.

## Candidates

Selection rule: {report["selection_rule"]}. The test split was not used for selection.

{table}

*Baseline log loss* is the score of always predicting the test set's base rate
({test["positive_rate"]:.1%} up moves); a useful model must beat it.

## Most influential features (model-native importance)

{importance_md}

## Intended use

Research and education on short-horizon market microstructure. The output is a probability,
not an order: it ignores fees, latency, queue position and slippage, and has not been
evaluated as a trading strategy.

## Limitations

- Trained on a single symbol and a single recording window; market regimes change.
- "Up" vs "not up" folds flat and down moves together; in quiet markets most labels are 0.
- Metrics come from one chronological split, not a walk-forward backtest.
"""
    path.write_text(content, encoding="utf-8")
    return path
