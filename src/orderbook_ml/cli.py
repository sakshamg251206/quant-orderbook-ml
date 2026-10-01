"""Command-line interface: ``obml <command>``.

Run ``obml --help`` for the list of commands and ``obml <command> --help`` for their options.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from orderbook_ml import __version__
from orderbook_ml.config import ConfigError, Settings, Workspace
from orderbook_ml.logging_utils import configure_logging, get_logger

logger = get_logger("cli")

DEMO_SUBDIR = "demo"


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    return settings.with_overrides(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        symbol=args.symbol.upper() if args.symbol else None,
        log_level=args.log_level,
    )


# --------------------------------------------------------------------------- commands


def cmd_collect(args: argparse.Namespace, settings: Settings) -> int:
    from orderbook_ml.collector import BinanceOrderBookStream
    from orderbook_ml.storage import SnapshotWriter

    ws = settings.workspace.ensure()
    max_snapshots = None
    if args.minutes:
        max_snapshots = int(args.minutes * 60 / settings.snapshot_interval_sec)
    logger.info(
        "Collecting %s every %d ms into %s%s (Ctrl+C to stop)",
        settings.symbol,
        settings.snapshot_interval_ms,
        ws.snapshots,
        f" for {args.minutes:g} min" if args.minutes else "",
    )

    async def collect() -> int:
        stream = BinanceOrderBookStream(settings)
        with SnapshotWriter(ws.snapshots, settings.symbol, settings.batch_size) as writer:
            try:
                async for record in stream.snapshots(max_snapshots):
                    if writer.add(record) is not None:
                        logger.info(
                            "Saved %s snapshots (mid %.2f, spread %.2f)",
                            f"{writer.rows_written:,}",
                            (record["best_bid"] + record["best_ask"]) / 2,
                            record["spread"],
                        )
            finally:
                writer.flush()
                logger.info("Stopped. %s snapshots saved.", f"{writer.rows_written:,}")
        return 0

    return asyncio.run(collect())


def cmd_simulate(args: argparse.Namespace, settings: Settings) -> int:
    from orderbook_ml.storage import SnapshotWriter
    from orderbook_ml.synthetic import generate_snapshots

    out_dir = Path(args.output) if args.output else settings.workspace.ensure().snapshots
    df = generate_snapshots(args.minutes * 60, seed=args.seed, start=args.start)
    with SnapshotWriter(out_dir, str(df["symbol"].iloc[0]), batch_size=6_000) as writer:
        for record in df.to_dict("records"):
            writer.add(record)
    logger.info(
        "Simulated %s synthetic snapshots (%g min, seed %d) into %s",
        f"{len(df):,}",
        args.minutes,
        args.seed,
        out_dir,
    )
    return 0


def cmd_validate(args: argparse.Namespace, settings: Settings) -> int:
    from orderbook_ml.storage import load_snapshots, write_json_atomic
    from orderbook_ml.validation import DataValidator

    ws = settings.workspace.ensure()
    df = load_snapshots(ws.snapshots, args.symbol)
    if df.empty:
        logger.error(
            "No snapshots in %s. Run `obml collect` or `obml simulate` first.", ws.snapshots
        )
        return 1
    validator = DataValidator()
    report = validator.report(df)
    write_json_atomic(report.to_dict(), ws.reports / "data_quality.json")
    chart = validator.plot(df, ws.reports / "data_quality.png")
    width = max(len(k) for k in report.to_dict())
    print("\nData quality report")
    for key, value in report.to_dict().items():
        print(f"  {key:<{width}}  {value}")
    print(f"\nChart: {chart}")
    return 0


def cmd_build_dataset(args: argparse.Namespace, settings: Settings) -> int:
    from orderbook_ml.dataset import build_dataset

    meta = build_dataset(settings.workspace, args.symbol, horizons_sec=args.horizons)
    print(f"\nDataset for {meta.symbol}: " + ", ".join(f"{k}={v:,}" for k, v in meta.rows.items()))
    for split, balance in meta.class_balance.items():
        rates = ", ".join(f"{k} {v['positive_rate']:.1%} up" for k, v in balance.items())
        print(f"  {split:<10} {rates}")
    return 0


def cmd_train(args: argparse.Namespace, settings: Settings) -> int:
    from orderbook_ml.modeling import train_models

    report = train_models(settings.workspace, args.target, args.models, args.seed)
    print(f"\nTarget {report['target']} - selected: {report['selected_model']}\n")
    print(f"  {'model':<20} {'val AUC':>8} {'test AUC':>9} {'test logloss':>13} {'baseline':>9}")
    for name, result in report["candidates"].items():
        val, test = result["validation"], result["test"]
        marker = "*" if name == report["selected_model"] else " "
        print(
            f"{marker} {result['display_name']:<20} {val['auc'] or float('nan'):>8.4f} "
            f"{test['auc'] or float('nan'):>9.4f} {test['log_loss']:>13.4f} "
            f"{test['baseline_log_loss']:>9.4f}"
        )
    print(f"\nModel card: {settings.workspace.reports / 'model_card.md'}")
    return 0


def cmd_explain(args: argparse.Namespace, settings: Settings) -> int:
    from orderbook_ml.explain import explain

    importance = explain(settings.workspace, max_samples=args.max_samples)
    print("\nTop features by mean |SHAP|:")
    for i, (name, value) in enumerate(list(importance.items())[:10], 1):
        print(f"  {i:>2}. {name:<28} {value:.5f}")
    return 0


def cmd_predict(args: argparse.Namespace, settings: Settings) -> int:
    from orderbook_ml.inference import PredictionSink, Predictor, run_live, run_replay
    from orderbook_ml.modeling import ModelBundle
    from orderbook_ml.storage import load_snapshots

    ws = settings.workspace.ensure()
    settings = settings.with_overrides(
        buy_threshold=args.buy_threshold, sell_threshold=args.sell_threshold
    )
    bundle = ModelBundle.load(ws.model_path)
    predictor = Predictor(bundle, settings.buy_threshold, settings.sell_threshold)
    sink = PredictionSink(ws.predictions)
    logger.info(
        "Model: %s for %s (%ss ahead); BUY >= %.2f, SELL <= %.2f",
        bundle.model_name,
        bundle.symbol,
        bundle.horizon_sec,
        settings.buy_threshold,
        settings.sell_threshold,
    )

    if args.replay:
        snapshots = load_snapshots(Path(args.replay))
        if snapshots.empty:
            logger.error("No snapshot files found at %s", args.replay)
            return 1
        replay_symbol = str(snapshots["symbol"].iloc[0])
        if replay_symbol != bundle.symbol:
            logger.error(
                "Replay data is %s but the model was trained on %s", replay_symbol, bundle.symbol
            )
            return 1
        count = run_replay(
            snapshots.to_dict("records"),
            predictor,
            sink,
            realtime_interval_sec=settings.snapshot_interval_sec if args.realtime else None,
            max_snapshots=args.max_snapshots,
        )
    else:
        from orderbook_ml.collector import BinanceOrderBookStream

        if bundle.synthetic:
            logger.error(
                "This model was trained on synthetic data; use --replay, or train on "
                "live data collected with `obml collect`."
            )
            return 1
        stream = BinanceOrderBookStream(settings).snapshots(args.max_snapshots)
        count = asyncio.run(run_live(stream, predictor, sink, settings))
    logger.info("Scored %s snapshots. Outputs in %s", f"{count:,}", ws.predictions)
    return 0


def cmd_demo(args: argparse.Namespace, settings: Settings) -> int:
    """Runs the full pipeline offline on synthetic data."""
    from orderbook_ml.storage import snapshot_files

    if not args.data_dir:
        settings = settings.with_overrides(data_dir=settings.data_dir / DEMO_SUBDIR)
    ws = settings.workspace.ensure()
    if snapshot_files(ws.snapshots) and not args.force:
        logger.error("%s already contains snapshots. Use --force to rebuild the demo.", ws.root)
        return 1
    for path in snapshot_files(ws.snapshots):
        path.unlink()

    replay_dir = ws.root / "replay"
    for path in snapshot_files(replay_dir):
        path.unlink()

    session_end = pd.Timestamp.now(tz="UTC").floor("s")
    train_start = session_end - pd.Timedelta(minutes=args.minutes + args.replay_minutes)
    replay_start = session_end - pd.Timedelta(minutes=args.replay_minutes)
    base = {"data_dir": str(ws.root), "symbol": None, "log_level": args.log_level}

    steps: list[tuple[str, Callable[[argparse.Namespace, Settings], int], dict]] = [
        (
            "Simulating training data",
            cmd_simulate,
            {"minutes": args.minutes, "seed": args.seed, "output": None, "start": str(train_start)},
        ),
        ("Building dataset", cmd_build_dataset, {"horizons": list(settings.horizons_sec)}),
        (
            "Training models",
            cmd_train,
            {"target": args.target, "models": args.models, "seed": args.seed},
        ),
        ("Explaining the champion model", cmd_explain, {"max_samples": 2_000}),
        (
            "Simulating an unseen session to replay",
            cmd_simulate,
            {
                "minutes": args.replay_minutes,
                "seed": args.seed + 1,
                "output": str(replay_dir),
                "start": str(replay_start),
            },
        ),
        (
            "Scoring the unseen session",
            cmd_predict,
            {
                "replay": str(replay_dir),
                "realtime": False,
                "max_snapshots": None,
                "buy_threshold": None,
                "sell_threshold": None,
            },
        ),
    ]
    for i, (title, func, extra) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {title}...", flush=True)
        code = func(argparse.Namespace(**base, **extra), settings)
        if code:
            return code
    hint = f" --data-dir {ws.root}" if args.data_dir else ""
    print(f"\nDemo workspace ready: {ws.root}\nOpen the dashboard with: obml dashboard{hint}")
    return 0


def cmd_status(args: argparse.Namespace, settings: Settings) -> int:
    from orderbook_ml.status import pipeline_status

    ws = settings.workspace
    print(f"Workspace: {ws.root.resolve()}\n")
    for i, step in enumerate(pipeline_status(ws), 1):
        mark = "[x]" if step.done else "[ ]"
        detail = f" - {step.detail}" if step.detail else ""
        optional = " (optional)" if step.optional else ""
        print(f"{mark} {i}. {step.title}{optional}{detail}")
        if not step.done:
            print(f"       $ {step.command}")
    return 0


def cmd_dashboard(args: argparse.Namespace, settings: Settings) -> int:
    app = Path(__file__).parent / "dashboard" / "app.py"
    env = {**os.environ, "OBML_DATA_DIR": str(settings.data_dir)}
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
        "--browser.gatherUsageStats",
        "false",
        "--client.toolbarMode",
        "viewer",
    ]
    return subprocess.call(cmd, env=env)


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="obml",
        description="Order book microstructure ML: collect -> build-dataset -> train -> predict.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", help="Workspace directory (default: $OBML_DATA_DIR or data)")
    common.add_argument("--symbol", help="Market symbol, e.g. BTCUSDT (default: $OBML_SYMBOL)")
    common.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    def add(name: str, func: Callable, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, parents=[common], help=help_text, description=help_text)
        p.set_defaults(func=func)
        return p

    p = add("collect", cmd_collect, "Record live Binance order book snapshots.")
    p.add_argument("--minutes", type=float, help="Stop after N minutes (default: run until Ctrl+C)")

    p = add("simulate", cmd_simulate, "Generate synthetic order book snapshots (offline).")
    p.add_argument("--minutes", type=float, default=30.0, help="Length of the session (default 30)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", help="Start timestamp (UTC); default ends the session now")
    p.add_argument("--output", help="Directory to write to (default: <workspace>/snapshots)")

    add("validate", cmd_validate, "Check snapshot data quality and plot it.")

    p = add("build-dataset", cmd_build_dataset, "Compute features/labels and split the data.")
    p.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        metavar="SEC",
        help="Label horizons in seconds (default: 1 5 10)",
    )

    def add_train_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--target", default="label_1s", help="Label column (default label_1s)")
        p.add_argument(
            "--models",
            nargs="+",
            default=["catboost", "xgboost", "logreg"],
            choices=["catboost", "xgboost", "logreg"],
            help="Candidates to compare",
        )
        p.add_argument("--seed", type=int, default=42)

    add_train_args(add("train", cmd_train, "Train candidate models and keep the best one."))

    p = add("explain", cmd_explain, "Explain the trained model with SHAP.")
    p.add_argument("--max-samples", type=int, default=5_000)

    p = add("predict", cmd_predict, "Score snapshots in real time (live or replay).")
    p.add_argument("--replay", metavar="PATH", help="Replay recorded snapshots instead of live")
    p.add_argument(
        "--realtime",
        action="store_true",
        help="With --replay: pace snapshots at the original sampling interval",
    )
    p.add_argument("--max-snapshots", type=int)
    p.add_argument("--buy-threshold", type=float)
    p.add_argument("--sell-threshold", type=float)

    p = add("demo", cmd_demo, "Run the whole pipeline on synthetic data (into <data-dir>/demo).")
    p.add_argument("--minutes", type=float, default=20.0, help="Training data length")
    p.add_argument("--replay-minutes", type=float, default=3.0, help="Unseen session length")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="Overwrite an existing demo workspace")
    p.add_argument("--target", default="label_1s", help="Label column (default label_1s)")
    p.add_argument(
        "--models",
        nargs="+",
        default=["catboost", "xgboost", "logreg"],
        choices=["catboost", "xgboost", "logreg"],
        help="Candidates to compare",
    )

    add("status", cmd_status, "Show which pipeline steps are complete.")

    p = add("dashboard", cmd_dashboard, "Open the Streamlit dashboard.")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--host", default="localhost")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = _settings(args)
    except ConfigError as exc:
        parser.error(str(exc))
    log_dir = Workspace(settings.data_dir).logs if args.command in {"collect", "predict"} else None
    configure_logging(settings.log_level, log_dir)

    try:
        return int(args.func(args, settings))
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        return 130
    except (FileNotFoundError, ValueError, RuntimeError, ConfigError) as exc:
        logger.debug("Command failed", exc_info=True)
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
