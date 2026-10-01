"""Logging setup shared by the CLI entry points."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "orderbook_ml"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Returns a child logger of the package logger (e.g. ``orderbook_ml.collector``)."""
    short = name.removeprefix(f"{LOGGER_NAME}.")
    return logging.getLogger(f"{LOGGER_NAME}.{short}")


def configure_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configures console logging and, optionally, a size-rotated log file.

    Safe to call more than once: existing handlers are replaced rather than duplicated.
    """
    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "obml.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.propagate = False
