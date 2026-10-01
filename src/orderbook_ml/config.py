"""Runtime configuration.

All settings come from environment variables (optionally loaded from a ``.env`` file) and are
validated once at startup, so a typo such as ``OBML_DEPTH=abc`` fails fast with a clear message
instead of surfacing later as a confusing runtime error.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}$")
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class ConfigError(ValueError):
    """Raised when a configuration value is missing or invalid."""


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Workspace:
    """Directory layout for one data workspace.

    Every artifact produced by the pipeline (raw snapshots, datasets, models, reports and
    predictions) lives under a single root, so a synthetic demo workspace and a live Binance
    workspace can never be mixed up.
    """

    root: Path

    @property
    def snapshots(self) -> Path:
        return self.root / "snapshots"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def model_path(self) -> Path:
        return self.models / "model.joblib"

    def ensure(self) -> Workspace:
        for path in (
            self.snapshots,
            self.processed,
            self.models,
            self.reports,
            self.predictions,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class Settings:
    """Validated application settings."""

    data_dir: Path = Path("data")
    symbol: str = "BTCUSDT"
    depth: int = 20
    snapshot_interval_ms: int = 100
    batch_size: int = 600
    binance_rest_url: str = "https://api.binance.com"
    binance_ws_url: str = "wss://stream.binance.com:9443"
    buy_threshold: float = 0.60
    sell_threshold: float = 0.40
    log_level: str = "INFO"
    horizons_sec: tuple[int, ...] = field(default=(1, 5, 10))

    def __post_init__(self) -> None:
        if not _SYMBOL_RE.match(self.symbol):
            raise ConfigError(
                f"Symbol must be 2-20 upper-case letters/digits (e.g. BTCUSDT), got {self.symbol!r}"
            )
        if not 1 <= self.depth <= 1000:
            raise ConfigError(f"Depth must be between 1 and 1000, got {self.depth}")
        if self.snapshot_interval_ms < 10:
            raise ConfigError(
                f"Snapshot interval must be >= 10 ms, got {self.snapshot_interval_ms}"
            )
        if self.batch_size < 1:
            raise ConfigError(f"Batch size must be positive, got {self.batch_size}")
        if not 0.0 < self.sell_threshold <= self.buy_threshold < 1.0:
            raise ConfigError(
                "Thresholds must satisfy 0 < sell_threshold <= buy_threshold < 1 "
                f"(got sell={self.sell_threshold}, buy={self.buy_threshold})"
            )
        if self.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ConfigError(f"Unknown log level {self.log_level!r}")
        if not self.horizons_sec or any(h <= 0 for h in self.horizons_sec):
            raise ConfigError(f"Horizons must be positive seconds, got {self.horizons_sec}")
        for name, secure, insecure in (
            ("binance_rest_url", "https", "http"),
            ("binance_ws_url", "wss", "ws"),
        ):
            url = urlparse(getattr(self, name))
            local = url.hostname in _LOCAL_HOSTS
            if url.scheme != secure and not (url.scheme == insecure and local):
                raise ConfigError(
                    f"{name} must use {secure}:// (plain {insecure}:// is only allowed for "
                    f"localhost), got {getattr(self, name)!r}"
                )

    @property
    def snapshot_interval_sec(self) -> float:
        return self.snapshot_interval_ms / 1000.0

    @property
    def workspace(self) -> Workspace:
        return Workspace(self.data_dir)

    def with_overrides(self, **overrides: object) -> Settings:
        """Returns a copy with the given non-``None`` values replaced (and re-validated)."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean) if clean else self  # type: ignore[arg-type]

    @classmethod
    def from_env(cls, env_file: str | os.PathLike[str] | None = ".env") -> Settings:
        """Builds settings from environment variables (and ``.env`` if present)."""
        if env_file is not None and Path(env_file).exists():
            load_dotenv(env_file, override=False)

        defaults = cls()
        return cls(
            data_dir=Path(_env("OBML_DATA_DIR", str(defaults.data_dir))),
            symbol=_env("OBML_SYMBOL", defaults.symbol).upper(),
            depth=_env_int("OBML_DEPTH", defaults.depth),
            snapshot_interval_ms=_env_int(
                "OBML_SNAPSHOT_INTERVAL_MS", defaults.snapshot_interval_ms
            ),
            batch_size=_env_int("OBML_BATCH_SIZE", defaults.batch_size),
            binance_rest_url=_env("OBML_BINANCE_REST_URL", defaults.binance_rest_url).rstrip("/"),
            binance_ws_url=_env("OBML_BINANCE_WS_URL", defaults.binance_ws_url).rstrip("/"),
            buy_threshold=_env_float("OBML_BUY_THRESHOLD", defaults.buy_threshold),
            sell_threshold=_env_float("OBML_SELL_THRESHOLD", defaults.sell_threshold),
            log_level=_env("OBML_LOG_LEVEL", defaults.log_level).upper(),
        )
