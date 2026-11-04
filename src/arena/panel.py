"""Daily ETF panel: traded vs reference prices split, explicit gap rows, returns only between real endpoints."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

import polars as pl

from src.core.config import ConfigError, load_config
from src.core.locks import data_lock
from src.core.paths import DataPaths

SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "ticker",
    "name",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trading_value",
    "nav",
    "shares_outstanding",
    "underlying_index_name",
)
_SOURCE_SCHEMA: Final = pl.Schema(
    {
        "date": pl.Date,
        "ticker": pl.String,
        "name": pl.String,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Int64,
        "trading_value": pl.Int64,
        "nav": pl.Float64,
        "shares_outstanding": pl.Int64,
        "underlying_index_name": pl.String,
    }
)
_TABLE: Final = "etf_daily"
_PIPELINE_LOCK: Final = "pipeline"
_PANEL_KEYS: Final = ("lock_timeout_s", "nav_divergence_max")


@dataclass(frozen=True)
class PanelConfig:
    lock_timeout_s: float
    nav_divergence_max: float


@dataclass(frozen=True)
class Panel:
    start: date
    end: date
    sessions: tuple[date, ...]
    frame: pl.DataFrame
    missing_tickers: tuple[str, ...]


class PanelSourceError(RuntimeError):
    """Raised when the silver ETF table is absent or lacks a required column."""


def _positive_float(section: Mapping[str, Any], key: str) -> float:
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"panel.{key} must be a number")
    if value <= 0:
        raise ConfigError(f"panel.{key} must be > 0")
    return float(value)


def load_panel_config(name: str = "arena") -> PanelConfig:
    """Load the `panel` mapping of `configs/<name>.yaml`; raises ConfigError on any problem."""
    section = load_config(name).get("panel")
    if not isinstance(section, Mapping):
        raise ConfigError(f"config {name} must define a panel mapping")
    if set(section) != set(_PANEL_KEYS):
        raise ConfigError(f"panel keys must be exactly {sorted(_PANEL_KEYS)}")
    return PanelConfig(
        lock_timeout_s=_positive_float(section, "lock_timeout_s"),
        nav_divergence_max=_positive_float(section, "nav_divergence_max"),
    )


def load_etf_silver(paths: DataPaths, cfg: PanelConfig) -> pl.DataFrame:
    """Read the SOURCE_COLUMNS of the silver ETF table under the shared pipeline lock."""
    source = paths.silver(_TABLE)
    if not source.is_file():
        raise PanelSourceError(f"silver table missing: {source}")
    with data_lock(paths.lock(_PIPELINE_LOCK), shared=True, timeout_s=cfg.lock_timeout_s):
        present = pl.read_parquet_schema(source)
        missing = [column for column in SOURCE_COLUMNS if column not in present]
        if missing:
            raise PanelSourceError(f"silver table {source} lacks columns {missing}")
        return pl.read_parquet(source, columns=list(SOURCE_COLUMNS))


def _check_source_schema(rows: pl.DataFrame) -> None:
    for name, dtype in _SOURCE_SCHEMA.items():
        if name not in rows.schema:
            raise ValueError(f"rows lack column {name}")
        if rows.schema[name] != dtype:
            raise ValueError(f"column {name} must be {dtype}, got {rows.schema[name]}")


def _previous(name: str) -> pl.Expr:
    return pl.col(name).shift(1).over("ticker")


def build_panel(
    rows: pl.DataFrame, tickers: Collection[str], start: date, end: date, cfg: PanelConfig
) -> Panel:
    """Build the per-session panel of `tickers` over [start, end]; raises ValueError on malformed or ambiguous input."""
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    if not tickers:
        raise ValueError("tickers must be non-empty")
    _check_source_schema(rows)
    rows = rows.filter(pl.col("date") <= end).select(SOURCE_COLUMNS)
    if rows.select("date", "ticker").is_duplicated().any():
        raise ValueError("duplicate (date, ticker) rows")

    wanted = pl.DataFrame({"ticker": sorted(set(tickers))}, schema={"ticker": pl.String})
    requested = rows.join(wanted, on="ticker", how="semi")
    bad = requested.filter(pl.col("close").is_null() | (pl.col("close") <= 0))
    if bad.height:
        first = bad.sort(["date", "ticker"]).row(0, named=True)
        raise ValueError(f"null or non-positive close at {first['date']} for {first['ticker']}")

    calendar = rows.select(pl.col("date").unique().sort())
    sessions = tuple(
        calendar.filter((pl.col("date") >= start) & (pl.col("date") <= end))["date"].to_list()
    )
    spans = requested.group_by("ticker").agg(
        span_start=pl.col("date").min(), span_end=pl.col("date").max()
    )
    missing = wanted.join(spans.select("ticker"), on="ticker", how="anti").sort("ticker")["ticker"].to_list()

    grid = (
        spans.join(calendar, how="cross")
        .filter((pl.col("date") >= pl.col("span_start")) & (pl.col("date") <= pl.col("span_end")))
        .select("ticker", "date")
        .join(requested.with_columns(listed=pl.lit(True)), on=["ticker", "date"], how="left")
        .with_columns(pl.col("listed").fill_null(False))
        .sort(["ticker", "date"])
    )
    traded = (
        pl.col("listed")
        & (pl.col("volume") > 0)
        & (pl.col("open") > 0)
        & (pl.col("high") > 0)
        & (pl.col("low") > 0)
    ).fill_null(False)
    grid = grid.with_columns(traded=traded).with_columns(
        open_t=pl.when(pl.col("traded")).then(pl.col("open")),
        high_t=pl.when(pl.col("traded")).then(pl.col("high")),
        low_t=pl.when(pl.col("traded")).then(pl.col("low")),
        close_t=pl.when(pl.col("traded")).then(pl.col("close")),
        close_ref=pl.when(pl.col("listed")).then(pl.col("close")),
        nav_v=pl.when(pl.col("listed") & (pl.col("nav") > 0)).then(pl.col("nav")),
    )
    grid = grid.with_columns(
        r_cc=pl.col("close_t") / _previous("close_t") - 1.0,
        r_on=pl.col("open_t") / _previous("close_t") - 1.0,
        r_id=pl.col("close_t") / pl.col("open_t") - 1.0,
        r_ref=pl.col("close_ref") / _previous("close_ref") - 1.0,
        r_nav=pl.col("nav_v") / _previous("nav_v") - 1.0,
    ).with_columns(
        nav_divergence=((pl.col("r_cc") - pl.col("r_nav")).abs() > cfg.nav_divergence_max).fill_null(False),
    )
    frame = grid.filter(pl.col("date") >= start).select(
        "date",
        "ticker",
        "name",
        "listed",
        "traded",
        pl.col("open_t").alias("open"),
        pl.col("high_t").alias("high"),
        pl.col("low_t").alias("low"),
        pl.col("close_t").alias("close"),
        "close_ref",
        pl.col("nav_v").alias("nav"),
        "volume",
        "trading_value",
        "r_cc",
        "r_on",
        "r_id",
        "r_ref",
        "r_nav",
        "nav_divergence",
    )
    return Panel(start=start, end=end, sessions=sessions, frame=frame, missing_tickers=tuple(missing))
