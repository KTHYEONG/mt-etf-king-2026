"""Research-only synthetic pre-listing history for single-stock leveraged ETFs from KRX stock daily bars."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Final, cast, get_args

import numpy as np
import polars as pl
from numpy import typing as npt

from src.arena.panel import SOURCE_COLUMNS, PanelConfig, PanelSourceError
from src.arena.simulate import Execution
from src.core.config import ConfigError, load_config
from src.core.locks import data_lock
from src.core.paths import DataPaths

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

_STOCK_TABLE: Final = "stock_daily"
_PIPELINE_LOCK: Final = "pipeline"
_STOCK_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "change_abs",
    "volume",
    "trading_value",
)
_CONFIG_KEYS: Final = frozenset(
    {
        "tag",
        "execution",
        "drag_bps_per_session",
        "max_abs_session_return",
        "min_overlap_sessions",
        "min_close_corr",
        "max_close_te_bps",
        "products",
    }
)
_PRODUCT_KEYS: Final = frozenset({"underlying", "multiple"})
_TICKER: Final = re.compile(r"[0-9A-Z]{6}")
_BPS_PER_UNIT: Final = 10_000.0


@dataclass(frozen=True)
class ProxyProduct:
    etf: str
    underlying: str
    multiple: float


@dataclass(frozen=True)
class ProxyConfig:
    tag: str
    execution: Execution
    drag_bps_per_session: float
    max_abs_session_return: float
    min_overlap_sessions: int
    min_close_corr: float
    max_close_te_bps: float
    products: tuple[ProxyProduct, ...]


@dataclass(frozen=True)
class ProxyCalibration:
    etf: str
    underlying: str
    multiple: float
    listed_first: date
    proxy_first: date | None
    proxy_last: date | None
    proxy_sessions: int
    overlap_sessions: int
    close_corr: float
    close_te_bps: float
    close_mean_bps: float
    open_corr: float
    open_te_bps: float
    open_mean_bps: float


class ProxyError(ValueError):
    """Raised on any proxy data or calibration failure; fails closed."""


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"{location} must be a finite number")
    return float(value)


def _products(value: Any) -> tuple[ProxyProduct, ...]:
    if not isinstance(value, Mapping) or not value:
        raise ConfigError("proxy.products must be a non-empty mapping")
    products: list[ProxyProduct] = []
    for etf, spec in value.items():
        if not isinstance(etf, str) or not _TICKER.fullmatch(etf):
            raise ConfigError(f"proxy.products key {etf!r} must be a 6-character ticker")
        if not isinstance(spec, Mapping) or set(spec) != _PRODUCT_KEYS:
            raise ConfigError(f"proxy.products.{etf} keys must be exactly underlying and multiple")
        underlying = spec["underlying"]
        if not isinstance(underlying, str) or not _TICKER.fullmatch(underlying):
            raise ConfigError(f"proxy.products.{etf}.underlying must be a 6-character ticker")
        multiple = _number(spec["multiple"], f"proxy.products.{etf}.multiple")
        if multiple == 0.0:
            raise ConfigError(f"proxy.products.{etf}.multiple must be non-zero")
        products.append(ProxyProduct(etf=etf, underlying=underlying, multiple=multiple))
    return tuple(sorted(products, key=lambda product: product.etf))


def load_proxy_config(name: str = "arena") -> ProxyConfig:
    """Load the `proxy` mapping of `configs/<name>.yaml`; raises ConfigError on any problem."""
    section = load_config(name).get("proxy")
    if not isinstance(section, Mapping):
        raise ConfigError(f"config {name} must define a proxy mapping")
    if set(section) != _CONFIG_KEYS:
        raise ConfigError(f"proxy keys must be exactly {sorted(_CONFIG_KEYS)}")
    tag = section["tag"]
    if not isinstance(tag, str) or not tag:
        raise ConfigError("proxy.tag must be a non-empty string")
    execution = section["execution"]
    if execution not in get_args(Execution):
        raise ConfigError("proxy.execution must be open or close")
    drag = _number(section["drag_bps_per_session"], "proxy.drag_bps_per_session")
    if drag < 0.0:
        raise ConfigError("proxy.drag_bps_per_session must be >= 0")
    max_return = _number(section["max_abs_session_return"], "proxy.max_abs_session_return")
    if not 0.0 < max_return < 1.0:
        raise ConfigError("proxy.max_abs_session_return must be in (0, 1)")
    overlap = section["min_overlap_sessions"]
    if type(overlap) is not int or overlap < 2:
        raise ConfigError("proxy.min_overlap_sessions must be an integer >= 2")
    min_corr = _number(section["min_close_corr"], "proxy.min_close_corr")
    if not 0.0 < min_corr <= 1.0:
        raise ConfigError("proxy.min_close_corr must be in (0, 1]")
    max_te = _number(section["max_close_te_bps"], "proxy.max_close_te_bps")
    if max_te <= 0.0:
        raise ConfigError("proxy.max_close_te_bps must be > 0")
    return ProxyConfig(
        tag=tag,
        execution=execution,
        drag_bps_per_session=drag,
        max_abs_session_return=max_return,
        min_overlap_sessions=overlap,
        min_close_corr=min_corr,
        max_close_te_bps=max_te,
        products=_products(section["products"]),
    )


def load_stock_silver(paths: DataPaths, cfg: PanelConfig, tickers: Collection[str]) -> pl.DataFrame:
    """Read the stock daily columns of `tickers` from silver under the shared pipeline lock."""
    source = paths.silver(_STOCK_TABLE)
    if not source.is_file():
        raise PanelSourceError(f"silver table missing: {source}")
    wanted = sorted(set(tickers))
    with data_lock(paths.lock(_PIPELINE_LOCK), shared=True, timeout_s=cfg.lock_timeout_s):
        present = pl.read_parquet_schema(source)
        missing = [column for column in _STOCK_COLUMNS if column not in present]
        if missing:
            raise PanelSourceError(f"silver table {source} lacks columns {missing}")
        return (
            pl.scan_parquet(source)
            .select(list(_STOCK_COLUMNS))
            .filter(pl.col("ticker").is_in(wanted))
            .collect()
        )


def underlying_returns(stock_rows: pl.DataFrame, ticker: str, cfg: ProxyConfig) -> pl.DataFrame:
    """Split-adjusted session returns of `ticker`; raises ProxyError on missing, duplicate or out-of-limit data."""
    rows = stock_rows.filter(pl.col("ticker") == ticker).sort("date")
    if rows.is_empty():
        raise ProxyError(f"no stock rows for {ticker}")
    if rows["date"].is_duplicated().any():
        raise ProxyError(f"duplicate stock rows for {ticker}")
    base = pl.col("close") - pl.col("change_abs")
    traded = (
        (pl.col("volume") > 0)
        & (pl.col("open") > 0)
        & (pl.col("high") > 0)
        & (pl.col("low") > 0)
        & (pl.col("close") > 0)
        & pl.col("change_abs").is_not_null()
        & (base > 0)
    ).fill_null(False)
    frame = rows.with_columns(traded=traded, base=base)
    if (frame["traded"] & frame["trading_value"].is_null()).any():
        raise ProxyError(f"null trading_value on a traded {ticker} session")
    out = frame.select(
        "date",
        "traded",
        r_cc=pl.when(pl.col("traded")).then(pl.col("close") / pl.col("base") - 1.0),
        r_on=pl.when(pl.col("traded")).then(pl.col("open") / pl.col("base") - 1.0),
        r_hi=pl.when(pl.col("traded")).then(pl.col("high") / pl.col("base") - 1.0),
        r_lo=pl.when(pl.col("traded")).then(pl.col("low") / pl.col("base") - 1.0),
        trading_value=pl.col("trading_value").cast(pl.Int64).fill_null(0),
    )
    breach = out.filter(pl.col("traded") & (pl.col("r_cc").abs() > cfg.max_abs_session_return))
    if breach.height:
        first = breach.row(0, named=True)
        raise ProxyError(f"{ticker} |r_cc| {first['r_cc']:.4f} at {first['date']} exceeds the price limit")
    return out


def _etf_rows(etf_rows: pl.DataFrame, etf: str) -> pl.DataFrame:
    rows = etf_rows.filter(pl.col("ticker") == etf).sort("date")
    if rows.is_empty():
        raise ProxyError(f"no ETF rows for {etf}")
    if rows["date"].is_duplicated().any():
        raise ProxyError(f"duplicate ETF rows for {etf}")
    return rows


def _corr(x: FloatArray, y: FloatArray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denom = math.sqrt(float(xc @ xc) * float(yc @ yc))
    return float(xc @ yc) / denom if denom > 0.0 else math.nan


def calibrate(
    etf_rows: pl.DataFrame, stock_rows: pl.DataFrame, product: ProxyProduct, cfg: ProxyConfig
) -> ProxyCalibration:
    """Compare the listed ETF returns with the multiple-scaled underlying; raises ProxyError when the fit fails closed."""
    rows = _etf_rows(etf_rows, product.etf)
    valid = (pl.col("volume") > 0) & (pl.col("open") > 0) & (pl.col("close") > 0)
    etf_returns = (
        rows.with_columns(valid=valid.fill_null(False))
        .with_columns(
            prev_close=pl.col("close").shift(1),
            prev_valid=pl.col("valid").shift(1).fill_null(False),
        )
        .filter(pl.col("valid") & pl.col("prev_valid"))
        .select(
            "date",
            etf_r_cc=pl.col("close") / pl.col("prev_close") - 1.0,
            etf_r_on=pl.col("open") / pl.col("prev_close") - 1.0,
        )
    )
    stock = underlying_returns(stock_rows, product.underlying, cfg).filter(pl.col("traded")).select("date", "r_cc", "r_on")
    joined = etf_returns.join(stock, on="date", how="inner")
    overlap = joined.height
    if overlap < cfg.min_overlap_sessions:
        raise ProxyError(f"{product.etf} overlap {overlap} is below {cfg.min_overlap_sessions}")

    multiple = product.multiple
    drag = cfg.drag_bps_per_session / _BPS_PER_UNIT
    etf_cc = joined["etf_r_cc"].to_numpy()
    und_cc = multiple * joined["r_cc"].to_numpy()
    etf_on = joined["etf_r_on"].to_numpy()
    und_on = multiple * joined["r_on"].to_numpy()
    close_resid = etf_cc - (und_cc - drag)
    open_resid = etf_on - und_on
    close_corr = _corr(etf_cc, und_cc)
    open_corr = _corr(etf_on, und_on)
    close_te = float(np.std(close_resid)) * _BPS_PER_UNIT
    close_mean = float(np.mean(close_resid)) * _BPS_PER_UNIT
    open_te = float(np.std(open_resid)) * _BPS_PER_UNIT
    open_mean = float(np.mean(open_resid)) * _BPS_PER_UNIT
    stats = (close_corr, open_corr, close_te, close_mean, open_te, open_mean)
    if not all(math.isfinite(value) for value in stats):
        raise ProxyError(f"{product.etf} calibration statistics are not finite")
    if close_corr < cfg.min_close_corr:
        raise ProxyError(f"{product.etf} close_corr {close_corr:.4f} is below {cfg.min_close_corr}")
    if close_te > cfg.max_close_te_bps:
        raise ProxyError(f"{product.etf} close_te_bps {close_te:.2f} exceeds {cfg.max_close_te_bps}")
    return ProxyCalibration(
        etf=product.etf,
        underlying=product.underlying,
        multiple=multiple,
        listed_first=rows["date"][0],
        proxy_first=None,
        proxy_last=None,
        proxy_sessions=0,
        overlap_sessions=overlap,
        close_corr=close_corr,
        close_te_bps=close_te,
        close_mean_bps=close_mean,
        open_corr=open_corr,
        open_te_bps=open_te,
        open_mean_bps=open_mean,
    )


def proxy_rows(etf_rows: pl.DataFrame, stock_rows: pl.DataFrame, product: ProxyProduct, cfg: ProxyConfig) -> pl.DataFrame:
    """Synthetic SOURCE_COLUMNS rows of `product.etf` on ETF-calendar dates before its first listed row; raises ProxyError."""
    rows = _etf_rows(etf_rows, product.etf)
    listing = rows.row(0, named=True)
    listed_first: date = listing["date"]
    stock = underlying_returns(stock_rows, product.underlying, cfg)
    calendar: list[date] = etf_rows.select(pl.col("date").unique().sort())["date"].to_list()
    pre_dates = [day for day in calendar if stock["date"][0] <= day < listed_first]
    grid = (
        pl.DataFrame({"date": [*pre_dates, listed_first]}, schema={"date": pl.Date})
        .join(stock, on="date", how="left")
    )
    if grid["traded"].is_null().any():
        missing = grid.filter(pl.col("traded").is_null())["date"][0]
        raise ProxyError(f"no underlying row for {product.underlying} on {missing}")
    if not grid["traded"][-1]:
        raise ProxyError(f"{product.underlying} is not traded on listing day {listed_first}")

    multiple = product.multiple
    drag = cfg.drag_bps_per_session / _BPS_PER_UNIT
    traded = grid["traded"].to_numpy()
    r_cc = grid["r_cc"].fill_null(0.0).to_numpy()
    growth = np.where(traded, 1.0 + multiple * r_cc - drag, 1.0)
    if not (growth > 0.0).all():
        raise ProxyError(f"{product.etf} daily growth factor must be > 0")

    listing_close = listing["close"]
    if not listing_close or listing_close <= 0.0:
        raise ProxyError(f"{product.etf} listing close must be > 0")
    tail = np.cumprod(growth[::-1])[::-1]
    close = listing_close / tail[1:]
    prev = close / growth[:-1]

    pre_traded = traded[:-1]
    r_on = grid["r_on"].fill_null(0.0).to_numpy()[:-1]
    r_hi = grid["r_hi"].fill_null(0.0).to_numpy()[:-1]
    r_lo = grid["r_lo"].fill_null(0.0).to_numpy()[:-1]
    up, down = (r_hi, r_lo) if multiple > 0.0 else (r_lo, r_hi)
    open_ = prev * (1.0 + multiple * r_on)
    high = prev * (1.0 + multiple * up)
    low = prev * (1.0 + multiple * down)
    if (pre_traded & ((open_ <= 0.0) | (high <= 0.0) | (low <= 0.0))).any():
        raise ProxyError(f"{product.etf} proxy price must be > 0")

    underlying_tv = grid["trading_value"].fill_null(0).to_numpy()[:-1].astype(np.float64)
    proxy_tv = np.where(pre_traded, np.rint(underlying_tv / abs(multiple)), 0.0)
    volume = np.where(pre_traded, np.maximum(1.0, np.rint(proxy_tv / close)), 0.0)

    count = len(pre_dates)
    frame = pl.DataFrame(
        [
            pl.Series("date", pre_dates, dtype=pl.Date),
            pl.Series("ticker", [product.etf] * count, dtype=pl.String),
            pl.Series("name", [listing["name"]] * count, dtype=pl.String),
            pl.Series("open", np.where(pre_traded, open_, np.nan), dtype=pl.Float64, nan_to_null=True),
            pl.Series("high", np.where(pre_traded, high, np.nan), dtype=pl.Float64, nan_to_null=True),
            pl.Series("low", np.where(pre_traded, low, np.nan), dtype=pl.Float64, nan_to_null=True),
            pl.Series("close", close, dtype=pl.Float64),
            pl.Series("volume", volume.astype(np.int64), dtype=pl.Int64),
            pl.Series("trading_value", proxy_tv.astype(np.int64), dtype=pl.Int64),
            pl.Series("nav", [None] * count, dtype=pl.Float64),
            pl.Series("shares_outstanding", [None] * count, dtype=pl.Int64),
            pl.Series("underlying_index_name", [listing["underlying_index_name"]] * count, dtype=pl.String),
        ]
    )
    return frame.select(list(SOURCE_COLUMNS))


def splice_proxy(
    etf_rows: pl.DataFrame, stock_rows: pl.DataFrame, cfg: ProxyConfig
) -> tuple[pl.DataFrame, tuple[ProxyCalibration, ...]]:
    """Return `etf_rows` with each product's proxy rows spliced in, sorted by (ticker, date); raises ProxyError."""
    frames = [etf_rows]
    calibrations: list[ProxyCalibration] = []
    for product in cfg.products:
        fit = calibrate(etf_rows, stock_rows, product, cfg)
        rows = proxy_rows(etf_rows, stock_rows, product, cfg)
        cal = replace(
            fit,
            proxy_first=cast(date | None, rows["date"].min()),
            proxy_last=cast(date | None, rows["date"].max()),
            proxy_sessions=rows.height,
        )
        logger.info(
            f"[DATA] proxy etf={cal.etf} underlying={cal.underlying} multiple={cal.multiple:g} "
            f"sessions={cal.proxy_sessions} overlap={cal.overlap_sessions} "
            f"close_corr={cal.close_corr:.6f} close_te_bps={cal.close_te_bps:.3f}"
        )
        calibrations.append(cal)
        frames.append(rows)
    spliced = pl.concat(frames, how="vertical").sort(["ticker", "date"])
    if spliced.select("date", "ticker").is_duplicated().any():
        raise ProxyError("duplicate (date, ticker) rows after splicing the proxy")
    return spliced, tuple(calibrations)


def _implied_factor(stock_rows: pl.DataFrame, product: ProxyProduct, cfg: ProxyConfig) -> pl.DataFrame:
    traded = underlying_returns(stock_rows, product.underlying, cfg).filter(pl.col("traded")).select("date")
    prices = stock_rows.filter(pl.col("ticker") == product.underlying).select("date", "open", "close")
    implied = traded.join(prices, on="date", how="inner").select(
        "date", factor=1.0 + product.multiple * (pl.col("close") / pl.col("open") - 1.0)
    )
    bad = implied.filter(pl.col("factor") <= 0.0)
    if bad.height:
        raise ProxyError(f"{product.etf} implied intraday factor is not positive on {bad['date'][0]}")
    return implied


def imply_open(rows: pl.DataFrame, stock_rows: pl.DataFrame, cfg: ProxyConfig) -> pl.DataFrame:
    """Set each proxied ETF's open to its multiple-implied intraday price on traded underlying sessions; raises ProxyError."""
    out = rows
    for product in cfg.products:
        factor = _implied_factor(stock_rows, product, cfg)
        implied = (
            out.filter(pl.col("ticker") == product.etf)
            .join(factor, on="date", how="inner")
            .select("date", "ticker", implied_open=pl.col("close") / pl.col("factor"))
        )
        out = (
            out.join(implied, on=["date", "ticker"], how="left")
            .with_columns(open=pl.coalesce(pl.col("implied_open"), pl.col("open")))
            .drop("implied_open")
        )
    return out
