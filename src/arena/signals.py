"""Causal signal helpers: each output at row d reads only rows at or before d and is null while history is short."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from typing import Final

import numpy as np
import polars as pl
from numpy import typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from src.arena.panel import Panel

FloatArray = npt.NDArray[np.float64]
_CLIP: Final = 1.0


def wide(panel: Panel, column: str, tickers: Sequence[str]) -> pl.DataFrame:
    """Return one row per panel session with `date` and one column per ticker; absent tickers are all null."""
    frame = pl.DataFrame({"date": list(panel.sessions)}, schema={"date": pl.Date})
    for ticker in dict.fromkeys(tickers):
        column_frame = panel.frame.filter(pl.col("ticker") == ticker).select("date", pl.col(column).alias(ticker))
        frame = frame.join(column_frame, on="date", how="left")
    return frame


def _float(series: pl.Series) -> FloatArray:
    values: FloatArray = series.cast(pl.Float64).fill_null(float("nan")).to_numpy()
    return values


def _output(values: FloatArray, name: str) -> pl.Series:
    return pl.Series(name, values, dtype=pl.Float64, nan_to_null=True)


def _trailing(values: FloatArray, window: int) -> FloatArray:
    """Row d holds values[d-window+1 .. d]; rows without a full window hold NaN."""
    padded = np.concatenate([np.full(window - 1, np.nan), values])
    windows: FloatArray = sliding_window_view(padded, window)
    return windows


def _trailing_std(values: FloatArray, window: int) -> FloatArray:
    windows = _trailing(values, window)
    std: FloatArray = windows.std(axis=1, ddof=1)
    return std


def _shifted(values: FloatArray, lag: int) -> FloatArray:
    shifted = np.full(values.shape, np.nan)
    if lag < values.size:
        shifted[lag:] = values[: values.size - lag]
    return shifted


def trend_score(prices: pl.Series, returns: pl.Series, horizons: Sequence[int], vol_window: int) -> pl.Series:
    """Mean over horizons of the clipped return/(vol*sqrt(h)) z-score; null when any horizon or sigma is unavailable or zero."""
    price = _float(prices)
    sigma = _trailing_std(_float(returns), vol_window)
    sigma = np.where(sigma > 0.0, sigma, np.nan)
    z_scores: list[FloatArray] = []
    for horizon in horizons:
        past = _shifted(price, horizon)
        z = (price / past - 1.0) / (sigma * sqrt(horizon))
        z_scores.append(np.clip(z, -_CLIP, _CLIP))
    return _output(np.mean(np.stack(z_scores), axis=0), "trend_score")


def efficiency_ratio(prices: pl.Series, window: int) -> pl.Series:
    """Net move over the window divided by the summed absolute moves; null when that path length is zero."""
    price = _float(prices)
    steps = np.full(price.shape, np.nan)
    steps[1:] = np.abs(np.diff(price))
    path = _trailing(steps, window).sum(axis=1)
    net = np.full(price.shape, np.nan)
    net[window:] = np.abs(price[window:] - price[:-window])
    ratio = np.full(price.shape, np.nan)
    np.divide(net, path, out=ratio, where=path > 0.0)
    return _output(ratio, "efficiency_ratio")


def flow_imbalance(
    lev_shares: pl.Series, lev_price: pl.Series, inv_shares: pl.Series, inv_price: pl.Series, window: int
) -> pl.Series:
    """Net creation value of the leverage leg minus the inverse leg over the window, as a share of their combined value."""
    lev_n = _float(lev_shares)
    inv_n = _float(inv_shares)
    lev_p = _float(lev_price)
    inv_p = _float(inv_price)
    flow = np.full(lev_n.shape, np.nan)
    flow[1:] = np.diff(lev_n) * lev_p[1:] - np.diff(inv_n) * inv_p[1:]
    total = _trailing(flow, window).sum(axis=1)
    value = lev_n * lev_p + inv_n * inv_p
    ratio = np.full(lev_n.shape, np.nan)
    np.divide(total, value, out=ratio, where=value > 0.0)
    return _output(ratio, "flow_imbalance")


def rolling_z(x: pl.Series, window: int) -> pl.Series:
    """Z-score of each value against the trailing window that ends on it; null when the window is incomplete or flat."""
    values = _float(x)
    windows = _trailing(values, window)
    mean = windows.mean(axis=1)
    std = windows.std(axis=1, ddof=1)
    z = np.full(values.shape, np.nan)
    np.divide(values - mean, std, out=z, where=std > 0.0)
    return _output(z, "rolling_z")
