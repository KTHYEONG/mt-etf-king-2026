"""Vehicle return panel for the contest simulator.

Each vehicle carries an overnight-gap leg and an intraday (open-to-close) leg on
the KOSPI200 session grid. A switch executed at the open earns the old holding's
gap leg and the new holding's intraday leg, which is how manual HTS orders at
09:00 are filled. Listed ETF prices are used where they traded; earlier rows use
the configured synthetic daily-reset multiple of the underlying (minus a daily
fee). Only the local reference parquet is read here, never the network.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

_GRID_INDEX: Final[str] = "코스피 200"
_TRADING_DAYS_PER_YEAR: Final[float] = 252.0


class PanelGapError(RuntimeError):
    """Raised when a tradable vehicle has no leg inside the contest period (fail-closed)."""


@dataclass(frozen=True)
class VehiclePanel:
    """Daily overnight-gap and intraday legs for every contest vehicle on the KRX session grid.

    A switch executed at the open earns the old holding's gap leg and the new holding's intraday leg, which is how
    manual HTS orders at 09:00 are filled. Listed ETF prices are used where they traded; earlier rows use the
    configured synthetic daily-reset multiple of the underlying (minus a daily fee).
    """

    dates: tuple[date, ...]
    names: tuple[str, ...]
    gap: np.ndarray  # [T, V] float32
    intraday: np.ndarray  # [T, V] float32
    log_nav: np.ndarray  # [T, V] float64 cumulative log close-to-close

    def row(self, session: date) -> int:
        """Return the panel row for a session date, or raise ValueError when absent."""
        for i, d in enumerate(self.dates):
            if d == session:
                return i
        raise ValueError(f"session not in panel: {session.isoformat()}")

    def index(self, alias: str) -> int:
        """Return the vehicle column for an alias, or raise KeyError when absent."""
        for i, n in enumerate(self.names):
            if n == alias:
                return i
        raise KeyError(f"vehicle not in panel: {alias}")


def _aligned_oc(frame: pl.DataFrame, grid: list[date]) -> tuple[np.ndarray, np.ndarray]:
    base = pl.DataFrame({"date": grid}, schema={"date": pl.Date})
    joined = base.join(frame, on="date", how="left").sort("date")
    raw_o = joined.get_column("open").to_numpy().astype(float)
    raw_c = joined.get_column("close").to_numpy().astype(float)
    valid_o = np.isfinite(raw_o) & (raw_o > 0)
    valid_c = np.isfinite(raw_c) & (raw_c > 0)
    return np.where(valid_o, raw_o, np.nan), np.where(valid_c, raw_c, np.nan)


def _legs_actual(o: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(o)
    gap = np.full(n, np.nan)
    prev_ok = np.isfinite(c[:-1]) & (c[:-1] > 0) & np.isfinite(o[1:]) & (o[1:] > 0)
    gap[1:][prev_ok] = o[1:][prev_ok] / c[:-1][prev_ok] - 1.0
    now_ok = np.isfinite(o) & (o > 0) & np.isfinite(c) & (c > 0)
    intraday = np.full(n, np.nan)
    intraday[now_ok] = c[now_ok] / o[now_ok] - 1.0
    return gap, intraday


def _legs_synth(o: np.ndarray, c: np.ndarray, multiple: float, fee: float) -> tuple[np.ndarray, np.ndarray]:
    gap, _ = _legs_actual(o, c)
    gap = np.where(np.isfinite(gap), gap * multiple, np.nan)
    now_ok = np.isfinite(o) & (o > 0) & np.isfinite(c) & (c > 0)
    intraday = np.full(len(o), np.nan)
    intraday[now_ok] = multiple * (c[now_ok] / o[now_ok] - 1.0) - fee
    return gap, intraday


def _splice(
    listed: tuple[np.ndarray, np.ndarray], synth: tuple[np.ndarray, np.ndarray] | None
) -> tuple[np.ndarray, np.ndarray]:
    if synth is None:
        return listed
    ok = np.isfinite(listed[0]) & np.isfinite(listed[1])
    return np.where(ok, listed[0], synth[0]), np.where(ok, listed[1], synth[1])


def _synth_legs(
    alias: str,
    vehicle: Mapping[str, Any],
    grid: list[date],
    by_index: dict[str, pl.DataFrame],
    by_symbol: dict[str, pl.DataFrame],
    fee: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    spec = vehicle.get("synthetic")
    if not isinstance(spec, Mapping):
        return None
    underlying = str(spec["underlying"])
    multiple = float(spec["multiple"])
    if underlying in by_symbol:
        o, c = _aligned_oc(by_symbol[underlying], grid)
    elif underlying in by_index:
        o, c = _aligned_oc(by_index[underlying], grid)
    else:
        raise PanelGapError(f"unknown synthetic underlying for {alias}: {underlying!r}")
    return _legs_synth(o, c, multiple, fee)


def build_vehicle_panel(
    etf_daily: pl.DataFrame,
    index_daily: pl.DataFrame,
    reference: pl.DataFrame,
    vehicles: Mapping[str, Any],
    fee_annual: float,
) -> VehiclePanel:
    """Assemble the panel from the normalized KRX tables and the single-stock reference.

    Raises:
        PanelGapError: a tradable vehicle has no leg for any session in [contest.start_date, last panel date]
        (fail-closed: live decisions must never run on imputed contest-period data).
    """
    from src.core.config import load_config

    contest_cfg = load_config("contest").get("contest", {})
    contest_start = date.fromisoformat(str(contest_cfg.get("start_date", "2026-09-21")))
    grid_df = (
        index_daily.filter(pl.col("index_name") == _GRID_INDEX).select("date").unique().sort("date")
    )
    grid: list[date] = grid_df.get_column("date").to_list()
    if not grid:
        raise ValueError("empty session grid: index_daily lacks 코스피 200 rows")
    fee = float(fee_annual) / _TRADING_DAYS_PER_YEAR

    by_index: dict[str, pl.DataFrame] = {}
    for name in index_daily.get_column("index_name").unique().to_list():
        by_index[str(name)] = index_daily.filter(pl.col("index_name") == name).select("date", "open", "close").sort("date")
    by_symbol: dict[str, pl.DataFrame] = {}
    if reference.height and "symbol" in reference.columns:
        for sym in reference.get_column("symbol").unique().to_list():
            by_symbol[str(sym)] = reference.filter(pl.col("symbol") == sym).select("date", "open", "close").sort("date")
    by_ticker: dict[str, pl.DataFrame] = {}
    for ticker in etf_daily.get_column("ticker").unique().to_list():
        by_ticker[str(ticker)] = etf_daily.filter(pl.col("ticker") == ticker).select("date", "open", "close").sort("date")

    names = list(vehicles.keys())
    gaps: list[np.ndarray] = []
    intras: list[np.ndarray] = []
    for alias in names:
        vehicle = vehicles[alias]
        ticker = vehicle.get("ticker")
        listed: tuple[np.ndarray, np.ndarray] | None = None
        if isinstance(ticker, str) and ticker in by_ticker:
            listed = _legs_actual(*_aligned_oc(by_ticker[ticker], grid))
        synth = _synth_legs(alias, vehicle, grid, by_index, by_symbol, fee)
        if str(vehicle.get("returns", "listed")) == "synthetic" or listed is None:
            if synth is None:
                raise PanelGapError(f"no legs available for {alias}")
            legs = synth
        else:
            legs = _splice(listed, synth)
        gap, intra = legs
        missing = ~np.isfinite(gap) | ~np.isfinite(intra)
        contest_rows = np.array([d >= contest_start for d in grid])
        bad = np.where(missing & contest_rows)[0]
        if len(bad):
            raise PanelGapError(f"missing legs for {alias} at {grid[int(bad[0])].isoformat()} (in contest period)")
        gaps.append(np.where(missing, 0.0, gap).astype(np.float32))
        intras.append(np.where(missing, 0.0, intra).astype(np.float32))

    gap_m = np.column_stack(gaps)
    intra_m = np.column_stack(intras)
    with np.errstate(divide="ignore", invalid="ignore"):
        cc = (1 + gap_m.astype(np.float64)) * (1 + intra_m.astype(np.float64)) - 1.0
        log_nav = np.cumsum(np.log1p(np.clip(cc, -0.999999, None)), axis=0)
    panel = VehiclePanel(dates=tuple(grid), names=tuple(names), gap=gap_m, intraday=intra_m, log_nav=log_nav)
    logger.info(f"[DATA] contest_panel sessions={len(grid)} vehicles={len(names)} start={grid[0].isoformat()}")
    return panel


def neutralize_drift(panel: VehiclePanel, start: date, end: date, keep_sessions: Sequence[date]) -> VehiclePanel:
    """Copy whose intraday legs are shifted so each vehicle's mean log daily return over [start, end] is zero;
    legs of `keep_sessions` (realized contest sessions) and `log_nav` (pre-decision momentum history) stay unchanged."""
    rows = [i for i, d in enumerate(panel.dates) if start <= d <= end]
    if not rows:
        raise ValueError(f"empty neutralize window: {start.isoformat()}..{end.isoformat()}")
    keep = {panel.row(s) for s in keep_sessions if s in panel.dates}
    fit = [r for r in rows if r not in keep]
    if not fit:
        raise ValueError("neutralize window fully covered by keep_sessions")
    gap = panel.gap.astype(np.float64)
    intra = panel.intraday.astype(np.float64)
    cc = (1 + gap[fit]) * (1 + intra[fit]) - 1.0
    mu = np.log1p(np.clip(cc, -0.999999, None)).mean(axis=0)
    shifted = (1 + intra) * np.exp(-mu)[None, :] - 1.0
    shifted = shifted.astype(np.float32).astype(np.float64)
    cc2 = (1 + gap[fit]) * (1 + shifted[fit]) - 1.0
    residual = np.log1p(np.clip(cc2, -0.999999, None)).mean(axis=0)
    shifted = (1 + shifted) * np.exp(-residual)[None, :] - 1.0
    out_intra = shifted.astype(np.float32)
    for r in keep:
        out_intra[r] = panel.intraday[r]
    return VehiclePanel(dates=panel.dates, names=panel.names, gap=panel.gap, intraday=out_intra, log_nav=panel.log_nav)
