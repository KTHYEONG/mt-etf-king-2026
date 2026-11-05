"""Two-execution simulator: an intraday book set at the open and an overnight book set at the close of each session."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from itertools import pairwise
from typing import Any, Final, Literal, get_args

import numpy as np
import polars as pl
from numpy import typing as npt

from src.arena.panel import Panel
from src.arena.simulate import (
    _BPS_PER_UNIT,
    _CODE_ADV_CAP,
    _CODE_CASH,
    _CODE_NO_ADV,
    _CODE_NOT_LISTED,
    _CODE_NOT_TRADED,
    _MAX_LOSS,
    ReturnMode,
    SimConfig,
    SimResult,
    SimulationDataError,
    _blocked_frame,
    _demean,
    _Dense,
    _dense_panel,
    _target_matrix,
)
from src.core.config import ConfigError, load_config

Leg = Literal["intraday", "overnight"]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]

_SEGMENT_KEYS: Final = frozenset({"cost_bps_per_side", "stress_cost_bps"})
_TARGET_COLUMNS: Final[dict[str, pl.DataType]] = {
    "decision_date": pl.Date(),
    "leg": pl.String(),
    "ticker": pl.String(),
    "weight": pl.Float64(),
}
_LEG_COLUMNS: Final = ("decision_date", "ticker", "weight")


@dataclass(frozen=True)
class SegmentConfig:
    cost_bps_per_side: float
    stress_cost_bps: float


@dataclass(frozen=True)
class _Session:
    leg: FloatArray
    leg_ok: BoolArray
    intra: FloatArray
    open_tradable: BoolArray
    close_tradable: BoolArray
    open_codes: IntArray
    close_codes: IntArray
    adv: FloatArray


@dataclass(frozen=True)
class _Fill:
    holdings: FloatArray
    turnover: float
    cols: IntArray
    codes: IntArray
    wanted: FloatArray
    executed: FloatArray


@dataclass
class _Blocks:
    days: list[IntArray] = field(default_factory=list)
    cols: list[IntArray] = field(default_factory=list)
    codes: list[IntArray] = field(default_factory=list)
    wanted: list[FloatArray] = field(default_factory=list)
    executed: list[FloatArray] = field(default_factory=list)

    def add(self, day: int, fill: _Fill) -> None:
        self.days.append(np.full(fill.cols.size, day, dtype=np.int64))
        self.cols.append(fill.cols)
        self.codes.append(fill.codes)
        self.wanted.append(fill.wanted)
        self.executed.append(fill.executed)

    def frame(self, sessions: tuple[date, ...], tickers: tuple[str, ...]) -> pl.DataFrame:
        return _blocked_frame(sessions, self.days, self.cols, self.codes, self.wanted, self.executed, tickers)


def _non_negative(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ConfigError(f"segment.{key} must be a finite number >= 0")
    return float(value)


def load_segment_config(name: str = "arena") -> SegmentConfig:
    """Load the `segment` mapping of `configs/<name>.yaml`; raises ConfigError on any problem."""
    section = load_config(name).get("segment")
    if not isinstance(section, Mapping):
        raise ConfigError(f"config {name} must define a segment mapping")
    if set(section) != _SEGMENT_KEYS:
        raise ConfigError(f"segment keys must be exactly {sorted(_SEGMENT_KEYS)}")
    base = _non_negative(section["cost_bps_per_side"], "cost_bps_per_side")
    stress = _non_negative(section["stress_cost_bps"], "stress_cost_bps")
    if stress < base:
        raise ConfigError("segment.stress_cost_bps must be >= segment.cost_bps_per_side")
    return SegmentConfig(cost_bps_per_side=base, stress_cost_bps=stress)


def _validate_targets(targets: pl.DataFrame) -> None:
    for name, dtype in _TARGET_COLUMNS.items():
        if targets.schema.get(name) != dtype:
            raise ValueError(f"targets column {name} must be {dtype}")
    if any(targets[name].null_count() for name in _TARGET_COLUMNS):
        raise ValueError("targets must not contain nulls")
    if not set(targets["leg"].unique().to_list()) <= set(get_args(Leg)):
        raise ValueError(f"targets leg must be one of {get_args(Leg)}")


def _leg_matrix(
    panel: Panel, targets: pl.DataFrame, leg: Leg, cfg: SimConfig, tickers: tuple[str, ...]
) -> tuple[FloatArray, BoolArray]:
    part = targets.filter(pl.col("leg") == leg).select(_LEG_COLUMNS)
    return _target_matrix(panel, part, cfg, tickers)


def _block_codes(listed: BoolArray, traded: BoolArray, adv_ok: BoolArray) -> IntArray:
    codes = np.select(
        [~listed, listed & ~traded, traded & ~adv_ok],
        [_CODE_NOT_LISTED, _CODE_NOT_TRADED, _CODE_NO_ADV],
        default=-1,
    )
    return codes.astype(np.int64)


def _session_legs(dense: _Dense, first: int, drift_neutral: bool, segment: IntArray) -> _Session:
    def window(x: FloatArray) -> FloatArray:
        sliced = x[first:]
        return _demean(sliced, segment) if drift_neutral else sliced

    r_on, r_id, r_cc, r_ref = (window(x) for x in (dense.r_on, dense.r_id, dense.r_cc, dense.r_ref))
    if drift_neutral:
        r_id = np.where(np.isfinite(r_on) & np.isfinite(r_cc), (1.0 + r_cc) / (1.0 + r_on) - 1.0, r_id)
    for returns in (r_on, r_id, r_cc, r_ref):
        if bool(np.any(returns <= _MAX_LOSS)):
            raise SimulationDataError("a leg return is <= -1 after drift removal")

    listed = dense.listed[first:]
    adv = dense.adv[first:]
    adv_ok = np.isfinite(adv)
    open_traded = listed & np.isfinite(r_on) & np.isfinite(r_id)
    close_traded = listed & np.isfinite(r_cc)
    leg_raw = np.where(open_traded, r_on, np.where(listed, r_ref, 0.0))
    leg_ok = listed & np.isfinite(leg_raw)
    return _Session(
        leg=np.where(leg_ok, leg_raw, 0.0),
        leg_ok=leg_ok,
        intra=np.where(open_traded, r_id, 0.0),
        open_tradable=open_traded & adv_ok,
        close_tradable=close_traded & adv_ok,
        open_codes=_block_codes(listed, open_traded, adv_ok),
        close_codes=_block_codes(listed, close_traded, adv_ok),
        adv=adv,
    )


def _execute(
    holdings: FloatArray,
    target: FloatArray,
    tradable: BoolArray,
    block_codes: IntArray,
    adv: FloatArray,
    pre_equity: float,
    cfg: SimConfig,
    capital_krw: int,
) -> _Fill:
    delta = target - holdings
    mag = np.abs(delta)
    cap = cfg.participation_max * adv / (pre_equity * capital_krw)
    over_cap = tradable & (mag > cap)
    allowed = np.where(tradable, np.minimum(mag, cap), 0.0)
    signed = np.sign(delta) * allowed
    sells = np.minimum(signed, 0.0)
    buys = np.maximum(signed, 0.0)
    room = max(0.0, 1.0 - float((holdings + sells).sum()))
    buy_total = float(buys.sum())
    scale = 1.0 if buy_total <= room else room / buy_total
    executed = sells + buys * scale
    cash_hit = (buys > 0.0) & (scale < 1.0)
    codes = np.where(
        block_codes >= 0,
        block_codes,
        np.where(over_cap, _CODE_ADV_CAP, np.where(cash_hit, _CODE_CASH, -1)),
    )
    wrong = np.flatnonzero(np.abs(delta - executed) > cfg.weight_tolerance)
    return _Fill(
        holdings=holdings + executed,
        turnover=float(np.abs(executed).sum()),
        cols=wrong.astype(np.int64),
        codes=codes[wrong].astype(np.int64),
        wanted=delta[wrong],
        executed=executed[wrong],
    )


def simulate_segments(
    panel: Panel,
    targets: pl.DataFrame,
    sim_cfg: SimConfig,
    cost_bps_per_side: float,
    *,
    capital_krw: int,
    mode: ReturnMode = "actual",
    drift_segments: Sequence[date] = (),
) -> SimResult:
    """Run intraday and overnight books per session; raises ValueError on bad inputs and SimulationDataError on held-position gaps."""
    _validate_targets(targets)
    if targets.height == 0:
        raise ValueError("targets must be non-empty")
    if capital_krw <= 0:
        raise ValueError("capital_krw must be > 0")
    if not math.isfinite(cost_bps_per_side) or cost_bps_per_side < 0.0:
        raise ValueError("cost_bps_per_side must be a finite number >= 0")
    if any(b <= a for a, b in pairwise(drift_segments)):
        raise ValueError("drift_segments must be strictly increasing")

    sessions = panel.sessions
    tickers = tuple(sorted(panel.frame["ticker"].unique().to_list()))
    dense = _dense_panel(panel, tickers, sim_cfg.adv_window)
    intra_mat, has_intra = _leg_matrix(panel, targets, "intraday", sim_cfg, tickers)
    over_mat, has_over = _leg_matrix(panel, targets, "overnight", sim_cfg, tickers)
    has = has_intra | has_over
    first = int(np.flatnonzero(has)[0]) + 1
    segment = np.searchsorted(
        np.array(drift_segments, dtype="datetime64[D]"),
        np.array(sessions[first:], dtype="datetime64[D]"),
        side="right",
    ).astype(np.int64)
    legs = _session_legs(dense, first, mode == "drift_neutral", segment)

    # A session runs on the latest decision strictly before it; each leg of that decision is its book.
    latest = np.maximum.accumulate(np.where(has, np.arange(len(sessions)), -1))
    book = latest[first - 1 : -1]
    intra_targets = intra_mat[book]
    over_targets = over_mat[book]

    n = len(sessions) - first
    cost = cost_bps_per_side / _BPS_PER_UNIT
    holdings = np.zeros(len(tickers))
    equity = 1.0
    gross = np.zeros(n)
    costs = np.zeros(n)
    net = np.zeros(n)
    turnover = np.zeros(n)
    weights = np.zeros((n, len(tickers)))
    blocks = _Blocks()

    for j in range(n):
        held_bad = (holdings > 0.0) & ~legs.leg_ok[j]
        if bool(held_bad.any()):
            bad = int(np.argmax(held_bad))
            raise SimulationDataError(f"{tickers[bad]} is held into {sessions[first + j]} but is not listed or priced")
        r_overnight = float(holdings @ legs.leg[j])
        holdings = holdings * (1.0 + legs.leg[j]) / (1.0 + r_overnight)
        pre_open = equity * (1.0 + r_overnight)
        day = first + j

        opened = _execute(
            holdings,
            intra_targets[j],
            legs.open_tradable[j],
            legs.open_codes[j],
            legs.adv[j],
            pre_open,
            sim_cfg,
            capital_krw,
        )
        blocks.add(day, opened)
        r_intra = float(opened.holdings @ legs.intra[j])
        holdings = opened.holdings * (1.0 + legs.intra[j]) / (1.0 + r_intra)
        pre_close = pre_open * (1.0 - cost * opened.turnover) * (1.0 + r_intra)

        closed = _execute(
            holdings,
            over_targets[j],
            legs.close_tradable[j],
            legs.close_codes[j],
            legs.adv[j],
            pre_close,
            sim_cfg,
            capital_krw,
        )
        blocks.add(day, closed)
        holdings = closed.holdings

        growth = (1.0 + r_overnight) * (1.0 + r_intra)
        multiplier = growth * (1.0 - cost * opened.turnover) * (1.0 - cost * closed.turnover)
        if multiplier <= 0.0:
            raise SimulationDataError(f"non-positive equity multiplier at {sessions[day]}")
        gross[j] = growth - 1.0
        costs[j] = growth - multiplier
        net[j] = multiplier - 1.0
        turnover[j] = opened.turnover + closed.turnover
        weights[j] = holdings
        equity *= multiplier

    rows, cols = np.nonzero(weights > 0.0)
    weights_df = pl.DataFrame(
        {
            "date": pl.Series(list(sessions), dtype=pl.Date).gather(rows + first),
            "ticker": [tickers[c] for c in cols.tolist()],
            "weight": weights[rows, cols],
        },
        schema={"date": pl.Date, "ticker": pl.String, "weight": pl.Float64},
    )
    dropped = (sessions[-1],) if has[-1] else ()
    return SimResult(
        sessions=tuple(sessions[first:]),
        net_returns=net,
        costs=costs,
        gross_returns=gross,
        turnover=turnover,
        weights=weights_df,
        blocked=blocks.frame(sessions, tickers),
        dropped_decisions=dropped,
        mode=mode,
    )
