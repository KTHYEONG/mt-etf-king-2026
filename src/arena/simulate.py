"""Return-space simulator: next-session execution of target weights with friction, ADV caps and drift-neutral legs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from typing import Any, Final, Literal

import numpy as np
import polars as pl
from numpy import typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from src.arena.panel import Panel
from src.core.config import ConfigError, load_config

Execution = Literal["open", "close"]
ReturnMode = Literal["actual", "drift_neutral"]
BlockReason = Literal["not_listed", "not_traded", "adv_cap", "no_adv", "cash"]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]

_SIM_KEYS: Final = (
    "execution",
    "cost_bps_per_side",
    "participation_max",
    "adv_window",
    "weight_levels",
    "weight_tolerance",
    "capacity_equity_max",
)
_BPS_PER_UNIT: Final = 10_000.0
_MAX_LOSS: Final = -1.0
_REASONS: Final[tuple[BlockReason, ...]] = ("not_listed", "not_traded", "no_adv", "adv_cap", "cash")
_CODE_NOT_LISTED: Final = 0
_CODE_NOT_TRADED: Final = 1
_CODE_NO_ADV: Final = 2
_CODE_ADV_CAP: Final = 3
_CODE_CASH: Final = 4
_TARGET_COLUMNS: Final[dict[str, pl.DataType]] = {
    "decision_date": pl.Date(),
    "ticker": pl.String(),
    "weight": pl.Float64(),
}
_BLOCKED_SCHEMA: Final = {
    "date": pl.Date,
    "ticker": pl.String,
    "reason": pl.String,
    "wanted": pl.Float64,
    "executed": pl.Float64,
}


@dataclass(frozen=True)
class SimConfig:
    execution: Execution
    cost_bps_per_side: float
    participation_max: float
    adv_window: int
    weight_levels: tuple[float, ...]
    weight_tolerance: float
    capacity_equity_max: float = math.inf


@dataclass(frozen=True)
class SimResult:
    sessions: tuple[date, ...]
    net_returns: FloatArray
    costs: FloatArray
    gross_returns: FloatArray
    turnover: FloatArray
    weights: pl.DataFrame
    blocked: pl.DataFrame
    dropped_decisions: tuple[date, ...]
    mode: ReturnMode


class SimulationDataError(ValueError):
    """Raised when a held position cannot be carried through a session (delisting or missing price)."""


@dataclass(frozen=True)
class _Dense:
    listed: BoolArray
    r_on: FloatArray
    r_id: FloatArray
    r_cc: FloatArray
    r_ref: FloatArray
    adv: FloatArray


@dataclass(frozen=True)
class _Legs:
    leg: FloatArray
    leg_ok: BoolArray
    intra: FloatArray
    tradable: BoolArray
    block_code: IntArray
    adv: FloatArray
    targets: FloatArray
    has_decision: BoolArray


def _execution(value: Any) -> Execution:
    if value == "open":
        return "open"
    if value == "close":
        return "close"
    raise ConfigError("simulate.execution must be open or close")


def _number(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"simulate.{key} must be a number")
    return float(value)


def _levels(value: Any) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError("simulate.weight_levels must be a non-empty list")
    levels = tuple(_number(item, "weight_levels") for item in value)
    if not all(0.0 <= w <= 1.0 for w in levels) or not all(b > a for a, b in pairwise(levels)):
        raise ConfigError("simulate.weight_levels must be strictly increasing within [0, 1]")
    if 0.0 not in levels:
        raise ConfigError("simulate.weight_levels must contain 0.0")
    return levels


def load_sim_config(name: str = "arena") -> SimConfig:
    """Load the `simulate` mapping of `configs/<name>.yaml`; raises ConfigError on any problem."""
    section = load_config(name).get("simulate")
    if not isinstance(section, Mapping):
        raise ConfigError(f"config {name} must define a simulate mapping")
    if set(section) != set(_SIM_KEYS):
        raise ConfigError(f"simulate keys must be exactly {sorted(_SIM_KEYS)}")
    cost = _number(section["cost_bps_per_side"], "cost_bps_per_side")
    if cost < 0.0:
        raise ConfigError("simulate.cost_bps_per_side must be >= 0")
    participation = _number(section["participation_max"], "participation_max")
    if not 0.0 < participation <= 1.0:
        raise ConfigError("simulate.participation_max must be in (0, 1]")
    window = section["adv_window"]
    if type(window) is not int or window < 1:
        raise ConfigError("simulate.adv_window must be an integer >= 1")
    tolerance = _number(section["weight_tolerance"], "weight_tolerance")
    if tolerance <= 0.0:
        raise ConfigError("simulate.weight_tolerance must be > 0")
    capacity = _number(section["capacity_equity_max"], "capacity_equity_max")
    if capacity < 1.0:
        raise ConfigError("simulate.capacity_equity_max must be >= 1")
    return SimConfig(
        execution=_execution(section["execution"]),
        cost_bps_per_side=cost,
        participation_max=participation,
        adv_window=window,
        weight_levels=_levels(section["weight_levels"]),
        weight_tolerance=tolerance,
        capacity_equity_max=capacity,
    )


def _demean(x: FloatArray, segment: IntArray) -> FloatArray:
    demeaned: FloatArray = np.array(x, copy=True)
    for seg in np.unique(segment):
        rows = segment == seg
        block = x[rows]
        finite = np.isfinite(block)
        count = finite.sum(axis=0)
        total = np.where(finite, block, 0.0).sum(axis=0)
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        demeaned[rows] = block - mean
    return demeaned


def _adv(listed: BoolArray, trading_value: FloatArray, window: int) -> FloatArray:
    sessions_n = listed.shape[0]
    pad = ((window, 0), (0, 0))
    count = sliding_window_view(np.pad(listed.astype(np.float64), pad), window, axis=0).sum(axis=-1)
    value = np.where(listed, trading_value, 0.0)
    total = sliding_window_view(np.pad(value, pad), window, axis=0).sum(axis=-1)
    return np.where(count[:sessions_n] == window, total[:sessions_n] / window, np.nan)


def _dense_panel(panel: Panel, tickers: tuple[str, ...], window: int) -> _Dense:
    sessions_n = len(panel.sessions)
    day_index = pl.DataFrame({"date": list(panel.sessions)}, schema={"date": pl.Date}).with_row_index("day")
    ticker_index = pl.DataFrame({"ticker": list(tickers)}, schema={"ticker": pl.String}).with_row_index("col")
    indexed = panel.frame.join(day_index, on="date", how="left").join(ticker_index, on="ticker", how="left")
    rows = indexed["day"].to_numpy().astype(np.int64)
    cols = indexed["col"].to_numpy().astype(np.int64)

    def dense(column: str) -> FloatArray:
        grid = np.full((sessions_n, len(tickers)), np.nan)
        grid[rows, cols] = indexed[column].cast(pl.Float64).fill_null(float("nan")).to_numpy()
        return grid

    listed = np.zeros((sessions_n, len(tickers)), dtype=bool)
    listed[rows, cols] = indexed["listed"].fill_null(False).to_numpy()
    return _Dense(
        listed=listed,
        r_on=dense("r_on"),
        r_id=dense("r_id"),
        r_cc=dense("r_cc"),
        r_ref=dense("r_ref"),
        adv=_adv(listed, dense("trading_value"), window),
    )


def _validate_targets(targets: pl.DataFrame) -> None:
    for name, dtype in _TARGET_COLUMNS.items():
        if targets.schema.get(name) != dtype:
            raise ValueError(f"targets column {name} must be {dtype}")
    if any(targets[name].null_count() for name in _TARGET_COLUMNS):
        raise ValueError("targets must not contain nulls")


def _target_matrix(
    panel: Panel, targets: pl.DataFrame, cfg: SimConfig, tickers: tuple[str, ...]
) -> tuple[FloatArray, BoolArray]:
    sessions_n = len(panel.sessions)
    day_index = pl.DataFrame({"date": list(panel.sessions)}, schema={"date": pl.Date}).with_row_index("day")
    ticker_index = pl.DataFrame({"ticker": list(tickers)}, schema={"ticker": pl.String}).with_row_index("col")
    indexed = (
        targets.rename({"decision_date": "date"})
        .join(day_index, on="date", how="left")
        .join(ticker_index, on="ticker", how="left")
    )
    if indexed["day"].null_count():
        raise ValueError("decision_date must be a panel session")
    if indexed.select("date", "ticker").is_duplicated().any():
        raise ValueError("duplicate (decision_date, ticker) in targets")

    raw = indexed["weight"].to_numpy()
    levels = np.asarray(cfg.weight_levels, dtype=np.float64)
    distance = np.abs(raw[:, None] - levels[None, :])
    nearest = np.argmin(distance, axis=1)
    fits = distance[np.arange(raw.size), nearest] <= cfg.weight_tolerance
    if not fits.all():
        bad = int(np.argmin(fits))
        raise ValueError(f"weight {raw[bad]} for {indexed['ticker'][bad]} is not a weight level")
    weights = levels[nearest]

    day = indexed["day"].to_numpy().astype(np.int64)
    sums = np.bincount(day, weights=weights, minlength=sessions_n)
    over = sums > 1.0 + cfg.weight_tolerance
    if over.any():
        bad_day = int(np.argmax(over))
        raise ValueError(f"weights on {panel.sessions[bad_day]} sum to {sums[bad_day]} > 1")
    if indexed["col"].null_count():
        raise ValueError("target ticker is not in the panel frame")

    matrix = np.zeros((sessions_n, len(tickers)))
    matrix[day, indexed["col"].to_numpy().astype(np.int64)] = weights
    has_decision = np.zeros(sessions_n, dtype=bool)
    has_decision[day] = True
    return matrix, has_decision


def _legs(
    dense: _Dense,
    targets: FloatArray,
    has: BoolArray,
    first: int,
    cfg: SimConfig,
    drift_neutral: bool,
    segment: IntArray,
) -> _Legs:
    def window(x: FloatArray) -> FloatArray:
        sliced = x[first:]
        return _demean(sliced, segment) if drift_neutral else sliced

    r_on, r_id, r_cc, r_ref = (window(x) for x in (dense.r_on, dense.r_id, dense.r_cc, dense.r_ref))
    if drift_neutral:
        # Demeaning the overnight and intraday legs separately leaves their covariance as drift in a
        # held position; deriving the intraday leg from the demeaned close-to-close keeps holds at zero mean.
        r_id = np.where(np.isfinite(r_on) & np.isfinite(r_cc), (1.0 + r_cc) / (1.0 + r_on) - 1.0, r_id)
    for returns in (r_on, r_id, r_cc, r_ref):
        if bool(np.any(returns <= _MAX_LOSS)):
            raise SimulationDataError("a leg return is <= -1 after drift removal")

    listed = dense.listed[first:]
    if cfg.execution == "open":
        traded = listed & np.isfinite(r_on) & np.isfinite(r_id)
        primary = r_on
        intra = np.where(traded, r_id, 0.0)
    else:
        traded = listed & np.isfinite(r_cc)
        primary = r_cc
        intra = np.zeros_like(r_cc)

    leg_raw = np.where(traded, primary, np.where(listed, r_ref, 0.0))
    leg_ok = listed & np.isfinite(leg_raw)
    adv = dense.adv[first:]
    adv_ok = np.isfinite(adv)
    block_code = np.select(
        [~listed, listed & ~traded, traded & ~adv_ok],
        [_CODE_NOT_LISTED, _CODE_NOT_TRADED, _CODE_NO_ADV],
        default=-1,
    )
    return _Legs(
        leg=np.where(leg_ok, leg_raw, 0.0),
        leg_ok=leg_ok,
        intra=intra,
        tradable=traded & adv_ok,
        block_code=block_code,
        adv=adv,
        targets=targets[first - 1 : -1],
        has_decision=has[first - 1 : -1],
    )


def _blocked_frame(
    sessions: tuple[date, ...],
    days: list[IntArray],
    cols: list[IntArray],
    codes: list[IntArray],
    wanted: list[FloatArray],
    executed: list[FloatArray],
    tickers: tuple[str, ...],
) -> pl.DataFrame:
    day_idx = np.concatenate([np.empty(0, dtype=np.int64), *days])
    col_idx = np.concatenate([np.empty(0, dtype=np.int64), *cols])
    code_idx = np.concatenate([np.empty(0, dtype=np.int64), *codes])
    wanted_arr = np.concatenate([np.empty(0), *wanted])
    executed_arr = np.concatenate([np.empty(0), *executed])
    return pl.DataFrame(
        {
            "date": pl.Series(list(sessions), dtype=pl.Date).gather(day_idx),
            "ticker": [tickers[c] for c in col_idx.tolist()],
            "reason": [_REASONS[c] for c in code_idx.tolist()],
            "wanted": wanted_arr,
            "executed": executed_arr,
        },
        schema=_BLOCKED_SCHEMA,
    )


def simulate(
    panel: Panel,
    targets: pl.DataFrame,
    cfg: SimConfig,
    *,
    capital_krw: int,
    mode: ReturnMode = "actual",
    drift_segments: Sequence[date] = (),
) -> SimResult:
    """Simulate target-weight decisions on a panel; raises ValueError on bad targets and SimulationDataError on held-position gaps."""
    _validate_targets(targets)
    if targets.height == 0:
        raise ValueError("targets must be non-empty")
    if capital_krw <= 0:
        raise ValueError("capital_krw must be > 0")
    sessions = panel.sessions
    tickers = tuple(sorted(panel.frame["ticker"].unique().to_list()))
    dense = _dense_panel(panel, tickers, cfg.adv_window)
    targets_matrix, has = _target_matrix(panel, targets, cfg, tickers)

    k0 = int(np.flatnonzero(has)[0])
    first = k0 + 1
    if any(b <= a for a, b in pairwise(drift_segments)):
        raise ValueError("drift_segments must be strictly increasing")
    # Each segment is demeaned on its own so a window's evaluation never carries another window's drift.
    segment = np.searchsorted(
        np.array(drift_segments, dtype="datetime64[D]"),
        np.array(sessions[first:], dtype="datetime64[D]"),
        side="right",
    ).astype(np.int64)
    legs = _legs(dense, targets_matrix, has, first, cfg, mode == "drift_neutral", segment)
    n = len(sessions) - first
    cost = cfg.cost_bps_per_side / _BPS_PER_UNIT

    h = np.zeros(len(tickers))
    equity = 1.0
    gross = np.zeros(n)
    costs = np.zeros(n)
    net = np.zeros(n)
    turnover = np.zeros(n)
    weights = np.zeros((n, len(tickers)))
    blk_days: list[IntArray] = []
    blk_cols: list[IntArray] = []
    blk_codes: list[IntArray] = []
    blk_wanted: list[FloatArray] = []
    blk_executed: list[FloatArray] = []

    for j in range(n):
        held_bad = (h > 0.0) & ~legs.leg_ok[j]
        if bool(held_bad.any()):
            bad = int(np.argmax(held_bad))
            raise SimulationDataError(f"{tickers[bad]} is held into {sessions[first + j]} but is not listed or priced")
        r_a = float(h @ legs.leg[j])
        h = h * (1.0 + legs.leg[j]) / (1.0 + r_a)
        pre_equity = equity * (1.0 + r_a)

        target = np.where(legs.has_decision[j], legs.targets[j], h)
        delta = target - h
        mag = np.abs(delta)
        # Evaluation windows restart at the contest capital; long-history compounding must not shrink capacity.
        cap = cfg.participation_max * legs.adv[j] / (min(pre_equity, cfg.capacity_equity_max) * capital_krw)
        over_cap = legs.tradable[j] & (mag > cap)
        allowed = np.where(legs.tradable[j], np.minimum(mag, cap), 0.0)
        signed = np.sign(delta) * allowed
        sells = np.minimum(signed, 0.0)
        buys = np.maximum(signed, 0.0)
        room = max(0.0, 1.0 - float((h + sells).sum()))
        buy_total = float(buys.sum())
        scale = 1.0 if buy_total <= room else room / buy_total
        executed = sells + buys * scale
        cash_hit = (buys > 0.0) & (scale < 1.0)
        code = np.where(
            legs.block_code[j] >= 0,
            legs.block_code[j],
            np.where(over_cap, _CODE_ADV_CAP, np.where(cash_hit, _CODE_CASH, -1)),
        )
        wrong = np.flatnonzero(np.abs(delta - executed) > cfg.weight_tolerance)
        blk_days.append(np.full(wrong.size, first + j, dtype=np.int64))
        blk_cols.append(wrong.astype(np.int64))
        blk_codes.append(code[wrong].astype(np.int64))
        blk_wanted.append(delta[wrong])
        blk_executed.append(executed[wrong])

        w_post = h + executed
        tau = float(np.abs(executed).sum())
        r_b = float(w_post @ legs.intra[j])
        growth = (1.0 + r_a) * (1.0 + r_b)
        multiplier = growth * (1.0 - cost * tau)
        if multiplier <= 0.0:
            raise SimulationDataError(f"non-positive equity multiplier at {sessions[first + j]}")
        h = w_post * (1.0 + legs.intra[j]) / (1.0 + r_b)
        gross[j] = growth - 1.0
        costs[j] = growth - multiplier
        net[j] = multiplier - 1.0
        turnover[j] = tau
        weights[j] = h
        equity *= multiplier

    sim_sessions = sessions[first:]
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
        sessions=tuple(sim_sessions),
        net_returns=net,
        costs=costs,
        gross_returns=gross,
        turnover=turnover,
        weights=weights_df,
        blocked=_blocked_frame(sessions, blk_days, blk_cols, blk_codes, blk_wanted, blk_executed, tickers),
        dropped_decisions=dropped,
        mode=mode,
    )
