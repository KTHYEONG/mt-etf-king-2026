"""Tournament-policy championship sleeve classifier (research-only, never a production gate)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final

import polars as pl

from src.features.pit import PitViolationError
from src.research.audit_regime import annualize_daily_realized_vol


class ChampionshipSleeve(StrEnum):
    CRASH_REBOUND = "CRASH_REBOUND"
    LOTTERY_ON = "LOTTERY_ON"
    INACTIVE = "INACTIVE"
    UNCERTAIN = "UNCERTAIN"


CHAMPIONSHIP_SLEEVE_IS_PRODUCTION_GATE: Final[bool] = False
P27_REGIME_IS_PRODUCTION_GATE: Final[bool] = False
CRASH_REBOUND_RV_ANNUALIZED_MIN: Final[float] = 0.25
CRASH_REBOUND_MOM20_MIN: Final[float] = 0.03
LOTTERY_ON_MOM60_MIN: Final[float] = 0.08


@dataclass(frozen=True, slots=True)
class ChampionshipRegimeSnapshot:
    as_of: date
    sleeve: ChampionshipSleeve
    mom60: float | None
    mom20: float | None
    rv20_annualized: float | None


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return float(number)


def classify_championship_sleeve(
    *,
    decision_date: date,
    mom60: float | None,
    mom20: float | None,
    rv20_daily: float | None,
) -> ChampionshipRegimeSnapshot:
    m60 = _finite_or_none(mom60)
    m20 = _finite_or_none(mom20)
    rv_ann = annualize_daily_realized_vol(_finite_or_none(rv20_daily))
    if m60 is None or m20 is None or rv_ann is None:
        return ChampionshipRegimeSnapshot(as_of=decision_date, sleeve=ChampionshipSleeve.UNCERTAIN, mom60=m60, mom20=m20, rv20_annualized=rv_ann)
    if rv_ann > CRASH_REBOUND_RV_ANNUALIZED_MIN and m20 > CRASH_REBOUND_MOM20_MIN and m60 < 0:
        sleeve = ChampionshipSleeve.CRASH_REBOUND
    elif m60 > LOTTERY_ON_MOM60_MIN:
        sleeve = ChampionshipSleeve.LOTTERY_ON
    else:
        sleeve = ChampionshipSleeve.INACTIVE
    return ChampionshipRegimeSnapshot(as_of=decision_date, sleeve=sleeve, mom60=m60, mom20=m20, rv20_annualized=rv_ann)


def classify_championship_sleeve_from_maps(
    *,
    decision_date: date,
    mom60_by_date: Mapping[date, float | None],
    mom20_by_date: Mapping[date, float | None],
    rv20_daily_by_date: Mapping[date, float | None],
) -> ChampionshipRegimeSnapshot:
    for mapping in (mom60_by_date, mom20_by_date, rv20_daily_by_date):
        for key in mapping:
            if key > decision_date:
                raise PitViolationError(f"PIT violation: sleeve date {key} exceeds decision_date {decision_date}")
    return classify_championship_sleeve(
        decision_date=decision_date,
        mom60=mom60_by_date.get(decision_date),
        mom20=mom20_by_date.get(decision_date),
        rv20_daily=rv20_daily_by_date.get(decision_date),
    )


def classify_championship_sleeve_series(
    *,
    mom60_by_date: Mapping[date, float | None],
    mom20_by_date: Mapping[date, float | None],
    rv20_daily_by_date: Mapping[date, float | None],
) -> tuple[ChampionshipRegimeSnapshot, ...]:
    ordered_days = sorted(set(mom60_by_date) | set(mom20_by_date) | set(rv20_daily_by_date))
    snapshots: list[ChampionshipRegimeSnapshot] = []
    for as_of in ordered_days:
        cut60 = {day: mom60_by_date[day] for day in mom60_by_date if day <= as_of}
        cut20 = {day: mom20_by_date[day] for day in mom20_by_date if day <= as_of}
        cutrv = {day: rv20_daily_by_date[day] for day in rv20_daily_by_date if day <= as_of}
        snapshots.append(classify_championship_sleeve_from_maps(decision_date=as_of, mom60_by_date=cut60, mom20_by_date=cut20, rv20_daily_by_date=cutrv))
    return tuple(snapshots)


def kospi_sleeve_feature_maps(index_daily: pl.DataFrame) -> tuple[dict[date, float | None], dict[date, float | None], dict[date, float | None]]:
    empty: tuple[dict[date, float | None], dict[date, float | None], dict[date, float | None]] = ({}, {}, {})
    if index_daily.height == 0:
        return empty
    cols = set(index_daily.columns)
    if not {"index_name", "date", "close"} <= cols:
        return empty
    sub = index_daily.filter(pl.col("index_name") == "KOSPI").sort("date")
    dates: list[date] = list(sub.get_column("date").to_list())
    closes: list[float] = [float(c) for c in sub.get_column("close").to_list()]
    mom60_by_date: dict[date, float | None] = {}
    mom20_by_date: dict[date, float | None] = {}
    rv20_daily_by_date: dict[date, float | None] = {}
    rets: list[float] = [0.0] * len(dates)
    for i in range(1, len(dates)):
        rets[i] = float(closes[i]) / float(closes[i - 1]) - 1.0
    for i, d in enumerate(dates):
        mom60: float | None = None
        if i >= 60:
            base = closes[i - 60]
            cur = closes[i]
            if base > 0.0:
                mom60 = float(float(cur) / float(base) - 1.0)
        mom20: float | None = None
        if i >= 20:
            base = closes[i - 20]
            cur = closes[i]
            if base > 0.0:
                mom20 = float(float(cur) / float(base) - 1.0)
        rv: float | None = None
        if i >= 20:
            window = rets[i - 19 : i + 1]
            finite = [float(x) for x in window]
            if len(finite) >= 20:
                mean = float(sum(finite) / 20.0)
                var = float(sum((float(x) - mean) ** 2 for x in finite) / 20.0)
                rv = float(math.sqrt(var))
        mom60_by_date[d] = mom60
        mom20_by_date[d] = mom20
        rv20_daily_by_date[d] = rv
    return (mom60_by_date, mom20_by_date, rv20_daily_by_date)


def build_championship_sleeve_map(*, mom60_by_date: Mapping[date, float | None], mom20_by_date: Mapping[date, float | None], rv20_daily_by_date: Mapping[date, float | None]) -> dict[date, str]:
    out: dict[date, str] = {}
    for d in sorted(set(mom60_by_date) | set(mom20_by_date) | set(rv20_daily_by_date)):
        snap = classify_championship_sleeve(decision_date=d, mom60=mom60_by_date.get(d), mom20=mom20_by_date.get(d), rv20_daily=rv20_daily_by_date.get(d))
        out[d] = snap.sleeve.value
    return out


def championship_sleeve_from_cache(cache: object, decision_date: date) -> str | None:
    mapping = getattr(cache, "championship_sleeves", None)
    if not isinstance(mapping, Mapping):
        return None
    raw = mapping.get(decision_date)
    if isinstance(raw, str) and raw != "":
        return raw
    return None


def abs_mom_rebound_bypass_allowed(*, sleeve: str | None, config_enabled: bool) -> bool:
    return bool(CHAMPIONSHIP_SLEEVE_IS_PRODUCTION_GATE and config_enabled and sleeve == ChampionshipSleeve.CRASH_REBOUND.value)


def sleeve_conditional_table(
    *,
    sleeves: Sequence[ChampionshipSleeve | str],
    terminal_returns: Sequence[float],
    oracle_returns: Sequence[float],
    ruin_threshold: float = -0.25,
    overlap_horizon: int = 36,
) -> pl.DataFrame:
    if not (len(sleeves) == len(terminal_returns) == len(oracle_returns)):
        raise ValueError(f"length mismatch: sleeves={len(sleeves)} terminals={len(terminal_returns)} oracle={len(oracle_returns)}")
    order = (ChampionshipSleeve.CRASH_REBOUND, ChampionshipSleeve.LOTTERY_ON, ChampionshipSleeve.INACTIVE, ChampionshipSleeve.UNCERTAIN)
    normalized = [item.value if isinstance(item, ChampionshipSleeve) else str(item) for item in sleeves]
    terminals = [float(value) for value in terminal_returns]
    oracles = [float(value) for value in oracle_returns]
    rows: list[dict[str, object]] = []
    for bucket in order:
        idx = [i for i, name in enumerate(normalized) if name == bucket.value]
        n_windows = len(idx)
        n_effective = float(n_windows) / float(overlap_horizon)
        if n_windows == 0:
            rows.append({"sleeve": bucket.value, "n_windows": 0, "n_effective": 0.0, "p27_p30": 0.0, "p27_p40": 0.0, "p27_p45": 0.0, "p27_p50": 0.0, "p27_ruin25": 0.0, "executable_oracle_p50": 0.0, "capture_50": 0.0})
            continue
        bucket_terminals = [terminals[i] for i in idx]
        bucket_oracles = [oracles[i] for i in idx]
        p30 = sum(1 for v in bucket_terminals if v > 0.30) / n_windows
        p40 = sum(1 for v in bucket_terminals if v > 0.40) / n_windows
        p45 = sum(1 for v in bucket_terminals if v > 0.45) / n_windows
        p50 = sum(1 for v in bucket_terminals if v > 0.50) / n_windows
        ruin = sum(1 for v in bucket_terminals if v < ruin_threshold) / n_windows
        oracle_hits = sum(1 for v in bucket_oracles if v > 0.50)
        oracle_p50 = oracle_hits / n_windows
        if oracle_hits == 0:
            capture = 0.0
        else:
            capture = sum(1 for t, o in zip(bucket_terminals, bucket_oracles, strict=True) if t > 0.50 and o > 0.50) / oracle_hits
        rows.append({"sleeve": bucket.value, "n_windows": n_windows, "n_effective": n_effective, "p27_p30": float(p30), "p27_p40": float(p40), "p27_p45": float(p45), "p27_p50": float(p50), "p27_ruin25": float(ruin), "executable_oracle_p50": float(oracle_p50), "capture_50": float(capture)})
    return pl.DataFrame(
        rows,
        schema={"sleeve": pl.String, "n_windows": pl.Int64, "n_effective": pl.Float64, "p27_p30": pl.Float64, "p27_p40": pl.Float64, "p27_p45": pl.Float64, "p27_p50": pl.Float64, "p27_ruin25": pl.Float64, "executable_oracle_p50": pl.Float64, "capture_50": pl.Float64},
        strict=False,
    )


class P27RegimeState(StrEnum):
    ON = "ON"
    UNCERTAIN = "UNCERTAIN"
    OFF = "OFF"


@dataclass(frozen=True, slots=True)
class P27RegimeInputs:
    kospi_mom20: float | None
    kospi_mom60: float | None
    kosdaq_mom20: float | None
    kosdaq_mom60: float | None
    kospi_eff60: float | None
    kosdaq_eff60: float | None
    kospi_eff60_prior_median: float | None
    kosdaq_eff60_prior_median: float | None
    leader_mom20: float | None
    leader_mom60: float | None
    breadth20: float | None
    breadth60: float | None
    kospi_dd60: float | None
    kospi_rv20_daily: float | None


@dataclass(frozen=True, slots=True)
class P27RegimeSnapshot:
    as_of: date
    state: P27RegimeState
    confidence: float
    components: Mapping[str, float]


def _zero_components() -> dict[str, float]:
    return {"beta_direction": 0.0, "trend_quality": 0.0, "opportunity": 0.0, "risk_stability": 0.0}


def classify_p27_regime(*, decision_date: date, inputs: P27RegimeInputs | None) -> P27RegimeSnapshot:
    if inputs is None:
        return P27RegimeSnapshot(as_of=decision_date, state=P27RegimeState.UNCERTAIN, confidence=0.0, components=_zero_components())
    kospi_mom20 = _finite_or_none(inputs.kospi_mom20)
    kospi_mom60 = _finite_or_none(inputs.kospi_mom60)
    kosdaq_mom20 = _finite_or_none(inputs.kosdaq_mom20)
    kosdaq_mom60 = _finite_or_none(inputs.kosdaq_mom60)
    kospi_eff60 = _finite_or_none(inputs.kospi_eff60)
    kosdaq_eff60 = _finite_or_none(inputs.kosdaq_eff60)
    kospi_prior = _finite_or_none(inputs.kospi_eff60_prior_median)
    kosdaq_prior = _finite_or_none(inputs.kosdaq_eff60_prior_median)
    leader_mom20 = _finite_or_none(inputs.leader_mom20)
    leader_mom60 = _finite_or_none(inputs.leader_mom60)
    breadth20 = _finite_or_none(inputs.breadth20)
    breadth60 = _finite_or_none(inputs.breadth60)
    kospi_dd60 = _finite_or_none(inputs.kospi_dd60)
    kospi_rv20_daily = _finite_or_none(inputs.kospi_rv20_daily)
    values = (kospi_mom20, kospi_mom60, kosdaq_mom20, kosdaq_mom60, kospi_eff60, kosdaq_eff60, kospi_prior, kosdaq_prior, leader_mom20, leader_mom60, breadth20, breadth60, kospi_dd60, kospi_rv20_daily)
    if any(v is None for v in values):
        return P27RegimeSnapshot(as_of=decision_date, state=P27RegimeState.UNCERTAIN, confidence=0.0, components=_zero_components())
    assert breadth20 is not None
    assert breadth60 is not None
    assert kospi_rv20_daily is not None
    if not (0.0 <= breadth20 <= 1.0 and 0.0 <= breadth60 <= 1.0 and kospi_rv20_daily >= 0.0):
        return P27RegimeSnapshot(as_of=decision_date, state=P27RegimeState.UNCERTAIN, confidence=0.0, components=_zero_components())
    assert kospi_mom20 is not None
    assert kospi_mom60 is not None
    assert kosdaq_mom20 is not None
    assert kosdaq_mom60 is not None
    assert kospi_eff60 is not None
    assert kosdaq_eff60 is not None
    assert kospi_prior is not None
    assert kosdaq_prior is not None
    assert leader_mom20 is not None
    assert leader_mom60 is not None
    assert kospi_dd60 is not None
    beta_direction = sum(1 for v in (kospi_mom20, kospi_mom60, kosdaq_mom20, kosdaq_mom60) if v > 0.0) / 4.0
    trend_quality = sum(1 for v, m in ((kospi_eff60, kospi_prior), (kosdaq_eff60, kosdaq_prior)) if v > m) / 2.0
    opportunity = sum(1 for ok in (leader_mom20 > 0.0, leader_mom60 > 0.0, breadth20 > 0.5, breadth60 > 0.5) if ok) / 4.0
    rv_ann = annualize_daily_realized_vol(kospi_rv20_daily)
    assert rv_ann is not None
    risk_stability = sum(1 for ok in (kospi_dd60 > -0.15, rv_ann < 0.32) if ok) / 2.0
    confidence = float((beta_direction + trend_quality + opportunity + risk_stability) / 4.0)
    if beta_direction == 1.0 and trend_quality == 1.0 and opportunity == 1.0:
        state = P27RegimeState.ON
    elif kospi_mom60 <= 0.0 and kosdaq_mom60 <= 0.0:
        state = P27RegimeState.OFF
    else:
        state = P27RegimeState.UNCERTAIN
    components: dict[str, float] = {"beta_direction": float(beta_direction), "trend_quality": float(trend_quality), "opportunity": float(opportunity), "risk_stability": float(risk_stability)}
    return P27RegimeSnapshot(as_of=decision_date, state=state, confidence=confidence, components=components)


def p27_regime_conditional_table(*, states: Sequence[P27RegimeState | str], confidences: Sequence[float], terminal_returns: Sequence[float], overlap_horizon: int = 36) -> pl.DataFrame:
    state_list = list(states)
    conf_list = [float(v) for v in confidences]
    term_list = [float(v) for v in terminal_returns]
    if len(state_list) == 0 or not (len(state_list) == len(conf_list) == len(term_list)):
        raise ValueError(f"length mismatch or empty: states={len(state_list)} confidences={len(conf_list)} terminals={len(term_list)}")
    if not isinstance(overlap_horizon, int) or isinstance(overlap_horizon, bool) or overlap_horizon < 1:
        raise ValueError(f"invalid overlap_horizon {overlap_horizon!r}: must be integer >= 1")
    for v in term_list:
        if not math.isfinite(v):
            raise ValueError(f"non-finite terminal return {v!r}")
    for v in conf_list:
        if not math.isfinite(v) or not (0.0 <= v <= 1.0):
            raise ValueError(f"invalid confidence {v!r}: must be finite in [0,1]")
    normalized = [item.value if isinstance(item, P27RegimeState) else str(item) for item in state_list]
    order = (P27RegimeState.ON, P27RegimeState.UNCERTAIN, P27RegimeState.OFF)
    rows: list[dict[str, object]] = []
    for bucket in order:
        idx = [i for i, name in enumerate(normalized) if name == bucket.value]
        n_windows = len(idx)
        n_effective = float(n_windows) / float(overlap_horizon)
        if n_windows == 0:
            rows.append({"state": bucket.value, "n_windows": 0, "n_effective": 0.0, "mean_confidence": 0.0, "p30": 0.0, "p40": 0.0, "p50": 0.0, "p60": 0.0, "ruin25": 0.0})
            continue
        bucket_conf = [conf_list[i] for i in idx]
        bucket_term = [term_list[i] for i in idx]
        mean_confidence = float(sum(bucket_conf) / n_windows)
        p30 = sum(1 for v in bucket_term if v > 0.30) / n_windows
        p40 = sum(1 for v in bucket_term if v > 0.40) / n_windows
        p50 = sum(1 for v in bucket_term if v > 0.50) / n_windows
        p60 = sum(1 for v in bucket_term if v > 0.60) / n_windows
        ruin25 = sum(1 for v in bucket_term if v < -0.25) / n_windows
        rows.append({"state": bucket.value, "n_windows": n_windows, "n_effective": float(n_effective), "mean_confidence": float(mean_confidence), "p30": float(p30), "p40": float(p40), "p50": float(p50), "p60": float(p60), "ruin25": float(ruin25)})
    return pl.DataFrame(
        rows,
        schema={"state": pl.String, "n_windows": pl.Int64, "n_effective": pl.Float64, "mean_confidence": pl.Float64, "p30": pl.Float64, "p40": pl.Float64, "p50": pl.Float64, "p60": pl.Float64, "ruin25": pl.Float64},
        strict=False,
    )
