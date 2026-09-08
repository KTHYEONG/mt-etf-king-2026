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
