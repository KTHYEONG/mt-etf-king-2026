"""Research protocol: causal vol regimes, causality guard, registered evidence views and the prize-line gate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from math import sqrt
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import numpy as np
import polars as pl
from numpy import typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from src.arena.evaluation import (
    DegenerateSeriesError,
    EvalConfig,
    EvaluationReport,
    InsufficientDataError,
)
from src.arena.panel import Panel
from src.arena.registry import evaluate_and_register
from src.arena.simulate import ReturnMode, SimConfig, SimResult, simulate
from src.core.config import ConfigError, load_config

Regime = Literal["low", "mid", "high", "extreme"]
REGIMES: Final[tuple[Regime, ...]] = ("low", "mid", "high", "extreme")
Label = Literal["VALIDATED", "UNVALIDATED"]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

_RESEARCH_KEYS: Final = (
    "discovery_start",
    "holdout_start",
    "regime_reference",
    "regime_window",
    "regime_edges",
    "regime_min_sessions",
    "stress_cost_bps",
    "gate_alpha",
    "gate_min_plateau_share",
    "causality_cuts",
    "annualization",
)
_REF_COLUMN: Final = "r_ref"
_MIN_YEARS: Final = 2
_TRIAL_KEYS: Final = frozenset({"strategy", "view", "cost_bps", "mode"})


@dataclass(frozen=True)
class ResearchConfig:
    discovery_start: date
    holdout_start: date
    regime_reference: str
    regime_window: int
    regime_edges: tuple[float, float, float]
    regime_min_sessions: int
    stress_cost_bps: float
    gate_alpha: float
    gate_min_plateau_share: float
    causality_cuts: int
    annualization: int


class Strategy(Protocol):
    @property
    def family(self) -> str: ...

    @property
    def strategy_id(self) -> str: ...

    @property
    def params(self) -> Mapping[str, Any]: ...

    def targets(self, panel: Panel) -> pl.DataFrame: ...


@dataclass(frozen=True)
class Evidence:
    strategy_id: str
    family: str
    discovery_dn: EvaluationReport | None
    holdout_dn: EvaluationReport | None
    full_dn: EvaluationReport | None
    holdout_stress: EvaluationReport | None
    regimes: Mapping[Regime, EvaluationReport]
    missing: tuple[str, ...]


@dataclass(frozen=True)
class Criterion:
    name: str
    value: float | None
    threshold: float
    passed: bool


@dataclass(frozen=True)
class GateVerdict:
    strategy_id: str
    passed: bool
    criteria: tuple[Criterion, ...]
    lift_at_band: tuple[float, float] | None


@dataclass(frozen=True)
class Selection:
    strategy_id: str
    label: Label
    lift_at_band_lo: float


class CausalityError(RuntimeError):
    """Raised when a strategy's decisions at or before a cut change once later data is removed."""


def _number(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"research.{key} must be a number")
    return float(value)


def _integer(value: Any, key: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"research.{key} must be an integer >= {minimum}")
    return value


def _day(value: Any, key: str) -> date:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise ConfigError(f"research.{key} must be a date")
    return value


def _edges(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ConfigError("research.regime_edges must be a list of 3 numbers")
    low, mid, high = (_number(item, "regime_edges") for item in value)
    if not 0.0 < low < mid < high:
        raise ConfigError("research.regime_edges must be strictly increasing and > 0")
    return low, mid, high


def load_research_config(name: str = "arena") -> ResearchConfig:
    """Load and validate the `research` mapping of `configs/<name>.yaml`; raises ConfigError on any problem."""
    section = load_config(name).get("research")
    if not isinstance(section, Mapping):
        raise ConfigError(f"config {name} must define a research mapping")
    if set(section) != set(_RESEARCH_KEYS):
        raise ConfigError(f"research keys must be exactly {sorted(_RESEARCH_KEYS)}")
    reference = section["regime_reference"]
    if not isinstance(reference, str) or not reference:
        raise ConfigError("research.regime_reference must be a non-empty string")
    cfg = ResearchConfig(
        discovery_start=_day(section["discovery_start"], "discovery_start"),
        holdout_start=_day(section["holdout_start"], "holdout_start"),
        regime_reference=reference,
        regime_window=_integer(section["regime_window"], "regime_window", 2),
        regime_edges=_edges(section["regime_edges"]),
        regime_min_sessions=_integer(section["regime_min_sessions"], "regime_min_sessions", 1),
        stress_cost_bps=_number(section["stress_cost_bps"], "stress_cost_bps"),
        gate_alpha=_number(section["gate_alpha"], "gate_alpha"),
        gate_min_plateau_share=_number(section["gate_min_plateau_share"], "gate_min_plateau_share"),
        causality_cuts=_integer(section["causality_cuts"], "causality_cuts", 1),
        annualization=_integer(section["annualization"], "annualization", 1),
    )
    if cfg.discovery_start >= cfg.holdout_start:
        raise ConfigError("research.discovery_start must be before holdout_start")
    if cfg.stress_cost_bps < 0.0:
        raise ConfigError("research.stress_cost_bps must be >= 0")
    if not 0.0 < cfg.gate_alpha < 1.0:
        raise ConfigError("research.gate_alpha must be in (0, 1)")
    if not 0.0 <= cfg.gate_min_plateau_share <= 1.0:
        raise ConfigError("research.gate_min_plateau_share must be in [0, 1]")
    return cfg


def regime_labels(panel: Panel, cfg: ResearchConfig) -> pl.DataFrame:
    """Return `(date, regime)` per session from the causal reference vol; regime is null until enough returns exist."""
    sessions = panel.sessions
    count = len(sessions)
    reference = panel.frame.filter(pl.col("ticker") == cfg.regime_reference).select("date", _REF_COLUMN)
    aligned = pl.DataFrame({"date": list(sessions)}, schema={"date": pl.Date}).join(reference, on="date", how="left")
    returns = aligned[_REF_COLUMN].cast(pl.Float64).fill_null(np.nan).to_numpy()

    window = cfg.regime_window
    padded = np.concatenate([np.full(window, np.nan), returns])
    past = sliding_window_view(padded, window)[:count]
    finite = np.isfinite(past)
    n_finite = finite.sum(axis=1)
    mean = np.where(finite, past, 0.0).sum(axis=1) / np.maximum(n_finite, 1)
    squares = np.where(finite, (past - mean[:, None]) ** 2, 0.0).sum(axis=1)
    std = np.sqrt(squares / np.maximum(n_finite - 1, 1))
    vol = np.where(n_finite >= window, std * sqrt(cfg.annualization), np.nan)

    low, mid, high = cfg.regime_edges
    vol_series = pl.Series("vol", vol, dtype=pl.Float64, nan_to_null=True)
    frame = pl.DataFrame({"date": list(sessions), "vol": vol_series}, schema={"date": pl.Date, "vol": pl.Float64})
    return frame.select(
        "date",
        pl.when(pl.col("vol").is_null())
        .then(pl.lit(None, dtype=pl.String))
        .when(pl.col("vol") < low)
        .then(pl.lit("low"))
        .when(pl.col("vol") < mid)
        .then(pl.lit("mid"))
        .when(pl.col("vol") < high)
        .then(pl.lit("high"))
        .otherwise(pl.lit("extreme"))
        .alias("regime"),
    )


def truncate_panel(panel: Panel, cut: date) -> Panel:
    """Return the panel as it would have looked at `cut`; raises ValueError when `cut` is outside the panel range."""
    if not panel.start <= cut <= panel.end:
        raise ValueError(f"cut {cut} is outside [{panel.start}, {panel.end}]")
    return replace(
        panel,
        end=cut,
        sessions=tuple(s for s in panel.sessions if s <= cut),
        frame=panel.frame.filter(pl.col("date") <= cut),
    )


def _up_to(targets: pl.DataFrame, cut: date) -> pl.DataFrame:
    return targets.filter(pl.col("decision_date") <= cut).sort(["decision_date", "ticker"])


def check_causality(strategy: Strategy, panel: Panel, cfg: ResearchConfig) -> pl.DataFrame:
    """Return the strategy's full-panel targets after proving decisions up to each cut ignore later data."""
    if not panel.sessions:
        raise ValueError("panel has no sessions")
    full = strategy.targets(panel)
    count = len(panel.sessions)
    for k in range(1, cfg.causality_cuts + 1):
        cut = panel.sessions[k * count // (cfg.causality_cuts + 1)]
        expected = _up_to(full, cut)
        observed = _up_to(strategy.targets(truncate_panel(panel, cut)), cut)
        if not expected.equals(observed):
            raise CausalityError(f"strategy {strategy.strategy_id} changes decisions at or before {cut}")
    return full


def _window(sessions: Sequence[date], lo: date | None, hi: date | None) -> BoolArray:
    return np.array([(lo is None or day >= lo) and (hi is None or day < hi) for day in sessions], dtype=bool)


def _select(result: SimResult, mask: BoolArray) -> tuple[list[date], FloatArray, FloatArray]:
    picked = np.flatnonzero(mask)
    return [result.sessions[i] for i in picked], result.net_returns[picked], result.costs[picked]


def gather_evidence(
    strategy: Strategy,
    panel: Panel,
    sim_cfg: SimConfig,
    eval_cfg: EvalConfig,
    cfg: ResearchConfig,
    *,
    track: str,
    capital_krw: int,
    registry: Path,
    recorded_at: datetime,
    trial_tags: Mapping[str, str] | None = None,
) -> Evidence:
    """Run the causality guard, simulate, and register every evidence view; views with too little data are None."""
    tags = dict(trial_tags or {})
    clash = sorted(set(tags) & (_TRIAL_KEYS | set(strategy.params)))
    if clash:
        raise ValueError(f"trial tags collide with trial params: {clash}")
    targets = check_causality(strategy, panel, cfg)
    if targets.is_empty():
        return Evidence(
            strategy_id=strategy.strategy_id,
            family=strategy.family,
            discovery_dn=None,
            holdout_dn=None,
            full_dn=None,
            holdout_stress=None,
            regimes={},
            missing=("all views: the strategy emits no decision on this panel",),
        )
    stress_cfg = replace(sim_cfg, cost_bps_per_side=cfg.stress_cost_bps)
    drift = simulate(
        panel, targets, sim_cfg, capital_krw=capital_krw, mode="drift_neutral", drift_segments=(cfg.holdout_start,)
    )
    drift_full = simulate(panel, targets, sim_cfg, capital_krw=capital_krw, mode="drift_neutral")
    stress = simulate(panel, targets, stress_cfg, capital_krw=capital_krw, mode="actual")
    missing: list[str] = []

    def run(
        view: str,
        picked: tuple[list[date], FloatArray, FloatArray],
        cost_bps: float,
        mode: ReturnMode,
        view_cfg: EvalConfig = eval_cfg,
    ) -> EvaluationReport | None:
        sessions, net, costs = picked
        params = {**strategy.params, **tags, "strategy": strategy.family, "view": view, "cost_bps": cost_bps, "mode": mode}
        try:
            return evaluate_and_register(
                registry,
                strategy.strategy_id,
                f"{track}/{view}",
                params,
                sessions,
                net,
                costs,
                view_cfg,
                recorded_at=recorded_at,
            )
        except (InsufficientDataError, DegenerateSeriesError) as exc:
            missing.append(f"{view}: {exc}")
            return None

    base_cost = sim_cfg.cost_bps_per_side
    stress_cost = cfg.stress_cost_bps
    discovery_dn = run(
        "discovery_dn",
        _select(drift, _window(drift.sessions, cfg.discovery_start, cfg.holdout_start)),
        base_cost,
        "drift_neutral",
    )
    holdout_dn = run(
        "holdout_dn", _select(drift, _window(drift.sessions, cfg.holdout_start, None)), base_cost, "drift_neutral"
    )
    full_dn = run("full_dn", _select(drift_full, _window(drift_full.sessions, None, None)), base_cost, "drift_neutral")
    holdout_stress = run(
        "holdout_stress", _select(stress, _window(stress.sessions, cfg.holdout_start, None)), stress_cost, "actual"
    )

    labels = regime_labels(panel, cfg)
    label_of = dict(zip(labels["date"].to_list(), labels["regime"].to_list(), strict=True))
    stress_labels = [label_of.get(day) for day in stress.sessions]
    regimes: dict[Regime, EvaluationReport] = {}
    for regime in REGIMES:
        mask = np.array([label == regime for label in stress_labels], dtype=bool)
        if int(mask.sum()) < cfg.regime_min_sessions:
            missing.append(
                f"regime_{regime}: {int(mask.sum())} sessions < regime_min_sessions {cfg.regime_min_sessions}"
            )
            continue
        # Regime subsets are shorter than a full evaluation history by design; their floor is regime_min_sessions.
        regime_cfg = replace(eval_cfg, min_sessions=max(cfg.regime_min_sessions, eval_cfg.horizon_sessions))
        report = run(f"regime_{regime}", _select(stress, mask), stress_cost, "actual", regime_cfg)
        if report is not None:
            regimes[regime] = report

    return Evidence(
        strategy_id=strategy.strategy_id,
        family=strategy.family,
        discovery_dn=discovery_dn,
        holdout_dn=holdout_dn,
        full_dn=full_dn,
        holdout_stress=holdout_stress,
        regimes=regimes,
        missing=tuple(missing),
    )


def _p_criterion(name: str, report: EvaluationReport | None, alpha: float) -> Criterion:
    value = None if report is None else report.edge.p_value_adjusted
    return Criterion(name, value, alpha, value is not None and value <= alpha)


def leave_one_year_out(full: EvaluationReport | None) -> Criterion:
    if full is None or len(full.by_year) < _MIN_YEARS:
        return Criterion("leave_one_year_out", None, 0.0, False)
    n = np.array([y.n_sessions for y in full.by_year], dtype=np.float64)
    m = np.array([y.mean_gross for y in full.by_year], dtype=np.float64)
    others = ((n * m).sum() - n * m) / (n.sum() - n)
    value = float(others.min())
    return Criterion("leave_one_year_out", value, 0.0, value > 0.0)


def _lift_criterion(name: str, lift: float | None) -> Criterion:
    return Criterion(name, lift, 0.0, lift is not None and lift > 0.0)


def gate(evidence: Evidence, cfg: ResearchConfig, band: tuple[float, float], family: Sequence[Evidence]) -> GateVerdict:
    """Evaluate the declared criteria for one candidate; `family` is its same-family grid, including itself."""
    lo, hi = band
    stress = evidence.holdout_stress
    lift_at_band: tuple[float, float] | None = None
    if stress is not None:
        grid = stress.exceedance.thresholds
        if lo not in grid or hi not in grid:
            raise ValueError(f"band {band} must lie on the holdout_stress exceedance grid {grid}")
        lift_at_band = (stress.exceedance.lift[grid.index(lo)], stress.exceedance.lift[grid.index(hi)])
    if not any(e.strategy_id == evidence.strategy_id and e.family == evidence.family for e in family):
        raise ValueError(f"family must include {evidence.strategy_id} within {evidence.family}")

    same = [e for e in family if e.family == evidence.family]
    positive = sum(1 for e in same if e.full_dn is not None and e.full_dn.edge.mean_gross > 0.0)
    share = positive / len(same)
    criteria = (
        _p_criterion("discovery_edge", evidence.discovery_dn, cfg.gate_alpha),
        _p_criterion("holdout_edge", evidence.holdout_dn, cfg.gate_alpha),
        leave_one_year_out(evidence.full_dn),
        _lift_criterion("lift_lo", None if lift_at_band is None else lift_at_band[0]),
        _lift_criterion("lift_hi", None if lift_at_band is None else lift_at_band[1]),
        Criterion("plateau", share, cfg.gate_min_plateau_share, share >= cfg.gate_min_plateau_share),
    )
    return GateVerdict(
        strategy_id=evidence.strategy_id,
        passed=all(c.passed for c in criteria),
        criteria=criteria,
        lift_at_band=lift_at_band,
    )


def select(verdicts: Sequence[GateVerdict]) -> Selection | None:
    """Pick the best candidate by lift at the band's low threshold, preferring gate-passing ones; ties go to the smaller id."""
    scored = [(v, band[0]) for v in verdicts if (band := v.lift_at_band) is not None]
    if not scored:
        return None
    passed = [pair for pair in scored if pair[0].passed]
    pool = passed or scored
    best, lift = min(pool, key=lambda pair: (-pair[1], pair[0].strategy_id))
    label: Label = "VALIDATED" if passed else "UNVALIDATED"
    return Selection(strategy_id=best.strategy_id, label=label, lift_at_band_lo=lift)
