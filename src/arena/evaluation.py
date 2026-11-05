"""Timing-edge evaluation against a zero-edge twin over stationary-bootstrap resamples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from itertools import pairwise
from math import erfc, sqrt
from typing import Any, Final

import numpy as np
from numpy import typing as npt

from src.core.config import ConfigError, load_config

_EVAL_KEYS: Final = (
    "horizon_sessions",
    "bootstrap_mean_block",
    "bootstrap_draws",
    "seed",
    "ci_level",
    "exceedance_grid",
    "min_sessions",
    "lock_in",
    "lock_exit_cost_bps",
)
_SEED_STREAMS: Final = 3
_EDGE_STREAM: Final = 0
_HORIZON_STREAM: Final = 1
_SIGN_STREAM: Final = 2
_MAX_LOSS: Final = -1.0
_Q_LOW: Final = 0.05
_Q_MEDIAN: Final = 0.5
_Q_HIGH: Final = 0.95
_BPS_PER_UNIT: Final = 10_000.0


@dataclass(frozen=True)
class EvalConfig:
    horizon_sessions: int
    bootstrap_mean_block: int
    bootstrap_draws: int
    seed: int
    ci_level: float
    exceedance_grid: tuple[float, ...]
    min_sessions: int
    lock_in: bool = False
    lock_exit_cost_bps: float = 0.0


@dataclass(frozen=True)
class EdgeStats:
    n_sessions: int
    mean_gross: float
    ci_low: float
    ci_high: float
    se_boot: float
    t_stat: float
    p_value: float
    n_trials: int
    p_value_adjusted: float


@dataclass(frozen=True)
class Exceedance:
    horizon_sessions: int
    n_effective: int
    thresholds: tuple[float, ...]
    p_strategy: tuple[float, ...]
    p_twin: tuple[float, ...]
    lift: tuple[float, ...]
    q05: float
    q50: float
    q95: float
    twin_q50: float
    lock_in: bool
    p_strategy_hold: tuple[float, ...]
    p_twin_hold: tuple[float, ...]


@dataclass(frozen=True)
class YearStat:
    year: int
    n_sessions: int
    mean_gross: float


@dataclass(frozen=True)
class EvaluationReport:
    strategy_id: str
    config_hash: str
    first_session: date
    last_session: date
    edge: EdgeStats
    exceedance: Exceedance
    by_year: tuple[YearStat, ...]
    positive_year_share: float


class InsufficientDataError(ValueError):
    """Raised when the series is shorter than `min_sessions`."""


class DegenerateSeriesError(ValueError):
    """Raised when the bootstrap standard error is zero, so the t-statistic is undefined."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _as_int(value: Any, key: str) -> int:
    if type(value) is not int:
        raise ConfigError(f"evaluation.{key} must be an integer")
    return value


def _as_float(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"evaluation.{key} must be a number")
    return float(value)


def _as_bool(value: Any) -> bool:
    if type(value) is not bool:
        raise ConfigError("evaluation.lock_in must be a boolean")
    return value


def _as_grid(value: Any) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError("evaluation.exceedance_grid must be a non-empty list")
    return tuple(_as_float(item, "exceedance_grid") for item in value)


def load_eval_config(name: str = "arena") -> EvalConfig:
    """Load and validate the `evaluation` mapping of `configs/<name>.yaml`; raises ConfigError on any problem."""
    section = load_config(name).get("evaluation")
    if not isinstance(section, Mapping):
        raise ConfigError(f"config {name} must define an evaluation mapping")
    if set(section) != set(_EVAL_KEYS):
        raise ConfigError(f"evaluation keys must be exactly {sorted(_EVAL_KEYS)}")
    cfg = EvalConfig(
        horizon_sessions=_as_int(section["horizon_sessions"], "horizon_sessions"),
        bootstrap_mean_block=_as_int(section["bootstrap_mean_block"], "bootstrap_mean_block"),
        bootstrap_draws=_as_int(section["bootstrap_draws"], "bootstrap_draws"),
        seed=_as_int(section["seed"], "seed"),
        ci_level=_as_float(section["ci_level"], "ci_level"),
        exceedance_grid=_as_grid(section["exceedance_grid"]),
        min_sessions=_as_int(section["min_sessions"], "min_sessions"),
        lock_in=_as_bool(section["lock_in"]),
        lock_exit_cost_bps=_as_float(section["lock_exit_cost_bps"], "lock_exit_cost_bps"),
    )
    _require(cfg.horizon_sessions >= 1, "evaluation.horizon_sessions must be >= 1")
    _require(cfg.bootstrap_mean_block >= 1, "evaluation.bootstrap_mean_block must be >= 1")
    _require(cfg.bootstrap_draws >= 100, "evaluation.bootstrap_draws must be >= 100")
    _require(0.0 < cfg.ci_level < 1.0, "evaluation.ci_level must be in (0, 1)")
    _require(all(w > 0 for w in cfg.exceedance_grid), "evaluation.exceedance_grid must be > 0")
    _require(
        all(b > a for a, b in pairwise(cfg.exceedance_grid)),
        "evaluation.exceedance_grid must be strictly increasing",
    )
    _require(cfg.min_sessions >= cfg.horizon_sessions, "evaluation.min_sessions must be >= horizon_sessions")
    _require(
        0.0 <= cfg.lock_exit_cost_bps < _BPS_PER_UNIT,
        "evaluation.lock_exit_cost_bps must be in [0, 10000)",
    )
    return cfg


def stationary_bootstrap_indices(
    n: int, length: int, mean_block: int, draws: int, rng: np.random.Generator
) -> npt.NDArray[np.int64]:
    """Return `(draws, length)` circular stationary-bootstrap indices in [0, n) with geometric block lengths."""
    if min(n, length, mean_block, draws) < 1:
        raise ValueError("n, length, mean_block and draws must be >= 1")
    restart = rng.random((draws, length)) < 1.0 / mean_block
    restart[:, 0] = True
    starts = rng.integers(0, n, size=(draws, length))
    idx = np.empty((draws, length), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, length):
        idx[:, t] = np.where(restart[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def edge_stats(gross: npt.NDArray[np.float64], idx: npt.NDArray[np.int64], cfg: EvalConfig, n_trials: int) -> EdgeStats:
    """Bootstrap the mean gross return and test it one-sided against zero, with a Sidak trial haircut."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    mean_gross = float(gross.mean())
    boot_means = gross[idx].mean(axis=1)
    if np.ptp(boot_means) == 0.0:
        raise DegenerateSeriesError("bootstrap means do not vary; standard error is zero")
    se_boot = float(boot_means.std(ddof=1))
    tail = (1.0 - cfg.ci_level) / 2.0
    ci_low, ci_high = np.quantile(boot_means, [tail, 1.0 - tail])
    t_stat = mean_gross / se_boot
    p_value = 0.5 * erfc(t_stat / sqrt(2.0))
    return EdgeStats(
        n_sessions=int(gross.size),
        mean_gross=mean_gross,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        se_boot=se_boot,
        t_stat=t_stat,
        p_value=p_value,
        n_trials=n_trials,
        p_value_adjusted=1.0 - (1.0 - p_value) ** n_trials,
    )


def lock_in_multiples(paths: npt.NDArray[np.float64], threshold: float, exit_cost: float) -> npt.NDArray[np.float64]:
    """Return each path's terminal multiple under lock-in at `threshold`; the flatten executes one session after the crossing."""
    if paths.ndim != 2 or paths.shape[1] < 1:
        raise ValueError("paths must be 2-D with at least one session")
    if not threshold > 0.0:
        raise ValueError("threshold must be > 0")
    if not 0.0 <= exit_cost < 1.0:
        raise ValueError("exit_cost must be in [0, 1)")
    horizon = paths.shape[1]
    crossed = paths >= threshold
    hit = crossed.any(axis=1)
    exit_idx = crossed.argmax(axis=1) + 1
    use_exit = hit & (exit_idx < horizon - 1)
    rows = np.arange(paths.shape[0])
    out = paths[:, -1].copy()
    out[use_exit] = paths[rows[use_exit], exit_idx[use_exit]] * (1.0 - exit_cost)
    return out


def _hit_rates(terminal: npt.NDArray[np.float64], thresholds: npt.NDArray[np.float64]) -> tuple[float, ...]:
    return tuple(float(p) for p in (terminal[:, None] >= thresholds[None, :]).mean(axis=0))


def exceedance(
    net: npt.NDArray[np.float64],
    costs: npt.NDArray[np.float64],
    idx: npt.NDArray[np.int64],
    signs: npt.NDArray[np.int8],
    cfg: EvalConfig,
) -> Exceedance:
    """Compare strategy and zero-edge twin multiples over the grid; `p_*` follow lock-in when enabled, quantiles stay hold-to-end."""
    r = np.maximum(net[idx], _MAX_LOSS)
    r0 = np.maximum(signs * np.abs(net + costs)[idx] - costs[idx], _MAX_LOSS)
    m = np.prod(1.0 + r, axis=1)
    m0 = np.prod(1.0 + r0, axis=1)
    thresholds = np.asarray(cfg.exceedance_grid, dtype=np.float64)
    p_strategy_hold = _hit_rates(m, thresholds)
    p_twin_hold = _hit_rates(m0, thresholds)
    if cfg.lock_in:
        exit_cost = cfg.lock_exit_cost_bps / _BPS_PER_UNIT
        cp = np.cumprod(1.0 + r, axis=1, dtype=np.float64)
        cp0 = np.cumprod(1.0 + r0, axis=1, dtype=np.float64)
        p_strategy = tuple(float(np.mean(lock_in_multiples(cp, float(w), exit_cost) >= w)) for w in cfg.exceedance_grid)
        p_twin = tuple(float(np.mean(lock_in_multiples(cp0, float(w), exit_cost) >= w)) for w in cfg.exceedance_grid)
    else:
        p_strategy, p_twin = p_strategy_hold, p_twin_hold
    q_low, q_median, q_high = np.quantile(m, [_Q_LOW, _Q_MEDIAN, _Q_HIGH])
    return Exceedance(
        horizon_sessions=cfg.horizon_sessions,
        n_effective=len(net) // cfg.horizon_sessions,
        thresholds=cfg.exceedance_grid,
        p_strategy=p_strategy,
        p_twin=p_twin,
        lift=tuple(s - t for s, t in zip(p_strategy, p_twin, strict=True)),
        q05=float(q_low),
        q50=float(q_median),
        q95=float(q_high),
        twin_q50=float(np.median(m0)),
        lock_in=cfg.lock_in,
        p_strategy_hold=p_strategy_hold,
        p_twin_hold=p_twin_hold,
    )


def evaluate(
    strategy_id: str,
    config_hash: str,
    sessions: Sequence[date],
    net_returns: npt.NDArray[np.float64],
    costs: npt.NDArray[np.float64],
    cfg: EvalConfig,
    n_trials: int,
) -> EvaluationReport:
    """Validate a per-session series and return its edge and exceedance report; raises ValueError or InsufficientDataError."""
    n = len(sessions)
    if not n == len(net_returns) == len(costs):
        raise ValueError("sessions, net_returns and costs must have equal lengths")
    if not all(b > a for a, b in pairwise(sessions)):
        raise ValueError("sessions must be strictly increasing")
    if not (np.isfinite(net_returns).all() and np.isfinite(costs).all()):
        raise ValueError("net_returns and costs must be finite")
    if bool((net_returns <= _MAX_LOSS).any()) or bool((costs < 0.0).any()):
        raise ValueError("net_returns must be > -1 and costs must be >= 0")
    if n < cfg.min_sessions:
        raise InsufficientDataError(f"{n} sessions < min_sessions {cfg.min_sessions}")

    seeds = np.random.SeedSequence(cfg.seed).spawn(_SEED_STREAMS)
    gross = net_returns + costs
    edge_idx = stationary_bootstrap_indices(
        n, n, cfg.bootstrap_mean_block, cfg.bootstrap_draws, np.random.default_rng(seeds[_EDGE_STREAM])
    )
    horizon_idx = stationary_bootstrap_indices(
        n,
        cfg.horizon_sessions,
        cfg.bootstrap_mean_block,
        cfg.bootstrap_draws,
        np.random.default_rng(seeds[_HORIZON_STREAM]),
    )
    signs = np.random.default_rng(seeds[_SIGN_STREAM]).choice(
        np.array([-1, 1], dtype=np.int8), size=(cfg.bootstrap_draws, cfg.horizon_sessions)
    )

    years = np.fromiter((s.year for s in sessions), dtype=np.int64, count=n)
    by_year = tuple(
        YearStat(
            year=int(year),
            n_sessions=int(np.count_nonzero(years == year)),
            mean_gross=float(gross[years == year].mean()),
        )
        for year in np.unique(years)
    )
    positive_year_share = sum(1 for y in by_year if y.mean_gross > 0.0) / len(by_year)

    return EvaluationReport(
        strategy_id=strategy_id,
        config_hash=config_hash,
        first_session=sessions[0],
        last_session=sessions[-1],
        edge=edge_stats(gross, edge_idx, cfg, n_trials),
        exceedance=exceedance(net_returns, costs, horizon_idx, signs, cfg),
        by_year=by_year,
        positive_year_share=positive_year_share,
    )


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


def report_to_dict(report: EvaluationReport) -> dict[str, Any]:
    """Return a JSON-safe dict with ISO dates and lists in place of tuples."""
    result: dict[str, Any] = _plain(asdict(report))
    return result
