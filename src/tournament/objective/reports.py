# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

from src.core.config import config_value
from src.tournament.distribution import ReturnDistribution, ruin_probability
from src.tournament.objective_core import BOOTSTRAP_EXPECTED_BLOCK
from src.tournament.objective_core import _dist_exceedance
@dataclass(frozen=True)
class ChampionshipObjectiveConfig:
    thresholds: tuple[float, ...]
    scenario_weights: Mapping[str, tuple[float, ...]]
    primary_scenario: str
    ruin_threshold: float = -0.25
    ruin_max: float = 0.05
    max_effective_gross: float = 1.60
    bootstrap_expected_block: int = BOOTSTRAP_EXPECTED_BLOCK
    bootstrap_resamples: int = 2000
    seed: int = 0
    min_era_effective: int = 5

    def __init__(
        self,
        thresholds: tuple[float, ...],
        scenario_weights: Mapping[str, tuple[float, ...]] | Mapping[str, float],
        primary_scenario: str,
        ruin_threshold: float = -0.25,
        ruin_max: float = 0.05,
        max_effective_gross: float = 1.60,
        bootstrap_expected_block: int = BOOTSTRAP_EXPECTED_BLOCK,
        bootstrap_resamples: int = 2000,
        seed: int = 0,
        min_era_effective: int = 5,
        bootstrap_samples: int | None = None,
        bootstrap_seed: int | None = None,
        thresholds_alias: tuple[float, ...] | None = None,
    ) -> None:
        # alias handling
        if bootstrap_samples is not None:
            bootstrap_resamples = int(bootstrap_samples)
        if bootstrap_seed is not None:
            seed = int(bootstrap_seed)
        # normalize thresholds
        thr = tuple(float(x) for x in thresholds) if thresholds is not None else ()  # type: ignore[arg-type]
        # normalize scenario_weights values to tuple if float
        norm_sw: dict[str, tuple[float, ...]] = {}
        for k, v in dict(scenario_weights).items():
            if isinstance(v, (list, tuple)):
                norm_sw[str(k)] = tuple(float(x) for x in v)  # type: ignore[arg-type]
            else:
                try:
                    norm_sw[str(k)] = (float(v),)  # type: ignore[arg-type]
                except Exception:
                    norm_sw[str(k)] = tuple()  # type: ignore[assignment]
        object.__setattr__(self, "thresholds", thr)
        object.__setattr__(self, "scenario_weights", norm_sw)
        object.__setattr__(self, "primary_scenario", str(primary_scenario))
        object.__setattr__(self, "ruin_threshold", float(ruin_threshold))
        object.__setattr__(self, "ruin_max", float(ruin_max))
        object.__setattr__(self, "max_effective_gross", float(max_effective_gross))
        object.__setattr__(self, "bootstrap_expected_block", int(bootstrap_expected_block))
        object.__setattr__(self, "bootstrap_resamples", int(bootstrap_resamples))
        object.__setattr__(self, "seed", int(seed))
        object.__setattr__(self, "min_era_effective", int(min_era_effective))

    @classmethod
    def from_yaml(cls, gates_path: Path, portfolio_path: Path) -> "ChampionshipObjectiveConfig":
        import math as _math

        # load gates
        with open(gates_path, encoding="utf-8") as f:
            raw_g = yaml.safe_load(f) or {}
        if not isinstance(raw_g, dict):
            raise ValueError("gates yaml root must be mapping")
        champ: dict | None = None
        if "championship" in raw_g and isinstance(raw_g["championship"], dict):
            champ = raw_g["championship"]
        elif "gates" in raw_g and isinstance(raw_g["gates"], dict) and "championship" in raw_g["gates"]:
            champ = raw_g["gates"]["championship"]
        if champ is None or not isinstance(champ, dict):
            raise ValueError("championship config missing in gates yaml")
        # thresholds
        thr_raw = champ.get("thresholds")
        if thr_raw is None:
            raise ValueError("championship.thresholds missing")
        if not isinstance(thr_raw, (list, tuple)):
            raise ValueError("championship.thresholds must be sequence")
        thresholds = tuple(float(x) for x in thr_raw)  # type: ignore[arg-type]
        if len(thresholds) == 0:
            raise ValueError("thresholds empty")
        for v in thresholds:
            if not _math.isfinite(v):
                raise ValueError("thresholds must be finite")
        # scenario_weights
        sw_raw = champ.get("scenario_weights")
        if sw_raw is None:
            raise ValueError("championship.scenario_weights missing")
        if not isinstance(sw_raw, dict):
            raise ValueError("scenario_weights must be mapping")
        scenario_weights: dict[str, tuple[float, ...]] = {}
        for k, vals in sw_raw.items():
            if not isinstance(vals, (list, tuple)):
                raise ValueError(f"scenario_weights[{k}] must be sequence")
            w = tuple(float(x) for x in vals)  # type: ignore[arg-type]
            if len(w) != len(thresholds):
                raise ValueError(f"scenario_weights[{k}] length must equal thresholds")
            s = sum(w)
            if abs(s - 1.0) > 1e-9:
                raise ValueError(f"scenario_weights[{k}] must sum to 1.0, got {s}")
            for vv in w:
                if not _math.isfinite(vv) or vv < -1e-12:
                    raise ValueError(f"scenario_weights[{k}] contains invalid weight {vv}")
            scenario_weights[str(k)] = w
        primary = champ.get("primary_scenario")
        if not isinstance(primary, str) or not primary:
            raise ValueError("championship.primary_scenario missing or invalid")
        if primary not in scenario_weights:
            raise ValueError("primary_scenario must exist in scenario_weights")
        # bootstrap params
        beb = champ.get("bootstrap_expected_block")
        bres = champ.get("bootstrap_resamples")
        sd = champ.get("seed")
        mer = champ.get("min_era_effective")
        # allow fallback to gates.g2a etc? but require present
        if beb is None or bres is None or sd is None or mer is None:
            # try alternative keys? fail
            raise ValueError("championship bootstrap/seed/min_era_effective missing")
        try:
            beb_i = int(beb)  # type: ignore[arg-type]
            bres_i = int(bres)  # type: ignore[arg-type]
            seed_i = int(sd)  # type: ignore[arg-type]
            mer_i = int(mer)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(f"bootstrap/seed/min_era invalid: {exc}") from exc
        if not _math.isfinite(float(beb_i)) or beb_i <= 0:
            raise ValueError("bootstrap_expected_block must be >0")
        if beb_i < BOOTSTRAP_EXPECTED_BLOCK:
            raise ValueError(f"bootstrap_expected_block must be >={BOOTSTRAP_EXPECTED_BLOCK}")
        if bres_i <= 0:
            raise ValueError("bootstrap_resamples must be >0")
        if mer_i < 0:
            raise ValueError("min_era_effective must be >=0")
        # ruin thresholds from gates
        gates_block = raw_g.get("gates") if isinstance(raw_g.get("gates"), dict) else raw_g
        if not isinstance(gates_block, dict):
            gates_block = {}
        # try to get g2a fields
        ruin_threshold = -0.25
        ruin_max = 0.05
        if "g2a_ruin_threshold" in gates_block:
            try:
                ruin_threshold = float(gates_block["g2a_ruin_threshold"])  # type: ignore[arg-type]
            except Exception as exc:
                raise ValueError(f"g2a_ruin_threshold invalid: {exc}") from exc
            if not _math.isfinite(ruin_threshold):
                raise ValueError("g2a_ruin_threshold must be finite")
        if "g2a_max_prob" in gates_block:
            try:
                ruin_max = float(gates_block["g2a_max_prob"])  # type: ignore[arg-type]
            except Exception as exc:
                raise ValueError(f"g2a_max_prob invalid: {exc}") from exc
            if not _math.isfinite(ruin_max) or ruin_max < 0 or ruin_max > 1:
                raise ValueError("g2a_max_prob must be in [0,1]")
        # max_effective_gross from portfolio
        with open(portfolio_path, encoding="utf-8") as f:
            raw_p = yaml.safe_load(f) or {}
        if not isinstance(raw_p, dict):
            raise ValueError("portfolio yaml root must be mapping")
        port = raw_p.get("portfolio") if isinstance(raw_p.get("portfolio"), dict) else raw_p
        if not isinstance(port, dict):
            raise ValueError("portfolio block missing")
        for kk in ("max_gross_exposure", "max_single_weight", "min_cash"):
            if kk not in port:
                raise ValueError(f"portfolio.{kk} missing")
        try:
            max_gross = float(port["max_gross_exposure"])  # type: ignore[arg-type]
            max_single = float(port["max_single_weight"])  # type: ignore[arg-type]
            min_cash = float(port["min_cash"])  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(f"portfolio limit invalid: {exc}") from exc
        if not _math.isfinite(max_gross) or not _math.isfinite(max_single) or not _math.isfinite(min_cash):
            raise ValueError("portfolio limits must be finite")
        if max_gross <= 0 or max_single <= 0 or min_cash < 0 or min_cash > 1:
            raise ValueError("portfolio limits out of range")
        max_effective_gross = float(max_gross)
        # strict check: expected 1.60 for current portfolio
        return cls(
            thresholds=thresholds,
            scenario_weights=dict(scenario_weights),
            primary_scenario=str(primary),
            ruin_threshold=float(ruin_threshold),
            ruin_max=float(ruin_max),
            max_effective_gross=float(max_effective_gross),
            bootstrap_expected_block=int(beb_i),
            bootstrap_resamples=int(bres_i),
            seed=int(seed_i),
            min_era_effective=int(mer_i),
        )


@dataclass(frozen=True)
class ChampionshipTailReport:
    thresholds: tuple[float, ...]
    exceedance: Mapping[float, float]
    exceedance_ci: Mapping[float, tuple[float, float]]
    scenario_scores: Mapping[str, float]
    ruin_probability: float
    n_windows: int
    n_effective: int


@dataclass(frozen=True)
class ChampionshipAdoptionResult:
    status: str
    failures: tuple[str, ...]
    candidate: ChampionshipTailReport | None
    incumbent: ChampionshipTailReport | None
    raw: ChampionshipTailReport | None
    scenario_delta_ci: Mapping[str, tuple[float, float]]
    era_deltas: Mapping[str, float]


def championship_tail_report(
    returns: Sequence[float], horizon: int, config: ChampionshipObjectiveConfig
) -> ChampionshipTailReport:
    import math as _math

    if not returns:
        raise ValueError("returns empty")
    # check non-finite
    for v in returns:
        try:
            fv = float(v)  # type: ignore[arg-type]
        except Exception:
            raise ValueError("returns contains non-numeric")
        if not _math.isfinite(fv):
            raise ValueError("returns contains non-finite")
    try:
        h = int(horizon)
    except Exception:
        raise ValueError("horizon must be int")
    if h <= 0:
        raise ValueError("horizon must be >0")
    if config.bootstrap_expected_block < h:
        raise ValueError("bootstrap_expected_block must be >= horizon")
    n_windows = int(len(returns))
    # effective sample
    from src.tournament.distribution import effective_sample_size, ruin_probability, stationary_bootstrap_ci

    n_effective = int(effective_sample_size(n_windows, h)) if n_windows else 0
    # exceedance
    exceedance: dict[float, float] = {}
    exceedance_ci: dict[float, tuple[float, float]] = {}
    for thr in config.thresholds:
        ft = float(thr)
        cnt = sum(1 for r in returns if float(r) > ft)
        exceedance[ft] = float(cnt / n_windows) if n_windows else 0.0
        # bootstrap CI for exceedance statistic
        def _stat(sample: Sequence[float], _thr: float = ft) -> float:
            if not sample:
                return 0.0
            c = sum(1 for x in sample if float(x) > _thr)
            return float(c) / float(len(sample)) if len(sample) else 0.0

        try:
            ci = stationary_bootstrap_ci(
                returns,
                _stat,
                expected_block=int(config.bootstrap_expected_block),
                n_resamples=int(config.bootstrap_resamples),
                seed=int(config.seed),
            )
        except Exception:
            ci = (0.0, 0.0)
        exceedance_ci[ft] = (float(ci[0]), float(ci[1]))
    # scenario scores
    scenario_scores: dict[str, float] = {}
    for scen, weights in config.scenario_weights.items():
        score = 0.0
        for thr, w in zip(config.thresholds, weights):
            score += float(w) * float(exceedance.get(float(thr), 0.0))
        scenario_scores[str(scen)] = float(score)
    ruin_prob = float(ruin_probability(returns, float(config.ruin_threshold)))
    return ChampionshipTailReport(
        thresholds=tuple(float(x) for x in config.thresholds),
        exceedance=dict(exceedance),
        exceedance_ci=dict(exceedance_ci),
        scenario_scores=dict(scenario_scores),
        ruin_probability=float(ruin_prob),
        n_windows=int(n_windows),
        n_effective=int(n_effective),
    )


def paired_scenario_delta_ci(
    candidate_returns: Sequence[float],
    control_returns: Sequence[float],
    thresholds: Sequence[float],
    weights: Sequence[float],
    *,
    expected_block: int,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    import math as _math
    import random as _random

    if not candidate_returns or not control_returns:
        raise ValueError("returns empty")
    if len(candidate_returns) != len(control_returns):
        raise ValueError("length mismatch")
    if not thresholds or not weights:
        raise ValueError("thresholds/weights empty")
    if len(thresholds) != len(weights):
        raise ValueError("thresholds and weights length mismatch")
    try:
        eb = int(expected_block)
    except Exception:
        raise ValueError("expected_block must be int")
    if eb <= 0:
        raise ValueError("expected_block must be >0")
    try:
        nr = int(n_resamples)
    except Exception:
        raise ValueError("n_resamples must be int")
    if nr <= 0:
        raise ValueError("n_resamples must be >0")
    # weights sum to 1
    w_sum = sum(float(x) for x in weights)  # type: ignore[arg-type]
    if abs(w_sum - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1.0, got {w_sum}")
    for w in weights:
        try:
            fv = float(w)  # type: ignore[arg-type]
        except Exception:
            raise ValueError("weights contain non-numeric")
        if not _math.isfinite(fv):
            raise ValueError("weights contain non-finite")
    for t in thresholds:
        try:
            fv = float(t)  # type: ignore[arg-type]
        except Exception:
            raise ValueError("thresholds contain non-numeric")
        if not _math.isfinite(fv):
            raise ValueError("thresholds contain non-finite")
    n = len(candidate_returns)
    # check non-finite returns
    for v in candidate_returns:
        try:
            fv = float(v)  # type: ignore[arg-type]
        except Exception:
            raise ValueError("candidate_returns non-numeric")
        if not _math.isfinite(fv):
            raise ValueError("candidate_returns non-finite")
    for v in control_returns:
        try:
            fv = float(v)  # type: ignore[arg-type]
        except Exception:
            raise ValueError("control_returns non-numeric")
        if not _math.isfinite(fv):
            raise ValueError("control_returns non-finite")
    # compute weighted exceedance indicator per window
    diffs: list[float] = []
    for idx in range(n):
        cand_score = 0.0
        ctrl_score = 0.0
        for thr, w in zip(thresholds, weights):
            ft = float(thr)
            fw = float(w)
            cand_ind = 1.0 if float(candidate_returns[idx]) > ft else 0.0  # type: ignore[index]
            ctrl_ind = 1.0 if float(control_returns[idx]) > ft else 0.0  # type: ignore[index]
            cand_score += fw * cand_ind
            ctrl_score += fw * ctrl_ind
        diffs.append(float(cand_score - ctrl_score))
    # stationary bootstrap on diffs
    p = 1.0 / float(eb)
    rng = _random.Random(int(seed))
    stats: list[float] = []
    arr = list(diffs)
    for _ in range(nr):
        sample: list[float] = []
        while len(sample) < n:
            start = rng.randrange(n)
            length = 1
            while length < n - len(sample):
                if rng.random() < p:
                    break
                length += 1
            length = max(1, length)
            for k in range(length):
                if len(sample) >= n:
                    break
                sample.append(arr[(start + k) % n])
        sample = sample[:n]
        s = float(sum(sample) / len(sample)) if sample else 0.0
        stats.append(s)
    stats.sort()
    lower_idx = int(_math.floor(0.025 * nr))
    upper_idx = int(_math.ceil(0.975 * nr)) - 1
    lower_idx = max(0, min(lower_idx, nr - 1))
    upper_idx = max(0, min(upper_idx, nr - 1))
    lower = float(stats[lower_idx])
    upper = float(stats[upper_idx])
    if lower > upper:
        lower, upper = upper, lower
    return (lower, upper)


GROSS_METRIC_UNAVAILABLE: str = "GROSS_METRIC_UNAVAILABLE"


@dataclass(frozen=True)
class FieldRelativeReport:
    n_windows: int
    n_agents: int
    n_effective: int
    win_rate: float
    top2_rate: float
    median_rank_percentile: float


def field_relative_report(
    candidate_returns: Sequence[float],
    rival_returns: Mapping[str, Sequence[float]],
    *,
    horizon: int,
) -> FieldRelativeReport:
    import math as _math
    import statistics as _stats

    from src.tournament.distribution import effective_sample_size

    try:
        h = int(horizon)  # type: ignore[arg-type]
    except Exception:
        raise ValueError("horizon must be >0")
    if h <= 0:
        raise ValueError("horizon must be >0")
    if not isinstance(candidate_returns, Sequence) or len(candidate_returns) == 0:
        raise ValueError("candidate empty")
    if not isinstance(rival_returns, Mapping) or len(rival_returns) == 0:
        raise ValueError("rival empty")
    n_windows = len(candidate_returns)  # type: ignore[arg-type]
    # check candidate finite
    for v in candidate_returns:  # type: ignore[union-attr]
        try:
            fv = float(v)  # type: ignore[arg-type]
        except Exception:
            raise ValueError("candidate contains non-numeric")
        if not _math.isfinite(fv):
            raise ValueError("candidate contains non-finite")
    # check rivals: identical lengths, finite
    for key, seq in rival_returns.items():  # type: ignore[union-attr]
        if not isinstance(seq, Sequence):
            raise ValueError("rival sequence invalid")
        if len(seq) != n_windows:  # type: ignore[arg-type]
            raise ValueError("length mismatch")
        for v in seq:  # type: ignore[union-attr]
            try:
                fv = float(v)  # type: ignore[arg-type]
            except Exception:
                raise ValueError("rival contains non-numeric")
            if not _math.isfinite(fv):
                raise ValueError("rival contains non-finite")
    n_agents = 1 + len(rival_returns)
    n_effective = int(effective_sample_size(n_windows, h))
    # compute per window
    win_count = 0
    top2_count = 0
    rank_percentiles: list[float] = []
    rival_lists = list(rival_returns.values())  # type: ignore[union-attr]
    for i in range(n_windows):
        cand = float(candidate_returns[i])  # type: ignore[index]
        rival_vals = [float(r[i]) for r in rival_lists]  # type: ignore[index]
        max_rival = max(rival_vals) if rival_vals else float("-inf")
        if cand > max_rival:
            win_count += 1
        cnt_ge = sum(1 for rv in rival_vals if rv >= cand)
        rank = 1 + cnt_ge
        if rank <= 2:
            top2_count += 1
        rank_percentiles.append(float(rank) / float(n_agents))
    win_rate = float(win_count) / float(n_windows) if n_windows else 0.0
    top2_rate = float(top2_count) / float(n_windows) if n_windows else 0.0
    median_rank_percentile = float(_stats.median(rank_percentiles)) if rank_percentiles else 0.0
    return FieldRelativeReport(
        n_windows=int(n_windows),
        n_agents=int(n_agents),
        n_effective=int(n_effective),
        win_rate=float(win_rate),
        top2_rate=float(top2_rate),
        median_rank_percentile=float(median_rank_percentile),
    )
