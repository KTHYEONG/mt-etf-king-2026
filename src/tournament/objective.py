# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.tournament.distribution import ReturnDistribution, ruin_probability


@dataclass(frozen=True)
class ObjectiveGateConfig:
    g1_prob_threshold: float
    g1_min_improvement: float
    g2a_ruin_threshold: float
    g2a_max_prob: float

    @classmethod
    def from_yaml(cls, path: Path) -> ObjectiveGateConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        gates = raw.get("gates") if isinstance(raw, dict) else None
        if gates is None:
            gates = raw if isinstance(raw, dict) else {}
        # Expected keys: g1_prob_threshold, g1_min_improvement, g2a_ruin_threshold, g2a_max_prob
        g1_prob = float(gates.get("g1_prob_threshold", 0.30))
        g1_min = float(gates.get("g1_min_improvement", 0.02))
        g2a_thresh = float(gates.get("g2a_ruin_threshold", -0.25))
        g2a_max = float(gates.get("g2a_max_prob", 0.05))
        return cls(g1_prob_threshold=g1_prob, g1_min_improvement=g1_min, g2a_ruin_threshold=g2a_thresh, g2a_max_prob=g2a_max)


@dataclass(frozen=True)
class ObjectiveGateResult:
    status: str
    failures: tuple[str, ...]
    p_tail: float
    baseline_p_tail: float
    ruin_probability: float
    paired_delta_ci: tuple[float, float] | None


def paired_tail_delta_ci(
    candidate_returns: Sequence[float],
    control_returns: Sequence[float],
    threshold: float,
    *,
    expected_block: int,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    if not candidate_returns or not control_returns:
        return (0.0, 0.0)
    n = min(len(candidate_returns), len(control_returns))
    if n == 0:
        return (0.0, 0.0)
    if expected_block <= 0:
        raise ValueError("expected_block must be >0")
    p = 1.0 / float(expected_block)
    rng = random.Random(seed)
    # Build paired differences of indicator
    diffs: list[float] = []
    for i in range(n):
        c = float(candidate_returns[i]) > float(threshold)
        b = float(control_returns[i]) > float(threshold)
        diffs.append(float(int(c) - int(b)))
    # stationary bootstrap on diffs circular
    base = float(sum(diffs) / len(diffs)) if diffs else 0.0
    stats: list[float] = []
    arr = list(diffs)
    for _ in range(n_resamples):
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
                idx = (start + k) % n
                sample.append(arr[idx])
        sample = sample[:n]
        try:
            s = float(sum(sample) / len(sample)) if sample else 0.0
        except Exception:
            s = 0.0
        stats.append(s)
    stats.sort()
    lower_idx = int(math.floor(0.025 * n_resamples))
    upper_idx = int(math.ceil(0.975 * n_resamples)) - 1
    lower_idx = max(0, min(lower_idx, n_resamples - 1))
    upper_idx = max(0, min(upper_idx, n_resamples - 1))
    lower = float(stats[lower_idx])
    upper = float(stats[upper_idx])
    if lower > upper:
        lower, upper = upper, lower
    return (lower, upper)


def evaluate_objective_gates(
    candidate: ReturnDistribution,
    baseline_b0: ReturnDistribution,
    config: ObjectiveGateConfig,
    *,
    paired_delta_ci: tuple[float, float] | None = None,
) -> ObjectiveGateResult:
    # INSUFFICIENT handling
    if candidate.n_effective < 1 or baseline_b0.n_effective < 1:
        return ObjectiveGateResult(
            status="INSUFFICIENT_EVIDENCE",
            failures=("INSUFFICIENT_SAMPLE",),
            p_tail=0.0,
            baseline_p_tail=0.0,
            ruin_probability=1.0,
            paired_delta_ci=paired_delta_ci,
        )
    # Check missing lineage/windows artifacts? For this function, we consider paired_delta_ci None but candidate returns maybe empty?
    # Actually if candidate returns empty, treat insufficient
    if not candidate.returns or not baseline_b0.returns:
        return ObjectiveGateResult(
            status="INSUFFICIENT_EVIDENCE",
            failures=("MISSING_ARTIFACT",),
            p_tail=0.0,
            baseline_p_tail=0.0,
            ruin_probability=1.0,
            paired_delta_ci=paired_delta_ci,
        )
    # Compute P(R > g1 threshold)
    thresh = float(config.g1_prob_threshold)
    # exceedance probability
    p_tail = float(sum(1 for r in candidate.returns if float(r) > thresh) / len(candidate.returns)) if candidate.returns else 0.0
    baseline_p = float(sum(1 for r in baseline_b0.returns if float(r) > thresh) / len(baseline_b0.returns)) if baseline_b0.returns else 0.0
    ruin_prob = ruin_probability(candidate.returns, float(config.g2a_ruin_threshold))
    failures: list[str] = []
    # G1: candidate P >= baseline + min_improvement
    if not (p_tail >= baseline_p + float(config.g1_min_improvement) - 1e-12):
        failures.append("G1_TAIL")
    # G2a: ruin <= max_prob
    # If ruin > max, fail with G2A_RUIN
    if not (ruin_prob <= float(config.g2a_max_prob) + 1e-12):
        failures.append("G2A_RUIN")
    status = "PASS" if not failures else "FAIL"
    # If failures empty but missing artifacts would have been handled above, return PASS else FAIL
    return ObjectiveGateResult(
        status=status,
        failures=tuple(failures),
        p_tail=float(p_tail),
        baseline_p_tail=float(baseline_p),
        ruin_probability=float(ruin_prob),
        paired_delta_ci=paired_delta_ci,
    )


@dataclass(frozen=True)
class P15AdoptionReport:
    status: str
    failures: tuple[str, ...]
    objective: ObjectiveGateResult | None = None


def _dist_exceedance(dist: ReturnDistribution, threshold: float) -> float:
    if not dist.exceedance:
        return 0.0
    direct = dist.exceedance.get(threshold)
    if direct is not None:
        return float(direct)
    for key, value in dist.exceedance.items():
        try:
            if abs(float(key) - float(threshold)) < 1e-9:
                return float(value)
        except Exception:
            continue
    return 0.0


def evaluate_p15_adoption_report(
    *,
    p15: ReturnDistribution,
    b1: ReturnDistribution,
    b0: ReturnDistribution,
    p14: ReturnDistribution,
    config: ObjectiveGateConfig,
    max_positive_family_count: int,
    multi_family_rate: float,
    artifacts_complete: bool,
    leverage_scenarios: Sequence[str],
    era_p40_deltas: Mapping[str, float] | None = None,
    era_b1_tail_event_counts: Mapping[str, int] | None = None,
) -> P15AdoptionReport:
    failures: list[str] = []
    if not artifacts_complete:
        failures.append("MISSING_ARTIFACT")
    if (
        p15.n_effective < 1
        or b1.n_effective < 1
        or b0.n_effective < 1
        or p14.n_effective < 1
        or not p15.returns
        or not b1.returns
        or not b0.returns
        or not p14.returns
    ):
        return P15AdoptionReport(
            status="INSUFFICIENT_EVIDENCE",
            failures=tuple(failures or ("INSUFFICIENT_SAMPLE",)),
            objective=None,
        )
    if set(leverage_scenarios) < {"aggressive", "conservative"}:
        failures.append("LEVERAGE_SCENARIOS")
    p30_p15 = _dist_exceedance(p15, 0.30)
    p40_p15 = _dist_exceedance(p15, 0.40)
    p30_b1 = _dist_exceedance(b1, 0.30)
    p40_b1 = _dist_exceedance(b1, 0.40)
    if p40_p15 < p40_b1 + 0.02 - 1e-12:
        failures.append("P40_VS_B1")
    if p30_p15 < p30_b1 - 1e-12:
        failures.append("P30_VS_B1")
    ruin_p15 = ruin_probability(p15.returns, config.g2a_ruin_threshold)
    ruin_p14 = ruin_probability(p14.returns, config.g2a_ruin_threshold)
    if ruin_p15 > float(config.g2a_max_prob) + 1e-12:
        failures.append("G2A_RUIN")
    if ruin_p15 > ruin_p14 + 0.01 + 1e-12:
        failures.append("RUIN_VS_P14")
    if max_positive_family_count > 1:
        failures.append("FAMILY_COUNT")
    if multi_family_rate > 1e-12:
        failures.append("MULTI_FAMILY")
    if era_p40_deltas and era_b1_tail_event_counts:
        for era, delta in era_p40_deltas.items():
            if int(era_b1_tail_event_counts.get(era, 0)) >= 5 and float(delta) < -1e-12:
                failures.append(f"ERA_{era}")
    objective = evaluate_objective_gates(p15, b0, config)
    if objective.status == "INSUFFICIENT_EVIDENCE":
        failures.append("INSUFFICIENT_SAMPLE")
    elif objective.status == "FAIL":
        failures.extend(list(objective.failures))
    if failures:
        status = (
            "INSUFFICIENT_EVIDENCE"
            if all(f in {"MISSING_ARTIFACT", "INSUFFICIENT_SAMPLE"} for f in failures)
            else "FAIL"
        )
        return P15AdoptionReport(status=status, failures=tuple(failures), objective=objective)
    return P15AdoptionReport(status="PASS", failures=(), objective=objective)


@dataclass(frozen=True)
class P16AdoptionReport:
    status: str
    failures: tuple[str, ...]
    objective: ObjectiveGateResult | None = None


def evaluate_p16_adoption_report(
    *,
    p16: ReturnDistribution,
    b1: ReturnDistribution,
    b0: ReturnDistribution,
    p14: ReturnDistribution,
    config: ObjectiveGateConfig,
    artifacts_complete: bool,
    leverage_scenarios: Sequence[str],
    skip_capacity_violations: int,
    vehicle_mult2_rate: float,
) -> P16AdoptionReport:
    failures: list[str] = []
    if not artifacts_complete:
        failures.append("MISSING_ARTIFACT")
    if (
        p16.n_effective < 1
        or b1.n_effective < 1
        or b0.n_effective < 1
        or p14.n_effective < 1
        or not p16.returns
        or not b1.returns
        or not b0.returns
        or not p14.returns
    ):
        return P16AdoptionReport(
            status="INSUFFICIENT_EVIDENCE",
            failures=tuple(failures or ("INSUFFICIENT_SAMPLE",)),
            objective=None,
        )
    if not {"aggressive", "conservative"} <= set(leverage_scenarios):
        failures.append("LEVERAGE_SCENARIOS")
    p30_p16 = _dist_exceedance(p16, 0.30)
    p40_p16 = _dist_exceedance(p16, 0.40)
    p50_p16 = _dist_exceedance(p16, 0.50)
    p30_b1 = _dist_exceedance(b1, 0.30)
    p40_b1 = _dist_exceedance(b1, 0.40)
    p50_b1 = _dist_exceedance(b1, 0.50)
    if p30_p16 < p30_b1 - 1e-12:
        failures.append("P30_VS_B1")
    if p40_p16 < p40_b1 - 1e-12:
        failures.append("P40_VS_B1")
    if p50_p16 < p50_b1 - 1e-12:
        failures.append("P50_VS_B1")
    ruin_p16 = ruin_probability(p16.returns, config.g2a_ruin_threshold)
    ruin_p14 = ruin_probability(p14.returns, config.g2a_ruin_threshold)
    if ruin_p16 > float(config.g2a_max_prob) + 1e-12:
        failures.append("G2A_RUIN")
    if ruin_p16 > ruin_p14 + 0.01 + 1e-12:
        failures.append("RUIN_VS_P14")
    if int(skip_capacity_violations) > 0:
        failures.append("CAPACITY_ON_CONVEXITY")
    if float(vehicle_mult2_rate) < 0.25 - 1e-12:
        failures.append("VEHICLE_ACTIVITY")
    objective = evaluate_objective_gates(p16, b0, config)
    if objective.status == "INSUFFICIENT_EVIDENCE":
        failures.append("INSUFFICIENT_SAMPLE")
    elif objective.status == "FAIL":
        failures.extend(list(objective.failures))
    if failures:
        status = (
            "INSUFFICIENT_EVIDENCE"
            if all(f in {"MISSING_ARTIFACT", "INSUFFICIENT_SAMPLE"} for f in failures)
            else "FAIL"
        )
        return P16AdoptionReport(status=status, failures=tuple(failures), objective=objective)
    return P16AdoptionReport(status="PASS", failures=(), objective=objective)


@dataclass(frozen=True)
class ChampionshipObjectiveConfig:
    thresholds: tuple[float, ...]
    scenario_weights: Mapping[str, tuple[float, ...]]
    primary_scenario: str
    ruin_threshold: float
    ruin_max: float
    max_effective_gross: float
    bootstrap_expected_block: int
    bootstrap_resamples: int
    seed: int
    min_era_effective: int

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
        if beb_i < 36:
            raise ValueError("bootstrap_expected_block must be >=36")
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


def evaluate_championship_adoption(
    *,
    candidate_returns: Sequence[float],
    incumbent_returns: Sequence[float],
    raw_returns: Sequence[float],
    horizon: int,
    config: ChampionshipObjectiveConfig,
    execution_parity: bool,
    gross_violation_count: int,
    era_pairs: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
) -> ChampionshipAdoptionResult:
    import math as _math

    failures: list[str] = []
    # fail-closed missing or non-finite
    try:
        if not candidate_returns or not incumbent_returns or not raw_returns:
            return ChampionshipAdoptionResult(
                status="INSUFFICIENT_EVIDENCE",
                failures=("MISSING_ARTIFACT",),
                candidate=None,
                incumbent=None,
                raw=None,
                scenario_delta_ci={},
                era_deltas={},
            )
        # check non-finite
        for seq in (candidate_returns, incumbent_returns, raw_returns):
            for v in seq:
                fv = float(v)  # type: ignore[arg-type]
                if not _math.isfinite(fv):
                    return ChampionshipAdoptionResult(
                        status="INSUFFICIENT_EVIDENCE",
                        failures=("MISSING_ARTIFACT",),
                        candidate=None,
                        incumbent=None,
                        raw=None,
                        scenario_delta_ci={},
                        era_deltas={},
                    )
    except Exception:
        return ChampionshipAdoptionResult(
            status="INSUFFICIENT_EVIDENCE",
            failures=("MISSING_ARTIFACT",),
            candidate=None,
            incumbent=None,
            raw=None,
            scenario_delta_ci={},
            era_deltas={},
        )
    # length check - require same length for paired comparisons? If mismatch, INSUFFICIENT
    if len(candidate_returns) != len(incumbent_returns) or len(candidate_returns) != len(raw_returns):
        return ChampionshipAdoptionResult(
            status="INSUFFICIENT_EVIDENCE",
            failures=("MISSING_ARTIFACT",),
            candidate=None,
            incumbent=None,
            raw=None,
            scenario_delta_ci={},
            era_deltas={},
        )
    try:
        h = int(horizon)
        if h <= 0:
            raise ValueError
    except Exception:
        return ChampionshipAdoptionResult(
            status="INSUFFICIENT_EVIDENCE",
            failures=("MISSING_ARTIFACT",),
            candidate=None,
            incumbent=None,
            raw=None,
            scenario_delta_ci={},
            era_deltas={},
        )
    if config.bootstrap_expected_block < h:
        failures.append("BLOCK_SIZE")
    # execution parity
    if not bool(execution_parity):
        failures.append("EXECUTION_PARITY")
    # gross violation
    try:
        gvc = int(gross_violation_count)  # type: ignore[arg-type]
    except Exception:
        gvc = 1
    if gvc != 0:
        failures.append("GROSS_EXPOSURE")
    # tail reports (may raiseValueError for empty/non-finite already handled)
    try:
        cand_report = championship_tail_report(candidate_returns, h, config)
        inc_report = championship_tail_report(incumbent_returns, h, config)
        raw_report = championship_tail_report(raw_returns, h, config)
    except Exception as exc:
        return ChampionshipAdoptionResult(
            status="INSUFFICIENT_EVIDENCE",
            failures=("MISSING_ARTIFACT",),
            candidate=None,
            incumbent=None,
            raw=None,
            scenario_delta_ci={},
            era_deltas={},
        )
    # ruin check
    if cand_report.ruin_probability > float(config.ruin_max) + 1e-12:
        failures.append("RUIN")
    # scenario non-inferiority: candidate must be >= incumbent and >= raw for all scenarios
    for scen in config.scenario_weights.keys():
        cand_s = float(cand_report.scenario_scores.get(scen, 0.0))
        inc_s = float(inc_report.scenario_scores.get(scen, 0.0))
        raw_s = float(raw_report.scenario_scores.get(scen, 0.0))
        if cand_s + 1e-12 < inc_s:
            failures.append(f"SCENARIO_{scen.upper()}_VS_INCUMBENT")
        if cand_s + 1e-12 < raw_s:
            failures.append(f"SCENARIO_{scen.upper()}_VS_RAW")
    # primary paired CI lower >=0 vs incumbent and vs raw
    scenario_delta_ci: dict[str, tuple[float, float]] = {}
    era_deltas: dict[str, float] = {}
    try:
        weights_primary = config.scenario_weights[config.primary_scenario]
        thresholds = config.thresholds
        ci_inc = paired_scenario_delta_ci(
            candidate_returns,
            incumbent_returns,
            thresholds,
            weights_primary,
            expected_block=int(config.bootstrap_expected_block),
            n_resamples=int(config.bootstrap_resamples),
            seed=int(config.seed),
        )
        ci_raw = paired_scenario_delta_ci(
            candidate_returns,
            raw_returns,
            thresholds,
            weights_primary,
            expected_block=int(config.bootstrap_expected_block),
            n_resamples=int(config.bootstrap_resamples),
            seed=int(config.seed),
        )
        scenario_delta_ci[config.primary_scenario] = ci_inc
        # also store raw comparison under different key for completeness
        scenario_delta_ci[f"{config.primary_scenario}_vs_raw"] = ci_raw
        if ci_inc[0] < -1e-12:
            failures.append("PRIMARY_CI_VS_INCUMBENT")
        if ci_raw[0] < -1e-12:
            failures.append("PRIMARY_CI_VS_RAW")
    except Exception:
        failures.append("CI_ERROR")
        scenario_delta_ci = {}
    # era deltas
    if era_pairs:
        for era, pair in era_pairs.items():
            try:
                cand_era, inc_era = pair  # type: ignore[misc]
                cand_era = list(cand_era)  # type: ignore[arg-type]
                inc_era = list(inc_era)  # type: ignore[arg-type]
                # compute effective
                from src.tournament.distribution import effective_sample_size

                n_eff = int(effective_sample_size(len(cand_era), h))
                if n_eff < int(config.min_era_effective):
                    continue
                # scenario score for primary scenario in era
                # compute exceedance for era
                # quick: use championship_tail_report era? Instead compute primary weighted score
                cand_exceed: dict[float, float] = {}
                inc_exceed: dict[float, float] = {}
                for thr in config.thresholds:
                    ft = float(thr)
                    cand_exceed[ft] = float(sum(1 for r in cand_era if float(r) > ft) / len(cand_era)) if cand_era else 0.0
                    inc_exceed[ft] = float(sum(1 for r in inc_era if float(r) > ft) / len(inc_era)) if inc_era else 0.0
                w_primary = config.scenario_weights[config.primary_scenario]
                cand_score_era = sum(float(w) * cand_exceed[float(thr)] for thr, w in zip(config.thresholds, w_primary))
                inc_score_era = sum(float(w) * inc_exceed[float(thr)] for thr, w in zip(config.thresholds, w_primary))
                delta = float(cand_score_era - inc_score_era)
                era_deltas[str(era)] = float(delta)
                if delta < -1e-12:
                    failures.append(f"ERA_{str(era).upper()}")
            except Exception:
                failures.append(f"ERA_{str(era).upper()}_ERROR")
                continue
    # dedup failures preserve order
    seen = set()
    uniq_failures: list[str] = []
    for f in failures:
        if f not in seen:
            seen.add(f)
            uniq_failures.append(f)
    if uniq_failures:
        # if any failure besides block size, status FAIL; if only missing artifact? Already handled
        return ChampionshipAdoptionResult(
            status="FAIL",
            failures=tuple(uniq_failures),
            candidate=cand_report,
            incumbent=inc_report,
            raw=raw_report,
            scenario_delta_ci=dict(scenario_delta_ci),
            era_deltas=dict(era_deltas),
        )
    return ChampionshipAdoptionResult(
        status="PASS",
        failures=(),
        candidate=cand_report,
        incumbent=inc_report,
        raw=raw_report,
        scenario_delta_ci=dict(scenario_delta_ci),
        era_deltas=dict(era_deltas),
    )


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
