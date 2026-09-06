# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.tournament.objective.reports import ChampionshipAdoptionResult, ChampionshipObjectiveConfig, GROSS_METRIC_UNAVAILABLE, championship_tail_report, paired_scenario_delta_ci


def evaluate_championship_adoption(
    *,
    candidate_returns: Sequence[float],
    incumbent_returns: Sequence[float],
    raw_returns: Sequence[float],
    horizon: int,
    config: ChampionshipObjectiveConfig,
    execution_parity: bool,
    gross_violation_count: int | None,
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
    if gross_violation_count is None:
        return ChampionshipAdoptionResult(
            status="INSUFFICIENT_EVIDENCE",
            failures=(GROSS_METRIC_UNAVAILABLE,),
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
