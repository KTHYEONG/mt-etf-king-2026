# ruff: noqa
from pathlib import Path


def test_championship_config_loads_curve_and_portfolio_limits() -> None:
    from pathlib import Path

    from src.tournament.objective import ChampionshipObjectiveConfig

    config = ChampionshipObjectiveConfig.from_yaml(Path('configs/gates.yaml'), Path('configs/portfolio.yaml'))

    assert config.thresholds == (0.30, 0.40, 0.50, 0.60)
    assert config.primary_scenario == 'championship'
    assert abs(sum(config.scenario_weights['championship']) - 1.0) < 1e-12
    assert config.bootstrap_expected_block >= 36
    assert config.max_effective_gross == 1.60


def test_championship_tail_report_preserves_full_curve() -> None:
    from pathlib import Path
    from dataclasses import replace

    from src.tournament.objective import ChampionshipObjectiveConfig, championship_tail_report

    config = ChampionshipObjectiveConfig.from_yaml(Path('configs/gates.yaml'), Path('configs/portfolio.yaml'))
    config = replace(config, bootstrap_expected_block=2, bootstrap_resamples=100)
    returns = [0.65, 0.55, 0.45, 0.35, 0.0, -0.30] * 4
    report = championship_tail_report(returns, 2, config)

    assert tuple(report.exceedance) == config.thresholds
    assert tuple(report.exceedance_ci) == config.thresholds
    assert set(report.scenario_scores) == set(config.scenario_weights)
    assert report.exceedance[0.30] > report.exceedance[0.60]
    assert report.ruin_probability > 0.0


def test_championship_adoption_requires_parity_and_zero_gross_violations() -> None:
    from dataclasses import replace
    from pathlib import Path

    from src.tournament.objective import ChampionshipObjectiveConfig, evaluate_championship_adoption

    config = ChampionshipObjectiveConfig.from_yaml(Path('configs/gates.yaml'), Path('configs/portfolio.yaml'))
    config = replace(config, bootstrap_expected_block=2, bootstrap_resamples=100, min_era_effective=1)
    candidate = [0.65] * 20
    incumbent = [0.0] * 20
    result = evaluate_championship_adoption(candidate_returns=candidate, incumbent_returns=incumbent, raw_returns=incumbent, horizon=2, config=config, execution_parity=False, gross_violation_count=1)

    assert result.status == 'FAIL'
    assert 'EXECUTION_PARITY' in result.failures
    assert 'GROSS_EXPOSURE' in result.failures


def test_championship_adoption_uses_incumbent_and_primary_ci() -> None:
    from dataclasses import replace
    from pathlib import Path

    from src.tournament.objective import ChampionshipObjectiveConfig, evaluate_championship_adoption

    config = ChampionshipObjectiveConfig.from_yaml(Path('configs/gates.yaml'), Path('configs/portfolio.yaml'))
    config = replace(config, bootstrap_expected_block=2, bootstrap_resamples=100, min_era_effective=1)
    candidate = [0.65] * 20
    incumbent = [0.0] * 20
    result = evaluate_championship_adoption(candidate_returns=candidate, incumbent_returns=incumbent, raw_returns=incumbent, horizon=2, config=config, execution_parity=True, gross_violation_count=0)

    assert result.status == 'PASS'
    assert result.failures == ()
    assert result.scenario_delta_ci[config.primary_scenario][0] > 0.0
