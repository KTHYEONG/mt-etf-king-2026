from __future__ import annotations


def test_champion_research_inputs_builds_real_adaptive_runtime() -> None:
    import argparse
    from src.cli.commands.champion_research import _build_champion_research_inputs
    from src.universe.provider import LiquidityAdmissionMode

    args = argparse.Namespace(start="2018-01-02", end="2026-08-27", candidate_mode="adaptive_specialists")
    runtime = _build_champion_research_inputs(args)["runtime"]
    filters = runtime.backtest_config.filters
    assert runtime.candidate_mode == "adaptive_specialists"
    assert runtime.dataset_config.label_horizon == 36
    assert filters.liquidity_admission is LiquidityAdmissionMode.STAGED_EXECUTION
    assert filters.max_order_to_adv == 0.01
    assert filters.max_position_weight == 0.80
    assert filters.allow_leverage and filters.allow_inverse


def test_adaptive_specialists_real_runtime_evaluates_all_windows() -> None:
    import argparse
    from src.cli.commands.champion_research import _build_champion_research_inputs
    from src.tournament.adaptive_specialists import run_adaptive_specialist_research

    args = argparse.Namespace(start="2018-01-02", end="2026-08-27", candidate_mode="adaptive_specialists")
    runtime = _build_champion_research_inputs(args)["runtime"]
    result = run_adaptive_specialist_research(runtime)
    assert result.status == "RESEARCH_ONLY"
    assert result.extra["eligible_window_count"] == 2088
    assert result.extra["evaluated_window_count"] == 2088
    assert result.extra["router_reset_count"] == 2088
    assert result.extra["controller_decision_count"] == 2088 * 36
    assert result.extra["shadow_transition_count"] == 2088 * 36 * 4
    assert result.extra["ledger_transition_parity"] is True
    assert result.artifact_integrity is bool(result.extra["artifact_integrity"])
