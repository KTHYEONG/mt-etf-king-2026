from __future__ import annotations


def test_champion_research_cli_preserves_p27_when_candidate_research_only(monkeypatch, tmp_path) -> None:
    import argparse
    from src.cli.commands import champion_research as _cr
    from src.cli.commands.champion_research import cmd_champion_research
    from src.cli.constants import CHAMPION_STRATEGY
    from src.strategies.ids import STICKY_MOM60_RAW

    monkeypatch.setattr(_cr, '_build_champion_research_inputs', lambda _: {})
    monkeypatch.setattr(_cr, 'run_champion_walk_forward', lambda **_: type('Result', (), {'status': 'RESEARCH_ONLY', 'write': lambda self, _: tmp_path / 'promotion.json'})())
    args = argparse.Namespace(start='2024-01-02', end='2026-08-27', log_level='ERROR', trace=False)
    result = cmd_champion_research(args)

    assert result == 0
    assert CHAMPION_STRATEGY == STICKY_MOM60_RAW


def test_champion_research_cli_passes_real_runtime(monkeypatch) -> None:
    import argparse

    from src.cli.commands import champion_research as _cr
    from src.cli.commands.champion_research import cmd_champion_research
    from src.tournament.champion_eval import ChampionEvaluation

    captured: dict[str, object] = {}
    monkeypatch.setattr(_cr, "_build_champion_research_inputs", lambda _: {"runtime": object()})
    monkeypatch.setattr(_cr, "run_champion_walk_forward", lambda **kwargs: captured.update(kwargs) or ChampionEvaluation())

    assert cmd_champion_research(argparse.Namespace(start="2024-01-02", end="2026-08-27", log_level="ERROR", trace=False)) == 0
    assert "runtime" in captured


def test_champion_research_requires_explicit_p27_matched_mode() -> None:
    import argparse

    import pytest

    from src.cli.commands.champion_research import _build_champion_research_inputs

    with pytest.raises(ValueError, match='candidate_mode'):
        _build_champion_research_inputs(argparse.Namespace(start='2026-01-02', end='2026-08-27', candidate_mode='invalid'))


def test_champion_research_filters_match_executable_labels() -> None:
    from src.cli.commands.champion_research import build_champion_research_filters
    from src.universe.provider import LiquidityAdmissionMode, UniverseFilters

    result = build_champion_research_filters(UniverseFilters(capital=1_000_000_000), "adaptive_specialists")
    assert result.liquidity_admission is LiquidityAdmissionMode.STAGED_EXECUTION
    assert result.max_order_to_adv == 0.01
    assert result.max_position_weight == 0.80
    assert result.allow_leverage and result.allow_inverse
