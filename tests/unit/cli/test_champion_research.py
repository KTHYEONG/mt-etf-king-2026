# ruff: noqa
from __future__ import annotations


def test_champion_research_cli_preserves_p27_when_candidate_research_only(monkeypatch, tmp_path) -> None:
    import argparse
    from src.cli.commands import champion_research as _cr
    from src.cli.commands.champion_research import cmd_champion_research
    from src.cli.constants import CHAMPION_STRATEGY
    from src.strategies.ids import STICKY_MOM60_POST_CRASH_ANCHOR

    monkeypatch.setattr(_cr, '_build_champion_research_inputs', lambda _: {})
    monkeypatch.setattr(_cr, 'run_champion_walk_forward', lambda **_: type('Result', (), {'status': 'RESEARCH_ONLY', 'write': lambda self, _: tmp_path / 'promotion.json'})())
    args = argparse.Namespace(start='2024-01-02', end='2026-08-27', log_level='ERROR', trace=False)
    result = cmd_champion_research(args)

    assert result == 0
    assert CHAMPION_STRATEGY == STICKY_MOM60_POST_CRASH_ANCHOR


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

def test_champion_research_frontier_cli_real_local_data(tmp_path) -> None:
    from datetime import date
    import polars as pl
    from src.core.calendar import get_calendar
    from src.tournament.frontier import run_frontier_research
    sessions=get_calendar().sessions(date(2024,1,2),date(2024,6,28))
    panel=pl.DataFrame({"date":sessions,"ticker":["069500"]*len(sessions),"name":["KODEX 200"]*len(sessions),"underlying_index_name":["KOSPI 200"]*len(sessions),"open":[100.0]*len(sessions),"close":[100.0]*len(sessions),"trading_value":[1e12]*len(sessions),"is_tradable":[True]*len(sessions)})
    import argparse
    import json
    from src.cli.commands.champion_research import cmd_champion_research
    data_root=tmp_path/"data"
    (data_root/"normalized").mkdir(parents=True)
    panel.write_parquet(data_root/"normalized"/"etf_daily.parquet")
    out=tmp_path/"frontier"
    args=argparse.Namespace(candidate_mode="capacity_frontier",start=str(sessions[0]),end=str(sessions[-1]),data_root=str(data_root),output=str(out))
    assert cmd_champion_research(args) == 0
    report=json.loads((out/"report.json").read_text())
    assert report["status"] == "RESEARCH_ONLY"
    assert report["session_count"] == len(sessions)
    windows=pl.read_parquet(out/"windows.parquet")
    assert windows.height == len(sessions)-35
    assert not (out/"promotion.json").exists()

def test_champion_research_frontier_rejects_missing_data(tmp_path) -> None:
    import argparse
    from src.cli.commands.champion_research import cmd_champion_research
    args=argparse.Namespace(candidate_mode="capacity_frontier",start="2024-01-02",end="2024-12-30",data_root=str(tmp_path/"absent"),output=str(tmp_path/"out"))
    assert cmd_champion_research(args) == 1
    assert not (tmp_path/"out"/"report.json").exists()
