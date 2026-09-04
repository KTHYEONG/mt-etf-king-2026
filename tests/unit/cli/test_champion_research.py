from __future__ import annotations


def test_champion_research_cli_preserves_p27_when_candidate_research_only(monkeypatch, tmp_path) -> None:
    import argparse
    from src.cli.constants import CHAMPION_STRATEGY
    from src.cli._impl import cmd_champion_research
    from src.strategies.ids import STICKY_MOM60_RAW

    monkeypatch.setattr('src.cli._impl._build_champion_research_inputs', lambda _: {})
    monkeypatch.setattr('src.cli._impl.run_champion_walk_forward', lambda **_: type('Result', (), {'status': 'RESEARCH_ONLY', 'write': lambda self, _: tmp_path / 'promotion.json'})())
    args = argparse.Namespace(start='2024-01-02', end='2026-08-27', log_level='ERROR', trace=False)
    result = cmd_champion_research(args)

    assert result == 0
    assert CHAMPION_STRATEGY == STICKY_MOM60_RAW
