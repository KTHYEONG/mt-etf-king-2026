def test_p28b_cli_championship_wires_p27_champion() -> None:
    import re
    from pathlib import Path

    from src.cli import STICKY_ADOPTION_MODELS
    from src.cli.commands.backtest.families.sticky_cash import _hook_mom60_abs_cash

    hook_src = Path("src/cli/commands/backtest/families/sticky_cash.py").read_text(encoding="utf-8")
    assert "sticky.mom60_abs_cash" in STICKY_ADOPTION_MODELS
    block = inspect_block(hook_src, "def _hook_mom60_abs_cash")
    assert "evaluate_championship_adoption" in block
    assert "sticky.mom60_raw" in block
    assert "run_rolling" in block
    assert re.search(r"^\s*_ = diagnostics\b", block, flags=re.M) is None


def inspect_block(source: str, marker: str) -> str:
    idx = source.find(marker)
    assert idx > 0
    return source[idx : idx + 8000]
