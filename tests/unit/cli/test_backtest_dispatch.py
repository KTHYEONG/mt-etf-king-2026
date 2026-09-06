import ast
import inspect
from pathlib import Path


def test_cmd_backtest_dispatches_by_family_not_by_id() -> None:
    from src.cli.commands.backtest import FAMILY_RUNNERS, cmd_backtest

    source = inspect.getsource(cmd_backtest)
    tree = ast.parse(source)
    statements = sum(1 for node in ast.walk(tree) if isinstance(node, ast.stmt))
    assert statements <= 80, f"cmd_backtest still holds {statements} statements"

    compares = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare) and any(isinstance(c, ast.Constant) and isinstance(c.value, str) for c in node.comparators)
    ]
    assert compares == [], "cmd_backtest still branches on literal strategy ids"

    assert set(FAMILY_RUNNERS) >= {"baseline", "portfolio", "sticky", "convex"}
    assert Path("src/cli/_impl.py").exists() is False


def test_family_runners_are_real_implementations() -> None:
    from src.cli.commands.backtest import FAMILY_RUNNERS

    assert FAMILY_RUNNERS, "family runner table is empty"
    for family, runner in sorted(FAMILY_RUNNERS.items()):
        assert callable(runner), family
        assert runner.__name__ != "<lambda>", f"{family} is a placeholder lambda"
        body = inspect.getsource(runner)
        statements = body.count("\n")
        assert statements > 3, f"{family} runner is a stub"
        signature = inspect.signature(runner)
        assert list(signature.parameters) == ["ctx"], f"{family} runner signature drifted"
