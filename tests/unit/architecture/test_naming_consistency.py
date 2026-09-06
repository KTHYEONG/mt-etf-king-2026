import ast
from pathlib import Path

BANNED = {"model_key": "strategy_id", "strategy_key": "strategy_id", "as_of": "decision_date", "cal": "calendar"}


def test_no_banned_parameter_aliases() -> None:
    offenders: list[str] = []
    for path in sorted(Path("src").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if node.name.startswith("_"):
                continue
            args = node.args
            all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
            offenders.extend(
                f"{path}:{node.lineno}:{node.name}({arg.arg}) -> use {BANNED[arg.arg]}"
                for arg in all_args
                if arg.arg in BANNED
            )

    assert offenders == [], "parameter naming drift: " + ", ".join(offenders[:20])


BANNED_FLOATS = {0.30, 0.40, 0.50, 0.60}


def test_no_hardcoded_gate_thresholds() -> None:
    offenders: list[str] = []
    for path in sorted(Path("src/tournament").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            value = node.value
            if isinstance(value, bool):
                continue
            if isinstance(value, float) and value in BANNED_FLOATS:
                offenders.append(f"{path}:{node.lineno}:{value}")
            if isinstance(value, int) and value == 36:
                offenders.append(f"{path}:{node.lineno}:36")

    assert offenders == [], "gate constants must come from configs/: " + ", ".join(offenders[:20])
